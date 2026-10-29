# DMA ring 时序模型与 runtime 接口

本次修改包含 gem5 C++ 时序模型、SimObject 参数和 runtime 接口。
独立构建目标为 `build/qwen_DMA_ring/gem5.opt`。新模型的微测试不能当成
Qwen prefill/decode 的性能结果；已有冻结实验二进制保持原样。

## 实现范围

CPU 在 DDR 中预建 descriptor，通过一次 ring tail / doorbell 发布批次。
DMA 通过真实 timing system port 自动读取 descriptor，经过有限队列和依赖检查后执行。
descriptor 读取具有 cache 行切分、访存时延和 outstanding 占用，不使用 functional read。
满足依赖的 descriptor 可并发，输入和输出共享系统总线及全局 credit。
调度每个 DMA 周期最多发出 `issue_width` 个 payload chunk，在活动 descriptor 间轮转；
bank credit 不足时尝试其他 descriptor，SPM 背压遵循 timing retry 协议。

AME、RVV 仍只访问 local SPM。事件为 DMA 可见的 64 位 level bitset：
runtime 读取输入完成事件，AME 指令完成后由 CPU 写 event set，解除输出 DMA 依赖。
**尚未实现 AME 硬件 command queue、DMA→AME 直接唤醒或中断线。**
因此不能宣称整个阶段 CPU 只参与一次提交与一次最终等待。

## 参数与时序口径

| 参数 / CLI | 默认值 | 含义 |
|---|---:|---|
| `burst_bytes` / `--dma-burst` | 64B | 请求大小上限，仍按 DDR cache line、burst 边界、SPM stripe、row 边界切分 |
| `descriptor_queue_depth` / `--dma-queue-depth` | 8 | queued、active、取回中及等待顺序退休的已完成 descriptor 共享额度 |
| `max_active_descriptors` / `--dma-active-descriptors` | 1 | 同时工作 descriptor 数；设为 2 可允许输入/输出重叠 |
| `max_outstanding_system` / `--dma-outstanding` | 16 | payload 从首次发出至 DDR/SPM 两段都完成的额度；ring 取回占其中 1 个额度 |
| `max_outstanding_per_bank` / `--dma-bank-outstanding` | 4 | 每个 SPM bank 在途 payload chunk 上限 |
| `issue_width` / `--dma-issue-width` | 16 | 每 DMA clock 的 payload chunk 发出上限 |

outstanding 为保守的端到端 credit，不等于 DDR 控制器实际排队请求数。
使用 gem5 packet/timing 机制，没有新增逐 beat AXI 信号级模型。
当前 cache line 和 bank stripe 均为 64B，仅将 burst 上限改为 128B 不会生成
128B payload packet，不能预设带宽翻倍。128B ring descriptor 分块顺序取回，
全部到达后才入队。

## MMIO ABI v1

基地址 `0x40010000`，寄存器为 64 位。旧 `0x00–0x58` 接口保留。

| 偏移 | 寄存器 | 语义 |
|---|---|---|
| `0x08` | STATUS | bit0 busy，bit1 done，bit2 sticky error，bits15:8 error code |
| `0x50` | DESCRIPTOR_ID | 活动 descriptor 用户 ID，**不能用于 completion fence** |
| `0x60` | COMPLETED_ID | 最近连续退休的用户 ID |
| `0x68` | SUBMITTED_SEQUENCE | 已验证并接受的全局序号 |
| `0x70` | COMPLETED_SEQUENCE | 连续完成前缀；后提交的先完成不能越过空洞 |
| `0x78` | RING_BASE | DDR descriptor 数组地址，64B 对齐；空闲时设置 |
| `0x80` | RING_CAPACITY | 1–65536 项，0 禁用；空闲时设置并清 head/tail |
| `0x88` | RING_HEAD | 已取回的位置；此前 descriptor 存储槽可复用 |
| `0x90` | RING_TAIL | 单生产者 doorbell，绝对递增，tail-head 不超过容量 |
| `0x98` | EVENTS | 完成/软件事件位 |
| `0xa0 / 0xa8` | EVENT_SET / CLEAR | 置位 / 清位，事件复用前清除上一代状态 |
| `0xb0` | ACTIVE_DESCRIPTORS | 活动 descriptor 数 |
| `0xb8` | FREE_DESCRIPTORS | 可用 descriptor credit |
| `0xc0` | ABI_VERSION | 返回 1；旧模型保留寄存器返回 0 |

每个 descriptor 为 16 个 little-endian uint64：

```text
id, system_addr, spm_addr, row_bytes, rows, system_stride, spm_stride,
flags, wait_sequence, wait_events, signal_events, reserved[5]
```

flags bit0=1 表示 SPM→DDR，0 表示 DDR→SPM，其余位及 reserved 必须为 0。
wait_sequence 只能引用之前已接受的序号；0 无顺序依赖。
wait_events 全部满足才可启动，成功完成时置 signal_events。
相关数据必须显式声明依赖，并发调度不自动检测地址冲突。
事件的生产者必须能进入有限队列，避免等待尚未入队的未来工作而死锁。

错误码：1 queue full、2 empty transfer、3 address overflow、4 SPM out of range、
5 SPM response error、6 invalid ring、7 invalid descriptor、8 system out of range。
错误保持至 CPU 写 CONTROL bit2；ring 遇错误停止取回，非法项已消费但不分配序号。
runtime 必须先检查错误，不能继续等未接受的 ticket。已有在途数据不会凭空取消。

## runtime 与显式 buffer 管理

`tools/gem5_dma_ring.h` 提供 `qwen_ring_init / put / submit / complete`：
先多次 put 准备 descriptor 和各自 ticket，再一次 submit 发布 tail。
complete 查询连续完成前缀并在完成后执行内存 fence。同一 batch 不要混用 ring 与 MMIO 提交。

descriptor 槽在 head 越过后可复用，**payload / SPM buffer 不能据此复用**。
`qwen_spm_buffer` 提供状态转换检查：

```text
FREE → FILLING(ticket) → READY → COMPUTING → DRAINING(ticket) → FREE
                                           └─ 无输出 DMA → FREE
```

acquire/reclaim 检查 DMA completion；release 由算子在 AME 真实完成后调用。
这些状态不提供硬件访问保护，也不自动判断 AME 是否完成。
DDR 源在输入完成前不覆盖，DDR 输出在输出完成前不读取，SPM 输入/输出窗口不重叠。
4096B 间隔的两个地址在当前 8×64B bank 映射下仍落在同一 bank，不能只看 buffer 编号。

`gem5_dma_tile.c` 的 MMIO 路径已修复活动 ID fence 问题：新模型查询 sequence / credit，
旧模型退回 busy 等待以确保正确性。构建 guest 时增加 `--dma-ring`，可将全模型
经过 `qwen_dma_submit/copy` 的请求切换到内存 ring。当前适配每次 submit 发布一个
descriptor，保留现有双缓冲依赖边界；尚未将算子统一改成多 descriptor 单次 doorbell，
也未实现自主 AME command queue。

本次 Qwen 运行位于 `runs/qwen_dma_ring_20260908/full_model`，使用新 ring guest、
4×4 连续注入、group 内 Acc、双缓冲，burst 上限 128B、outstanding 16、active 2，
4-token prefill + 1-token decode。完整数值比较等待该运行结束后执行。

## 新增统计

- `ringFetches / ringFetchBytes / ringDoorbells`：descriptor 取回次数、请求字节、批提交次数，含取回后验证失败的项。
- `activeDescriptorTicks / outstandingChunkTicks`：活动数量 / payload credit 占用的时间积分。
- `concurrentDescriptorTicks`：至少两个 descriptor 活动的 tick 数。
- `bidirectionalTicks`：输入与输出 descriptor 同时活动的 tick 数，不等于总线满带宽时间。
- `maxActiveDescriptors / maxOutstandingChunks`：峰值，后者不含 ring metadata credit。
- `maxSystemCredits / systemCreditTicks`：包括 metadata 的共享额度峰值 / 时间积分，用于核对全局 outstanding 上限。
- `completionEvents`：成功 descriptor 发布事件的次数。

有效流量仍用 `systemToSpmBytes / spmToSystemBytes`；descriptor 流量单独报告。
当前 1GHz 下 **1000 tick = 1 cycle**。带宽使用字节 / ROI 时间，并结合 cache、
SystemXBar、DDR、SPM 各层统计。descriptor 重叠不能替代链路利用率或 bank conflict 分析。

## 构建与回归

```bash
bash tests/spm_compute_soc/build_dma_ring.sh
python3 tests/spm_compute_soc/dma_ring.py --outdir runs/qwen_dma_ring_20260908/regression
PATH=/home/xiangyy/001_intern/llvm-project_riscv/build/bin:$PATH \
python3 tests/spm_compute_soc/run.py \
  --simulator build/qwen_DMA_ring/gem5.opt \
  --clang /home/xiangyy/001_intern/llvm-project_riscv/build/bin/clang \
  --outdir runs/qwen_dma_ring_20260908/shared_spm
```

ring 回归覆盖乱序完成、有限额度、双向并发、ring 回绕、事件门控、本地 AME 数值、
非法 buffer 状态、非法 tail / SPM 地址；现有共享 SPM 回归验证 RVV→AME 数据流和
两种 engine 直接 DDR 访问的拒绝。输出均保存为未压缩 `trace.out`。
