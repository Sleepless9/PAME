# Qwen3-0.6B：4×4 基线与后续 A/B/C/D 实验

本文件按 2026-09-08 最新要求更新实施顺序，优先于 `NEXT_MODIFICATION_PLAN.md` 的旧顺序。当前先修复模拟器基础问题、验证严格 local SPM，再建立 prefill/decode 基线；A/B/C/D 暂未启用。

## 当前进展

已使用 `/home/xiangyy/001_intern/llvm-project_riscv/build/bin/clang-21` 完成源代码重建；不再调用旧 ELF 内的算子或 libc helper。用户随后允许必要的算子修改，因此本轮已加入阻塞 DMA/SPM 适配和索引外提。

正式基线正在 `runs/qwen3_baseline_4x4_20260908/full_model` 后台运行，输入采用固定种子 20260907、prefill=4、decode=1。**当前尚未完成整模型 prefill/decode，也尚未取得 A/B/C/D 数据；不能把下述微测试数字当作完整阶段结果。**

| 验证 | 结果 |
|---|---|
| AME 单元测试 | 20 项通过，含未来 PE issue 及取消计数测试 |
| RVV 单元测试 | 33 项通过 |
| Minor inst queue 单元测试 | 5 项通过 |
| 共享 SPM 与 DDR 越界 | 正向数值测试和两个负向测试通过 |
| 写资源预留 | 等待中读写、队列重试、不提前发布数据、正常写均通过 |
| phase 导出与 trace 窗口 | 两阶段边界、真实结果导出、窗口长度及统计索引验证通过 |
| K=4 算子适配 | 944 次 tile，241664 个实际 PE issue，AME squash=0，DMA error=0；RVV copy/QK/AV 和矩阵投影数值通过 |
| 重复索引外提 | 6 组尺寸、548 个输出，与外提前源代码逐位一致；覆盖 N=1024/2048/3072、尾块和跨输出块 |
| 模型身份 | 新构建使用的 blob 与原 ELF 内嵌 blob 的 SHA256 相同：`2be6a748111407a0f3c6e351083ef24385641d4026d8517bd74f69c98822274a` |
| 同算法宿主参考 | 两阶段共 140 layer passes 已完成；每阶段 151936 个 logits 全部有限且非零 |

宿主参考的 prefill logits FNV64 为 `19B8F9BE17F65642`，argmax=50340；decode 为 `60028B55C9AAF186`，argmax=131131。宿主参考保留 K=4 缩放次序，并模拟 VLEN=512/e32/m1 的 16 lane QK 累加次序。它用于验证与现有算法一致性，不是另一套模型实现的精度评测。

完整基线预期 prefill 有 119824384 次 tile、decode 有 37249024 次 tile。短算子测试的宿主速度不能直接当作完整模型速度；早期全量进度也不是最终阶段数据。全量时间预计为多天，`run.json` 每 30 秒刷新运行状态和已完成 tile。

## 1. 基线边界

| 项目 | 基线要求 |
|---|---|
| 物理阵列 | 4×4 PE，1×1 tile，PSS |
| 内部 K block | 4，保留每块完成/排空后推进的流程 |
| 指令逻辑 K | int8 下为 16；原算子只填一个 square，有效 K=4 |
| MAC | latency=3、II=1；保持有限 operation/ingress 容量 |
| RVV/AME 访存 | 两者 timing port 只连接同一 BankedScratchpad；提交数据访问也检查 local SPM 范围 |
| DDR ↔ SPM | 显式 DMA；阻塞等待；不做双缓冲或与计算重叠 |
| 算法 | 保留原 tile 布局、K=4 软件循环及逐 tile 浮点缩放顺序 |
| 公共修复 | 统计、正确的写资源占用、取消计数、串行化下发、必要地址适配及重复索引修正；这些不能作为 A/B/C/D 的收益 |
| trace | 未压缩 `trace.out`；整模型常规运行记录必要事件，逐 PE 细节采用定向短窗口 |

旧 ELF 的 DDR/栈操作数无法通过 local-SPM 检查；当前通过源代码中的最小搬运适配解决。没有将引擎接回系统总线，也没有使用旧 checkpoint。

guest 的普通 C 和 libc 以 `rv64gc` 编译，确保 compiler lowering 不会偷偷生成访问 DDR 的 RVV load/store。原 attention 内联 RVV 指令保留，通过 SPM 指针访问；独立 RVV memcpy 以 `rv64gcv` 编译，Ztt 汇编以 `rv64gcv_xxiangshanztt` 编译。全部比较构建固定 `-ffp-contract=off`，保留显式浮点运算顺序；此公共编译条件也不能计入 A/B/C/D 收益。

SPM [0,192) 为 A/B/C tile，4/8/12 KiB 处各留 4 KiB RVV 工作区；所有使用者同步完成后才复用。矩阵 packing 当前由 CPU 标量在 DDR 中准备，然后 DMA 搬运；不引入 packed squares 或 group 累加。

## 2. 本批模拟器修复

- AME/RVV context squash 只在实际存在并取消 context 时累加。广播到其他引擎、未知 key、重复取消不再算作实际 context squash。
- 加速器 timing store 改为 `WriteReq + Request::NO_ACCESS`。这是本地 SPM 写资源预留契约：走写延迟、队列、bank 仲裁和响应，不在投机执行阶段写入数据。Minor 按序提交时发布真实数据。
- SPM functional 读写不会从尚未提交的 reservation 包中取数，也不会把它当成已提交写覆盖。真实 DMA 写仍按普通 timing write 更新 SPM。
- AME `SerializeBefore` 等待旧指令和 LSQ；串行化 offload 在退休/取消之前阻止年轻指令下发。保留原 ISA 序列化标志以及提交端的分支刷新行为，不直接放开寄存器动态跨度依赖。
- 增加实际 PE issue/result 计数。未来预排的 PE 事件到达对应周期后才计数；取消后的未来事件不计数。保留旧整块完成 MAC 计数并修改说明。
- 加入 RVV 活跃/完成/取消与访存计数、CPU 下发阻塞主因、SPM bank 等待及端口字节、DMA 已完成 descriptor 的服务时长。
- workbegin/workend 均输出累计统计和 `phase_events.json`。不 reset 后端生命周期计数，阶段数据使用边界差分。

写资源预留仍属于 timing/functional 分离模型：模拟器尚未把实际矩阵数值计算迁入 PE timing 流水。此限制必须在基线及后续所有比较中保持一致。它不等价于一个可在任意时间提前发布写数据的真实 store buffer。

## 3. 计数解释

| 指标 | 定义与用途 |
|---|---|
| 阶段 cycles | `(end_tick - begin_tick) / clock_period`，使用完整阶段的实际时间 |
| `saPeIssues` | 已到实际发射周期的 PE MAC 数，含 padding/零值运算 |
| PE 发射利用率 | `ΔsaPeIssues / (阶段 cycles × PE 数 / MAC II)` |
| `saPeResults` | 实际 PE pipeline 结果事件数；不能用旧 tile 完成数替代 |
| `saActiveCycles` | SA 有工作时的周期，含 operand/排空等待；不等于 PE 利用率 |
| AME/RVV `activeCycles` | 引擎有在途工作时的周期；不等于算术单元利用率 |
| `memoryRequests/Accepted/Responses` | 唯一生成、对端接受、返回；重试另计 |
| SPM `bytesWritten` | timing 写请求字节，包含写资源预留 |
| SPM `storeReservationBytes` | 其中无投机数据副作用的写资源预留字节 |
| SPM `functionalBytesWritten` | functional 写访问字节，包含提交的加速器实际数据；不能再加到 timing 字节上 |
| SPM `portReadBytes/portWriteBytes` | 各输入端口的请求字节，端口所属引擎由 config.json 对照 |
| `bankWaitTicks` | 各 bank 接受请求的排队等待总和；可重叠，不是总 stall |
| `completedRequestTicks` | 已服务请求的接受到服务完成延迟总和，不含响应端口额外背压 |
| DMA `completedDescriptorTicks` | 已退休 descriptor 从启动到结束的时间总和，包含失败 descriptor；不含启动前队列等待 |
| CPU offload stall 计数 | 被对应主因阻挡的尝试次数；提交等待可与下发阻塞重叠，不能直接相加做 CPI 分解 |

当前 RVV LSU 按 cache-line 粒度合并请求：例如 VL=16 的 e8 store 有效数据为 16 B，但 timing 请求为 64 B。DMA 有效搬运字节、引擎有效写字节、SPM timing 请求字节是三个不同口径。

现有 `AMEPE` trace 的 `PE_MAC_ISSUE` 行可能在排入未来事件时打印，行首 tick 不一定是实际 issue 时刻。利用率使用修正后的 `saPeIssues/saPeResults`，不能直接按 trace 行数统计；短窗口末端尤其可能包含尚未实际发生的 issue。

## 4. 正确性与基线运行门槛

1. AME、RVV、Minor inst queue 单元测试通过；新增 PE 计数测试检查取消之前与之后的实际 issue/result。
2. `tests/spm_compute_soc/run.py` 验证 DMA → RVV → AME → DMA 的数值结果以及两个 DDR 越界负例，核对写资源字节、PE 数及 AME 零 squash。
3. `tests/spm_compute_soc/store_reservation.py` 使用真实 timing requestor，验证等待中的预留写、bank 队列重试、functional 读写不被投机包污染、正常 timing 写仍生效。
4. 原 ELF 做严格 SPM 兼容性探测；若失败，记录首个不合法访问。该探测不作为 Qwen 性能 baseline。
5. 全模型基线必须覆盖所有 RVV/AME DDR 操作数迁移。只有 tile 的 DMA wrapper 不够，还包括原始编译产物中的 RVV packing/清零、memcpy、attention QK/AV 等。
6. 重复索引外提已在 `kernel/src/ztt_matmul.c` 实施：stride_groups 每次调用计算一次，四行权重基地址每个 group 计算一次；K=4 循环、缩放及输出顺序保持不变。已检查生成汇编，除法位于函数/批处理准备区，内层 K tile 不再重复除法。
7. 完整 prefill 和 decode 分别留下 workbegin/workend、模型/ELF/模拟器 hash、token 数、固定输入及数值验证记录。只能把正常完成并验证的数据列入正式比较。

阶段抽取工具：

```sh
python3 example/qwen3-0.6B/tools/analyze_baseline.py runs/实际整模型目录
```

输出 `phase_summary.json` 和 `phase_summary.md`。没有结束边界的阶段标为 incomplete，没有开始边界的阶段标为 not_started。微测试目录中的阶段标记仅用于验证统计链路，不是模型的 prefill/decode 数据。

重建和启动入口（从 gem5 根目录执行，重跑时使用新的输出目录）：

```sh
python3 example/qwen3-0.6B/tools/build_baseline.py runs/新构建目录/baseline.elf --prefill-tokens=4 --decode-steps=1
python3 example/qwen3-0.6B/tools/build_baseline.py runs/新构建目录/reference --host-reference --prefill-tokens=4 --decode-steps=1
runs/新构建目录/reference example/qwen3-0.6B/core/model_data/qwen3-0.6b-int4-tile.bin runs/新构建目录/reference_output
python3 example/qwen3-0.6B/tools/run_baseline.py --binary runs/新构建目录/baseline.elf --outdir runs/新结果目录 --reference-prefix runs/新构建目录/reference_output
```

运行器先冻结 gem5 binary、guest ELF、配置、参考输出和符号表到 `artifacts/`，防止后续修改影响这次运行。全量 trace 为未压缩的 `trace.out`；另外在 `validation/prefill_trace_sample` 补采同一冻结 binary 的短 trace，覆盖 prefill 准备期之后的实际 AME 运算。

该补采在 2 ms 模拟时间处主动截断（退出码 124），共完成 1138 次 tile，实际 PE issue/result 均为 291328，AME squash 和 DMA error 均为 0；详细 trace 只覆盖 phase 开始后的 1 ms。`sample_metadata.json` 保存窗口、统计快照和冻结产物 hash。它用于检查事件及访存链路，不作为完整 prefill 性能结果。

每阶段结束先执行 fence，再保存累计统计、guest tile/layer 计数和 logits。运行结束后自动检查两阶段 logits 与宿主参考逐位一致、tile 数与模型工作量一致，并生成 JSON/Markdown 阶段报告；未完成或校验失败均不得标为 passed。

## 5. 后续四项优化

| 名称 | 变更 | 主要修改层 |
|---|---|---|
| A | K=16 packed squares，填满原指令的 4 个 int8 squares | 算子 packing、K 循环 |
| B | K=128 量化 group 内保留 Acc，group 结束后缩放 | 算子/汇编；记录浮点累加顺序变化的误差 |
| C | 8A：PSS 连续注入 | 模拟器 SA/MOFU、有限容量、buffer 生命周期、取消处理 |
| D | 8B：DMA 双缓冲与计算重叠 | 算子/运行时，必要时修改 DMA 调度模型 |

所有实验保持 4×4；本轮不混入 M=4 多 token 映射、第 9 步阵列扩大等额外收益。每个结果必须列出实际启用集合，尤其不能把包含 A 的 B 结果写成 B 单项收益。A/B/C/D 尚未实施、运行或取得收益数字。

待 baseline 和四项有效结果齐备后，用 Python 绘制按 prefill/decode 分开的阶段时延、相对加速比、PE 利用率、指令/下发量和 DMA/SPM 字节图，保存 PNG/PDF 及原始 CSV，再编写最终 Markdown 分析。缺测列不能用估计值或微测试时延代替。
