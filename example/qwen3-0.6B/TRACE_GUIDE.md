# AME / DMA / SPM trace 阅读说明

以下命令在 `/home/xiangyy/001_intern/gem5` 执行。

## 解析与筛选

先用已经关闭 gzip 的短程恢复测试学习格式；它不是完整模型成功记录。

```sh
python3 example/qwen3-0.6B/tools/summarize_trace.py \
  runs/qwen3_original_restore_smoke_20260908/trace.out.gz \
  --output runs/qwen3_original_restore_smoke_20260908/analysis_flow \
  --all-events
```

输入也支持未压缩的 `trace.out`。输出 `summary.json` 包含事件计数及延迟
count/min/max/mean/p50/p95/p99；`events.csv` 包含 tick、component、event、
fields（JSON 字符串）。默认 CSV 只导出 DMA/SPM、供数 stall 和最终累加器授权；
查看 AME 全流程时加 `--all-events`。事件计数与延迟汇总始终针对全部输入。
CSV 中少数复合消息只保留首个事件词，精确子阶段仍应查看原文。

```sh
# 单条 AME 动态指令；边界避免把 inst=110 当成 inst=11。
gzip -dc runs/qwen3_original_restore_smoke_20260908/trace.out.gz | rg '\binst=11\b'

# 未压缩 trace 的关键流程。
rg 'AME_DISPATCH_ACCEPT|TILE_CMD_ACCEPT|KSTEP_INJECT|ACC_FINAL_GRANT|AME_COMPLETE|SQUASH' trace.out

# 外部访存与反压。
rg 'MLSU_REQ|MLSU_RESP|system\.spm([.:])|system\.spm_dma:|SUPPLY_STALL' trace.out
```

运行中的主记录位于 `runs/qwen3_original_final_20260908/trace.out.gz`。
它可能尚未写入 gzip 尾部，解析器将报告 `gzip_complete=false` 并保留可读部分；
缺少结束事件也可能只是采样边界。`gzip_complete=true` 仅表示压缩流完整，
`trace_read_complete` 仅表示读取结束；两者都不代表模型完成。
完整模型结果检查见该运行的 `run.json`、`run.log` 和 `system.platform.terminal`。
正在增长的大 trace 使用 `--all-events` 会产生较大的 CSV。

## 时间与关联键

原文格式：`tick: component: cycle=... inst=... EVENT fields...`。

- `tick` 是全局仿真时间。当前 `stats.txt` 的 `simFreq=1000000000000`，
  因此 1000 ticks = 1 ns；当前 AME 1 GHz 下也等于一个时钟周期。
- `cycle` 是 AME pipeline 的本地推进计数，空闲时引擎会停止调度，不能把它
  当成全局绝对周期。跨模块、跨空闲间隔的耗时用 tick 差。
- `inst` 是 AME 动态上下文 ID，`exec` 关联 CPU 动态执行序号；
  `slot` 是可复用的 AME 槽位，不能用它作为指令唯一键。
- AME 访存用 `txn` 配对 `MLSU_REQ/MLSU_RESP`，同时保留 `inst`、`desc`。
- DMA 用 `descriptor` 分组，再结合 `row`、地址、bytes 区分 chunk。
  同一 row 可以有多个 chunk，不可仅按 row 配对。
- SPM 目前没有贯穿所有事件的唯一请求 ID。同地址并发请求仅凭地址无法
  无歧义关联；至少结合 port、顺序和上游事务，严格追踪需要补充请求 ID。

## 一条矩阵乘指令的流程

```text
AME_DISPATCH_ACCEPT → AME_ID_ALLOC → AME_GLOBAL_BROADCAST → TARGET_ACCEPT
  → TILE_CMD_ACCEPT → 取操作数 / 累加器初值
  → KSTEP_INJECT（多个 kblock/kstep）
  → TILE_DRAIN_ENQUEUE → 累加器 WRITE_GRANT
  → ACC_FINAL_GRANT → AME_COMPLETE
```

load/store、config、move 等指令走各自目标单元，不会都有 `KSTEP_INJECT`。
当前样本中的 matrix_multiply `inst=11`：

| 事件 | 本地 cycle |
| --- | ---: |
| AME_DISPATCH_ACCEPT | 1137 |
| TILE_CMD_ACCEPT | 1140 |
| ACC_SEED_LOAD | 1154 |
| 首次 KSTEP_INJECT | 1155 |
| 最后一次 KSTEP_INJECT | 1230 |
| TILE_DRAIN_ENQUEUE | 1250 |
| ACC_FINAL_GRANT | 1258 |
| AME_COMPLETE | 1260 |

该指令 `latency=123`，是接收至 AME 完成的本地周期数，并非纯 MAC 延迟。
对应摘录在恢复测试目录 `analysis_flow/inst11.csv`。
`AME_COMPLETE` 是加速器时序完成，不是 CPU commit 或模型层完成。
`SQUASH` 表示取消；dispatch 没有 complete 时先检查 squash。
短程样本有 13413 次 dispatch、9468 次 complete、3945 次 squash；
不能将所有取消指令计作 starvation。

## DMA、bank 和供数 stall

| 事件/字段 | 观察含义 |
| --- | --- |
| MLSU_REQ → MLSU_RESP | AME 对外访存事务往返，按 txn 配对 |
| SYSTEM_READ_DONE latency | DMA chunk 发出至系统侧读完成，单位 tick |
| SPM_REQUEST → SPM_RESPONSE | DMA 被接受的 SPM 请求至响应；latency 单位 tick |
| CHUNK_DONE latency | 整个 DMA chunk 的传输延迟，包含串联阶段，不能再与各阶段重复相加 |
| system.spm: BANK wait | start − arrival，bank 调度等待，单位 tick |
| system.spm: BANK occupancy | 接收当前请求后该 bank 在途请求数，不只是等待队列长度 |
| bank_capacity / bank_credit | SPM bank 容量 / DMA 每 bank 在途 credit 约束 |
| system_credit / spm_retry | DMA 系统侧在途限制 / SPM 请求被拒绝等待重试 |
| response_capacity / response_backpressure | 响应空间不足 / 下游暂时无法接收响应 |
| SUPPLY_STALL_BEGIN/END | SA 检查操作数 block 未就绪的等待区间；END.cycles 是本地周期数 |

SPM 的 `COMPLETE` 表示存储体服务完成，`RESPONSE` 表示响应成功发给下游；
两者间隔大应查响应反压。`BANK wait=0` 并不表示没有延迟或没有反压。
一个跨 bank 请求可能打印多个 `BANK`，因此 BANK 计数不等于请求数。

`READ_GRANT/WRITE_GRANT bank=...` 来自 AME 内部 MRF/AccRF，
**不是外部 banked SPM**。`SUPPLY_STALL` 也不是 SPM 专属信号：它只覆盖
当前代码的操作数就绪检查；内部寄存器供数也可能触发。没有该事件不代表
CPU、MLSU、DMA 或所有 SA 阶段都没有等待。还应查看 `stats.txt` 中的
`mrfConflictCycles`、`accConflictCycles`、`saInjectionStalls`、`saDrainStalls`、
`memoryOutstanding` 及 SPM/DMA retry 统计。

诊断停滞时，先找没有 COMPLETE/SQUASH 的 inst，再查其最后阶段；
检查是否有未返回 txn、持续 bank/response 反压或未闭合 SUPPLY_STALL。
确认仿真 tick 仍在推进并排除 trace 截止边界后，才能判断是持续阻塞。
单凭 KSTEP 之间有空隙不能认定为 starvation，空隙也可能属于正常流水延迟。

## 当前记录能说明什么

原始 Qwen binary 的恢复测试样本中 `system.spm.timingRequests=0`，
DMA descriptor 数为 0；MLSU 示例地址为 `0x863fe280`（DRAM 栈），
不在当前 SPM 范围 `[0xc0000000, 0xc0100000)`。
该 binary 未编程 DMA，不能用这份模型记录评估 DMA/SPM 瓶颈。

已有 DMA/SPM 链路示例：
`runs/qwen3_ame_20260907/tile_dma_coherent/trace.out.gz`。
它是独立两 tile 诊断，不是完整 Qwen 模型流量。其中 DMA chunk 延迟
14–15 ns，SPM 请求响应 1–2 ns，bank wait 为 0，但存在响应反压事件。

主运行开启 `AMEFlow,SPMVirtDma,BankedSPM`，没有开启 `AMEPE`，
因此可以观察指令与访存流程，无法从已有 trace 恢复每个 PE 的 MAC 事件。
