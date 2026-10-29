# 2026-09-08 实施顺序更新

最新要求以 [BASELINE_EXPERIMENT.md](BASELINE_EXPERIMENT.md) 为准：先完成公共基础修复，建立 4×4、无连续注入的 prefill/decode 基线，再测 A（K=16）、B（group K=128 Acc）、C（PSS 连续注入）、D（DMA 双缓冲）。本文下方的旧批次顺序、M=4 与阵列扩展不作为本次默认执行范围。

**Qwen3-0.6B：共享 SPM、AME 算子与 systolic 流水下一步修改计划**

日期：2026-09-08。状态：实施计划；除下述已经完成的拓扑修正和测试外，其余工作均待实现。本文件合并此前 9 项优化、连续注入设计及矩阵分块建议，作为后续修改和验收依据。

目标是先建立完整、正确的 local-SPM 推理基线，再减少软件准备与指令开销，改善阵列连续工作和 DMA 重叠，最后评估阵列规模。首轮以现有 PSS、int8 操作数和 int32 累加为对象；编译器后端自动优化暂不作为前置条件。

**一、已完成的工作与必须保留的架构约束**

| 项目 | 当前状态 |
|---|---|
| RVV、AME memory_port 直连同一个 BankedScratchpad | 已完成 |
| DMA 的 SPM 侧共享同一 bank 仲裁，系统侧经互联访问 DDR | 已保留 |
| AME 独立入口统一调用共享 SPM 配置 | 已完成 |
| DMA→SPM→RVV→AME→DMA 数据正确性测试 | 已通过，验证 16 个输出 |
| RVV、AME 分别直接访问 DDR 的负例 | 已通过，均按预期被 SPM 地址检查拒绝 |
| 原 Qwen 完整算子的 SPM 迁移 | 未完成 |
| 修正后拓扑的完整 prefill/decode 性能基线 | 尚未建立 |

```text
RVV VLSU ──┐
           ├── 共享 Banked SPM ⇄ DMA ⇄ coherence cache ⇄ SystemXBar ⇄ DDR
AME M-LSU ─┘
```

CPU 标量访存保留 cache/DDR 路径，SPM 仍位于 CPU 地址空间中。RVV/AME 的计算访存只能使用 SPM 地址。禁止通过重新连接 engine 到 membus、functional-only 模式或隐藏的 DDR 访问来绕过这个约束。

本轮后续修改不得覆盖原始 ELF、历史日志或旧 checkpoint。新的 guest、拓扑或 ISA 能力必须使用独立运行目录；旧 checkpoint 中的 DDR 操作数不能代替软件迁移。修改 Python 配置不会改变已经启动的仿真实例。

现有验证入口：[tests/spm_compute_soc/run.py](/home/xiangyy/001_intern/gem5/tests/spm_compute_soc/run.py)。历史分析：[report.md](/home/xiangyy/001_intern/gem5/runs/qwen3_original_final_20260908/performance_analysis/report.md)。

**二、历史证据只用于定位，不作为新拓扑的加速比分母**

固定旧快照为恢复后 2.85 s、285 个完整累计统计块；指令级样本为恢复后前 20 ms。

| 旧运行现象 | 后续对应工作 |
|---|---|
| SPM/DMA 请求和搬运量全部为零 | 第 2 步：完整软件迁移 |
| 每 tile 约 279 条 CPU 指令、13 条 IntDiv | 第 3 步：索引外提与地址递增 |
| 指令执行 K=16，但仅有效 K=4，A 仅一行有用 | 第 4、6 步：packing 与跨 token 分块 |
| 每 tile 反复 zero/move/store/scale | 第 5 步：量化 group 内累加 |
| 每 tile 真实取消 3 load、1 config、1 release | 第 7 步：下发与序列化 |
| K-block 间出现 20 拍无新注入 | 第 8A 步：注入/完成解耦 |
| operationDepth 与入口深度绑定，默认只有 2 | 第 8A 步：独立在途容量与真实 PE 发射验证 |
| DDR 总线利用率仅 2.73% | 不优先扩大 DDR 带宽 |

旧数据的建模 MAC 峰值利用率约 1.3874%，按内核结构剔除填零后约 0.0867%。这不是新 local-SPM 架构的利用率，也不是整个 Qwen 的总运算吞吐。后续加速比均相对于同一 workload、同一统计口径的 local-SPM 基线。

**三、模型矩阵、K 的定义与候选尺寸**

原始 ELF 模型头：28 层，hidden=1024，Q heads=16，KV heads=8，head_dim=128，q_dim=2048，kv_dim=1024，FFN=3072，vocab=151936，量化 group_size=128。以下使用 A[M,K]×B[K,N] 的计算布局；prefill 的 M 是同时处理的 token 数，单请求 decode 的 M=1。

| 线性算子 | K | N |
|---|---:|---:|
| Q | 1024 | 2048 |
| K、V，各自 | 1024 | 1024 |
| 合并 QKV | 1024 | 4096 |
| Attention 输出投影 | 2048 | 1024 |
| FFN gate、up，各自 | 1024 | 3072 |
| FFN down | 3072 | 1024 |
| LM head | 1024 | 151936 |

这些线性层的 K、N 均可被 128 整除。Attention 的 QK、AV 还受实际序列长度、head 和 KV cache 布局影响；其 SPM 迁移属于第 2 步，但首轮不将它们直接替换为未经验证的 AME 整数矩阵乘。

必须区分下列四个 K：

| 名称 | 含义 | 当前值／首轮目标 |
|---|---|---|
| 算子总 K | 一整个投影的归约维度 | 1024、2048、3072 等 |
| 软件 K 块 | 一次按量化 scale 边界处理的数据 | 128 |
| 单条矩阵指令逻辑 K | AMENLEN×max(1,AMEUDSZ/元素位宽)，这里为 int8 | 当前 4×4=16；8×8 候选为 8×4=32 |
| 内部 k_block | MOFU/SA 内部供数分块粒度 | 当前 4；与单条逻辑 K 分开实验 |

在现有指令 K=16 下，8 次 mmulacc 可以累加完 K=128，不要求扩展 ISA。跨量化 group 的 scale 不同，不能直接把全部整数结果混成一个未缩放累加值。

| 配置 | 基础 PE tile | tile 网格 | AMENLEN | 单条 int8 K | 软件工作块 |
|---|---|---|---:|---:|---|
| 当前硬件基线 | 4×4 | 1×1 | 4 | 16 | 保留原分块，先修正搬运 |
| 首轮优化，decode | 4×4 | 1×1 | 4 | 16 | M=1，N 可按 64 组织，K=128 |
| 首轮优化，prefill | 4×4 | 1×1 | 4 | 16 | M=4，N=64，K=128 |
| 第 9 步 prefill 候选 | 4×4 | 2×2 | 8 | 32 | M=8，N=64，K=128 |

后三项的算子布局均待实现。当前 batch API 仍逐 token 调用，函数名为 batch 并不意味着已经使用了多个 M 行。

8×8 候选下 A[8,32] int8、B[32,8] int8、C[8,8] int32 各为 256 B，刚好落在当前 M/Acc 容量内；这只证明基本容量可容纳，不能替代完整 ISA、布局与 timing 验证。AMENLEN/AMEUDSZ 当前为只读能力 CSR，8×8 需要新增并贯通模拟器能力配置，不能通过 guest 普通 CSR 写入实现，也不能只增加 tile_rows/tile_cols。

对软件块 M=8、N=64、K=128，int8 A=1 KiB、int8 B=8 KiB，一份 32-bit C=2 KiB；A/B 双缓冲加一份 C 共 20 KiB。M=4 时同口径为 18 KiB。实际 allocator 还必须计入 scale、对齐、RVV packing scratch、未缩放 int32 tile 与跨 group FP32 输出累加，不能直接把 20 KiB 当作总容量上限。W4 模型在当前矩阵计算路径使用展开后的 int8 tile，容量按实际传输表示计算。

**四、实施顺序和职责**

| 步骤 | 工作 | 主要修改层 | 进入条件 |
|---|---|---|---|
| 1 | 统计、访存 timing 与提交口径 | 模拟器 | 已有 local-SPM 小测试 |
| 2 | 完整 Qwen 阻塞 DMA/SPM 基线 | 算子／运行时，必要的模型修正 | 第 1 步口径确定 |
| 3 | CPU 重复索引计算外提 | 算子源码，检查编译产物 | 第 2 步正确性通过 |
| 4 | 填满 K=16 packed squares | 算子／packing | 第 3 步结果留档 |
| 5 | K=128 group 内保留 Acc | 算子／汇编调度 | 第 4 步通过 |
| 6 | 多 token prefill 使用 M=4 | 算子／调用层 | 第 5 步通过 |
| 7 | 消除无效下发，逐类放宽序列化 | 模拟器 | 有稳定算子与依赖测试 |
| 8A | PSS 连续注入 | 模拟器／微架构模型 | 统计、依赖、取消语义稳定 |
| 8B | DMA 双缓冲与计算重叠 | 算子／运行时；按需增强 DMA 模型 | 第 2 步正确，第 8A 步结果留档 |
| 9 | 8×8 prefill 与资源参数实验 | 配置、ISA 能力、模型及算子 | 首轮软件／流水优化完成 |

保留单项消融结果，再合并收益。开发中可以用短算子先验证局部设计，但正式集成与整模型比较按表中顺序推进。每一步均是独立、可回退的修改；未完成验收不得把候选参数标为默认最优值。

**第 1 步：修正计数，明确真实访存时序。**

修改重点为 `AME/engine.cc`、`RVV/engine.cc`、`AME/p1_pipeline.*`、`cpu/minor/inst_queue.*`、`cpu/minor/execute.cc`、`exec_context.hh` 及 SPM/DMA 统计。

- 取消通知按 engine/context 归属计数。AME 的取消函数只有确实取消 context 时才增加 context squash；未进入 context 的队列取消单独记录。
- 将接受、真正启动、完成、取消按 opcode 分开。当前 SA 某些命名为 issue 的统计在 completion 时增加，应明确命名或分成真实 issue/result 两套计数；不能在预排队时把未来 PE issue 算进当前窗口。
- 增加 CPU/offload 的队列满、源依赖、序列化、等待完成、等待提交等原因；明确可重叠统计与互斥主因，禁止直接相加构造错误 CPI 分解。
- 分别记录 SA 入口注入、PE 实际 issue、result、operation 容量阻塞、operand 等待、边界背压和最终 drain。
- 增加 DMA 描述符延迟、有效字节、请求字节、最大/平均 outstanding；SPM 记录每 bank 的等待、服务及请求来源。
- AME 和 RVV 的 store timing 都需核对：当前 SoftPFExReq 与 functional commit 不等同于真实 SPM 写事务。先确定 store 的资源占用、写延迟及可见性契约，再实现；不能简单换成会提前修改存储内容的投机 WriteReq。
- 检查 architectural functional 访问是否始终对应合法 SPM 地址，避免 timing 与最终数据访问地址/范围不一致。

验收：短 trace 与计数逐项对齐；取消不产生可见的错误写入；DMA 不得读取尚未提交的输出。阶段统计使用累计计数差分或实现正确 reset，记录时钟、起止 tick 和统计定义版本。对有重叠的时间段使用区间并集，不把请求延迟总和解释为总 stall。

交付：计数定义表、最小 trace 对账结果、store timing/commit 设计说明和相关回归。

**第 2 步：完成全算子 SPM 迁移，建立阻塞基线。**

首先建立算子访存清单，覆盖 QKV/O/FFN/LM head，以及清零、packing、量化、RMSNorm、RoPE、softmax、QK/AV、KV cache 读写和 memcpy。对每个调用记录：输入/输出位置、RVV/AME 指令、临时空间和 DMA 方向。

实现一个明确的 SPM 工作区分配与生命周期接口。第一版严格按以下顺序执行：

```text
CPU 准备 DDR 输入 → DMA 搬入 → 等待完成
→ RVV/AME 在 SPM 上执行 → 等待结果提交
→ DMA 搬回 → 等待完成 → 复用 buffer
```

已有 `tools/gem5_dma_tile.c` 可作为阻塞搬运起点；`gem5_packed_backend.c` 的 packing 仍有 DDR 向量操作，不能原样当作 local-only 实现。需审计旧 ELF helper ABI 和每个实际保留的 helper，不能只包装最外层 AME 调用。

DDR 上的初始化、拷贝可由 CPU 标量执行；检查生成汇编，避免编译器自动向量化这些路径。显式 RVV 操作全部迁移到 SPM，遇到当前未支持的向量访存形式时做有验证的 lowering，不使用绕开 timing 的 fallback 隐藏问题。

验收按单 tile→投影算子→完整层→完整 prefill→decode 推进。所有 engine 请求范围必须包含在 SPM 内；两个 engine 的 DDR requestor 流量为零。验证每层/算子输出和最终 logits，不能只检查退出码或 token 是否相同。建立完整 run manifest、阶段边界和 guest 完成标记。

交付：SPM-aware guest、buffer 布局、阻塞 DMA 调用层、完整 local-SPM 基线。该基线包含真实新增搬运成本，不要求先优于绕过 DMA 的历史运行。

**第 3 步：减少 CPU 准备成本。**

主要修改 `kernel/src/ztt_matmul.c` 的 `matmul_one()`、`weight_tile()` 及相关调用层。

外提 stride_groups、group/block/row 基址；内层使用指针递增；对固定 group_size=128 和已知维度提供专用路径，保留必要的通用尾块路径。保持当前 packing、DMA 策略和计算顺序，单独测量这一项。

验收：反汇编中的重复 divw 明显减少；记录 IntDiv/IntMult/总提交指令数、准备区间周期和总周期。整数结果一致，浮点路径按既定误差标准通过。不要仅根据优化级别或源码变短判定成功。

**第 4 步：使单条 K=16 真正承载 16 个有效 K。**

保持 AMENLEN=4、AMEUDSZ=32、物理 4×4 不变。将连续四段 K 数据填入四个 4×4 packed squares，循环从每次推进 4 改为推进 16。核对 A/B 的 square 拼接和转置关系；M=1 时仍只填一行有效 activation，其他行清零，尾部 K 才补零。

相关位置：`ztt_matmul.c`、`ztt_matmul.S`、`gem5_packed_backend.c`，以及新 SPM packing 调用层。输入仍为 64 B/操作数；提升的是有效填充率，而不是新增更大的指令。

验收：对同一个数学矩阵，矩阵乘指令数约降至有效 K=4 版本的四分之一；测试 K=16/32/128 及非整倍数尾块，检查有效 MAC 数和布局。分别测 CPU packing 与 RVV/SPM packing 的成本，保留实际更好的实现。

**第 5 步：量化 group 内累计，减少控制和输出往返。**

以 `tools/gem5_ztt_group.S` 为起点，把一组 K=128 改为一次 zero、8 次 K=16 的 load/multiply-accumulate、一次 move/store，最后应用这一 group 的 activation scale 和 weight scale。复用类型配置和 acquire/release 的范围必须符合寄存器所有权与调用约定。

```text
原路径：每个小 K 块都 zero / multiply / move / store / scale
目标：  zero / 8 次 K=16 累加 / move / store / group scale
```

不同 group 仍分别缩放，再累计到 FP32 输出。核对 int32 的累加范围；对 int8 最坏值和 K=128，参考界为 128×128×128=2,097,152，实际数据类型/算术模式仍需按实现确认。改变浮点累加顺序后记录最大绝对/相对误差，不预设逐位相同。

验收：在结果通过的前提下，zero/move/store/scale 的次数与量化 group 数一致，DMA 输出往返减少；保持整数 Acc 跨指令累加正确。这里不要求跨指令 SA 无缝接续，后者不是简单循环可以保证的功能。

**第 6 步：prefill 按 token 分块，先使用 M=4。**

修改 `ztt_matmul_quant_batch*`、`core/ztt_qwen_backend.c` 和 prefill 调用层，将四个 token 的 activation 打包为 A 的四行，共享 B 权重。不能继续在 batch 函数里逐 token 调用 GEMV。每行使用自己的 activation scale，每个输出列使用对应 weight scale。

软件首选实验块为 M=4、N=64、K=128；N=64 对齐现有权重输出块布局。处理不足 4 个 token 的尾块，保证输出屏蔽和 buffer 边界正确。此时仍使用一个 4×4 物理 tile，先衡量权重复用和有效行利用率。

验收：T=1/3/4/5/8/9 等短 workload 的输出通过；prefill 与 decode 分开报告。T=1 decode 保留独立路径，不通过计算虚构 token 填满 PE。

**第 7 步：减少 SerializeAfter 引起的无效下发。**

主要修改 Minor 的 issue/commit/scoreboard、共享 inst_queue，以及 Ztt 指令属性和 AME scoreboard。

第一阶段保留正确性约束，让 host 下发尊重序列化边界，避免已经知道将被刷新却提前发送的 load/config/release。第二阶段再逐类建模动态 M register span、类型配置、Acc 和 memory 依赖，放宽不必要的全局串行化。

验收覆盖 RAW/WAR/WAW、动态类型/寄存器跨度、分支取消、异常、store 可见性和 checkpoint/drain。目标是消除确定性的重复请求；但“取消数少”并不必然意味着快，必须同时比较总周期，防止只把投机取消变成更长的串行等待。

不允许一次性删除所有 SerializeBefore/SerializeAfter。软件减少控制指令与模拟器修正序列化是不同改动，分别留档。

**第 8A 步：实现 PSS 单条指令内部连续注入。**

这是微架构模型修改，主文件为 `AME/systolic_array.*`、`operand_subsystem.*`、`AMEEngine.py`、`engine.cc` 和相关测试。目标是同一 context、同一 output tile 内跨 K-block 重叠，首次不扩展为多条矩阵指令同时共享驻留 Acc，也不直接推广到 WS。

8A-1：把 operationDepth 与 ingressDepth 解耦。当前 operationDepth=ingress_depth=2，两个 wave 未完成就阻塞新启动。新增独立的 tile operation 容量参数，保留原容量作为对照，先实验 16。4×4 tile 最远 PE 的传播加 MAC 延迟约 9 拍，持续 II=1 还需要容纳完成传递；16 是验证起点，不是硬件最终容量结论。

8A-2：拆分注入游标与完成计数。保留 nextKBlock/nextKStep，按 token 的 contextId/outputBlock/kBlock/kStep/tile 记录完成。换注入块时不再清掉尚在执行的上一块完成信息；给重复 completion、跨 output/context completion 增加校验。

8A-3：按输入握手推进。下一块数据就绪且入口有容量时，当前块最后一步之后即可发下一块；gridIdleFor 只保留在最终 drain 或需要真正切换驻留状态的位置。

```cpp
collectCompletionsByTag();
if (!allInjected && nextOperandsReady() && ingressCanAccept()) {
    injectNextStep();
    advanceInjectionCursor();
}
if (allInjected && allCompleted && gridIdleFor(context)) {
    beginFinalDrain();
}
```

此为拟议控制逻辑，不是已经提交的实现。理想输入条件下，K=16 的注入序列应为 0…15 连续推进，而不是每 4 步等待整个阵列排空。

8A-4：定义 operand buffer 所有权。现有两个 MOFU buffer 的 consumeBlock 在计算完成后才调用。若要提前回收，必须保证最后一份 A/B 已被复制到有容量约束的独立 FIFO/锁存器。当前 WaveToken 主要是标签，不能把“标签入队”当作真实数据已经搬完；若消费者仍引用原 buffer，则增加引用计数并延迟复用。根据实际 refill 延迟决定两个 buffer 是否足够。

8A-5：确认 PSS 累加反馈满足目标 II。当前 PE 主要建模 token、时序和驻留状态，最终数值通过 ISA 路径执行；数值测试不能证明假设的硬件累加反馈可实现。明确 int8 乘法流水与 int32 累加反馈结构；若反馈需要多拍，应设计可验证的交错部分和及最终归并，或使用真实可实现的 II。不要把 macLatency 直接减为 1 来掩盖依赖。

8A-6：保持有限资源和完成语义。入口 FIFO、在途 operation、边界 FIFO、completion 和 drain 都继续限制容量；任何背压下不得丢失或重复 token。全部 K 的结果到达前，不能清除驻留 psum、写回 Acc 或复用输出。取消时清理所有块的在途引用，保证后续 context 可以运行。

验收：先单 tile、PSS、输入始终就绪，再加随机供数延迟、backpressure、取消、尾 K 和多物理 tile。检查真实 PE_MAC_ISSUE 的 issue=时刻与 result，而不只是 KSTEP_INJECT。分别测试更大 operationDepth、跨块注入及二者组合，证明收益来自哪里。实际 II、最长断流、operand/operation/edge stall 均需报告。

跨多条 mmulacc 的 Acc 驻留与连续注入、WS 权重双驻留为后续独立设计，不计入本步骤完成条件。

**第 8B 步：DMA 双缓冲与计算重叠。**

在阻塞实现上拆出 submit/wait 接口，返回 descriptor 标识或等价完成凭据。不能把全局 BUSY 清零当成任意指定 buffer 已完成；先核对现有 descriptor ID、queued descriptors 和完成状态寄存器的语义。当前 DMA 有描述符队列但只有一个 active descriptor，第一版目标是“DMA 与计算重叠”，不假设多个描述符能够独立并行执行。

为 A/B ping-pong buffer 和输出区定义状态与所有权，例如 FREE、DMA_FILLING、READY、COMPUTING、DRAINING；只有相关消费者与输出 DMA 都完成，才允许覆盖对应区间。

```text
启动：搬入第 0 块并等待
稳态：计算当前块，同时 DMA 搬入下一块
切换：等待下一块 READY，再交换 buffer
收尾：等待最终计算提交及输出搬运完成
```

按块/行批量提交搬运，减少细粒度 MMIO；边界对齐与 A/B bank 分布应结合真实冲突统计确定。CPU fence 用于保证所需的访存顺序，不能替代 DMA 指定操作完成或 accelerator 提交完成通知。

验收：反复使用 buffer、输入输出别名边界、部分块和队列背压均正确；记录 DMA-only、compute-only、两者重叠及计算等待 DMA 的区间。比较阻塞/双缓冲，核对总传输量，防止通过遗漏搬运得到虚假加速。

**第 9 步：阵列与资源参数实验。**

先以优化后的 4×4 为固定基线，再测试 4×4 基础 tile、2×2 网格、AMENLEN=8、AMEUDSZ=32 的 prefill 候选。新增能力参数需要贯通 RiscvISA.py、ISA 初始化、配置快照、checkpoint 序列化/兼容性检查以及 engine snapshot；保持 capability CSR 只读。guest 按能力选择匹配布局，不能在 capability 与 kernel 不匹配时继续执行。

8×8 试验先以单条 K=32、group K=128、M=8/N=64/K=128 软件块进行，验证 A/B/C 各 256 B 的寄存器边界。比较 T=1/4/8/16/32 等，尾部单独统计。两个布局的有效行利用率、指令工作量和硬件成本不同，既要比较同一 workload 总时间，也要比较每 PE 的有效吞吐。

供数约束：int8 下满行列每拍注入一个 K-step，4×4 需要约 8 B/拍，8×8 约 16 B/拍，16×16 约 32 B/拍，指的是阵列边界 operand 供给，不是 DDR 带宽。当前 MOFU 为 2 请求/拍、8 B/word，名义上只有 16 B/拍；8×8 已没有多少仲裁/请求气泡余量，需要测量真实 MRF 吞吐。不能把名义带宽当成无冲突保证。

| 参数类别 | 首轮实验方式 | 扩容依据 |
|---|---|---|
| tile operation 容量 | 独立于 ingress，保留 2 的对照并测试 16，随后收缩 | PE 启动被在途容量限制 |
| 内部 k_block | 基线 4，分别测试 8/16 等合法值 | refill/块边界开销；不能改变有效工作量 |
| MOFU request width、MRF bank/端口 | 单独扫描 | operand 等待与真实 bank 服务能力 |
| SPM bank、interleave、队列 | 先保持 8 bank、64 B 条带基线 | 实际 bank 等待、并发冲突和布局偏斜 |
| DMA outstanding、描述符粒度 | 保持相同有效传输后单项扫描 | 队列/DDR/SPM 哪一侧限制供数 |
| AME context/slot 数量 | 第 7 步之后扫描 | 真实队列满或可利用的并行任务 |
| 8×8 prefill 阵列 | 与优化后 4×4 同 workload 比较 | M 足够、带宽匹配、单位资源收益 |

单 token decode 不以扩大方阵作为默认方向。正常 M×N 映射时 4×4/8×8/16×16 的有效行比例分别只有 1/4、1/8、1/16。输出通道并行的独立 tile 调度、单行宽输出映射或 K 维分担归并需要新的设计与测试，不能只修改 tile 网格参数获得。16×16 逻辑输出仅 int32 C 就需要 1024 B，超过当前 256 B Acc，留作后续容量/ISA 扩展课题。

**五、各层修改边界和主要文件**

以下路径相对于 gem5 根目录；列出的是预计涉及范围，不要求每一步都修改所有文件。

| 层次 | 文件／目录 | 职责 |
|---|---|---|
| SoC 配置 | `configs/example/spm_compute_soc.py` | local-only 连接、参数贯通、阶段运行配置 |
| ISA 能力 | `src/arch/riscv/RiscvISA.py`、`isa.cc`、`types.hh`、`regs/matrix.hh` | AMENLEN、AMEUDSZ、容量、序列化兼容性 |
| Ztt 语义 | `src/arch/riscv/isa/formats/ztt_*.isa` | 指令属性、动态跨度、功能执行 |
| host 下发／提交 | `src/cpu/minor/inst_queue.*`、`execute.*`、`scoreboard.*`、`exec_context.hh` | 依赖、取消、内存提交与可见性 |
| AME 引擎 | `src/computing_cluster/AME/engine.*`、`p1_pipeline.*`、`ame_dispatcher.*`、`ame_scoreboard.*` | 计数、context 生命周期、指令衔接 |
| SA 连续注入 | `src/computing_cluster/AME/systolic_array.*`、`operand_subsystem.*`、`accumulator.*`、`AMEEngine.py` | 注入/完成解耦、在途容量、buffer 和 Acc 生命周期 |
| RVV 引擎 | `src/computing_cluster/RVV/engine.*` 及 VLSU 路径 | local 访存、store timing、有效 lane 统计 |
| SPM/DMA 模型 | `src/mem/banked_scratchpad.*`、`src/dev/spm_virt_dma.*` | bank 服务、完成接口、真实搬运计数 |
| 算子 | `example/qwen3-0.6B/kernel/src/ztt_matmul.c`、`ztt_matmul.S` | packing、地址生成、group 累加、batch 布局 |
| guest 搬运／汇编 | `example/qwen3-0.6B/tools/gem5_dma_tile.c`、`gem5_packed_backend.c`、`gem5_ztt_group.S` | 阻塞/异步 DMA、SPM staging、核内循环 |
| 模型调用层 | `example/qwen3-0.6B/core/ztt_qwen_backend.c`、`qwen3_bare_w4a8_prefill_all.c` | prefill/decode 分路、所有 RVV 算子迁移、阶段标记 |
| 模型回归 | `AME/ame_backend.test.cc`、`RVV/rvv_backend.test.cc`（位于 computing_cluster）及 `src/cpu/minor/inst_queue.test.cc` | 依赖、取消、流水及背压验证 |
| SoC 集成回归 | `tests/spm_compute_soc/` | 共享数据正确性和 DDR 访问负例 |

第 3–6、8B 步先用 C、汇编及运行时接口实现，不先修改 LLVM 后端。改变编译参数或检查汇编不等于修改编译器。自动 DMA 插入、SPM 分配、分块、寄存器调度和软件流水化，在手工 kernel 的契约、正确性与收益稳定后再迁移到编译器。

第 8A 步和资源扩容代表所建模微架构的改变；如需对接 FPGA/RTL，必须实现相同的有限资源、部分和依赖、带宽和流水时序。gem5 的功能输出正确不等同于该硬件结构已经实现。

**六、统一实验与验收规范**

每个 run 保存：代码版本或 patch 标识、gem5/guest ELF 哈希、编译选项、模型哈希、输入 token IDs、prefill 长度、decode 起始 KV 长度及生成步数、所有硬件参数、CSR 能力、统计版本、阶段起止 tick、退出原因和数值对比结果。测试中的随机数据固定种子。

所有结果按相同数学工作量比较。更改 tile 后软件 tile 数和建模 MAC 数可能改变，应以有效输出与有效 MAC 为主；padding、重发和被取消的运算单独计数。记录是否包含 DMA、packing、scale、最终搬回和同步，禁止用仅 kernel 时间替代全阶段时间。

| 每次实验必须记录 | 解释 |
|---|---|
| 全阶段周期、prefill/decode 吞吐 | 完整阶段完成后才能计算；明确 token 分母 |
| 有效 MAC、执行 MAC、每全局周期利用率 | 区分数学工作、padding 与重发；注明 issue/result 口径 |
| CPU 指令数、IntDiv、准备区间 | 定位软件开销，不把不同单元忙时相加 |
| 各 opcode 接受/完成/取消 | 判断指令放大与控制成本 |
| SA 注入间隔、实际 PE issue 间隔、operation 占用 | 区分入口排队和真正计算 |
| SPM 每 bank 字节、等待、冲突、来源 | 不能用 MRF conflict 代替 SPM 竞争 |
| DMA 双向有效字节、请求字节、描述符延迟与 outstanding | 区分有用数据、粒度浪费和并发 |
| DMA/计算重叠区间、计算等待 DMA | 区间并集计算，避免时间重复相加 |
| MRF/Acc grant、等待和容量 | 识别片内供数限制 |
| 数值误差与完成证据 | 整数参考、浮点误差、算子/层/logits、正常退出 |
| 宿主耗时与日志配置 | 仿真器速度单独报告，不作为芯片利用率 |

整数 tile 使用独立标量参考，优先逐元素严格一致。浮点步骤在变更前确定并记录误差指标和阈值来源；累加顺序变化要记录最大绝对/相对误差，必要时与高精度参考比较。正确性未通过的性能结果不得进入收益结论。

正常性能运行使用必要统计与短窗口 trace；供数、取消和连续注入问题用定向 AMEFlow/AMEPE/RVV/SPM/DMA trace 复核。不要为每个整模型实验记录无限制逐 PE trace。原有计数不足或模型语义改变后，应补跑对应基线，不能将不同统计版本直接相除。

现有 SoC 回归命令（在 gem5 根目录执行）：

```sh
python3 tests/spm_compute_soc/run.py --outdir /tmp/qwen-local-spm-validation
```

该命令只验证现有小规模路径，不代表未来整模型迁移通过。新的 CSR、operation depth 和 guest 参数尚未实现，本文不提供假定已经存在的运行选项。

**七、下一批提交的具体范围与停止条件**

第一批只完成第 1 步和第 2 步的最小纵向路径：统计/取消口径、store timing 契约与回归、SPM buffer 接口、一个投影算子的全数据搬运与参考输出。随后扩展到全部算子并取得完整 prefill/decode 阻塞基线。

第二批依次完成第 3–6 步，每项保留独立数值和性能结果。第三批完成第 7 步、第 8A 步，再集成第 8B 步。第四批才引入 AMENLEN=8 和 2×2 tile 的 prefill 候选。

当前不实施的扩展：全局直接删除序列化标志、跨指令无限制 Acc 驻留、WS 连续注入、16×16 大阵列、decode 行重映射、新编译器自动调度。这些需要独立设计，不能作为本轮必要正确性修复的附带改动。

最终验收不是“所有参数都变大”，而是：完整 Qwen 在严格 local-SPM 路径上正确执行；每项收益可由同口径数据解释；软件减少的工作量与硬件提高的吞吐可分别量化；prefill、decode 的合适配置有独立实验依据。
