# Qwen3-0.6B + AME SoC 面试快问快答手册

这份手册对应当前仓库中的 source-built Qwen3-0.6B workload、4×4 AME、RVV 和共享 local SPM 模型。回答参数时要先说明实验边界：AME/RVV 的 timing port 只能访问 SPM，DDR 到 SPM 由 DMA 完成；当前 baseline 是 4×4、PSS、有效 K=4、阻塞 DMA、无连续注入、无 DMA 重叠。

## 一分钟项目介绍

项目把 Qwen3-0.6B 的 bare-metal 推理 workload 接到 gem5 模拟的 RISC-V computing cluster。标量控制和普通 cache 访问由 Minor CPU 完成，量化矩阵乘映射到 AME，attention 中的向量计算映射到 RVV。AME 和 RVV 共享一个多 bank SPM；它们不能直接通过系统总线取操作数，DMA 是 DDR 与 SPM 之间唯一的数据搬运通路。性能分析按 prefill/decode 阶段边界差分统计 PE issue/result、引擎活跃周期、SPM bank 等待、DMA descriptor 和 CPU 下发阻塞。

## 模型和张量尺寸

当前模型 blob 的运行时检查为：

| 参数 | 数值 |
|---|---:|
| hidden size `H` | 1024 |
| transformer layers `L` | 28 |
| vocabulary `V` | 151936 |
| query heads `Nq` | 16 |
| key/value heads `Nkv` | 8 |
| head dimension `Dh` | 128 |
| Q 投影维度 | 16×128 = 2048 |
| K/V 投影维度 | 8×128 = 1024 |
| FFN intermediate `I` | 3072 |
| quantization group size | 128 |
| RVV VLEN | 512 bit = 16 个 FP32 lane |

因此 GQA 的每个 KV head 被 2 个 query head 复用。单 token 的 Q、K、V 元素数分别是 2048、1024、1024；FP32 下分别为 8 KiB、4 KiB、4 KiB。KV cache 每层每个 token 是 `(Nkv×Dh)×2 = 2048` 个 FP16/FP32 逻辑元素，实际字节数取决于存储格式。

## 计算数学

### RMSNorm

对长度 `H=1024` 的向量 `x`：

```text
rms = sqrt((1/H) * Σ_i x_i² + ε)
y_i = γ_i * x_i / rms
```

工作量约为 `H` 次平方、`H-1` 次加法、一次倒数平方根和 `H` 次缩放。它是典型的 reduction，算术强度低，适合 RVV 分块归约；AME 不适合承担这种不规则的标量归约。

### Q/K/V 投影和输出投影

对 token 矩阵 `X∈R^{T×1024}`，权重矩阵为：

```text
Q = X Wq + bq,  Wq ∈ R^{1024×2048}
K = X Wk + bk,  Wk ∈ R^{1024×1024}
V = X Wv + bv,  Wv ∈ R^{1024×1024}
O = Attn W_o + bo, W_o ∈ R^{2048×1024}
```

每个输出元素是长度 1024 或 2048 的点积。量化权重按 group size 128 保存 int4 权重和 scale；当前 AME kernel 使用 int8 tile 形式供给，软件负责解包、scale 和 DMA staging。

### RoPE

对每个 head 的偶数/奇数坐标：

```text
y[2i]   = x[2i] cos θ_i - x[2i+1] sin θ_i
y[2i+1] = x[2i] sin θ_i + x[2i+1] cos θ_i
```

每个 Q head 和 KV head 都处理 `Dh=128`。这是连续向量运算，RVV 可以用 `e32` 向量化；DMA 只需在 RVV workspace 不足或跨阶段时搬运。

### Attention

第 `t` 个 token、query head `h` 的 GQA attention 为：

```text
s_{h,j} = (Q_{t,h} · K_{j,h/2}) / sqrt(Dh),  j∈[0,t]
p_{h,j} = softmax_j(s_{h,j})
z_{t,h} = Σ_{j=0}^t p_{h,j} V_{j,h/2}
```

prefill 对 `T` 个 token 形成近似 `T²` 的 QK/AV 工作；decode 每步只计算一个 query，但要读取历史 KV，因此延迟随上下文长度线性增长。QK 是长度 128 的 dot product，RVV 的 16-lane FP32 reduction 很合适；AV 是对 `Dh=128` 的加权累加，也适合 RVV。softmax 的最大值归约、指数、求和、归一化有数据依赖，通常保留在 RVV/标量辅助单元。

### Softmax

为避免溢出：

```text
m = max_j s_j
e_j = exp(s_j - m)
p_j = e_j / Σ_k e_k
```

它包含两次 reduction 和指数函数。应使用 RVV 分块处理，保存局部最大值/和；不要为了填满 AME 而把 softmax 强行改成矩阵乘。

### SwiGLU FFN

Qwen3 的门控 FFN 可写成：

```text
u = SiLU(X W_gate)
v = X W_up
h = u ⊙ v
Y = h W_down
```

其中 `W_gate,W_up∈R^{1024×3072}`，`W_down∈R^{3072×1024}`。AME 最适合三个大矩阵乘；SiLU、逐元素乘和 RMSNorm 由 RVV 处理。FFN 的矩阵乘通常比 attention 的小向量归约更能摊薄 DMA 开销。

### Prefill 和 decode

prefill 一次处理 `T` 个输入 token，矩阵乘的 M 维为 `T`，可以批量复用权重和提高 AME 利用率；decode 一次处理 1 个 token，M=1，主要瓶颈是权重/激活搬运、KV cache 和下发气泡。当前实验采用 prefill=4、decode=1 作为固定可复现实验输入，不能把两阶段混成一个平均 CPI。

## Tile、K block 和 SPM 参数怎么答

### 为什么当前是 4×4 tile

4×4 是当前 Ztt 指令和 AME physical PE array 的共同粒度：一个 tile 产生 16 个 int32 输出，`macLatency=3`、`macII=1` 时每个 PE 可按每周期一项发射。它的优点是 SPM footprint 小、边界简单、验证容易；缺点是 Qwen 的大矩阵会产生很多 tile，且 M=1 decode 很难填满阵列。

### K=4、K=16、K=128 分别代表什么

`k_block` 是 AME 内部供数和 drain 的分块，不等于模型量化 group。当前 baseline 每次只处理 4 个 K 元素。A 把一个逻辑 K=16 的指令填满四个 4×4 packed squares；B 把量化 group K=128 的 32 个 K=4 子块在 Acc 中保留，group 结束时才做 scale/drain。K 越大，DMA 和 drain 的固定开销越容易摊薄，但 SPM/MRF/Acc 容量、取消恢复和溢出风险越高。

### SPM 容量和布局

当前 SPM 为 1 MiB、8 banks、64 B interleave；AME scratch 使用低地址区，RVV workspace 使用 `0xc0001000`、`0xc0002000`、`0xc0003000` 三个 4 KiB 区域。每个 buffer 必须满足：起始地址按 cache line 对齐、不同活跃 producer 不发生 bank 热点、DMA 源仍在 DDR、消费者只读 SPM。

### DMA burst 和 outstanding

当前 DMA burst 为 64 B，descriptor queue=8，单 bank outstanding=4，系统侧 outstanding=16。64 B 与 SPM interleave/cache line 对齐，适合 4×4 int8 tile（A/B 各 64 B）和 FP32 RVV 向量的合并请求。调大 burst 可以减少 descriptor 数，但会增加尾部浪费和 bank 占用；调大 outstanding 可以隐藏 DDR 延迟，但超过 SPM bank queue 或 response queue 后只会增加重试和拥塞。面试中应强调必须用 `requestAccepted/requestRetries/bankWaitTicks` 验证，而不是只看理论带宽。

## 当前 AME SoC 参数及影响

| 参数 | 当前值 | 影响 |
|---|---:|---|
| PE array | 4×4 | 16 个并行 MAC，适合 4×4 Ztt tile |
| physical tile grid | 1×1 | 没有跨 tile edge transfer，控制简单 |
| dataflow | PSS | Acc/psum 在阵列内传播，适合连续 K 注入 |
| MAC latency / II | 3 / 1 | 决定 PE pipeline 深度和理论 issue 上限 |
| tile ingress depth | 2 | 太小会造成 wavefront/injection stall |
| operation depth | 2 baseline；C 使用 12 | 连续注入的在途 wave 数 |
| MRF banks | 8 | A/B operand 读取并行度 |
| Acc banks | 4 | PSS group accumulation 的读写冲突来源 |
| SPM read/write latency | 2 / 1 cycles | 影响 DMA 响应和 RVV/AME 供数间隔 |
| AME request width | 2/cycle | MOFU operand 发射上限 |
| AME max inflight | 8 | CPU 下发和取消恢复的窗口 |

AME 的 `saActiveCycles` 是阵列有供数、计算或排空工作时的周期；PE 利用率必须用实际到达发射周期的 `saPeIssues/(cycles×16/II)` 计算。不能用 AME activeCycles 代替算术利用率。

## A/B/C/D 优化如何回答

### A：K=16 packed squares

算子把连续四个 K=4 子块打包进一条 AME 供数序列，减少指令、DMA descriptor 和每块初始化。主要改算子 packing、Ztt operand layout 和有效 MAC 计数；模拟器只需正确处理 `nSquares/kLogical`。风险是尾 K、权重 stride、scale 顺序和 padding 错误。

### B：K=128 group 内保留 Acc

group 内执行 32 个 K=4 子块，只在 group 完成后执行：

```text
acc_j = Σ_{k=0}^{127} a_k b_{k,j}
out_j += float(acc_j) * x_scale * w_scale_j
```

它减少 Acc seed/drain 和 scale 次数，但改变浮点转换位置，必须与 K=4 参考比较误差；整数累加要检查最大值和 Acc 位宽。

### C：PSS 连续注入

保持前一个 K block 的 psum resident，在前一 wave 排空前注入后续 wave。需要扩大 ingress/operation depth，并保证 token 的 context、output block、k block 标签不乱序。收益来自减少 drain/reseed 气泡；风险是 FIFO 满、边界背压和取消清理。

### D：DMA 双缓冲与计算重叠

使用 buffer 0 计算时，DMA 将下一 group 搬入 buffer 1，随后交换。必须保持 source lifetime，等待正确的 descriptor 完成，并避免同 bank 冲突。D 的有效收益条件是：

```text
DMA_next_time <= compute_current_time
```

否则双缓冲只增加控制复杂度，不会降低阶段时延。

## AME + RVV 联合优化方向

最有效的联合策略是按算子特性分工：AME 负责 Q/K/V、O 和 FFN 的大矩阵乘；RVV 负责量化/反量化、RMSNorm、RoPE、QK/AV 小向量归约、softmax 和 SiLU。两者通过 SPM 生产者/消费者队列连接，避免中间结果回 DDR。

下一步可做：

1. 在 prefill 中以 M=4 或更大 token batch 映射 AME 的 M 维，decode 仍保留 M=1 专用路径。
2. 对 QK/AV 采用 RVV 双缓冲 workspace，使下一 head 的 DMA 与当前 head 的向量计算重叠。
3. 让 AME 输出直接落入 RVV 可消费的 SPM bank stripe，减少一次 DMA 回写再读。
4. 为 SPM 建立 bank-aware allocator，分别给 AME A/B/C、RVV Q/K/V、KV cache 分配无冲突 bank。
5. 增加动态调度：根据阶段和上下文长度选择 K=16/K=128、burst 和 outstanding，而不是全模型固定一个 profile。
6. 对 softmax 使用 RVV online softmax，减少 attention score 的完整写回；对 FFN 使用 AME→RVV 融合流水。
7. 将 PE 实际 issue/result、RVV active、DMA wait 和 SPM bank wait 放到同一时间轴，按 overlap 分解瓶颈。

## 高频追问及标准回答

**问：为什么 AME 不直接接 DDR？**

答：架构约束是 AME/RVV 只能共享 local SPM。这样可以明确 scratchpad 延迟、bank 冲突和数据复用，避免 accelerator 绕过互联缓存制造不可比的带宽。DDR 访问由 DMA 显式完成，SPM 是唯一计算侧数据入口。

**问：为什么不能只看 AME 利用率？**

答：AME active 可能包含供数、排空和等待；真正算术利用率要用实际 PE issue 数除以理论 issue 槽位。还要同时看 DMA、SPM bank wait 和 CPU dispatch stall。

**问：prefill 一定比 decode 快吗？**

答：不一定。prefill 有更大的 M 维，能提高矩阵阵列利用率，但 attention 的 T² 工作也增长；decode 的 M=1 阵列利用率低，且 KV cache/权重搬运占比高。必须分别报告两个阶段。

**问：如何证明优化没有改坏结果？**

答：固定模型 blob、输入 token、seed、编译器和 `-ffp-contract=off`；每组建立匹配的 native reference，比较 prefill/decode logits、有限性、argmax 和最大绝对误差，同时检查 expected tile/layer count。

**问：后续最值得做什么？**

答：先完成 A→AB→ABC→ABCD 的阶段数据闭环，再根据计数决定方向：若 PE issue 低而 DMA wait 高，优先 D 和 bank-aware layout；若 drain/injection stall 高，优先 C；若指令和 scale 开销高，优先 A/B；若 decode 仍受 M=1 限制，则做 token batching 或 AME/RVV 专用 decode 数据流。

## 实验复现入口

```bash
python3 example/qwen3-0.6B/tools/build_baseline.py \
  runs/<variant>/guest.elf --variant A|AB|ABC|ABCD \
  --prefill-tokens=4 --decode-steps=1

python3 example/qwen3-0.6B/tools/run_baseline.py \
  --binary runs/<variant>/guest.elf \
  --outdir runs/<variant>/full_model \
  --reference-prefix runs/<variant>/reference_output \
  --simulator build/<independent-gem5-build>/gem5.opt
```

阶段完成后使用：

```bash
python3 example/qwen3-0.6B/tools/analyze_baseline.py runs/<variant>/full_model
```

输出中的 `cycles`、`pe_issue_utilization`、`dma_*`、`spm_*` 和 CPU stall 计数才是性能结论的依据；未完成阶段、微测试和截断 trace 不能冒充完整模型结果。
