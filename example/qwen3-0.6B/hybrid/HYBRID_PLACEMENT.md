# Qwen3-0.6B RVV + Ztt hybrid placement

This directory is self-contained for the hybrid build. It consumes the model,
tokenizer, generic Qwen front-end, and existing accelerator components from the
parent tree without adding hybrid targets to the parent `core/Makefile`.

## Operator placement

| Qwen operation | Engine | Reason |
|---|---|---|
| Prefill QKV, WO, W1/W3, W2, LM head | Ztt/AME | Multiple token rows reuse each 64-output weight block; four token rows occupy a 4x4 Ztt tile. |
| Decode QKV, WO, W1/W3, W2, LM head | RVV | Decode is `M=1`; Ztt would leave three matrix rows idle. |
| RMSNorm and Q/K norm | RVV | Vector reduction followed by elementwise scale. |
| RoPE | RVV | Elementwise trigonometric rotation across each head. |
| Attention QK, online softmax, AV | RVV | Reductions and irregular causal sequence lengths fit vector execution better than a fixed matrix tile. |
| Residual add and SiLU/gating | RVV | Contiguous elementwise vectors. |
| Quantize/dequantize and copies | RVV | Contiguous conversion and movement. |
| Tokenizer, sampling, loop/control | MinorCPU | Scalar control-heavy work. |

The batch Ztt kernel handles up to four prefill tokens in each matrix tile. The
single-row front-end entry always selects RVV, so decode does not accidentally
use the prefill path. Weights use the existing v6 layout
`[output/64][input/group][64][group]`; the DMA staging code preserves this
layout and only transposes the current 4x4 operand into the AME matrix register.

## KV cache and FlashAttention

K is normalized and RoPE-rotated before K/V are written to the persistent DDR
KV cache. FlashAttention does not create a second cache. It consumes the cache
in 16-token blocks:

1. `SPMVirtDma` stages Q and a K block from DDR into shared SPM.
2. RVV computes QK scores in SPM.
3. `SPMVirtDma` stages the matching V block.
4. RVV performs online softmax and AV accumulation in SPM.
5. `SPMVirtDma` writes the attention output to DDR.

This avoids materializing the full attention matrix and reuses each K/V block
across all query heads assigned to the same grouped-query KV head.

## SoC data path

```text
                         +---------------- MinorCPU ----------------+
                         | scalar pipeline -> shared InstQueue      |
                         |                    | RVV instructions     |
                         |                    +----------------------> RVVEngine
                         |                    | Ztt instructions     |
                         |                    +----------------------> AMEEngine
                         +-------------------------------------------+
                             | I/D              | memory      | memory
                             +------------------+-------------+
                                                |
                                           SystemXBar
                                  +-------------+-------------+
                                  |             |             |
                                DDR3       SPMVirtDma PIO   mapped SPM
                                  ^             |
                                  | DMA master  | private timing port
                                  +-------------+------> shared banked SPM
```

The mapped and private SPM ports share one `BankedScratchpad` byte store and
bank scheduler. RVV/AME accesses therefore contend with DMA at bank granularity.
The CPU data port bypasses a write-back cache, avoiding incoherent CPU/DMA
views. `MinorInstQueue` is the only dispatch/completion boundary for RVV and Ztt
instructions.

## Build and run

```bash
cd example/qwen3-0.6B/hybrid
./run.sh build-guest
./run.sh build-gem5
./run.sh run
```

The gem5 binary is built as `build/QWEN_HYBRID_RISCV/gem5.opt`; the existing
`build/RISCV/gem5.opt` is never rebuilt or overwritten. `GEN_STEPS` defaults to
2 because the first generated token consumes prefill logits and the second step
is the first decode forward pass.
