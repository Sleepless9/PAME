# Qwen3-0.6B Real W4A8 Port

This directory is a self-contained Qwen3-0.6B port for the repository's
example/common build and report flow. It contains the real FP checkpoint,
the reproducible W4A8 conversion, the real tokenizer, every model operator,
the Ztt matrix backend, and a complete prefill/decode front end.

## Included assets

| Path | Purpose |
| --- | --- |
| model/model.safetensors | Original FP checkpoint (1,503,300,328 bytes) |
| model/qwen3-0.6b-int4-v5.bin | Row-major per-group INT4 weights + FP32 scales |
| model/qwen3-0.6b-int4-tile-v6.bin | Runtime W4A8 image: INT8 tile expansion + group-major FP32 scales |
| tokenizer/data/ | Real Qwen vocabulary, merges, tokenizer JSON, and embedded binary |
| quantization/ | Safetensors-to-v5 quantizer, v5-to-v6 repacker, and packing tests |
| operators/ | Ztt 4x4 INT8 matrix kernel and full Qwen backend adapters |
| main.c | Token input, all 28 transformer layers, KV cache, LM head, sampling, decode |

model/SHA256SUMS and tools/verify_assets.py verify the checkpoint, quantized
images, tokenizer, header geometry, and bit widths. The configured model is
hidden=1024, layers=28, vocab=151936, q_heads=16, kv_heads=8, head_dim=128,
intermediate=3072, W4A8 group size 128.

## Default full-model regression

    make -C example/qwen3_0_6b test

The default bounded regression uses the real tokenizer mapping for Hello
(token_id=9707), runs one-token prefill through all 28 layers, then performs
one autoregressive decode forward through all 28 layers. GEN_STEPS=2 is
intentional: the first sampled token is the decode input and the second sample
observes its logits. The required result is therefore exactly 56 real layer
passes, 226 full quantized projections, the complete 151,936-row LM head, and
18,624,512 K16 4x4 tiles. The port uses the simulator's architectural
AMEUDSZ=32 default, so these map one-to-one to 18,624,512 physical
`mmulacc.2d` instructions. No simulator arithmetic or memory fast path is
required by this example.

The tokenizer assets remain embedded and checksummed. To keep the MinorCPU
regression focused on model execution, the already verified single-token
mapping is compiled into the guest instead of rebuilding the 151,643-entry
BPE hash tables under cycle-accurate simulation. tools/verify_assets.py checks
that tokenizer/data/vocab.json still maps Hello to 9707.

The Host binary executes the same v6 tiled weights with the scalar-equivalent
Ztt kernel. The guest repacks each real K128 group into eight architectural
K16 transfers and executes `mls.rm` plus `mmulacc.2d`; it does not replace any
real weights or MACs with procedural data. The default gem5
AtomicSimpleCPU validation executes the complete
guest ELF, including every real Ztt instruction, without detailed pipeline
timing; `make minor-run` runs the identical ELF on MinorCPU when cycle-level
performance simulation is required. Both write a compact
deterministic report containing prompt/generated IDs, layer count, state and
logit hashes, and top-5 token/logit bit patterns. common/case_rules.mk requires
the reports to match byte-for-byte.

## Build and validation targets

    make -C example/qwen3_0_6b asset-check  # hashes and binary headers
    make -C example/qwen3_0_6b all          # Host + RISC-V ELF + audit
    make -C example/qwen3_0_6b run          # Host/Atomic full byte comparison
    make -C example/qwen3_0_6b test         # trace assertions + report
    make -C example qwen3_0_6b              # standard suite entry
    make -C example/qwen3_0_6b minor-run    # optional detailed MinorCPU timing

The full guest ELF embeds the 697,524,352-byte v6 model and 6,223,797-byte
tokenizer (151,643 vocabulary records and all 151,387 tokenizer.json merges)
and runs with MEM_SIZE=1GiB. Socket listeners are disabled for
batch and sandbox execution.

## Reproduce quantization

The Python environment must provide PyTorch, NumPy, and safetensors. The
repository-local environment used for this port is the default in the
Makefile.

    make -C example/qwen3_0_6b quantize-v5
    make -C example/qwen3_0_6b repack-v6
    make -C example/qwen3_0_6b tokenizer-bin
    make -C example/qwen3_0_6b requantize

The v5 conversion uses symmetric signed INT4 [-7, 7], group size 128,
16-point MSE clipping, row chunks of 1024, and SmoothQuant-style input
smoothing (alpha=0.5, maximum scale 8.0). Activations are quantized to signed
INT8 at runtime. The v6 repacker expands matrix weights to tile-major INT8
[D/64][N/128][64][128] and transposes scales to group-major [N/128][D];
embedding rows remain packed INT4.

## Weight locations

The original source checkpoint provided for this port is
`/nfs/home/xiangyuanyang/001_research/ISA/test/ztt_llm/qwen3-0.6b/model/qwen3-w4a8/convert_tools/model.safetensors`.
The self-contained copy used by this example is `model/model.safetensors`;
runtime inference embeds `model/qwen3-0.6b-int4-tile-v6.bin`.

## Output

The standard report is build/reports/REPORT.md. Raw deterministic reports,
Minor trace, disassembly, build logs, and gem5 output stay under build/.
