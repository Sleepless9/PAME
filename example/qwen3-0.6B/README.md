# Qwen3-0.6B W4A8 on MinorCPU + RVVEngine / AMEEngine

This directory is a standalone bare-metal Qwen3-0.6B inference workload for
the cycle-accurate RVV and AME engines in this gem5 tree. The ELF embeds the prepacked
W4A8 model and tokenizer, runs the complete tokenizer, model-load, state-build,
prefill, generation, and output path, then exits through a gem5 pseudo-op.

The Qwen workload uses a pure RVV W4A8 backend. Quantized matrix tiles are
loaded with RVV byte operations, widened with `vwmul.vv`, and reduced with
`vwredsum.vs`; QK/AV, memset, memcpy, accumulation, and the transformer GEMMs
all use the same RVV execution path. RVV instructions are sent through
MinorCPU's instruction queue to the independent `RVVEngine` ClockedObject.
MinorCPU retains architectural execution and retirement; RVVEngine supplies
the lane/FU/dependency timing and instruction-queue completion handshake.

The AME backend is driven by `AMEEngine`, while its operator and kernel
implementation comes from the migrated `ztt-qwen3-w4a8` overlay. Ztt sources
are explicitly named with the `ztt` prefix (`core/ztt_*` and
`kernel/*/ztt_*`). The wrapper keeps the shared AME API used by the Qwen
front-end, and `ztt_matmul.S` emits `rv64gcv_xxiangshanztt` instructions that
enter the AMEEngine through MinorInstQueue.

## Reproduce

From this directory:

```sh
scripts/run.sh all
```

Run the AMEEngine + Ztt path with MinorCPU and timing DDR3:

```sh
BACKEND=ame CORE_BUILD=build/ame RUN_NAME=qwen3_ame_ztt \
  GEN_STEPS=1 MINIMAL_PROMPT=1 scripts/run.sh all
```

For an unattended end-to-end run which survives the invoking terminal, use:

```sh
BACKEND=ame CORE_BUILD=build/ame RUN_NAME=qwen3_ame_ztt_e2e \
  GEN_STEPS=1 MINIMAL_PROMPT=1 scripts/run.sh start

BACKEND=ame RUN_NAME=qwen3_ame_ztt_e2e scripts/run.sh status
```

The selected detail phase and cycle count control trace verbosity only. When
the detailed window ends, gem5 continues through prefill, generation, output,
`=== done ===`, and the final `m5_exit`. `QwenAMETrace`/`QwenRVVTrace` retain
phase and MinorInstQueue dispatch/completion/retirement events throughout the
run; only the high-volume internal engine events are windowed.

The default validation workload passes raw prompt `Hello` through the real
tokenizer and uses one generation step. It omits the 21-token chat wrapper but
still traverses all 28 transformer layers and the complete inference flow.
Longer generation, the full chat template, and a different lane topology can
be selected with:

```sh
MINIMAL_PROMPT=0 GEN_STEPS=8 NUM_LANES=8 VLEN=512 scripts/run.sh all
```

`MINIMAL_PROMPT=0` is substantially slower in a cycle-accurate simulator: the
default chat template contains 22 prefill tokens for `Hello`, versus one token
in validation mode.

Individual operations are available as `build-guest`, `build-gem5`, `run`,
and `analyze`. `RUN_DIR` selects an existing or new result directory. Changing
guest compile-time settings requires `build-guest`; the script cleans only its
dedicated `core/build/gem5` directory so stale compiler flags cannot be reused.
The gem5 build uses HiFive's 1-byte 8250 UART register spacing. For the source
FPGA platform, pass `UART_REG_FLAGS=-DFPGA_UART_4BYTE` directly to `make`.

## Trace and profiling

`QwenRVVTrace` and `QwenAMETrace` are the dedicated full-run lifecycle flags;
each contains `MinorInstQueue` and `PseudoInst`. The runner dynamically enables
the corresponding engine's detailed events for a bounded window, defaulting to
the first 250,000 prefill cycles. Set
`RVV_DETAIL_PHASE` and `RVV_DETAIL_CYCLES` to move or resize this window. This
keeps the full lifecycle trace while preventing multi-gigabyte lane logs. A
run preserves:

- `rvv_trace.log.gz` or `ame_trace.log.gz`: gzip-compressed instruction queue, dependency, lane, VRF/FU or AME pipeline,
  writeback, completion, retirement, and work-phase events.
- `uart.log`: guest progress, generated output, transformer stage timings,
  prefill/decode split, selected-layer detail, and backend/RVV kernel profiling.
- `phase_events.json`: exact gem5 begin/end ticks for all seven phases.
- `stats.txt` and `stats_<phase>.txt`: gem5 statistics reset and dumped at each
  guest phase boundary.
- `phase_profile.csv` and `phase_profile.md`: CPU/RVV event and latency summary.
- `rvv_instruction_latency.csv` or `ame_instruction_latency.csv`: dispatch-to-completion latency per accelerator instruction.
- `lane_profile.csv`: per-phase activity for each physical lane.
- `stall_profile.csv`: dependency/operand stall reason counts.

The RVV and ISA vector lengths must match. The supplied configuration sets both
to 512 bits and defaults to four 64-bit lanes with eight VRF banks per lane.
MinorCPU uses a 32 KiB 2-way I-cache and a 64 KiB 4-way D-cache in front of the
timing DDR3 controller; these fixed parameters are part of the reported model.

## Runtime contents

The copied runtime deliberately includes only inputs required to rebuild and
run the embedded ELF: source, platform code, linker files, packed tokenizer,
configuration metadata, and the 697,524,352-byte prepacked int4 tile model.
Raw safetensors and conversion intermediates are not required.

The RVV flavor builds large-block bare-metal `memset` with RVV. This removes a
scalar initialization bottleneck in tokenizer setup and makes memory
initialization visible in the same RVV lane trace. Small `memcpy` operations
remain scalar.

The timing model is intentionally cycle-detailed down to the lane/FU or AME
pipeline and DDR3
transactions. A full 0.6B inference is therefore much slower than native
execution. An interrupted run remains useful for auditing: `phase_events.json`
records only completed phases, while `analysis_summary.json` reports any
in-flight RVV instructions left at interruption. A completed run ends with all
seven phase entries, guest `=== done ===`, and a closed `rvv_trace.log.gz`.
