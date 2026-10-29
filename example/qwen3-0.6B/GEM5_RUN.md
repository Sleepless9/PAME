# Qwen3 Ztt timing run

## Current source-built local-SPM baseline

The source-built 4×4 baseline and its validation/run commands are documented in
[`BASELINE_EXPERIMENT.md`](BASELINE_EXPERIMENT.md). It uses the supplied Ztt Clang
toolchain, explicit blocking DMA for AME/RVV operands, effective K=4, and plain
`trace.out`. The prefill=4/decode=1 full-model run is in
`runs/qwen3_baseline_4x4_20260908/full_model`; consult its `run.json` for current
status. Starting a run does not establish full-model numerical correctness or
completed phase performance. A/B/C/D optimizations are not enabled.

## Local-SPM-only timing topology

`configs/example/spm_compute_soc.py` connects both accelerator memory ports
directly to the same `BankedScratchpad`. Only `SPMVirtDma` has a system-side
data-transfer port through the coherence cache and DDR interconnect. RVV and
AME loads/stores must use the SPM window; a DDR address is rejected by the
scratchpad's range check. CPU scalar accesses retain their system memory path.

The original FPGA ELF uses DDR/stack operands for AME and RVV and therefore
cannot run unchanged in this timing topology. Wrapping only the AME tile with
DMA is insufficient: RVV packing and the remaining vector kernels also need
SPM staging. The commands below document the earlier DDR-capable runs, not a
validated full-model run of the corrected topology. Old checkpoints preserve
those guest addresses and do not supply the missing DMA staging.

The `runs/qwen3_original_final_20260908` logs were produced before this topology
correction. Their zero SPM/DMA counters must not be used to characterize the
local-SPM architecture. `--cpu-mode=atomic` remains a functional diagnostic
without accelerator timing, not a validation of the local-only data path.

## Historical direct run of the supplied FPGA ELF

```sh
build/RISCV/gem5.opt --listener-mode=off -d runs/qwen3_original \
  --debug-flags=AMEFlow,PseudoInst,SPMVirtDma,BankedSPM \
  --debug-file=trace.out.gz configs/example/spm_compute_soc.py \
  example/qwen3_chat_w4a8_ame_fpga > runs/qwen3_original.log 2>&1
```

The default CPU is Minor with the RVV and AME timing engines. `AMEFlow`
records instruction/resource events and supply stalls; `AMEPE` adds every
PE MAC issue/result. `AMEEngine` enables both. `--cpu-mode=atomic` is a
separate functional diagnostic and supplies no AME timing evidence.

The configuration recognizes the supplied ELF by SHA-256
`39b8e434f8d48ef57c1623f9920bba2b0adf03d84139b01da165abef89cba67c`.
After loading it, two startup fixes are applied to simulated memory: the
stack initialization at `0x864000f2` is relocated below code, and the MS
enable constant at `0x86400520` uses bits 24:23. The original disk ELF,
model entry, kernels and weights are retained. `boot_compat.json` records
the exact original/replacement bytes. `--no-qwen-fpga-compat` disables this
behavior for diagnostics. Other ELF hashes receive no boot patches.

For an automatically checked run, use
`python3 example/qwen3-0.6B/tools/run_original_gem5.py --outdir runs/qwen3_original_checked`.
The wrapper saves a drained `checkpoint.prefill` at workbegin phase 5.
Use `--restore-checkpoint PATH` on the wrapper for a subsequent timing
retry from that point. The wrapper writes its command, input hash and eventual completion verdict
to `run.json`; progress lines report the original ELF's `g_tiles` and
`g_timing.n_layers` counters. UART output is `system.platform.terminal`. Successful completion requires
the actual model output/timing report and a zero-code `m5_exit`; a progress
tick or an existing trace file does not demonstrate completion. This
particular ELF was compiled with one generated token: its generation loop
samples the prefill logits without a second model forward. It does not
exercise DMA registers; DMA/SPM regression traces must be identified
separately from its model trace.

## Earlier generated workload diagnostics

The generated frontend and packed-kernel runs below are separate diagnostic
artifacts and are not evidence that the original FPGA ELF completed.

The supplied FPGA ELF embeds the complete v6 model. Its compiled generation
limit is one output token, which does not call the autoregressive `forward`
routine. Its original startup also places the stack inside `.model_blob`.

`tools/build_random_elf.py` builds a separate ELF with a new frontend below
the original image. It reuses the model loading, prefill and decode C routines
from `core/qwen3_bare_w4a8_prefill_all.c` and resolves the existing Ztt/RVV
kernel helpers at their original ELF addresses. This also works when the
`kernel/`, `platform/` and `tokenizer/` source directories are unavailable. Original weights are
preserved. Generated C, object files, the linker script/map and a JSON manifest
are kept beside the output ELF.

The workload input is one reproducible random token ID (xorshift32,
seed 20260907, token 125846). Tokenization is unnecessary for a token-ID input.
Prefill processes all 28 layers and the LM head. The highest-logit token is
then fed into a complete decode forward at position 1, using the populated KV
cache. Completion requires 56 layer passes and finite logits in both phases.
`QWEN_FULL_RANDOM PASS` is printed only after these checks.

With `--dma`, each invocation of the original Ztt instruction kernel is
wrapped by guest DMA transfers: 64 bytes of A to SPM bank 0, 64 bytes of B to
bank 1, AME execution, and 64 bytes of C from bank 2 to system memory. This is
a blocking supply baseline. A/B each contain four packed int8 squares; the
original kernel zeros the unused squares. DMA does actual data movement.

Run these commands from the gem5 repository root:

```sh
python3 example/qwen3-0.6B/tools/build_random_elf.py \
  example/qwen3_chat_w4a8_ame_fpga runs/qwen3_random_dma.elf --dma

build/RISCV/gem5.opt --listener-mode=off -d runs/qwen3_full \
  --debug-flags=AMEEngine,PseudoInst,SPMVirtDma,BankedSPM \
  --debug-file=trace.out.gz configs/example/spm_compute_soc.py \
  runs/qwen3_random_dma.elf > runs/qwen3_full.log 2>&1

python3 example/qwen3-0.6B/tools/summarize_trace.py \
  runs/qwen3_full/trace.out.gz --output runs/qwen3_full/analysis
```

The configuration uses 1 GiB DRAM at `0x80000000`, 1 MiB SPM at
`0xc0000000`, DMA registers at `0x40010000`, 512-bit RVV, and 32 KiB instruction
and data caches. SPM has eight banks, 64-byte interleaving, a one-cycle issue
interval, two-cycle reads and one-cycle writes. CPU and accelerator clocks
default to 1 GHz. Tick values in traces are picoseconds.
The DMA system port uses a 1 KiB, eight-way coherence cache with one-cycle
tag/data/response latency; this converts ordinary DMA writes into coherent
transactions against CPU caches. The frontend enables `mstatus.MS[24:23]`,
matching this repository's Ztt CSR implementation.

`--max-ticks` is a diagnostic limit; reaching it returns status 124 and is
not successful model completion. UART output is in `system.platform.terminal`.
The trace includes `SUPPLY_STALL_BEGIN/END`, DMA transaction completion and
latency, SPM `BANK` arrival/start/wait, resource `STALL`, and response events.
The summary emits JSON latency distributions and an event CSV; gzip integrity
does not by itself establish that the model completed.

For the small data-path regression, build with `--dma --tile-test`. It checks
constant and signed diagonal matrix products through the same supply path.

The RVV packing regression can emit BadAddressError warnings for speculative
scalar loads after a mispredicted function return. The captured example at
tick 212185000 is FLW PC 0x86000ad2, execution ID 82695, address 0x20.
Minor discards that instruction and its response at tick 212298000 after
the stream changes from 650 to 651. This is distinct from an architectural
address fault or an accelerator VLSU request; see the earlier diagnostic
`runs/qwen3_ame_20260907/zero_cpu_diagnostic/trace.out`.
