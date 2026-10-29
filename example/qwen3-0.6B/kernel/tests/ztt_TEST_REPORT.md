# Qwen3 W4A8 Ztt Operator Test Report

## Scope

This overlay dispatches the existing Qwen v6 W4A8 inference call contract to the Ztt tile primitive. The host executable is the x86 golden producer; each RISC-V guest also checks an independent scalar W4A8 oracle before writing its result report.

> This result is now part of the top-level `ISA/test` regression. The shared
> runner writes all operator artifacts under `ISA/test/build/`.

## Run Metadata

| Field | Value |
| --- | --- |
| UTC timestamp | 2026-08-20T06:44:15Z |
| Host golden | Native `cc` on `x86_64` |
| Target | `riscv64-linux-gnu`, `rv64gcv_xxiangshanztt`, static ELF |
| LLVM | Resolved from `TOOLCHAIN_ROOT/llvm-project_riscv/build` |
| gem5 | Resolved from `TOOLCHAIN_ROOT/AMEgem5/build/RISCV/gem5.opt` |
| Oracle | Scalar W4A8 reference using logical `[D][N]` weights, then byte-for-byte host/guest report comparison |

## Operator Build And Execution

| Stage | Status | Evidence |
| --- | --- | --- |
| Host + RISC-V operator build | PASS | `ISA/test/build/gemm_e2e_build.log` |
| Ztt v0.6 disassembly audit | PASS | `ame.acquire msettyp asettyp mzero.2d.acc mls.rm mmulacc.2d mmov.m.a mss.rm ame.release` in `ISA/test/build/disasm/qwen_w4a8_operator.dis` |
| x86 scalar golden | PASS | `33ef0a16ff19d2c5cafc103f1b9df4e63c09ada061320b8303f2c4c1abb2af31` |
| gem5 AtomicSimpleCPU | PASS / PASS | `33ef0a16ff19d2c5cafc103f1b9df4e63c09ada061320b8303f2c4c1abb2af31` |
| gem5 MinorCPU | PASS / PASS | `33ef0a16ff19d2c5cafc103f1b9df4e63c09ada061320b8303f2c4c1abb2af31` |
| gem5 RiscvO3CPU | PASS / PASS | `33ef0a16ff19d2c5cafc103f1b9df4e63c09ada061320b8303f2c4c1abb2af31` |

## Operator Cases

| Case | API Coverage | Geometry | Validation |
| --- | --- | --- | --- |
| Decode | `ztt_matmul_quant_i8_i8_f32` | `T=1 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |
| Prefill | `ztt_matmul_quant_batch_i8_i8_f32` | `T=2 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |
| Paired FFN | `ztt_matmul_quant_pair_i8_i8_f32` | `T=1 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |
| Batched paired FFN | `ztt_matmul_quant_batch_pair_i8_i8_f32` | `T=1 N=256 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |
| Tail regression | batch API, non-divisible N/D | `T=1 N=130 D=66 GS=64` | verifies ceil group indexing for batch scales |

## Full Qwen Integration

| Item | Status | Note |
| --- | --- | --- |
| Shared-infrastructure Ztt Qwen ELF | NOT RUN | Optional full-image build; the top-level regression validates the operator ELF and does not load the 666 MiB model blob |
| Full Qwen inference in gem5 | NOT RUN | The embedded v6 model blob is approximately 666 MiB; operator-level Atomic/Minor/O3 validation is the practical coverage gate for this run. |

## Result

**PASS (operator scope)** - all W4A8 operator cases passed their scalar oracle and matched the x86 golden in Atomic, Minor, and O3 gem5.
