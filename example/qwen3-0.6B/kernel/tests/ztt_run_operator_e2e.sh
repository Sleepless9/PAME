#!/bin/sh

set -u

TEST_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ZTT_ROOT=$(CDPATH= cd -- "$TEST_DIR/.." && pwd)
ISA_TEST_ROOT=$(CDPATH= cd -- "$TEST_DIR/../../../../.." && pwd)
TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT:-$TEST_DIR/../../../../../../..}
TOOLCHAIN_ROOT=$(CDPATH= cd -- "$TOOLCHAIN_ROOT" && pwd)
BUILD=${BUILD:-$ISA_TEST_ROOT/build}
LLVM_BUILD=${LLVM_BUILD:-$TOOLCHAIN_ROOT/llvm-project_riscv/build}
CLANG=${CLANG:-$LLVM_BUILD/bin/clang}
OBJDUMP=${OBJDUMP:-$LLVM_BUILD/bin/llvm-objdump}
GEM5=${GEM5:-$TOOLCHAIN_ROOT/AMEgem5/build/RISCV/gem5.opt}
HOST_CC=${HOST_CC:-cc}
REPORT=${REPORT:-$BUILD/QWEN3_W4A8_OPERATOR_TEST_REPORT.md}

HOST_BIN=$BUILD/host/w4a8_operator_test
TARGET_BIN=$BUILD/target/w4a8_operator_test
HOST_REPORT=$BUILD/reports/w4a8_operator.host.txt
ATOMIC_REPORT=$BUILD/reports/w4a8_operator.atomic.txt
MINOR_REPORT=$BUILD/reports/w4a8_operator.minor.txt
O3_REPORT=$BUILD/reports/w4a8_operator.o3.txt
BUILD_LOG=$BUILD/logs/build.log
FULL_BUILD_LOG=$BUILD/logs/full_qwen_build.log
ATOMIC_LOG=$BUILD/logs/atomic.log
MINOR_LOG=$BUILD/logs/minor.log
O3_LOG=$BUILD/logs/o3.log
DISASM=$BUILD/disasm/w4a8_operator_test.dis

mkdir -p "$BUILD/reports" "$BUILD/logs" "$BUILD/disasm" "$BUILD/m5out"

append() {
    printf '%s\n' "$1" >> "$REPORT"
}

tool_version() {
    "$1" --version 2>/dev/null | sed -n '1p'
}

file_hash() {
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
}

overall=0
build_status=PASS
host_status=SKIP
atomic_status=SKIP
minor_status=SKIP
o3_status=SKIP
disasm_status=SKIP
atomic_compare=SKIP
minor_compare=SKIP
o3_compare=SKIP
full_build_status=SKIP

: > "$REPORT"
append '# Qwen3 W4A8 Ztt Operator Test Report'
append ''
append '## Scope'
append ''
append 'This overlay dispatches the existing Qwen v6 W4A8 inference call contract to the Ztt tile primitive. The host executable is the x86 golden producer; each RISC-V guest also checks an independent scalar W4A8 oracle before writing its result report.'
append ''
append '## Run Metadata'
append ''
append '| Field | Value |'
append '| --- | --- |'
append "| UTC timestamp | $(date -u '+%Y-%m-%dT%H:%M:%SZ') |"
append "| Host golden | Native \`$HOST_CC\` on \`$(uname -m)\` |"
append "| Target | \`riscv64-linux-gnu\`, \`rv64gcv_xxiangshanztt\`, static ELF |"
append "| LLVM | \`$(tool_version "$CLANG")\` |"
append "| gem5 | \`$GEM5\` |"
append '| Oracle | Scalar W4A8 reference using logical `[D][N]` weights, then byte-for-byte host/guest report comparison |'
append ''
append '## Operator Build And Execution'
append ''

if [ ! -x "$CLANG" ] || [ ! -x "$OBJDUMP" ] || [ ! -x "$GEM5" ]; then
    build_status=FAIL
    overall=1
else
    if make -B -C "$TEST_DIR" BUILD="$BUILD" HOST_CC="$HOST_CC" \
        CLANG="$CLANG" OBJDUMP="$OBJDUMP" GEM5="$GEM5" host target \
        > "$BUILD_LOG" 2>&1; then
        build_status=PASS
    else
        build_status=FAIL
        overall=1
    fi
fi

if [ "$build_status" = PASS ]; then
    if "$OBJDUMP" -d --mattr=+xxiangshanztt "$TARGET_BIN" > "$DISASM" 2>&1 && \
       grep -Fq 'msettyp' "$DISASM" && grep -Fq 'asettyp' "$DISASM" && \
       grep -Fq 'mzero.2d.acc' "$DISASM" && grep -Fq 'mls.rm' "$DISASM" && \
       grep -Fq 'mmulacc.2d' "$DISASM" && grep -Fq 'mmov.m.a' "$DISASM" && \
       grep -Fq 'mss.rm' "$DISASM"; then
        disasm_status=PASS
    else
        disasm_status=FAIL
        overall=1
    fi

    rm -f "$HOST_REPORT" "$ATOMIC_REPORT" "$MINOR_REPORT" "$O3_REPORT"
    if "$HOST_BIN" "$HOST_REPORT" > "$BUILD/logs/host.log" 2>&1 && [ -s "$HOST_REPORT" ]; then
        host_status=PASS
        if PYTHONDONTWRITEBYTECODE=1 "$GEM5" -d "$BUILD/m5out/atomic" "$ISA_TEST_ROOT/gem5_se.py" \
            --cpu atomic --cmd "$TARGET_BIN $ATOMIC_REPORT" > "$ATOMIC_LOG" 2>&1 && \
            [ -s "$ATOMIC_REPORT" ]; then
            atomic_status=PASS
            if cmp -s "$HOST_REPORT" "$ATOMIC_REPORT"; then
                atomic_compare=PASS
            else
                atomic_compare=FAIL
                diff -u "$HOST_REPORT" "$ATOMIC_REPORT" > "$BUILD/logs/atomic.diff" || true
                overall=1
            fi
        else
            atomic_status=FAIL
            atomic_compare=SKIP
            overall=1
        fi
        if PYTHONDONTWRITEBYTECODE=1 "$GEM5" -d "$BUILD/m5out/minor" "$ISA_TEST_ROOT/gem5_se.py" \
            --cpu minor --cmd "$TARGET_BIN $MINOR_REPORT" > "$MINOR_LOG" 2>&1 && \
            [ -s "$MINOR_REPORT" ]; then
            minor_status=PASS
            if cmp -s "$HOST_REPORT" "$MINOR_REPORT"; then
                minor_compare=PASS
            else
                minor_compare=FAIL
                diff -u "$HOST_REPORT" "$MINOR_REPORT" > "$BUILD/logs/minor.diff" || true
                overall=1
            fi
        else
            minor_status=FAIL
            minor_compare=SKIP
            overall=1
        fi
        if PYTHONDONTWRITEBYTECODE=1 "$GEM5" -d "$BUILD/m5out/o3" "$ISA_TEST_ROOT/gem5_se.py" \
            --cpu o3 --cmd "$TARGET_BIN $O3_REPORT" > "$O3_LOG" 2>&1 && \
            [ -s "$O3_REPORT" ]; then
            o3_status=PASS
            if cmp -s "$HOST_REPORT" "$O3_REPORT"; then
                o3_compare=PASS
            else
                o3_compare=FAIL
                diff -u "$HOST_REPORT" "$O3_REPORT" > "$BUILD/logs/o3.diff" || true
                overall=1
            fi
        else
            o3_status=FAIL
            o3_compare=SKIP
            overall=1
        fi
    else
        host_status=FAIL
        overall=1
    fi
fi

if make -C "$ZTT_ROOT/core" BUILD="$BUILD/qwen3_w4a8_core" \
    TOOLCHAIN_ROOT="$TOOLCHAIN_ROOT" CLANG="$CLANG" qwen_ztt_fpga > "$FULL_BUILD_LOG" 2>&1; then
    full_build_status=PASS
else
    full_build_status=FAIL
    overall=1
fi

if [ -f "$HOST_REPORT" ]; then host_hash=$(file_hash "$HOST_REPORT"); else host_hash='-'; fi
if [ -f "$ATOMIC_REPORT" ]; then atomic_hash=$(file_hash "$ATOMIC_REPORT"); else atomic_hash='-'; fi
if [ -f "$MINOR_REPORT" ]; then minor_hash=$(file_hash "$MINOR_REPORT"); else minor_hash='-'; fi
if [ -f "$O3_REPORT" ]; then o3_hash=$(file_hash "$O3_REPORT"); else o3_hash='-'; fi

append '| Stage | Status | Evidence |'
append '| --- | --- | --- |'
append "| Host + RISC-V operator build | $build_status | \`$BUILD/logs/build.log\` |"
append "| Ztt disassembly audit | $disasm_status | \`msettyp asettyp mzero.2d.acc mls.rm mmulacc.2d mmov.m.a mss.rm\` in \`$BUILD/disasm/w4a8_operator_test.dis\` |"
append "| x86 scalar golden | $host_status | \`$host_hash\` |"
append "| gem5 AtomicSimpleCPU | $atomic_status / $atomic_compare | \`$atomic_hash\` |"
append "| gem5 MinorCPU | $minor_status / $minor_compare | \`$minor_hash\` |"
append "| gem5 RiscvO3CPU | $o3_status / $o3_compare | \`$o3_hash\` |"
append ''
append '## Operator Cases'
append ''
append '| Case | API Coverage | Geometry | Validation |'
append '| --- | --- | --- | --- |'
append '| Decode | `ztt_matmul_quant_i8_i8_f32` | `T=1 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |'
append '| Prefill | `ztt_matmul_quant_batch_i8_i8_f32` | `T=2 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |'
append '| Paired FFN | `ztt_matmul_quant_pair_i8_i8_f32` | `T=1 N=128 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |'
append '| Batched paired FFN | `ztt_matmul_quant_batch_pair_i8_i8_f32` | `T=1 N=256 D=64 GS=128` | scalar oracle + x86/Atomic/Minor/O3 byte match |'
append '| Tail regression | batch API, non-divisible N/D | `T=1 N=130 D=66 GS=64` | verifies ceil group indexing for batch scales |'
append ''
append '## Full Qwen Integration'
append ''
append "| Item | Status | Note |"
append '| --- | --- | --- |'
append "| Shared-infrastructure Ztt Qwen ELF | $full_build_status | \`$BUILD/qwen3_w4a8_core/qwen3_chat_w4a8_ztt_fpga\`; model/tokenizer and bare-metal front end remain in the AME project |"
append '| Full Qwen inference in gem5 | NOT RUN | The embedded v6 model blob is approximately 666 MiB; operator-level Atomic/Minor/O3 validation is the practical coverage gate for this run. |'
append ''
append '## Result'
append ''
if [ "$overall" -eq 0 ]; then
    append '**PASS** - all W4A8 operator cases passed their scalar oracle and matched the x86 golden in Atomic, Minor, and O3 gem5; the shared-infrastructure Ztt Qwen ELF also built.'
    printf 'Qwen3 W4A8 Ztt operator test: PASS\n'
else
    append "**FAIL** - inspect the build, simulator, and comparison logs under \`$BUILD/logs/\`."
    printf 'Qwen3 W4A8 Ztt operator test: FAIL\n' >&2
fi
exit "$overall"
