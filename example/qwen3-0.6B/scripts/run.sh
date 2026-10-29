#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
GEM5_ROOT=$(cd "${EXAMPLE_DIR}/../.." && pwd)
RESEARCH_ROOT=$(cd "${GEM5_ROOT}/.." && pwd)

ACTION=${1:-all}
BACKEND=${BACKEND:-rvv}
VLEN=${VLEN:-512}
NUM_LANES=${NUM_LANES:-4}
LANE_WIDTH_BITS=${LANE_WIDTH_BITS:-64}
ZTT_MARCH=${ZTT_MARCH:-rv64gcv_xxiangshanztt}
GEN_STEPS=${GEN_STEPS:-1}
PREFILL_TRACE_TOKENS=${PREFILL_TRACE_TOKENS:-1}
MINIMAL_PROMPT=${MINIMAL_PROMPT:-1}
RVV_DETAIL_PHASE=${RVV_DETAIL_PHASE:-prefill}
RVV_DETAIL_CYCLES=${RVV_DETAIL_CYCLES:-250000}
JOBS=${JOBS:-8}
RUN_NAME=${RUN_NAME:-qwen_${BACKEND}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${EXAMPLE_DIR}/output/${RUN_NAME}}
CORE_BUILD=${CORE_BUILD:-build/gem5}
ELF=${EXAMPLE_DIR}/core/${CORE_BUILD}/qwen3_chat_w4a8_${BACKEND}_fpga
GEM5=${GEM5_ROOT}/build/RISCV/gem5.opt
CLANG=${CLANG:-${RESEARCH_ROOT}/llvm-project_riscv/build_AME/bin/clang-21}

build_guest() {
    make -C "${EXAMPLE_DIR}/core" clean BUILD="${CORE_BUILD}"
    make -B -C "${EXAMPLE_DIR}/core" chat_${BACKEND}_fpga \
        BUILD="${CORE_BUILD}" \
        CLANG="${CLANG}" \
        RVV_VLEN_BITS="${VLEN}" \
        BARE_INJECT="-DGEN_STEPS=${GEN_STEPS} -DPREFILL_TRACE_TOKENS=${PREFILL_TRACE_TOKENS} -DGEM5_MINIMAL_PROMPT=${MINIMAL_PROMPT} -DQWEN_RVV_LIBC=1"
}

build_gem5() {
    mkdir -p "${GEM5_ROOT}/.tmp_gem5"
    (
        cd "${GEM5_ROOT}"
        TMPDIR="${GEM5_ROOT}/.tmp_gem5" scons \
            build/RISCV/gem5.opt -j"${JOBS}"
    )
}

run_gem5() {
    if [[ ! -x "${GEM5}" ]]; then
        echo "missing gem5 binary: ${GEM5}" >&2
        return 2
    fi
    if [[ ! -f "${ELF}" ]]; then
        echo "missing Qwen ELF: ${ELF}" >&2
        return 2
    fi
    mkdir -p "${RUN_DIR}"
    {
        echo "gem5=${GEM5}"
        echo "elf=${ELF}"
        echo "backend=${BACKEND}"
        echo "clock=1GHz"
        echo "vlen_bits=${VLEN}"
        echo "num_lanes=${NUM_LANES}"
        echo "lane_width_bits=${LANE_WIDTH_BITS}"
        echo "ztt_march=${ZTT_MARCH}"
        echo "generation_steps=${GEN_STEPS}"
        echo "minimal_prompt=${MINIMAL_PROMPT}"
        echo "rvv_detail_phase=${RVV_DETAIL_PHASE}"
        echo "rvv_detail_cycles=${RVV_DETAIL_CYCLES}"
        "${CLANG}" --version | head -n 1
    } > "${RUN_DIR}/environment.txt"

    set +e
    "${GEM5}" \
        --outdir="${RUN_DIR}" \
        --stats-file=stats.txt \
        --debug-flags="Qwen${BACKEND^^}Trace" \
        --debug-file="${BACKEND}_trace.log.gz" \
        "${EXAMPLE_DIR}/qwen_config.py" \
        --cmd "${ELF}" \
        --vlen "${VLEN}" \
        --num-lanes "${NUM_LANES}" \
        --lane-width-bits "${LANE_WIDTH_BITS}" \
        --rvv-detail-phase "${RVV_DETAIL_PHASE}" \
        --rvv-detail-cycles "${RVV_DETAIL_CYCLES}" \
        --backend "${BACKEND}" \
        2>&1 | tee "${RUN_DIR}/gem5.log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ -f "${RUN_DIR}/system.platform.terminal" ]]; then
        cp "${RUN_DIR}/system.platform.terminal" "${RUN_DIR}/uart.log"
    fi
    if [[ -f "${RUN_DIR}/phase_events.json" ]]; then
        python3 "${SCRIPT_DIR}/analyze_trace.py" "${RUN_DIR}" --backend "${BACKEND}"
    fi
    echo "Qwen ${BACKEND^^} run directory: ${RUN_DIR}"
    return "${status}"
}

start_background() {
    mkdir -p "${RUN_DIR}"
    export BACKEND VLEN NUM_LANES LANE_WIDTH_BITS ZTT_MARCH GEN_STEPS
    export PREFILL_TRACE_TOKENS MINIMAL_PROMPT RVV_DETAIL_PHASE
    export RVV_DETAIL_CYCLES JOBS RUN_NAME RUN_DIR CORE_BUILD CLANG
    nohup setsid "${BASH_SOURCE[0]}" all \
        > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &
    launcher_pid=$!
    echo "${launcher_pid}" > "${RUN_DIR}/launcher.pid"
    echo "Started Qwen ${BACKEND^^} end-to-end run: pid=${launcher_pid}"
    echo "Run directory: ${RUN_DIR}"
    echo "Launcher log: ${RUN_DIR}/launcher.log"
}

show_status() {
    pid_file="${RUN_DIR}/launcher.pid"
    if [[ ! -f "${pid_file}" ]]; then
        echo "missing launcher PID file: ${pid_file}" >&2
        return 2
    fi
    launcher_pid=$(<"${pid_file}")
    if kill -0 "${launcher_pid}" 2>/dev/null; then
        echo "Qwen ${BACKEND^^} end-to-end run is active: pid=${launcher_pid}"
    else
        echo "Qwen ${BACKEND^^} end-to-end run is not active: pid=${launcher_pid}"
    fi
    [[ -f "${RUN_DIR}/gem5.log" ]] && tail -n 20 "${RUN_DIR}/gem5.log"
    [[ -f "${RUN_DIR}/uart.log" ]] && tail -n 20 "${RUN_DIR}/uart.log"
}

case "${ACTION}" in
    build-guest)
        build_guest
        ;;
    build-gem5)
        build_gem5
        ;;
    run)
        run_gem5
        ;;
    analyze)
        python3 "${SCRIPT_DIR}/analyze_trace.py" "${RUN_DIR}" --backend "${BACKEND}"
        ;;
    start)
        start_background
        ;;
    status)
        show_status
        ;;
    all)
        build_guest
        build_gem5
        run_gem5
        ;;
    *)
        echo "usage: $0 [all|build-guest|build-gem5|run|start|status|analyze]" >&2
        exit 2
        ;;
esac
