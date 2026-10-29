#!/usr/bin/env bash
set -euo pipefail

HYBRID_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEM5_ROOT=$(cd "${HYBRID_DIR}/../../.." && pwd)
RESEARCH_ROOT=$(cd "${GEM5_ROOT}/.." && pwd)

ACTION=${1:-all}
VLEN=${VLEN:-512}
NUM_LANES=${NUM_LANES:-4}
LANE_WIDTH_BITS=${LANE_WIDTH_BITS:-64}
VRF_BANKS_PER_LANE=${VRF_BANKS_PER_LANE:-8}
INST_QUEUE_SIZE=${INST_QUEUE_SIZE:-32}
SPM_SIZE=${SPM_SIZE:-1MiB}
SPM_BANKS=${SPM_BANKS:-8}
SPM_INTERLEAVE=${SPM_INTERLEAVE:-64B}
SPM_ADDR=${SPM_ADDR:-0x70000000}
DMA_ADDR=${DMA_ADDR:-0x40010000}
GEN_STEPS=${GEN_STEPS:-2}
MINIMAL_PROMPT=${MINIMAL_PROMPT:-1}
PREFILL_TRACE_TOKENS=${PREFILL_TRACE_TOKENS:-1}
DETAIL_PHASE=${DETAIL_PHASE:-prefill}
DETAIL_CYCLES=${DETAIL_CYCLES:-250000}
JOBS=${JOBS:-8}
RUN_NAME=${RUN_NAME:-qwen_hybrid_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${HYBRID_DIR}/output/${RUN_NAME}}
GUEST_BUILD=${GUEST_BUILD:-build}
ELF=${HYBRID_DIR}/${GUEST_BUILD}/qwen3_chat_w4a8_hybrid_fpga
SMOKE_BUILD=${SMOKE_BUILD:-smoke-build}
SMOKE_ELF=${HYBRID_DIR}/${SMOKE_BUILD}/hybrid_smoke
GEM5_BUILD_NAME=${GEM5_BUILD_NAME:-QWEN_HYBRID_RISCV}
GEM5=${GEM5_ROOT}/build/${GEM5_BUILD_NAME}/gem5.opt
CLANG=${CLANG:-${RESEARCH_ROOT}/llvm-project_riscv/build_AME/bin/clang-21}

build_guest() {
    make -C "${HYBRID_DIR}" clean BUILD="${GUEST_BUILD}"
    make -C "${HYBRID_DIR}" -j"${JOBS}" \
        BUILD="${GUEST_BUILD}" \
        CLANG="${CLANG}" \
        RVV_VLEN_BITS="${VLEN}" \
        SPM_BASE_ADDR="${SPM_ADDR}ULL" \
        SPM_DMA_BASE_ADDR="${DMA_ADDR}ULL" \
        BARE_INJECT="-DGEN_STEPS=${GEN_STEPS} -DPREFILL_TRACE_TOKENS=${PREFILL_TRACE_TOKENS} -DGEM5_MINIMAL_PROMPT=${MINIMAL_PROMPT} -DQWEN_RVV_LIBC=1"
}

build_gem5() {
    mkdir -p "${GEM5_ROOT}/.tmp_qwen_hybrid"
    (
        cd "${GEM5_ROOT}"
        scons defconfig "build/${GEM5_BUILD_NAME}" build_opts/RISCV
        TMPDIR="${GEM5_ROOT}/.tmp_qwen_hybrid" \
            scons "build/${GEM5_BUILD_NAME}/gem5.opt" -j"${JOBS}"
    )
}

build_smoke() {
    make -C "${HYBRID_DIR}" smoke SMOKE_BUILD="${SMOKE_BUILD}" \
        CLANG="${CLANG}" RVV_VLEN_BITS="${VLEN}"
}

run_smoke() {
    if [[ ! -x "${GEM5}" ]]; then
        echo "missing isolated gem5 binary: ${GEM5}" >&2
        return 2
    fi
    if [[ ! -f "${SMOKE_ELF}" ]]; then
        build_smoke
    fi
    local smoke_dir=${RUN_DIR}_smoke
    mkdir -p "${smoke_dir}"
    "${GEM5}" --outdir="${smoke_dir}" --stats-file=stats.txt \
        --debug-flags="MinorInstQueue,SPMVirtDma,RVVEngine,AMEEngine" \
        --debug-file=hybrid_trace.log "${HYBRID_DIR}/qwen_hybrid_config.py" \
        --cmd "${SMOKE_ELF}" --vlen "${VLEN}" --num-lanes "${NUM_LANES}" \
        --lane-width-bits "${LANE_WIDTH_BITS}" \
        --vrf-banks-per-lane "${VRF_BANKS_PER_LANE}" \
        --inst-queue-size "${INST_QUEUE_SIZE}" --spm-addr "${SPM_ADDR}" \
        --spm-size "${SPM_SIZE}" --spm-banks "${SPM_BANKS}" \
        --spm-interleave "${SPM_INTERLEAVE}" --dma-addr "${DMA_ADDR}" \
        --detail-phase none --detail-cycles 0 2>&1 | tee "${smoke_dir}/gem5.log"
    echo "Hybrid smoke run directory: ${smoke_dir}"
}

run_gem5() {
    if [[ ! -x "${GEM5}" ]]; then
        echo "missing isolated gem5 binary: ${GEM5}" >&2
        echo "run: ${BASH_SOURCE[0]} build-gem5" >&2
        return 2
    fi
    if [[ ! -f "${ELF}" ]]; then
        echo "missing hybrid guest: ${ELF}" >&2
        return 2
    fi
    mkdir -p "${RUN_DIR}"
    {
        echo "gem5=${GEM5}"
        echo "elf=${ELF}"
        echo "backend=hybrid"
        echo "vlen_bits=${VLEN}"
        echo "num_lanes=${NUM_LANES}"
        echo "lane_width_bits=${LANE_WIDTH_BITS}"
        echo "vrf_banks_per_lane=${VRF_BANKS_PER_LANE}"
        echo "inst_queue_size=${INST_QUEUE_SIZE}"
        echo "spm_address=${SPM_ADDR}"
        echo "spm_size=${SPM_SIZE}"
        echo "spm_banks=${SPM_BANKS}"
        echo "spm_interleave=${SPM_INTERLEAVE}"
        echo "dma_address=${DMA_ADDR}"
        echo "generation_steps=${GEN_STEPS}"
        echo "minimal_prompt=${MINIMAL_PROMPT}"
        echo "detail_phase=${DETAIL_PHASE}"
        echo "detail_cycles=${DETAIL_CYCLES}"
        "${CLANG}" --version | head -n 1
    } > "${RUN_DIR}/environment.txt"

    set +e
    "${GEM5}" \
        --outdir="${RUN_DIR}" \
        --stats-file=stats.txt \
        --debug-flags="MinorInstQueue,PseudoInst,SPMVirtDma" \
        --debug-file=hybrid_trace.log \
        "${HYBRID_DIR}/qwen_hybrid_config.py" \
        --cmd "${ELF}" \
        --vlen "${VLEN}" \
        --num-lanes "${NUM_LANES}" \
        --lane-width-bits "${LANE_WIDTH_BITS}" \
        --vrf-banks-per-lane "${VRF_BANKS_PER_LANE}" \
        --inst-queue-size "${INST_QUEUE_SIZE}" \
        --spm-addr "${SPM_ADDR}" \
        --spm-size "${SPM_SIZE}" \
        --spm-banks "${SPM_BANKS}" \
        --spm-interleave "${SPM_INTERLEAVE}" \
        --dma-addr "${DMA_ADDR}" \
        --detail-phase "${DETAIL_PHASE}" \
        --detail-cycles "${DETAIL_CYCLES}" \
        2>&1 | tee "${RUN_DIR}/gem5.log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ -f "${RUN_DIR}/system.platform.terminal" ]]; then
        cp "${RUN_DIR}/system.platform.terminal" "${RUN_DIR}/uart.log"
    fi
    if [[ -f "${RUN_DIR}/phase_events.json" ]]; then
        python3 "${HYBRID_DIR}/analyze_trace.py" "${RUN_DIR}"
    fi
    echo "Qwen hybrid run directory: ${RUN_DIR}"
    return "${status}"
}

case "${ACTION}" in
    build-guest)
        build_guest
        ;;
    build-gem5)
        build_gem5
        ;;
    build-smoke)
        build_smoke
        ;;
    run)
        run_gem5
        ;;
    run-smoke)
        run_smoke
        ;;
    analyze)
        python3 "${HYBRID_DIR}/analyze_trace.py" "${RUN_DIR}"
        ;;
    all)
        build_guest
        [[ -x "${GEM5}" ]] || build_gem5
        run_gem5
        ;;
    *)
        echo "usage: $0 [all|build-guest|build-gem5|build-smoke|run|run-smoke|analyze]" >&2
        exit 2
        ;;
esac
