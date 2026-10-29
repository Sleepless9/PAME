#ifndef QWEN_HYBRID_PRELUDE_H
#define QWEN_HYBRID_PRELUDE_H

#include <stddef.h>
#include <stdint.h>

#include "backends/ops_lib.h"
#include "backends/rvv/core/matmul_quant_rvv_tile.h"
#include "hybrid_flash_attention.h"
#include "qwen_hybrid_runtime.h"
#include "spm_virt_dma.h"

#define QWEN_EXTERNAL_BACKEND_IMPL "qwen_hybrid_backend.inc"
#define QWEN_EXTERNAL_QUANTIZE_X(q, s, x, n, gs) \
    rvv_quantize_f32_i8_f32((q), (s), (x), (n), (gs))
#define QWEN_EXTERNAL_RMSNORM(o, x, w, n, eps) \
    rvv_rmsnorm_f32((o), (x), (w), (n), (eps))
#define QWEN_EXTERNAL_RMSNORM_INPLACE(x, w, n, eps) \
    rvv_rmsnorm_inplace_f32((x), (w), (n), (eps))
#define QWEN_EXTERNAL_SOFTMAX(x, n) rvv_softmax_f32((x), (n))
#define QWEN_EXTERNAL_ROPE_ROTATE(v, hd, pos, theta) \
    rvv_rope_rotate_f32((v), (hd), (pos), (theta))
#define QWEN_EXTERNAL_ADD_VEC(dst, lhs, rhs, n) \
    rvv_add_vec_f32((dst), (lhs), (rhs), (n))
#define QWEN_EXTERNAL_SILU_MUL_VEC(dst, rhs, n) do { \
    rvv_silu_vec_f32((dst), (dst), (n)); \
    rvv_mul_vec_f32((dst), (dst), (rhs), (n)); \
} while (0)
#define QWEN_EXTERNAL_ATTENTION hybrid_flash_attention
#define QWEN_EXTERNAL_STATS_RESET qwen_hybrid_stats_reset
#define QWEN_EXTERNAL_STATS_REPORT qwen_hybrid_stats_report

/* The generic front-end's RVV dot helper stores partials here. */
#define QWEN_EXTERNAL_QK_SCRATCH 1
#define g_qk_rvv_prod \
    ((float *)(uintptr_t)(SPM_BASE_ADDR + 0x4000ULL))
#define g_qk_rvv_acc \
    ((float *)(uintptr_t)(SPM_BASE_ADDR + 0x4400ULL))

#endif
