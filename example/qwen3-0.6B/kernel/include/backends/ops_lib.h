#ifndef OPS_LIB_H
#define OPS_LIB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPS_BACKEND_RVV 0
#define OPS_BACKEND_AME 1
#define OPS_BACKEND_SCALAR 2

float scalar_silu_f32(float x);
void scalar_silu_vec_f32(float *dst, const float *src, int n);
float scalar_add_f32(float x, float y);
void scalar_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float scalar_sub_f32(float x, float y);
void scalar_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float scalar_mul_f32(float x, float y);
void scalar_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
void scalar_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps);
void scalar_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps);
void scalar_softmax_f32(float *x, int n);
void scalar_rope_rotate_f32(float *v, int head_dim, int pos, float theta);
int scalar_argmax_f32_i32(const float *x, int n);
void scalar_quantize_f32_i8_f32(int8_t *q, float *s, const float *x, int n, int gs);
void scalar_dequantize_i8_f32(float *xout, const int8_t *q, const float *s, int n, int gs);
void scalar_matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs);
void scalar_matmul_ab_i8_i8_f32(const int8_t *A, const int8_t *B_kn, float *C, int m, int n, int k);
void scalar_matmul_abt_i8_i8_f32(const int8_t *A, const int8_t *B_nk, float *C, int m, int n, int k);
void scalar_accum_ws_xs_i32_f32(float *dst,
                                   const int32_t *src_i32,
                                   const float *ws_base,
                                   ptrdiff_t ws_stride_bytes,
                                   float xs,
                                   int n);
void scalar_zero_f32(float *dst, int n);
void scalar_zero_i32(int32_t *dst, int n);

float rvv_silu_f32(float x);
void rvv_silu_vec_f32(float *dst, const float *src, int n);
void rvv_exp_vec_f32(float *dst, const float *src, int n);
float rvv_sqrtf(float x);
void rvv_sqrt_vec_f32(float *dst, const float *src, int n);
float rvv_sinf(float x);
void rvv_sin_vec_f32(float *dst, const float *src, int n);
float rvv_cosf(float x);
void rvv_cos_vec_f32(float *dst, const float *src, int n);
float rvv_add_f32(float x, float y);
void rvv_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float rvv_sub_f32(float x, float y);
void rvv_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float rvv_mul_f32(float x, float y);
void rvv_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
void rvv_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps);
void rvv_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps);
void rvv_softmax_f32(float *x, int n);
void rvv_rope_rotate_f32(float *v, int head_dim, int pos, float theta);
void rvv_debug_get_rope_state_i32_i32_i32_f32(int *entered, int *half, int *vl, float *vals8);
void rvv_argmax_debug_set(int enabled);
int rvv_argmax_f32_i32(const float *x, int n);
void rvv_quantize_f32_i8_f32(int8_t *q, float *s, const float *x, int n, int gs);
void rvv_dequantize_i8_f32(float *x, const int8_t *q, const float *s, int n, int gs);
void rvv_matmul_quant_i8_i8_f32(float *xout,
                                const int8_t *xq,
                                const float *xs,
                                const int8_t *wq,
                                const float *ws,
                                int n,
                                int d,
                                int gs);
void rvv_matmul_quant_tile_i8_i8_f32(float *xout,
                                     const int8_t *xq,
                                     const float *xs,
                                     const int8_t *wq,
                                     const float *ws,
                                     int n,
                                     int d,
                                     int gs);
void rvv_tile_matmul_stats_reset(void);
uint64_t rvv_tile_matmul_stats_calls(void);
uint64_t rvv_tile_matmul_stats_macs(void);
uint64_t rvv_tile_matmul_stats_row_groups(void);
uint64_t rvv_tile_matmul_stats_vector_chunks(void);
void rvv_matmul_ref_i8_f32(float *o,
                   const int8_t *w,
                   const float *w_scales,
                   const float *x,
                   int m,
                   int n,
                   int group_size);
void rvv_accum_ws_xs_i32_f32(float *dst,
                                const int32_t *src_i32,
                                const float *ws_base,
                                ptrdiff_t ws_stride_bytes,
                                float xs,
                                int n);
void rvv_zero_f32(float *dst, int n);
void rvv_zero_i32(int32_t *dst, int n);
void rvv_memcpy_f32(float *dst, const float *src, int n);
void rvv_memcpy_f32_r2(float *dst, const float *src, int n);
void rvv_memcpy_f32_r4(float *dst, const float *src, int n);
void rvv_memcpy(void *dst, const void *src, size_t nbytes);
void rvv_memcpy_u8(void *dst, const void *src, size_t nbytes);
void rvv_memcpy_u8_r2(void *dst, const void *src, size_t nbytes);
void rvv_memcpy_u8_r4(void *dst, const void *src, size_t nbytes);
void ame_memcpy_i8_acc2(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc4(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc6(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i8_acc8(int8_t *dst, const int8_t *src, size_t count);
void ame_memcpy_i32_acc2(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc4(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc6(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_i32_acc8(int32_t *dst, const int32_t *src, size_t count);
void ame_memcpy_f32_acc2(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc4(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc6(float *dst, const float *src, size_t count);
void ame_memcpy_f32_acc8(float *dst, const float *src, size_t count);
/* Compatibility byte-count wrappers (count is bytes). */
void ame_memcpy_acc2(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc4(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc6(void *dst, const void *src, size_t nbytes);
void ame_memcpy_acc8(void *dst, const void *src, size_t nbytes);

void ame_matmul_quant_i8_i8_f32(float *xout,
                                const int8_t *xq,
                                const float *xs,
                                const int8_t *wq,
                                const float *ws,
                                int n,
                                int d,
                                int gs);

/* Decode-oriented GEMV entry (single-token x quantized weight matrix).
 * Mathematically equivalent to ame_matmul_quant_i8_i8_f32 for m=1.
 */
void ame_gemv_quant_i8_i8_f32(float *xout,
                              const int8_t *xq,
                              const float *xs,
                              const int8_t *wq,
                              const float *ws,
                              int n,
                              int d,
                              int gs,
                              int w_row_stride);
void ame_rope_rotate_f32(float *v, int head_dim, int pos, float theta);
void ame_matmul_quant_noblk_i8_i8_f32(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride);
void ame_matmul_quant_blocked_i8_i8_f32(float *xout,
                                        const int8_t *xq,
                                        const float *xs,
                                        const int8_t *wq,
                                        const float *ws,
                                        int n,
                                        int d,
                                        int gs,
                                        int w_row_stride);
/* Batched quantized matmul: xout[T, d] = X[T, n] * W^T[n, d] (per-token-row
 * activation scales, per-output-row weight scales). Uses the AME tile's
 * M dim for token batching, so a prefill of T tokens issues ~T/16 the mqma
 * count of T sequential single-token matmuls.
 *
 * Layouts (all row-major, contiguous):
 *   xout: [T, d]
 *   xq  : [T, n]      (caller quantizes per row)
 *   xs  : [T, gpr]    where gpr = ceil(n / gs)
 *   wq  : [d, w_row_stride]  (n columns used; stride lets caller pass a view)
 *   ws  : [d, w_row_groups]  where w_row_groups = ceil(w_row_stride / gs)
 *
 * Constraints:
 *   1 <= T  <= AME_MIX_BATCH_T_MAX (kernel-side compile-time cap)
 *   1 <= gs <= AME_MIX_BATCH_MAX_GS
 */
void ame_matmul_quant_batch_i8_i8_f32(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int T,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride);
int ame_matmul_abt_i8_i8_f32(const int8_t *A,
                          const int8_t *B,
                          float *C,
                          int m,
                          int n,
                          int k);
int ame_matmul_singleacc_abt_i8_i8_f32(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int m,
                                        int n,
                                        int k);
int ame_matmul_dualacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);
int ame_matmul_quadacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);
int ame_matmul_ab_i8_i8_f32(const int8_t *A,
                         const int8_t *B,
                         float *C,
                         int m,
                         int n,
                         int k);
int ame_matmul_singleacc_ab_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k);
int ame_matmul_quadacc_ab_i8_i8_f32(const int8_t *A,
                                    const int8_t *B,
                                    float *C,
                                    int m,
                                    int n,
                                    int k);
int ame_probe_matmul_ab_i8_i8_f32_once(const int8_t *A,
                                       const int8_t *B,
                                       float *C,
                                       int req_m,
                                       int req_n,
                                       int req_k,
                                       int *got_m,
                                       int *got_n,
                                       int *got_k);
int ame_probe_matmul_abt_i8_i8_f32_once(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int req_m,
                                        int req_n,
                                        int req_k,
                                        int *got_m,
                                        int *got_n,
                                        int *got_k);

float kernel_silu_f32(float x);
void kernel_silu_vec_f32(float *dst, const float *src, int n);
float kernel_add_f32(float x, float y);
void kernel_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float kernel_sub_f32(float x, float y);
void kernel_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
float kernel_mul_f32(float x, float y);
void kernel_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n);
void kernel_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps);
void kernel_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps);
void kernel_softmax_f32(float *x, int n);
void kernel_rope_rotate_f32(float *v, int head_dim, int pos, float theta);
int kernel_argmax_f32_i32(const float *x, int n);
void kernel_quantize_f32_i8_f32(int8_t *q, float *s, const float *x, int n, int gs);
void kernel_dequantize_i8_f32(float *x, const int8_t *q, const float *s, int n, int gs);
void kernel_matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs);
void kernel_matmul_quant_i8_i8_f32_backend(float *xout,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq,
                                           const float *ws,
                                           int n,
                                           int d,
                                           int gs,
                                           int backend);

/* Backend matmul kernel call statistics. Each backend keeps its own
 * counters; query the one matching the backend you compiled with. */
void rvv_matmul_stats_reset(void);
uint64_t rvv_matmul_stats_calls(void);
uint64_t rvv_matmul_stats_macs(void);
uint64_t rvv_matmul_stats_dots_x4(void);
uint64_t rvv_matmul_stats_dots_x1(void);

void ame_matmul_stats_reset(void);
uint64_t ame_matmul_stats_calls(void);
uint64_t ame_matmul_stats_macs(void);
uint64_t ame_matmul_stats_tile_calls(void);
uint64_t ame_matmul_noblk_cycles_total(void);
uint64_t ame_matmul_noblk_cycles_ame(void);
uint64_t ame_matmul_noblk_cycles_noname(void);

void scalar_matmul_stats_reset(void);
uint64_t scalar_matmul_stats_calls(void);
uint64_t scalar_matmul_stats_macs(void);

#ifdef __cplusplus
}
#endif

#endif
