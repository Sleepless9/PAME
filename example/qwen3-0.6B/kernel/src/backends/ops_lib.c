#include "backends/ops_lib.h"

/* Per-flavor dispatcher.
 *
 * Selection is by the compile-time macros the per-flavor Makefile sets:
 *   USE_AME=1  -> AME mix matmul + scalar elementwise/reduce (the AME app
 *                 is built with -march=rv64imafdc; AME accelerator does the
 *                 matmul, the host CPU keeps elementwise on scalar paths).
 *   USE_RVV=1  -> RVV matmul + RVV elementwise/reduce/core (whole flavor
 *                 built -march=rv64gcv).
 *   neither    -> scalar all the way.
 *
 * Branches gated by the macros are necessary because each flavor only
 * links the backend objects it actually uses; an always-three-way switch
 * would create unresolved external references at link time.
 */

#if defined(USE_RVV) && USE_RVV
#  define KERNEL_DISPATCH_VEC 1
#else
#  define KERNEL_DISPATCH_VEC 0
#endif

float kernel_silu_f32(float x) {
#if KERNEL_DISPATCH_VEC
  return rvv_silu_f32(x);
#else
  return scalar_silu_f32(x);
#endif
}

void kernel_silu_vec_f32(float *dst, const float *src, int n) {
#if KERNEL_DISPATCH_VEC
  rvv_silu_vec_f32(dst, src, n);
#else
  scalar_silu_vec_f32(dst, src, n);
#endif
}

float kernel_add_f32(float x, float y) {
#if KERNEL_DISPATCH_VEC
  return rvv_add_f32(x, y);
#else
  return scalar_add_f32(x, y);
#endif
}

void kernel_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
#if KERNEL_DISPATCH_VEC
  rvv_add_vec_f32(dst, lhs, rhs, n);
#else
  scalar_add_vec_f32(dst, lhs, rhs, n);
#endif
}

float kernel_sub_f32(float x, float y) {
#if KERNEL_DISPATCH_VEC
  return rvv_sub_f32(x, y);
#else
  return scalar_sub_f32(x, y);
#endif
}

void kernel_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
#if KERNEL_DISPATCH_VEC
  rvv_sub_vec_f32(dst, lhs, rhs, n);
#else
  scalar_sub_vec_f32(dst, lhs, rhs, n);
#endif
}

float kernel_mul_f32(float x, float y) {
#if KERNEL_DISPATCH_VEC
  return rvv_mul_f32(x, y);
#else
  return scalar_mul_f32(x, y);
#endif
}

void kernel_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
#if KERNEL_DISPATCH_VEC
  rvv_mul_vec_f32(dst, lhs, rhs, n);
#else
  scalar_mul_vec_f32(dst, lhs, rhs, n);
#endif
}

void kernel_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
#if KERNEL_DISPATCH_VEC
  rvv_rmsnorm_f32(o, x, w, n, eps);
#else
  scalar_rmsnorm_f32(o, x, w, n, eps);
#endif
}

void kernel_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps) {
#if KERNEL_DISPATCH_VEC
  rvv_rmsnorm_inplace_f32(x, w, n, eps);
#else
  scalar_rmsnorm_inplace_f32(x, w, n, eps);
#endif
}

void kernel_softmax_f32(float *x, int n) {
#if KERNEL_DISPATCH_VEC
  rvv_softmax_f32(x, n);
#else
  scalar_softmax_f32(x, n);
#endif
}

void kernel_rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
#if defined(USE_AME) && USE_AME
  ame_rope_rotate_f32(v, head_dim, pos, theta);
#elif KERNEL_DISPATCH_VEC
  rvv_rope_rotate_f32(v, head_dim, pos, theta);
#else
  scalar_rope_rotate_f32(v, head_dim, pos, theta);
#endif
}

int kernel_argmax_f32_i32(const float *x, int n) {
#if KERNEL_DISPATCH_VEC
  return rvv_argmax_f32_i32(x, n);
#else
  return scalar_argmax_f32_i32(x, n);
#endif
}

void kernel_quantize_f32_i8_f32(int8_t *q, float *s, const float *x, int n, int gs) {
#if KERNEL_DISPATCH_VEC
  rvv_quantize_f32_i8_f32(q, s, x, n, gs);
#else
  scalar_quantize_f32_i8_f32(q, s, x, n, gs);
#endif
}

void kernel_dequantize_i8_f32(float *x, const int8_t *q, const float *s, int n, int gs) {
#if KERNEL_DISPATCH_VEC
  rvv_dequantize_i8_f32(x, q, s, n, gs);
#else
  scalar_dequantize_i8_f32(x, q, s, n, gs);
#endif
}

void kernel_matmul_quant_i8_i8_f32_backend(float *xout,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq,
                                           const float *ws,
                                           int n,
                                           int d,
                                           int gs,
                                           int backend) {
#if defined(USE_AME) && USE_AME
  if (backend == OPS_BACKEND_AME) {
    ame_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
    return;
  }
#endif
#if defined(USE_RVV) && USE_RVV
  if (backend == OPS_BACKEND_RVV) {
    rvv_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
    return;
  }
#endif
  /* OPS_BACKEND_SCALAR or unsupported requested backend in this flavor. */
  (void)backend;
  scalar_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
}

void kernel_matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs) {
#if defined(USE_AME) && USE_AME
  ame_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
#elif defined(USE_RVV) && USE_RVV
  rvv_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
#else
  scalar_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs);
#endif
}
