#ifndef OPS_HOST_H
#define OPS_HOST_H

#include "backends/ops_lib.h"

static void quantize_f32_i8_f32(int8_t *q, float *s, const float *x, int n) {
  kernel_quantize_f32_i8_f32(q, s, x, n, GS);
}

static float silu_f32(float x) {
  return kernel_silu_f32(x);
}

static void silu_vec_f32(float *dst, const float *src, int n) {
  kernel_silu_vec_f32(dst, src, n);
}

static void rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
  kernel_rmsnorm_f32(o, x, w, n, eps);
}

static void rmsnorm_inplace_f32(float *x, const float *w, int n, float eps) {
  kernel_rmsnorm_inplace_f32(x, w, n, eps);
}

static void softmax_f32(float *x, int n) {
  kernel_softmax_f32(x, n);
}

static void rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
  kernel_rope_rotate_f32(v, head_dim, pos, theta);
}

static int argmax_f32_i32(const float *x, int n) {
  return kernel_argmax_f32_i32(x, n);
}

static int8_t *alloc_i8(int n) {
  return (int8_t *)xmalloc((size_t)n * sizeof(int8_t));
}

static float *alloc_f32(int n) {
  return (float *)xmalloc((size_t)((n + GS - 1) / GS) * sizeof(float));
}

static __attribute__((unused)) void dequantize_i8_f32(float *x, const int8_t *q, const float *s, int n) {
  kernel_dequantize_i8_f32(x, q, s, n, GS);
}

static void matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d) {
  kernel_matmul_quant_i8_i8_f32(xout, xq, xs, wq, ws, n, d, GS);
}

static void quantize_rows(int8_t *q_out, float *s_out, const float *x, int rows, int n) {
  const int groups = (n + GS - 1) / GS;
  for (int r = 0; r < rows; r++) {
    quantize_f32_i8_f32(q_out + (size_t)r * (size_t)n,
                        s_out + (size_t)r * (size_t)groups,
                        x + (size_t)r * (size_t)n,
                        n);
  }
}

static void matmul_quant_batch_i8_i8_f32(float *out,
                                         const int8_t *xq,
                                         const float *xs,
                                         const int8_t *wq,
                                         const float *ws,
                                         int rows,
                                         int n,
                                         int d) {
  const int groups_per_row = (n + GS - 1) / GS;
  for (int row = 0; row < rows; row++) {
    kernel_matmul_quant_i8_i8_f32(out + (size_t)row * (size_t)d,
                                  xq + (size_t)row * (size_t)n,
                                  xs + (size_t)row * (size_t)groups_per_row,
                                  wq,
                                  ws,
                                  n,
                                  d,
                                  GS);
  }
}

static void apply_qk_norm_and_rope(const Config *p,
                                   const Weights *w,
                                   int layer,
                                   int pos,
                                   float *q_out,
                                   float *k_out) {
  if (p->has_qk_norm) {
    for (int h = 0; h < p->q_heads; h++) {
      rmsnorm_inplace_f32(q_out + (size_t)h * p->head_dim, w->q_norm[layer], p->head_dim, p->rms_eps);
    }
    for (int h = 0; h < p->kv_heads; h++) {
      rmsnorm_inplace_f32(k_out + (size_t)h * p->head_dim, w->k_norm[layer], p->head_dim, p->rms_eps);
    }
  }

  for (int h = 0; h < p->q_heads; h++) {
    rope_rotate_f32(q_out + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
  }
  for (int h = 0; h < p->kv_heads; h++) {
    rope_rotate_f32(k_out + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
  }
}

#endif
