#include "backends/scalar/core/ops_scalar_core.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void scalar_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
  if (n <= 0 || !o || !x || !w) return;
  double ss = 0.0;
  for (int i = 0; i < n; i++) {
    double v = x[i];
    ss += v * v;
  }
  {
    const float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = w[i] * (x[i] * inv);
  }
}

void scalar_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps) {
  if (n <= 0 || !x || !w) return;
  double ss = 0.0;
  for (int i = 0; i < n; i++) {
    double v = x[i];
    ss += v * v;
  }
  {
    const float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = w[i] * (x[i] * inv);
  }
}

void scalar_softmax_f32(float *x, int n) {
  float maxv = x[0];
  for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
  {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      x[i] = expf(x[i] - maxv);
      sum += x[i];
    }
    {
      const float inv = 1.0f / (float)sum;
      for (int i = 0; i < n; i++) x[i] *= inv;
    }
  }
}

void scalar_rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
  /* Match RVV variant: refuse odd head_dim instead of silently dropping the
   * trailing element. Qwen3 uses head_dim=128 (even); odd dims would break
   * the (v[i], v[i+half]) pair convention. */
  if (!v || head_dim <= 0 || (head_dim & 1)) return;
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float exponent = (float)(2.0f * i) / (float)head_dim;
    float freq = powf(theta, -exponent);
    float ang = freq * (float)pos;
    float c = cosf(ang);
    float s = sinf(ang);
    float v0 = v[i];
    float v1 = v[i + half];
    v[i] = v0 * c - v1 * s;
    v[i + half] = v0 * s + v1 * c;
  }
}

void scalar_accum_ws_xs_i32_f32(float *dst,
                                   const int32_t *src_i32,
                                   const float *ws_base,
                                   ptrdiff_t ws_stride_bytes,
                                   float xs,
                                   int n) {
  if (ws_stride_bytes == (ptrdiff_t)sizeof(float)) {
    for (int i = 0; i < n; i++) dst[i] += (float)src_i32[i] * ws_base[i] * xs;
    return;
  }
  for (int i = 0; i < n; i++) {
    const float ws_val =
        *(const float *)((const uint8_t *)ws_base + (size_t)i * (size_t)ws_stride_bytes);
    dst[i] += (float)src_i32[i] * ws_val * xs;
  }
}

void scalar_zero_f32(float *dst, int n) {
  for (int i = 0; i < n; i++) dst[i] = 0.0f;
}

void scalar_zero_i32(int32_t *dst, int n) {
  for (int i = 0; i < n; i++) dst[i] = 0;
}

void scalar_matmul_ab_i8_i8_f32(const int8_t *A,
                                const int8_t *B_kn,
                                float *C,
                                int m,
                                int n,
                                int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float s = C[i * n + j];
      for (int kk = 0; kk < k; kk++) s += (float)A[i * k + kk] * (float)B_kn[kk * n + j];
      C[i * n + j] = s;
    }
  }
}

void scalar_matmul_abt_i8_i8_f32(const int8_t *A,
                                 const int8_t *B_nk,
                                 float *C,
                                 int m,
                                 int n,
                                 int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float s = C[i * n + j];
      for (int kk = 0; kk < k; kk++) s += (float)A[i * k + kk] * (float)B_nk[j * k + kk];
      C[i * n + j] = s;
    }
  }
}
