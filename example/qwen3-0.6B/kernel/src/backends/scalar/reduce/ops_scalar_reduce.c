#include "backends/scalar/reduce/ops_scalar_reduce.h"

#include <stdint.h>

static inline float scalar_absf(float x) {
  return x < 0.0f ? -x : x;
}

static inline int8_t scalar_round_clamp_i8(float x) {
  int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
  if (qi > 127) qi = 127;
  if (qi < -127) qi = -127;
  return (int8_t)qi;
}

int scalar_argmax_f32_i32(const float *x, int n) {
  int mi = 0;
  float mv = x[0];
  for (int i = 1; i < n; i++) {
    if (x[i] > mv) {
      mv = x[i];
      mi = i;
    }
  }
  return mi;
}

void scalar_quantize_f32_i8_f32(int8_t *q, float *scales, const float *x, int n, int gs) {
  int num_groups = (n + gs - 1) / gs;
  const float qmax = 127.0f;
  for (int g = 0; g < num_groups; g++) {
    int base = g * gs;
    int cnt = (base + gs <= n) ? gs : (n - base);
    float wmax = 0.0f;
    for (int i = 0; i < cnt; i++) {
      float v = scalar_absf(x[base + i]);
      if (v > wmax) wmax = v;
    }
    float scale = (wmax > 0.0f) ? (wmax / qmax) : 1.0f;
    scales[g] = scale;
    if (wmax == 0.0f) {
      for (int i = 0; i < cnt; i++) q[base + i] = 0;
    } else {
      float inv = 1.0f / scale;
      for (int i = 0; i < cnt; i++) q[base + i] = scalar_round_clamp_i8(x[base + i] * inv);
    }
  }
}
