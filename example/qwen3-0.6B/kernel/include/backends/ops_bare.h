#ifndef OPS_BARE_H
#define OPS_BARE_H

static float my_sqrtf(float x) {
  if (x <= 0.0f) return 0.0f;
  union { float f; uint32_t i; } u = { .f = x };
  u.i = (u.i >> 1) + 0x1FC00000;
  u.f = 0.5f * (u.f + x / u.f);
  u.f = 0.5f * (u.f + x / u.f);
  u.f = 0.5f * (u.f + x / u.f);
  return u.f;
}

static float my_expf(float x) {
  if (x < -88.0f) return 0.0f;
  if (x > 88.0f) return 1e38f;
  float ln2 = 0.6931471805599453f;
  float log2e = 1.4426950408889634f;
  float t = x * log2e;
  int n = (t >= 0) ? (int)(t + 0.5f) : (int)(t - 0.5f);
  float f = x - (float)n * ln2;
  float p = 1.0f + f * (1.0f + f * (0.5f + f * (0.16666667f + f * (0.041666668f + f * 0.008333334f))));
  union { float fv; uint32_t iv; } u;
  u.fv = p;
  u.iv += (uint32_t)n << 23;
  return u.fv;
}

static float my_logf(float x) {
  union { float f; uint32_t i; } u = { .f = x };
  int e = (int)(u.i >> 23) - 127;
  u.i = (u.i & 0x007FFFFF) | 0x3F800000;
  float m = u.f;
  float t = (m - 1.0f) / (m + 1.0f);
  float t2 = t * t;
  float ln_m = 2.0f * t * (1.0f + t2 * (0.3333333f + t2 * (0.2f + t2 * 0.142857f)));
  return ln_m + (float)e * 0.6931471805599453f;
}

static float my_powf(float base, float exp) {
  if (base <= 0.0f) return 0.0f;
  return my_expf(exp * my_logf(base));
}

static float my_fmodf(float x, float y) {
  return x - (float)(int)(x / y) * y;
}

static float my_sinf_inner(float x) {
  float x2 = x * x;
  return x * (1.0f + x2 * (-0.16666667f + x2 * (0.008333333f + x2 * (-0.0001984127f + x2 * 2.7557319e-06f))));
}

static float my_sinf(float x) {
  float pi = 3.14159265358979f;
  x = my_fmodf(x, 2.0f * pi);
  if (x < 0) x += 2.0f * pi;
  if (x > pi) { x -= pi; return -my_sinf_inner(x); }
  return my_sinf_inner(x);
}

static float my_cosf(float x) {
  return my_sinf(x + 1.5707963267948966f);
}

static inline float q_fabsf(float x) { return x < 0.0f ? -x : x; }

static inline int8_t q_round_clamp_i8(float x) {
  int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
  if (qi > 127) qi = 127;
  if (qi < -127) qi = -127;
  return (int8_t)qi;
}

static int8_t *alloc_i8(int n) {
  return (int8_t *)malloc((size_t)n * sizeof(int8_t));
}

static float *alloc_f32(int n) {
  return (float *)malloc((size_t)((n + GS - 1) / GS) * sizeof(float));
}

static void quantize_f32_i8_f32(int8_t *q, float *scales, const float *x, int n) {
  int num_groups = (n + GS - 1) / GS;
  for (int g = 0; g < num_groups; g++) {
    int start = g * GS;
    int end = start + GS;
    if (end > n) end = n;
    float maxabs = 0.0f;
    for (int i = start; i < end; i++) {
      float a = q_fabsf(x[i]);
      if (a > maxabs) maxabs = a;
    }
    float scale = (maxabs == 0.0f) ? 1.0f : (maxabs / 127.0f);
    scales[g] = scale;
    {
      float inv = 1.0f / scale;
      for (int i = start; i < end; i++) q[i] = q_round_clamp_i8(x[i] * inv);
    }
  }
}

static void rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; i++) { double v = x[i]; ss += v * v; }
  float inv = 1.0f / my_sqrtf((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) o[i] = w[i] * (x[i] * inv);
}

static void rmsnorm_inplace_f32(float *x, const float *w, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; i++) { double v = x[i]; ss += v * v; }
  float inv = 1.0f / my_sqrtf((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) x[i] = w[i] * (x[i] * inv);
}

static void softmax_f32(float *x, int n) {
  float maxv = x[0];
  for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
  double sum = 0.0;
  for (int i = 0; i < n; i++) { x[i] = my_expf(x[i] - maxv); sum += x[i]; }
  float inv = 1.0f / (float)sum;
  for (int i = 0; i < n; i++) x[i] *= inv;
}

static void rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
  int half = head_dim / 2;
  for (int i = 0; i < half; i++) {
    float exponent = (float)(2.0f * i) / (float)head_dim;
    float freq = my_powf(theta, -exponent);
    float ang = freq * (float)pos;
    float c = my_cosf(ang);
    float s = my_sinf(ang);
    float v0 = v[i], v1 = v[i + half];
    v[i] = v0 * c - v1 * s;
    v[i + half] = v0 * s + v1 * c;
  }
}

static inline void ops_rvv_matmul_quant_i8_i8_f32_ref(float *xout,
                                                      const int8_t *xq,
                                                      const float *xs,
                                                      const int8_t *wq,
                                                      const float *ws,
                                                      int n,
                                                      int d,
                                                      int gs,
                                                      int w_row_stride) {
  (void)w_row_stride;
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    int in = i * n;
    for (int j = 0; j < n; j++) {
      val += (float)xq[j] * (float)wq[in + j] * xs[j / gs] * ws[in / gs + j / gs];
    }
    xout[i] = val;
  }
}

#if (USE_BACKEND == OPS_BACKEND_AME)
#define OPS_BARE_QMATMUL_IMPL ame_matmul_quant_noblk_i8_i8_f32
#elif (USE_BACKEND == OPS_BACKEND_SCALAR)
#define OPS_BARE_QMATMUL_IMPL ops_rvv_matmul_quant_i8_i8_f32_ref
#else
#define OPS_BARE_QMATMUL_IMPL ops_rvv_matmul_quant_i8_i8_f32_ref
#endif

static void matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d) {
  OPS_BARE_QMATMUL_IMPL(xout, xq, xs, wq, ws, n, d, GS, n);
}

static void apply_qk_norm_and_rope(const Config *p, const Weights *w, int layer, int pos, float *q, float *k) {
  if (p->has_qk_norm) {
    for (int h = 0; h < p->q_heads; h++) rmsnorm_inplace_f32(q + (size_t)h * p->head_dim, w->q_norm[layer], p->head_dim, p->rms_eps);
    for (int h = 0; h < p->kv_heads; h++) rmsnorm_inplace_f32(k + (size_t)h * p->head_dim, w->k_norm[layer], p->head_dim, p->rms_eps);
  }
  for (int h = 0; h < p->q_heads; h++) rope_rotate_f32(q + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
  for (int h = 0; h < p->kv_heads; h++) rope_rotate_f32(k + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
}

static float silu_f32(float x) { return x / (1.0f + my_expf(-x)); }

static int argmax_f32_i32(const float *x, int n) {
  int mi = 0;
  float mv = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mv) { mv = x[i]; mi = i; }
  return mi;
}

#endif
