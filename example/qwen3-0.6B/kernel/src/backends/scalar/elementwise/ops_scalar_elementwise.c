#include "backends/scalar/elementwise/ops_scalar_elementwise.h"

#include <stdint.h>

static float scalar_expf_approx(float x) {
  if (x < -88.0f) return 0.0f;
  if (x > 88.0f) return 1e38f;
  const float ln2 = 0.6931471805599453f;
  const float log2e = 1.4426950408889634f;
  float t = x * log2e;
  int n = (t >= 0) ? (int)(t + 0.5f) : (int)(t - 0.5f);
  float f = x - (float)n * ln2;
  float p = 1.0f + f * (1.0f + f * (0.5f + f * (0.16666667f + f * (0.041666668f + f * 0.008333334f))));
  union {
    float fv;
    uint32_t iv;
  } u;
  u.fv = p;
  u.iv += (uint32_t)n << 23;
  return u.fv;
}

static float scalar_sqrtf_approx(float x) {
  if (x <= 0.0f) return 0.0f;
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  u.i = (u.i >> 1) + 0x1FC00000;
  u.f = 0.5f * (u.f + x / u.f);
  u.f = 0.5f * (u.f + x / u.f);
  u.f = 0.5f * (u.f + x / u.f);
  return u.f;
}

static float scalar_logf_approx(float x) {
  /* Domain guard: log is undefined for x <= 0. Caller (powf) already checks
   * for <= 0, but make the contract explicit so misuse fails loud, not silent
   * with garbage. -1e38f is the FP32 stand-in for -INFINITY (we don't pull
   * <math.h> macros since this file replaces libm). */
  if (x <= 0.0f) return -1e38f;
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  int e = (int)(u.i >> 23) - 127;
  u.i = (u.i & 0x007FFFFF) | 0x3F800000;
  float m = u.f;
  float t = (m - 1.0f) / (m + 1.0f);
  float t2 = t * t;
  float ln_m = 2.0f * t * (1.0f + t2 * (0.3333333f + t2 * (0.2f + t2 * 0.142857f)));
  return ln_m + (float)e * 0.6931471805599453f;
}

static float scalar_powf_approx(float base, float exp) {
  if (base <= 0.0f) return 0.0f;
  return scalar_expf_approx(exp * scalar_logf_approx(base));
}

static float scalar_fmodf_approx(float x, float y) {
  return x - (float)(int)(x / y) * y;
}

static inline float scalar_f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } cvt = {.u = bits};
  return cvt.f;
}

static float scalar_sinf_approx(float x) {
  const float inv_pi = scalar_f32_from_bits(0x3EA2F983u);
  const float pi_hi = scalar_f32_from_bits(0x40490FDBu);
  const float pi_lo = scalar_f32_from_bits(0xB3BBBD2Eu);
  const float c3 = scalar_f32_from_bits(0xBE2AAABBu);
  const float c5 = scalar_f32_from_bits(0x3C08873Eu);
  const float c7 = scalar_f32_from_bits(0xB94FB223u);
  const float c9 = scalar_f32_from_bits(0x362E9C5Bu);

  const int32_t k = (x >= 0.0f) ? (int32_t)(x * inv_pi + 0.5f) : (int32_t)(x * inv_pi - 0.5f);
  const float kf = (float)k;
  float r = x - kf * pi_hi;
  r -= kf * pi_lo;

  const float r2 = r * r;
  float poly = c9;
  poly = poly * r2 + c7;
  poly = poly * r2 + c5;
  poly = poly * r2 + c3;

  float sinv = poly * (r2 * r) + r;
  return (k & 1) ? -sinv : sinv;
}

static float scalar_cosf_approx(float x) {
  const float inv_pi = scalar_f32_from_bits(0x3EA2F983u);
  const float pi_hi = scalar_f32_from_bits(0x40490FDBu);
  const float pi_lo = scalar_f32_from_bits(0xB3BBBD2Eu);
  const float half_pi = scalar_f32_from_bits(0x3FC90FDBu);
  const float c2 = scalar_f32_from_bits(0xBEFFFFFFu);
  const float c4 = scalar_f32_from_bits(0x3D2AAAA5u);
  const float c6 = scalar_f32_from_bits(0xBAB60619u);
  const float c8 = scalar_f32_from_bits(0x37CCF5CEu);

  x += half_pi;
  const int32_t k = (x >= 0.0f) ? (int32_t)(x * inv_pi + 0.5f) : (int32_t)(x * inv_pi - 0.5f);
  const float kf = (float)k;
  float r = x - kf * pi_hi;
  r -= kf * pi_lo;
  const float r2 = r * r;
  float poly = c8;
  poly = poly * r2 + c6;
  poly = poly * r2 + c4;
  poly = poly * r2 + c2;
  float cosv = poly * r2 + 1.0f;
  return (k & 1) ? -cosv : cosv;
}

float expf(float x) { return scalar_expf_approx(x); }
float sqrtf(float x) { return scalar_sqrtf_approx(x); }
float powf(float x, float y) { return scalar_powf_approx(x, y); }
float sinf(float x) { return scalar_sinf_approx(x); }
float cosf(float x) { return scalar_cosf_approx(x); }
void sincosf(float x, float *s, float *c) {
  if (s) *s = scalar_sinf_approx(x);
  if (c) *c = scalar_cosf_approx(x);
}

float scalar_silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

void scalar_silu_vec_f32(float *dst, const float *src, int n) {
  for (int i = 0; i < n; i++) dst[i] = scalar_silu_f32(src[i]);
}

float scalar_add_f32(float x, float y) {
  return x + y;
}

void scalar_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  for (int i = 0; i < n; i++) dst[i] = lhs[i] + rhs[i];
}

float scalar_sub_f32(float x, float y) {
  return x - y;
}

void scalar_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  for (int i = 0; i < n; i++) dst[i] = lhs[i] - rhs[i];
}

float scalar_mul_f32(float x, float y) {
  return x * y;
}

void scalar_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  for (int i = 0; i < n; i++) dst[i] = lhs[i] * rhs[i];
}

void scalar_dequantize_i8_f32(float *xout, const int8_t *q, const float *s, int n, int gs) {
  for (int i = 0; i < n; i++) xout[i] = (float)q[i] * s[i / gs];
}
