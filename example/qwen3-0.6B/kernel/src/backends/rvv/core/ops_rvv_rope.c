#include "backends/rvv/core/ops_rvv_core.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#define RVV_ROPE_TMP_MAX 256

static float rvv_rope_cos_tmp[RVV_ROPE_TMP_MAX] __attribute__((aligned(64)));
static float rvv_rope_sin_tmp[RVV_ROPE_TMP_MAX] __attribute__((aligned(64)));
static float rvv_rope_ang_tmp[RVV_ROPE_TMP_MAX] __attribute__((aligned(64)));
static float rvv_rope_tmp_in[RVV_ROPE_TMP_MAX] __attribute__((aligned(64)));

static int rvv_rope_dbg_entered = 0;
static int rvv_rope_dbg_half = 0;
static int rvv_rope_dbg_vl = 0;
static float rvv_rope_dbg_vals[8];

extern void rvv_rope_apply(float *vlo, float *vhi,
                           const float *cos_t, const float *sin_t,
                           int chunk);

static inline size_t rvv_setvl_e32m1_tama(size_t avl) {
  size_t vl;
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma" : "=r"(vl) : "r"(avl));
  return vl;
}

static inline float rvv_rope_fmodf(float x, float y) {
  return x - (float)(int)(x / y) * y;
}

static float rvv_rope_logf(float x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  int e = (int)(u.i >> 23) - 127;
  u.i = (u.i & 0x007FFFFFu) | 0x3F800000u;
  float m = u.f;
  float t = (m - 1.0f) / (m + 1.0f);
  float t2 = t * t;
  float ln_m = 2.0f * t * (1.0f + t2 * (0.3333333f + t2 * (0.2f + t2 * 0.142857f)));
  return ln_m + (float)e * 0.6931471805599453f;
}

static float rvv_rope_expf(float x) {
  if (x < -88.0f) return 0.0f;
  if (x > 88.0f) return 1e38f;
  const float ln2 = 0.6931471805599453f;
  const float log2e = 1.4426950408889634f;
  float t = x * log2e;
  int n = (t >= 0.0f) ? (int)(t + 0.5f) : (int)(t - 0.5f);
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

static float rvv_rope_powf(float base, float exp) {
  if (base <= 0.0f) return 0.0f;
  return rvv_rope_expf(exp * rvv_rope_logf(base));
}

static float rvv_rope_sinf_inner(float x) {
  float x2 = x * x;
  return x * (1.0f + x2 * (-0.16666667f + x2 * (0.008333333f + x2 * (-0.0001984127f + x2 * 2.7557319e-06f))));
}

static float rvv_rope_sinf(float x) {
  const float pi = 3.14159265358979f;
  x = rvv_rope_fmodf(x, 2.0f * pi);
  if (x < 0.0f) x += 2.0f * pi;
  if (x > pi) {
    x -= pi;
    return -rvv_rope_sinf_inner(x);
  }
  return rvv_rope_sinf_inner(x);
}

static inline float rvv_rope_f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } cvt = {.u = bits};
  return cvt.f;
}

static inline uint32_t rvv_rope_f32_to_bits(float x) {
  union {
    uint32_t u;
    float f;
  } cvt = {.f = x};
  return cvt.u;
}

static inline int32_t rvv_rope_round_to_i32(float x) {
  return (x >= 0.0f) ? (int32_t)(x + 0.5f) : (int32_t)(x - 0.5f);
}

static void sinf_rvv(float *dst, const float *src, size_t n) {
  if (!dst || !src || n == 0) return;

  const uint32_t BITS_INV_PI = 0x3EA2F983u;
  const uint32_t BITS_PI_HI = 0x40490FDBu;
  const uint32_t BITS_PI_LO = 0xB3BBBD2Eu;
  const uint32_t C3 = 0xBE2AAABBu;
  const uint32_t C5 = 0x3C08873Eu;
  const uint32_t C7 = 0xB94FB223u;
  const uint32_t C9 = 0x362E9C5Bu;
  const size_t vlmax = __riscv_vsetvlmax_e32m1();

  vfloat32m1_t v_inv_pi = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_INV_PI, vlmax));
  vfloat32m1_t v_pi_hi = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_PI_HI, vlmax));
  vfloat32m1_t v_pi_lo = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_PI_LO, vlmax));
  vfloat32m1_t vc3 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C3, vlmax));
  vfloat32m1_t vc5 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C5, vlmax));
  vfloat32m1_t vc7 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C7, vlmax));
  vfloat32m1_t vc9 = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(C9, vlmax));

  size_t idx = 0;
  while (idx < n) {
    size_t vl = rvv_setvl_e32m1_tama(n - idx);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + idx, vl);

    vfloat32m1_t v_xpi = __riscv_vfmul_vv_f32m1(vx, v_inv_pi, vl);
    vint32m1_t vk = __riscv_vfcvt_x_f_v_i32m1(v_xpi, vl);
    vfloat32m1_t vk_f = __riscv_vfcvt_f_x_v_f32m1(vk, vl);

    vfloat32m1_t vr = __riscv_vfnmsac_vv_f32m1(vx, vk_f, v_pi_hi, vl);
    vr = __riscv_vfnmsac_vv_f32m1(vr, vk_f, v_pi_lo, vl);

    vint32m1_t v_sign_mask = __riscv_vand_vx_i32m1(vk, 1, vl);
    vint32m1_t v_sign_shift = __riscv_vsll_vx_i32m1(v_sign_mask, 31, vl);
    vuint32m1_t v_sign_bits = __riscv_vreinterpret_v_i32m1_u32m1(v_sign_shift);

    vfloat32m1_t vr2 = __riscv_vfmul_vv_f32m1(vr, vr, vl);
    vfloat32m1_t vpoly = vc9;
    vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc7, vl);
    vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc5, vl);
    vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc3, vl);

    vfloat32m1_t vr3 = __riscv_vfmul_vv_f32m1(vr2, vr, vl);
    vfloat32m1_t vsin = __riscv_vfmadd_vv_f32m1(vpoly, vr3, vr, vl);

    vuint32m1_t vsin_bits = __riscv_vreinterpret_v_f32m1_u32m1(vsin);
    vsin_bits = __riscv_vxor_vv_u32m1(vsin_bits, v_sign_bits, vl);
    vsin = __riscv_vreinterpret_v_u32m1_f32m1(vsin_bits);

    __riscv_vse32_v_f32m1(dst + idx, vsin, vl);
    idx += vl;
  }
}

/* cos(x) = sin(x + pi/2) */
static void cosf_rvv(float *dst, const float *src, size_t n) {
  if (!dst || !src || n == 0) return;

  const uint32_t BITS_HALF_PI = 0x3FC90FDBu;
  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t v_half_pi = __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vmv_v_x_u32m1(BITS_HALF_PI, vlmax));

  size_t idx = 0;
  while (idx < n) {
    size_t vl = rvv_setvl_e32m1_tama(n - idx);
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + idx, vl);
    vx = __riscv_vfadd_vv_f32m1(vx, v_half_pi, vl);
    __riscv_vse32_v_f32m1(rvv_rope_tmp_in, vx, vl);
    sinf_rvv(dst + idx, rvv_rope_tmp_in, vl);
    idx += vl;
  }
}

void rvv_rope_rotate_f32(float *v, int head_dim, int pos, float theta) {
  if (!v || head_dim <= 0 || (head_dim & 1)) return;

  const int half = head_dim >> 1;
  int done = 0;

  rvv_rope_dbg_entered = 1;
  rvv_rope_dbg_half = half;
  rvv_rope_dbg_vl = 0;

  while (done < half) {
    int chunk = half - done;
    if (chunk > RVV_ROPE_TMP_MAX) chunk = RVV_ROPE_TMP_MAX;

    for (int i = 0; i < chunk; i++) {
      const int idx = done + i;
      const float exponent = (float)(2.0f * idx) / (float)head_dim;
      const float freq = rvv_rope_powf(theta, -exponent);
      rvv_rope_ang_tmp[i] = (float)pos * freq;
    }

    sinf_rvv(rvv_rope_sin_tmp, rvv_rope_ang_tmp, (size_t)chunk);
    cosf_rvv(rvv_rope_cos_tmp, rvv_rope_ang_tmp, (size_t)chunk);

    if (done == 0 && chunk > 0) {
      rvv_rope_dbg_vals[0] = rvv_rope_ang_tmp[0];
      rvv_rope_dbg_vals[1] = rvv_rope_cos_tmp[0];
      rvv_rope_dbg_vals[2] = rvv_rope_sin_tmp[0];
      rvv_rope_dbg_vals[3] = v[0];
      rvv_rope_dbg_vals[4] = v[half];
      rvv_rope_dbg_vals[5] = v[0];
      rvv_rope_dbg_vals[6] = v[half];
      rvv_rope_dbg_vals[7] = (float)chunk;
    }

    rvv_rope_apply(v + done, v + half + done,
                   rvv_rope_cos_tmp, rvv_rope_sin_tmp, chunk);

    if (done == 0) rvv_rope_dbg_vl = chunk;
    done += chunk;
  }
}

void rvv_debug_get_rope_state_i32_i32_i32_f32(int *entered, int *half,
                                              int *vl, float *vals8) {
  if (entered) *entered = rvv_rope_dbg_entered;
  if (half)    *half    = rvv_rope_dbg_half;
  if (vl)      *vl      = rvv_rope_dbg_vl;
  if (vals8) {
    for (int i = 0; i < 8; ++i) vals8[i] = rvv_rope_dbg_vals[i];
  }
}
