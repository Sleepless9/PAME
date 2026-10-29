#include "backends/rvv/elementwise/ops_rvv_elementwise.h"

#include <riscv_vector.h>

#define RVV_ELEMWISE_TMP 256
#define RVV_SYNC_DELAY_CYCLES 0

static float rvv_elem_in_tmp[RVV_ELEMWISE_TMP] __attribute__((aligned(64)));
static float rvv_elem_rhs_tmp[RVV_ELEMWISE_TMP] __attribute__((aligned(64)));
static float rvv_elem_out_tmp[RVV_ELEMWISE_TMP] __attribute__((aligned(64)));
static float rvv_elem_y0_tmp[RVV_ELEMWISE_TMP] __attribute__((aligned(64)));
static int32_t rvv_elem_i32_tmp[RVV_ELEMWISE_TMP] __attribute__((aligned(64)));

extern float sqrtf(float);
extern float sinf(float);
extern float cosf(float);

static inline int rvv_elem_min_int(int a, int b) {
  return (a < b) ? a : b;
}

static inline void rvv_elem_sync_delay(void) {
  for (unsigned i = 0; i < RVV_SYNC_DELAY_CYCLES; ++i) {
    asm volatile("nop");
  }
  asm volatile("fence rw, rw" ::: "memory");
}

static inline vfloat32m1_t broadcast_u32_to_f32vec(uint32_t bits, size_t vl) {
  vuint32m1_t v_bits = __riscv_vmv_v_x_u32m1(bits, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_bits);
}

static inline float rvv_elem_f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } cvt = {.u = bits};
  return cvt.f;
}

static inline uint32_t rvv_elem_f32_to_bits(float x) {
  union {
    uint32_t u;
    float f;
  } cvt = {.f = x};
  return cvt.u;
}

static inline int32_t rvv_elem_round_to_i32(float x) {
  return (x >= 0.0f) ? (int32_t)(x + 0.5f) : (int32_t)(x - 0.5f);
}

static inline int rvv_sqrt_is_nan_f32(float x) {
  uint32_t u = rvv_elem_f32_to_bits(x);
  return ((u & 0x7F800000u) == 0x7F800000u) && ((u & 0x007FFFFFu) != 0);
}

static inline int rvv_sqrt_is_inf_f32(float x) {
  uint32_t u = rvv_elem_f32_to_bits(x);
  return (u & 0x7FFFFFFFu) == 0x7F800000u;
}

static inline void rvv_sqrt_precompute_scalar(float x,
                                              float *norm,
                                              float *scale,
                                              float *scale_inv,
                                              float *y0) {
  if (rvv_sqrt_is_nan_f32(x)) {
    *norm = x;
    *scale = 1.0f;
    *scale_inv = 1.0f;
    *y0 = x;
    return;
  }
  if (x <= 0.0f) {
    *norm = 0.0f;
    *scale = 1.0f;
    *scale_inv = 1.0f;
    *y0 = 0.0f;
    return;
  }
  if (rvv_sqrt_is_inf_f32(x)) {
    *norm = x;
    *scale = 1.0f;
    *scale_inv = 0.0f;
    *y0 = 0.0f;
    return;
  }

  union {
    float f;
    uint32_t u;
  } xu = {.f = x};
  uint32_t bits = xu.u;
  int exp = (int)((bits >> 23) & 0xFFu) - 127;
  int odd = exp & 1;
  int even = exp - odd;
  uint32_t norm_bits = (bits & 0x807FFFFFu) | (uint32_t)(odd + 127) << 23;
  union {
    uint32_t u;
    float f;
  } nb = {.u = norm_bits};
  *norm = nb.f;

  int half = even >> 1;
  union {
    uint32_t u;
    float f;
  } sb = {.u = (uint32_t)(half + 127) << 23};
  union {
    uint32_t u;
    float f;
  } sib = {.u = (uint32_t)(127 - half) << 23};
  *scale = sb.f;
  *scale_inv = sib.f;

  union {
    float f;
    uint32_t u;
  } yi = {.f = *norm};
  yi.u = 0x5F3759DFu - (yi.u >> 1);
  *y0 = yi.f;
}

static inline void rvv_elem_copy_in(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_elem_sync_delay();
    done += (int)vl;
  }
}

static inline void rvv_elem_store_out(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_elem_sync_delay();
    done += (int)vl;
  }
}

void rvv_exp_vec_f32(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;

  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + done, vl);

    {
      const vfloat32m1_t vlo = broadcast_u32_to_f32vec(0xC2B00000u, vl);
      const vfloat32m1_t vhi = broadcast_u32_to_f32vec(0x42B00000u, vl);
      const vfloat32m1_t vlog2e = broadcast_u32_to_f32vec(0x3FB8AA3Bu, vl);
      const vfloat32m1_t vln2 = broadcast_u32_to_f32vec(0x3F317218u, vl);
      const vfloat32m1_t vc120 = broadcast_u32_to_f32vec(0x3C088889u, vl);
      const vfloat32m1_t vc24 = broadcast_u32_to_f32vec(0x3D2AAAABu, vl);
      const vfloat32m1_t vc6 = broadcast_u32_to_f32vec(0x3E2AAAABu, vl);
      const vfloat32m1_t vc2 = broadcast_u32_to_f32vec(0x3F000000u, vl);
      const vfloat32m1_t vone = broadcast_u32_to_f32vec(0x3F800000u, vl);

      vx = __riscv_vfmax_vv_f32m1(vx, vlo, vl);
      vx = __riscv_vfmin_vv_f32m1(vx, vhi, vl);

      vfloat32m1_t vt = __riscv_vfmul_vv_f32m1(vx, vlog2e, vl);
      vint32m1_t vn = __riscv_vfcvt_x_f_v_i32m1(vt, vl);
      __riscv_vse32_v_i32m1(rvv_elem_i32_tmp, vn, vl);

      vfloat32m1_t vnf = __riscv_vfcvt_f_x_v_f32m1(vn, vl);
      vfloat32m1_t vf = __riscv_vfsub_vv_f32m1(vx, __riscv_vfmul_vv_f32m1(vnf, vln2, vl), vl);
      vfloat32m1_t poly = __riscv_vfmul_vv_f32m1(vc120, vf, vl);
      poly = __riscv_vfadd_vv_f32m1(poly, vc24, vl);
      poly = __riscv_vfmul_vv_f32m1(poly, vf, vl);
      poly = __riscv_vfadd_vv_f32m1(poly, vc6, vl);
      poly = __riscv_vfmul_vv_f32m1(poly, vf, vl);
      poly = __riscv_vfadd_vv_f32m1(poly, vc2, vl);
      poly = __riscv_vfmul_vv_f32m1(poly, vf, vl);
      poly = __riscv_vfadd_vv_f32m1(poly, vone, vl);
      poly = __riscv_vfmul_vv_f32m1(poly, vf, vl);
      poly = __riscv_vfadd_vv_f32m1(poly, vone, vl);

      __riscv_vse32_v_f32m1(rvv_elem_out_tmp, poly, vl);

      for (size_t i = 0; i < vl; i++) {
        int exponent = rvv_elem_i32_tmp[i] + 127;
        if (exponent <= 0) exponent = 0;
        if (exponent >= 255) exponent = 255;
        {
          union {
            uint32_t i;
            float f;
          } bits;
          bits.i = (uint32_t)(exponent << 23);
          rvv_elem_out_tmp[i] *= bits.f;
        }
      }

      __riscv_vse32_v_f32m1(dst + done, __riscv_vle32_v_f32m1(rvv_elem_out_tmp, vl), vl);
      rvv_elem_sync_delay();
    }

    done += (int)vl;
  }
}

float rvv_silu_f32(float x) {
  float src[1] = {x};
  float dst[1] = {0.0f};
  rvv_silu_vec_f32(dst, src, 1);
  return dst[0];
}

void rvv_silu_vec_f32(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;

  int done = 0;
  while (done < n) {
    const int chunk = rvv_elem_min_int(n - done, RVV_ELEMWISE_TMP);
    rvv_elem_copy_in(rvv_elem_in_tmp, src + done, chunk);
    for (int i = 0; i < chunk; i++) rvv_elem_rhs_tmp[i] = -rvv_elem_in_tmp[i];
    rvv_exp_vec_f32(rvv_elem_out_tmp, rvv_elem_rhs_tmp, chunk);
    for (int i = 0; i < chunk; i++) rvv_elem_out_tmp[i] = rvv_elem_in_tmp[i] / (1.0f + rvv_elem_out_tmp[i]);
    rvv_elem_store_out(dst + done, rvv_elem_out_tmp, chunk);
    done += chunk;
  }
}

float rvv_add_f32(float x, float y) {
  return x + y;
}

float rvv_sqrtf(float x) {
  float src[1] = {x};
  float dst[1] = {0.0f};
  rvv_sqrt_vec_f32(dst, src, 1);
  return dst[0];
}

void rvv_sqrt_vec_f32(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;

  int done = 0;
  while (done < n) {
    const int chunk = rvv_elem_min_int(n - done, RVV_ELEMWISE_TMP);
    for (int i = 0; i < chunk; i++) {
      rvv_sqrt_precompute_scalar(src[done + i],
                                 &rvv_elem_in_tmp[i],
                                 &rvv_elem_rhs_tmp[i],
                                 &rvv_elem_out_tmp[i],
                                 &rvv_elem_y0_tmp[i]);
    }

    size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    const float *p_norm = rvv_elem_in_tmp;
    const float *p_scale = rvv_elem_rhs_tmp;
    const float *p_scale_inv = rvv_elem_out_tmp;
    const float *p_y0 = rvv_elem_y0_tmp;
    const float *p_x = src + done;
    float *p_dst = dst + done;
    const uint64_t bits_half = 0x3F000000u;
    const uint64_t bits_onep5 = 0x3FC00000u;

    asm volatile(
        "vsetvli zero, %[vl], e32, m1, ta, ma\n\t"
        "vle32.v v8, (%[norm])\n\t"
        "vle32.v v11, (%[y0])\n\t"
        "vmv.v.x v9, %[half]\n\t"

        "vfmul.vv v10, v11, v11\n\t"
        "vfmul.vv v10, v8, v10\n\t"
        "vfmul.vv v10, v9, v10\n\t"
        "vmv.v.x v9, %[onep5]\n\t"
        "vfsub.vv v10, v9, v10\n\t"
        "vfmul.vv v11, v11, v10\n\t"
        "vmv.v.x v9, %[half]\n\t"

        "vfmul.vv v10, v11, v11\n\t"
        "vfmul.vv v10, v8, v10\n\t"
        "vfmul.vv v10, v9, v10\n\t"
        "vmv.v.x v9, %[onep5]\n\t"
        "vfsub.vv v10, v9, v10\n\t"
        "vfmul.vv v11, v11, v10\n\t"
        "vmv.v.x v9, %[half]\n\t"

        "vfmul.vv v10, v11, v11\n\t"
        "vfmul.vv v10, v8, v10\n\t"
        "vfmul.vv v10, v9, v10\n\t"
        "vmv.v.x v9, %[onep5]\n\t"
        "vfsub.vv v10, v9, v10\n\t"
        "vfmul.vv v11, v11, v10\n\t"
        "vmv.v.x v9, %[half]\n\t"

        "vfmul.vv v10, v11, v11\n\t"
        "vfmul.vv v10, v8, v10\n\t"
        "vfmul.vv v10, v9, v10\n\t"
        "vmv.v.x v9, %[onep5]\n\t"
        "vfsub.vv v10, v9, v10\n\t"
        "vfmul.vv v11, v11, v10\n\t"
        "vmv.v.x v9, %[half]\n\t"

        "vfmul.vv v10, v11, v11\n\t"
        "vfmul.vv v10, v8, v10\n\t"
        "vfmul.vv v10, v9, v10\n\t"
        "vmv.v.x v9, %[onep5]\n\t"
        "vfsub.vv v10, v9, v10\n\t"
        "vfmul.vv v11, v11, v10\n\t"

        "vle32.v v9, (%[scale])\n\t"
        "vle32.v v10, (%[scale_inv])\n\t"
        "vfmul.vv v8, v8, v11\n\t"
        "vfmul.vv v8, v8, v9\n\t"
        "vfmul.vv v10, v11, v10\n\t"
        "vle32.v v11, (%[x])\n\t"
        "vfmul.vv v9, v8, v8\n\t"
        "vfsub.vv v11, v11, v9\n\t"
        "vmv.v.x v9, %[half]\n\t"
        "vfmul.vv v11, v9, v11\n\t"
        "vfmul.vv v11, v11, v10\n\t"
        "vfadd.vv v8, v8, v11\n\t"
        "vse32.v v8, (%[dst])\n\t"
        :
        : [vl] "r"(vl),
          [norm] "r"(p_norm),
          [scale] "r"(p_scale),
          [scale_inv] "r"(p_scale_inv),
          [y0] "r"(p_y0),
          [x] "r"(p_x),
          [dst] "r"(p_dst),
          [half] "r"(bits_half),
          [onep5] "r"(bits_onep5)
        : "memory");
    rvv_elem_sync_delay();
    done += chunk;
  }
}

float rvv_sinf(float x) {
  float src[1] = {x};
  float dst[1] = {0.0f};
  rvv_sin_vec_f32(dst, src, 1);
  return dst[0];
}

static void rvv_sin_poly_vec_f32(float *dst, const float *src, int n, uint32_t phase_bits) {
  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + done, vl);

    if (phase_bits != 0u) {
      vfloat32m1_t vphase = broadcast_u32_to_f32vec(phase_bits, vl);
      vx = __riscv_vfadd_vv_f32m1(vx, vphase, vl);
    }

    {
      const vfloat32m1_t v_inv_pi = broadcast_u32_to_f32vec(0x3EA2F983u, vl);
      const vfloat32m1_t v_pi_hi = broadcast_u32_to_f32vec(0x40490FDBu, vl);
      const vfloat32m1_t v_pi_lo = broadcast_u32_to_f32vec(0xB3BBBD2Eu, vl);
      const vfloat32m1_t vc3 = broadcast_u32_to_f32vec(0xBE2AAABBu, vl);
      const vfloat32m1_t vc5 = broadcast_u32_to_f32vec(0x3C08873Eu, vl);
      const vfloat32m1_t vc7 = broadcast_u32_to_f32vec(0xB94FB223u, vl);
      const vfloat32m1_t vc9 = broadcast_u32_to_f32vec(0x362E9C5Bu, vl);

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

      __riscv_vse32_v_f32m1(dst + done, vsin, vl);
      rvv_elem_sync_delay();
    }

    done += (int)vl;
  }
}

static void rvv_cos_poly_vec_f32(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + done, vl);

    {
      const vfloat32m1_t v_inv_pi = broadcast_u32_to_f32vec(0x3EA2F983u, vl);
      const vfloat32m1_t v_pi_hi = broadcast_u32_to_f32vec(0x40490FDBu, vl);
      const vfloat32m1_t v_pi_lo = broadcast_u32_to_f32vec(0xB3BBBD2Eu, vl);
      const vfloat32m1_t v_half_pi = broadcast_u32_to_f32vec(0x3FC90FDBu, vl);
      const vfloat32m1_t vc2 = broadcast_u32_to_f32vec(0xBEFFFFFFu, vl);
      const vfloat32m1_t vc4 = broadcast_u32_to_f32vec(0x3D2AAAA5u, vl);
      const vfloat32m1_t vc6 = broadcast_u32_to_f32vec(0xBAB60619u, vl);
      const vfloat32m1_t vc8 = broadcast_u32_to_f32vec(0x37CCF5CEu, vl);
      const vfloat32m1_t vone = broadcast_u32_to_f32vec(0x3F800000u, vl);

      vfloat32m1_t vx_shifted = __riscv_vfadd_vv_f32m1(vx, v_half_pi, vl);
      vfloat32m1_t v_xpi = __riscv_vfmul_vv_f32m1(vx_shifted, v_inv_pi, vl);
      vint32m1_t vk = __riscv_vfcvt_x_f_v_i32m1(v_xpi, vl);
      vfloat32m1_t vk_f = __riscv_vfcvt_f_x_v_f32m1(vk, vl);

      vfloat32m1_t vr = __riscv_vfnmsac_vv_f32m1(vx_shifted, vk_f, v_pi_hi, vl);
      vr = __riscv_vfnmsac_vv_f32m1(vr, vk_f, v_pi_lo, vl);

      vint32m1_t v_sign_mask = __riscv_vand_vx_i32m1(vk, 1, vl);
      vint32m1_t v_sign_shift = __riscv_vsll_vx_i32m1(v_sign_mask, 31, vl);
      vuint32m1_t v_sign_bits = __riscv_vreinterpret_v_i32m1_u32m1(v_sign_shift);

      vfloat32m1_t vr2 = __riscv_vfmul_vv_f32m1(vr, vr, vl);
      vfloat32m1_t vpoly = vc8;
      vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc6, vl);
      vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc4, vl);
      vpoly = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vc2, vl);

      vfloat32m1_t vcos = __riscv_vfmadd_vv_f32m1(vpoly, vr2, vone, vl);

      vuint32m1_t vcos_bits = __riscv_vreinterpret_v_f32m1_u32m1(vcos);
      vcos_bits = __riscv_vxor_vv_u32m1(vcos_bits, v_sign_bits, vl);
      vcos = __riscv_vreinterpret_v_u32m1_f32m1(vcos_bits);

      __riscv_vse32_v_f32m1(dst + done, vcos, vl);
      rvv_elem_sync_delay();
    }

    done += (int)vl;
  }
}

void rvv_sin_vec_f32(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;
  rvv_sin_poly_vec_f32(dst, src, n, 0u);
}

float rvv_cosf(float x) {
  float src[1] = {x};
  float dst[1] = {0.0f};
  rvv_cos_vec_f32(dst, src, 1);
  return dst[0];
}

void rvv_cos_vec_f32(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;
  rvv_cos_poly_vec_f32(dst, src, n);
}

void rvv_add_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  if (!dst || !lhs || !rhs || n <= 0) return;

  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t va = __riscv_vle32_v_f32m1(lhs + done, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(rhs + done, vl);
    vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);
    __riscv_vse32_v_f32m1(dst + done, vc, vl);
    rvv_elem_sync_delay();
    done += (int)vl;
  }
}

float rvv_sub_f32(float x, float y) {
  float src_lhs[1] = {x};
  float src_rhs[1] = {y};
  float dst[1] = {0.0f};
  rvv_sub_vec_f32(dst, src_lhs, src_rhs, 1);
  return dst[0];
}

void rvv_sub_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  if (!dst || !lhs || !rhs || n <= 0) return;

  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t va = __riscv_vle32_v_f32m1(lhs + done, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(rhs + done, vl);
    vfloat32m1_t vc = __riscv_vfsub_vv_f32m1(va, vb, vl);
    __riscv_vse32_v_f32m1(dst + done, vc, vl);
    rvv_elem_sync_delay();
    done += (int)vl;
  }
}

float rvv_mul_f32(float x, float y) {
  float src_lhs[1] = {x};
  float src_rhs[1] = {y};
  float dst[1] = {0.0f};
  rvv_mul_vec_f32(dst, src_lhs, src_rhs, 1);
  return dst[0];
}

void rvv_mul_vec_f32(float *dst, const float *lhs, const float *rhs, int n) {
  if (!dst || !lhs || !rhs || n <= 0) return;

  int done = 0;
  while (done < n) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t va = __riscv_vle32_v_f32m1(lhs + done, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(rhs + done, vl);
    vfloat64m2_t vw = __riscv_vfwmul_vv_f64m2(va, vb, vl);
    vfloat32m1_t vc = __riscv_vfncvt_f_f_w_f32m1(vw, vl);
    __riscv_vse32_v_f32m1(dst + done, vc, vl);
    rvv_elem_sync_delay();
    done += (int)vl;
  }
}
