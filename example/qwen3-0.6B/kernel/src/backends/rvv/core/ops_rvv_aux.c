#include "backends/ops_mix.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#define RVV_AUX_CHUNK 256
#define RVV_SYNC_DELAY_CYCLES 0

static int32_t rvv_aux_i32_buf[RVV_AUX_CHUNK] __attribute__((aligned(64)));
static float rvv_aux_f32_buf0[RVV_AUX_CHUNK] __attribute__((aligned(64)));
static float rvv_aux_f32_buf1[RVV_AUX_CHUNK] __attribute__((aligned(64)));

static inline int rvv_aux_min_int(int a, int b) {
  return (a < b) ? a : b;
}

static inline void rvv_aux_sync_delay(void) {
  for (unsigned i = 0; i < RVV_SYNC_DELAY_CYCLES; ++i) {
    asm volatile("nop");
  }
  asm volatile("fence rw, rw" ::: "memory");
}

static inline vfloat32m1_t rvv_aux_broadcast_u32_to_f32vec(uint32_t bits, size_t vl) {
  vuint32m1_t v_bits = __riscv_vmv_v_x_u32m1(bits, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_bits);
}

static void rvv_aux_load_f32(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    const int chunk = rvv_aux_min_int(n - done, RVV_AUX_CHUNK);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_aux_sync_delay();
    done += (int)vl;
  }
}

static void rvv_aux_store_f32(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    const int chunk = rvv_aux_min_int(n - done, RVV_AUX_CHUNK);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_aux_sync_delay();
    done += (int)vl;
  }
}

static void rvv_aux_load_i32(int32_t *dst, const int32_t *src, int n) {
  int done = 0;
  while (done < n) {
    const int chunk = rvv_aux_min_int(n - done, RVV_AUX_CHUNK);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vint32m1_t v = __riscv_vle32_v_i32m1(src + done, vl);
    __riscv_vse32_v_i32m1(dst + done, v, vl);
    rvv_aux_sync_delay();
    done += (int)vl;
  }
}

void rvv_accum_ws_xs_i32_f32(float *dst,
                             const int32_t *src_i32,
                             const float *ws_base,
                             ptrdiff_t ws_stride_bytes,
                             float xs,
                             int n) {
  if (!dst || !src_i32 || !ws_base || n <= 0) return;

  if (ws_stride_bytes == (ptrdiff_t)sizeof(float)) {
    int done = 0;
    while (done < n) {
      const int chunk = rvv_aux_min_int(n - done, RVV_AUX_CHUNK);
      const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
      vint32m1_t vsrc_i32 = __riscv_vle32_v_i32m1(src_i32 + done, vl);
      vfloat32m1_t vsrc = __riscv_vfcvt_f_x_v_f32m1(vsrc_i32, vl);
      vfloat32m1_t vws = __riscv_vle32_v_f32m1(ws_base + done, vl);
      vfloat32m1_t vdst = __riscv_vle32_v_f32m1(dst + done, vl);
      union {
        float f;
        uint32_t u;
      } xs_bits = {.f = xs};
      vfloat32m1_t vxs = rvv_aux_broadcast_u32_to_f32vec(xs_bits.u, vl);
      vfloat32m1_t vscaled = __riscv_vfmul_vv_f32m1(vws, vxs, vl);
      vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(vsrc, vscaled, vl);
      vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(vdst, vprod, vl);
      __riscv_vse32_v_f32m1(dst + done, vout, vl);
      rvv_aux_sync_delay();
      done += (int)vl;
    }
    return;
  }

  int done = 0;
  while (done < n) {
    const int chunk = rvv_aux_min_int(n - done, RVV_AUX_CHUNK);
    rvv_aux_load_i32(rvv_aux_i32_buf, src_i32 + done, chunk);
    rvv_aux_load_f32(rvv_aux_f32_buf0, dst + done, chunk);
    for (int i = 0; i < chunk; i++) {
      const float ws_val =
          *(const float *)((const uint8_t *)ws_base +
                           (size_t)(done + i) * (size_t)ws_stride_bytes);
      rvv_aux_f32_buf1[i] = rvv_aux_f32_buf0[i] + (float)rvv_aux_i32_buf[i] * ws_val * xs;
    }
    rvv_aux_store_f32(dst + done, rvv_aux_f32_buf1, chunk);
    done += chunk;
  }
}

static void rvv_zero_e32_multi(void *dst, int n) {
  if (!dst || n <= 0) return;

  uint8_t *base = (uint8_t *)dst;
  int done = 0;
  while (n - done >= 4) {
    size_t vl;
    uint8_t *ptr = base + (size_t)done * sizeof(uint32_t);
    const size_t avl = (size_t)((n - done) / 4);
    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v9, 0\n\t"
        "vmv.v.i v10, 0\n\t"
        "vmv.v.i v11, 0\n\t"
        "slli t0, %[vl], 2\n\t"
        "add t1, %[ptr], t0\n\t"
        "add t2, t1, t0\n\t"
        "add t3, t2, t0\n\t"
        "vse32.v v8, (%[ptr])\n\t"
        "vse32.v v9, (t1)\n\t"
        "vse32.v v10, (t2)\n\t"
        "vse32.v v11, (t3)\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [ptr] "r"(ptr)
        : "v8", "v9", "v10", "v11", "t0", "t1", "t2", "t3", "memory");
    rvv_aux_sync_delay();
    done += (int)(vl * 4);
  }

  while (done < n) {
    size_t vl;
    uint8_t *ptr = base + (size_t)done * sizeof(uint32_t);
    const size_t avl = (size_t)(n - done);
    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vse32.v v8, (%[ptr])\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [ptr] "r"(ptr)
        : "v8", "memory");
    rvv_aux_sync_delay();
    done += (int)vl;
  }
}

void rvv_zero_f32(float *dst, int n) {
  rvv_zero_e32_multi(dst, n);
}

void rvv_zero_i32(int32_t *dst, int n) {
  rvv_zero_e32_multi(dst, n);
}

static void rvv_memcpy_u8_impl(void *dst, const void *src, size_t nbytes) {
  if (!dst || !src || nbytes == 0) return;

  uint8_t *d = (uint8_t *)dst;
  const uint8_t *s = (const uint8_t *)src;
  size_t done = 0;
  while (done < nbytes) {
    const size_t vl = __riscv_vsetvl_e8m1(nbytes - done);
    vuint8m1_t v = __riscv_vle8_v_u8m1(s + done, vl);
    __riscv_vse8_v_u8m1(d + done, v, vl);
    done += vl;
  }
}

void rvv_memcpy_u8_r2(void *dst, const void *src, size_t nbytes) {
  rvv_memcpy_u8_impl(dst, src, nbytes);
}

void rvv_memcpy_u8_r4(void *dst, const void *src, size_t nbytes) {
  rvv_memcpy_u8_impl(dst, src, nbytes);
}

void rvv_memcpy_u8(void *dst, const void *src, size_t nbytes) {
  rvv_memcpy_u8_r4(dst, src, nbytes);
}

void rvv_memcpy(void *dst, const void *src, size_t nbytes) {
  rvv_memcpy_u8_r4(dst, src, nbytes);
}

static void rvv_memcpy_f32_impl(float *dst, const float *src, int n) {
  if (!dst || !src || n <= 0) return;

  int done = 0;
  while (done < n) {
    const size_t vl = __riscv_vsetvl_e32m1((size_t)(n - done));
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    done += (int)vl;
  }
}

void rvv_memcpy_f32_r2(float *dst, const float *src, int n) {
  rvv_memcpy_f32_impl(dst, src, n);
}

void rvv_memcpy_f32_r4(float *dst, const float *src, int n) {
  rvv_memcpy_f32_impl(dst, src, n);
}

void rvv_memcpy_f32(float *dst, const float *src, int n) {
  rvv_memcpy_f32_r4(dst, src, n);
}
