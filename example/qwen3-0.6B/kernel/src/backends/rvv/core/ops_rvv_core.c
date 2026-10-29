#include "backends/rvv/core/ops_rvv_core.h"

#include <math.h>
#include <riscv_vector.h>

#if !defined(RVV_RMSNORM_DEBUG)
#define RVV_RMSNORM_DEBUG 0
#endif

#if !defined(RVV_RMSNORM_VEC_LEVEL)
#define RVV_RMSNORM_VEC_LEVEL 1
#endif

#define RVV_CORE_CHUNK 256
#ifndef RVV_SYNC_DELAY_CYCLES
#define RVV_SYNC_DELAY_CYCLES 16
#endif

extern void rvv_exp_vec_f32(float *dst, const float *src, int n);
extern void print_uart(const char *str) __attribute__((weak));
extern void write_serial(char c) __attribute__((weak));

static float rvv_core_x_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_w_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_lo_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_hi_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_cos_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_sin_buf[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_out_lo[RVV_CORE_CHUNK] __attribute__((aligned(64)));
static float rvv_core_out_hi[RVV_CORE_CHUNK] __attribute__((aligned(64)));

static inline int rvv_core_min_int(int a, int b) {
  return (a < b) ? a : b;
}

static inline vfloat32m1_t rvv_core_broadcast_u32_to_f32vec(uint32_t bits, size_t vl) {
  vuint32m1_t v_bits = __riscv_vmv_v_x_u32m1(bits, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_bits);
}

static inline void rvv_core_sync_delay(void) {
  for (unsigned i = 0; i < RVV_SYNC_DELAY_CYCLES; ++i) {
    asm volatile("nop");
  }
  asm volatile("fence rw, rw" ::: "memory");
}

#if RVV_RMSNORM_DEBUG
static void rvv_core_debug_putc(char c) {
  if (&write_serial) write_serial(c);
}

static void rvv_core_debug_print_str(const char *s) {
  if (!&write_serial) return;
  while (*s) rvv_core_debug_putc(*s++);
}

static void rvv_core_debug_print_u64_dec(uint64_t val) {
  if (!&write_serial) return;
  char buf[32];
  int i = 0;
  if (val == 0) {
    rvv_core_debug_putc('0');
    return;
  }
  while (val > 0 && i < (int)sizeof(buf)) {
    buf[i++] = (char)('0' + (val % 10));
    val /= 10;
  }
  while (i--) rvv_core_debug_putc(buf[i]);
}

static void rvv_core_debug_print_float(float x) {
  if (!&write_serial) return;
  union { float f; uint32_t u; } xu = { .f = x };
  uint32_t u = xu.u, exp = u & 0x7F800000u, frac = u & 0x007FFFFFu, sign = u >> 31;
  if (exp == 0x7F800000u) {
    if (frac == 0) {
      rvv_core_debug_print_str(sign ? "-inf" : "inf");
    } else {
      rvv_core_debug_print_str("nan");
    }
    return;
  }
  if ((u & 0x7FFFFFFFu) == 0) {
    rvv_core_debug_print_str("0.000000");
    return;
  }
  if (x < 0.0f) {
    rvv_core_debug_putc('-');
    x = -x;
  }
  uint32_t ip = (uint32_t)x;
  float fracf = x - (float)ip;
  uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
  if (frac6 >= 1000000u) {
    ip += 1u;
    frac6 -= 1000000u;
  }
  rvv_core_debug_print_u64_dec((uint64_t)ip);
  rvv_core_debug_putc('.');
  rvv_core_debug_putc((char)('0' + (frac6 / 100000) % 10));
  rvv_core_debug_putc((char)('0' + (frac6 / 10000) % 10));
  rvv_core_debug_putc((char)('0' + (frac6 / 1000) % 10));
  rvv_core_debug_putc((char)('0' + (frac6 / 100) % 10));
  rvv_core_debug_putc((char)('0' + (frac6 / 10) % 10));
  rvv_core_debug_putc((char)('0' + (frac6 % 10)));
}

static void rvv_core_debug_print_head(const char *tag, const float *a, int n) {
  if (!&write_serial) return;
  rvv_core_debug_print_str(tag);
  rvv_core_debug_print_str(" [");
  int limit = (n < 8) ? n : 8;
  for (int i = 0; i < limit; i++) {
    if (i) rvv_core_debug_print_str(", ");
    rvv_core_debug_print_float(a[i]);
  }
  rvv_core_debug_print_str("]\r\n");
}

static void rvv_core_debug_print_scalar(const char *tag, float v) {
  if (!&write_serial) return;
  rvv_core_debug_print_str(tag);
  rvv_core_debug_print_str("=");
  rvv_core_debug_print_float(v);
  rvv_core_debug_print_str("\r\n");
}

static void rvv_core_debug_print_u32(const char *tag, uint32_t v) {
  if (!&write_serial) return;
  rvv_core_debug_print_str(tag);
  rvv_core_debug_print_str("=");
  rvv_core_debug_print_u64_dec((uint64_t)v);
  rvv_core_debug_print_str("\r\n");
}
#endif

static void rvv_core_load_f32(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_core_sync_delay();
    done += (int)vl;
  }
}

static void rvv_core_store_f32(float *dst, const float *src, int n) {
  int done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(src + done, vl);
    __riscv_vse32_v_f32m1(dst + done, v, vl);
    rvv_core_sync_delay();
    done += (int)vl;
  }
}

void rvv_rmsnorm_f32(float *o, const float *x, const float *w, int n, float eps) {
  if (!o || !x || !w || n <= 0) return;

#if RVV_RMSNORM_DEBUG
  if (&print_uart) {
    print_uart("[rvv_rmsnorm_f32] begin\r\n");
    rvv_core_debug_print_u32("  n", (uint32_t)n);
    rvv_core_debug_print_scalar("  eps", eps);
    rvv_core_debug_print_head("  x_head", x, n);
    rvv_core_debug_print_head("  w_head", w, n);
  }
#endif

  double ss = 0.0;
#if RVV_RMSNORM_VEC_LEVEL >= 2
  int done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    int local = 0;
    while (local < chunk) {
      const size_t vl = __riscv_vsetvl_e32m1((size_t)(chunk - local));
      vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + done + local, vl);
      vfloat32m1_t vx2 = __riscv_vfmul_vv_f32m1(vx, vx, vl);
      __riscv_vse32_v_f32m1(rvv_core_buf + local, vx2, vl);
      rvv_core_sync_delay();
      local += (int)vl;
    }
    for (int i = 0; i < chunk; i++) ss += (double)rvv_core_buf[i];
    done += chunk;
  }
#else
  for (int i = 0; i < n; i++) {
    const double v = x[i];
    ss += v * v;
  }
#endif
#if RVV_RMSNORM_DEBUG
  const float sum_sq = (float)ss;
  if (&print_uart) rvv_core_debug_print_scalar("  sum_sq", sum_sq);
#endif

  const float mean = (float)(ss / (double)n) + eps;
#if RVV_RMSNORM_DEBUG
  if (&print_uart) rvv_core_debug_print_scalar("  mean", mean);
#endif

  const float rstd = 1.0f / sqrtf(mean);
#if RVV_RMSNORM_DEBUG
  if (&print_uart) rvv_core_debug_print_scalar("  rstd", rstd);
#endif

  #if RVV_RMSNORM_VEC_LEVEL >= 1
  union {
    float f;
    uint32_t u;
  } rstd_bits = {.f = rstd};
  int out_done = 0;
  while (out_done < n) {
    const size_t vl = __riscv_vsetvl_e32m1((size_t)(n - out_done));
    vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + out_done, vl);
    vfloat32m1_t vw = __riscv_vle32_v_f32m1(w + out_done, vl);
    vfloat32m1_t vrstd = rvv_core_broadcast_u32_to_f32vec(rstd_bits.u, vl);
    vfloat32m1_t result = __riscv_vfmul_vv_f32m1(vx, vrstd, vl);
    result = __riscv_vfmul_vv_f32m1(result, vw, vl);
    __riscv_vse32_v_f32m1(o + out_done, result, vl);
    rvv_core_sync_delay();
    out_done += (int)vl;
  }
#else
  for (int i = 0; i < n; i++) {
    o[i] = w[i] * (x[i] * rstd);
  }
#endif

#if RVV_RMSNORM_DEBUG
  if (&print_uart) {
    rvv_core_debug_print_head("  out_head", o, n);
    print_uart("[rvv_rmsnorm_f32] end\r\n");
  }
#endif
}

void rvv_rmsnorm_inplace_f32(float *x, const float *w, int n, float eps) {
  rvv_rmsnorm_f32(x, x, w, n, eps);
}

void rvv_softmax_f32(float *x, int n) {
  if (!x || n <= 0) return;

  float maxv = x[0];
  int done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    rvv_core_load_f32(rvv_core_buf, x + done, chunk);
    for (int i = 0; i < chunk; i++) {
      if (rvv_core_buf[i] > maxv) maxv = rvv_core_buf[i];
    }
    done += chunk;
  }

  double sum = 0.0;
  done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    rvv_core_load_f32(rvv_core_buf, x + done, chunk);
    for (int i = 0; i < chunk; i++) rvv_core_buf[i] -= maxv;
    rvv_exp_vec_f32(rvv_core_buf, rvv_core_buf, chunk);
    for (int i = 0; i < chunk; i++) sum += rvv_core_buf[i];
    rvv_core_store_f32(x + done, rvv_core_buf, chunk);
    done += chunk;
  }

  if (sum <= 0.0) {
    const float u = 1.0f / (float)n;
    done = 0;
    while (done < n) {
      const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
      for (int i = 0; i < chunk; i++) rvv_core_buf[i] = u;
      rvv_core_store_f32(x + done, rvv_core_buf, chunk);
      done += chunk;
    }
    return;
  }

  const float inv = 1.0f / (float)sum;
  done = 0;
  while (done < n) {
    const int chunk = rvv_core_min_int(n - done, RVV_CORE_CHUNK);
    rvv_core_load_f32(rvv_core_buf, x + done, chunk);
    for (int i = 0; i < chunk; i++) rvv_core_buf[i] *= inv;
    rvv_core_store_f32(x + done, rvv_core_buf, chunk);
    done += chunk;
  }
}

void rvv_rope_apply(float *vlo, float *vhi,
                    const float *cos_t, const float *sin_t,
                    int chunk) {
  if (!vlo || !vhi || !cos_t || !sin_t || chunk <= 0) return;

  int done = 0;
  while (done < chunk) {
    const int blk = rvv_core_min_int(chunk - done, RVV_CORE_CHUNK);
    rvv_core_load_f32(rvv_core_lo_buf, vlo + done, blk);
    rvv_core_load_f32(rvv_core_hi_buf, vhi + done, blk);
    rvv_core_load_f32(rvv_core_cos_buf, cos_t + done, blk);
    rvv_core_load_f32(rvv_core_sin_buf, sin_t + done, blk);
    for (int i = 0; i < blk; i++) {
      rvv_core_out_lo[i] = rvv_core_lo_buf[i] * rvv_core_cos_buf[i] - rvv_core_hi_buf[i] * rvv_core_sin_buf[i];
      rvv_core_out_hi[i] = rvv_core_lo_buf[i] * rvv_core_sin_buf[i] + rvv_core_hi_buf[i] * rvv_core_cos_buf[i];
    }
    rvv_core_store_f32(vlo + done, rvv_core_out_lo, blk);
    rvv_core_store_f32(vhi + done, rvv_core_out_hi, blk);
    done += blk;
  }
}
