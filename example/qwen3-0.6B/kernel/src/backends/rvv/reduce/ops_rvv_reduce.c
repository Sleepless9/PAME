#include "backends/rvv/reduce/ops_rvv_reduce.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv_vector)
#error "src/backends/rvv/reduce/ops_rvv_reduce.c requires RVV-enabled compilation"
#endif

#if defined(__clang__)
#define RVV_NO_AUTOVEC __attribute__((optnone, noinline))
#elif defined(__GNUC__)
#define RVV_NO_AUTOVEC __attribute__((optimize("O0"), noinline))
#else
#define RVV_NO_AUTOVEC
#endif

#define RVV_REDUCE_CHUNK 256
#define RVV_SYNC_DELAY_CYCLES 0
#ifndef RVV_ARGMAX_DEBUG_MAX_CHUNKS
#define RVV_ARGMAX_DEBUG_MAX_CHUNKS 256
#endif

static float rvv_tmp_f32_a[RVV_REDUCE_CHUNK] __attribute__((aligned(64)));
static float rvv_tmp_f32_b[RVV_REDUCE_CHUNK] __attribute__((aligned(64)));
extern void print_uart(const char *str) __attribute__((weak));
static int rvv_argmax_debug_enabled;

static inline int rvv_reduce_min_int(int a, int b) {
  return (a < b) ? a : b;
}

static inline float rvv_absf(float x) {
  return x < 0.0f ? -x : x;
}

static inline int8_t rvv_round_clamp_i8(float x) {
  int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
  if (qi > 127) qi = 127;
  if (qi < -127) qi = -127;
  return (int8_t)qi;
}

static inline void rvv_reduce_sync_delay(void) {
  for (unsigned i = 0; i < RVV_SYNC_DELAY_CYCLES; ++i) {
    asm volatile("nop");
  }
}

static inline size_t rvv_setvl_e32m1_tama(size_t avl) {
  size_t vl;
  asm volatile("vsetvli %0, %1, e32, m1, ta, ma"
               : "=r"(vl)
               : "r"(avl));
  return vl;
}

static inline vfloat32m1_t rvv_reduce_broadcast_u32_to_f32vec(uint32_t bits, size_t vl) {
  vuint32m1_t v_bits = __riscv_vmv_v_x_u32m1(bits, vl);
  return __riscv_vreinterpret_v_u32m1_f32m1(v_bits);
}

static inline uint32_t rvv_reduce_f32_to_bits(float x) {
  union {
    uint32_t u;
    float f;
  } cvt = {.f = x};
  return cvt.u;
}

void rvv_argmax_debug_set(int enabled) {
  rvv_argmax_debug_enabled = enabled ? 1 : 0;
}

static inline int rvv_argmax_dbg_on(void) {
  return rvv_argmax_debug_enabled && &print_uart;
}

static void rvv_argmax_dbg_putc(char c) {
  char s[2];
  s[0] = c;
  s[1] = '\0';
  print_uart(s);
}

static void rvv_argmax_dbg_i32(int32_t v) {
  char buf[16];
  int i = 0;
  if (v == 0) {
    rvv_argmax_dbg_putc('0');
    return;
  }
  if (v < 0) {
    rvv_argmax_dbg_putc('-');
    v = -v;
  }
  while (v && i < (int)sizeof(buf)) {
    buf[i++] = (char)('0' + (v % 10));
    v /= 10;
  }
  while (i--) rvv_argmax_dbg_putc(buf[i]);
}

static void rvv_argmax_dbg_u32_hex(uint32_t v) {
  static const char hex[] = "0123456789ABCDEF";
  print_uart("0x");
  for (int i = 7; i >= 0; --i) {
    rvv_argmax_dbg_putc(hex[(v >> (i * 4)) & 0xFu]);
  }
}

static void rvv_argmax_dbg_f32(float x) {
  union {
    float f;
    uint32_t u;
  } xu = {.f = x};
  const uint32_t exp = xu.u & 0x7F800000u;
  const uint32_t frac = xu.u & 0x007FFFFFu;
  const uint32_t sign = xu.u >> 31;
  if (exp == 0x7F800000u) {
    if (frac == 0) print_uart(sign ? "-inf" : "inf");
    else print_uart("nan");
    return;
  }
  if ((xu.u & 0x7FFFFFFFu) == 0) {
    print_uart("0.000000");
    return;
  }
  int neg = x < 0.0f;
  if (neg) x = -x;
  if (x >= 2147483000.0f) {
    if (neg) rvv_argmax_dbg_putc('-');
    rvv_argmax_dbg_u32_hex(xu.u);
    return;
  }
  int32_t ip = (int32_t)x;
  float fracf = x - (float)ip;
  uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
  if (frac6 >= 1000000u) {
    ip += 1;
    frac6 -= 1000000u;
  }
  if (neg) rvv_argmax_dbg_putc('-');
  rvv_argmax_dbg_i32(ip);
  rvv_argmax_dbg_putc('.');
  rvv_argmax_dbg_putc((char)('0' + (frac6 / 100000u) % 10u));
  rvv_argmax_dbg_putc((char)('0' + (frac6 / 10000u) % 10u));
  rvv_argmax_dbg_putc((char)('0' + (frac6 / 1000u) % 10u));
  rvv_argmax_dbg_putc((char)('0' + (frac6 / 100u) % 10u));
  rvv_argmax_dbg_putc((char)('0' + (frac6 / 10u) % 10u));
  rvv_argmax_dbg_putc((char)('0' + frac6 % 10u));
}

static void rvv_argmax_dbg_input_head(const float *x, int n) {
  const int lim = n < 16 ? n : 16;
  print_uart("[rvv_argmax] input_head [");
  for (int i = 0; i < lim; ++i) {
    if (i) print_uart(", ");
    rvv_argmax_dbg_f32(x[i]);
  }
  print_uart("]\r\n");
}

static void rvv_argmax_dbg_mask_head(const float *x, int base, int vl, float maxv) {
  const int lim = vl < 16 ? vl : 16;
  print_uart("  eq_mask_head [");
  for (int i = 0; i < lim; ++i) {
    if (i) print_uart(", ");
    rvv_argmax_dbg_i32(x[base + i] == maxv ? 1 : 0);
  }
  print_uart("]\r\n");
}

static inline void rvv_mul_f32_block(float *dst, const float *src, size_t avl, float scale) {
  size_t vl = rvv_setvl_e32m1_tama(avl);
  if (vl == 0) return;
  union {
    float f;
    uint32_t u;
  } scale_bits = {.f = scale};
  vfloat32m1_t vx = __riscv_vle32_v_f32m1(src, vl);
  vfloat32m1_t vs = rvv_reduce_broadcast_u32_to_f32vec(scale_bits.u, vl);
  vfloat32m1_t vy = __riscv_vfmul_vv_f32m1(vx, vs, vl);
  __riscv_vse32_v_f32m1(dst, vy, vl);
}

static inline float rvv_reduce_extract_lane0_f32(vfloat32m1_t v0) {
  vuint32m1_t vu = __riscv_vreinterpret_v_f32m1_u32m1(v0);
  uint32_t bits = __riscv_vmv_x_s_u32m1_u32(vu);
  union {
    uint32_t u;
    float f;
  } cvt = {.u = bits};
  return cvt.f;
}

static float rvv_max_abs_f32(const float *src, int n) {
  if (!src || n <= 0) return 0.0f;
  float m = 0.0f;
  for (int i = 0; i < n; ++i) {
    float a = rvv_absf(src[i]);
    if (a > m) m = a;
  }
  return m;
}

int rvv_argmax_f32_i32(const float *x, int n) {
  if (!x || n <= 0) return 0;

  const size_t vlmax = __riscv_vsetvlmax_e32m1();
  if (vlmax == 0) return 0;
  vfloat32m1_t vneg_inf = rvv_reduce_broadcast_u32_to_f32vec(0xFF800000u, vlmax);

  float best_val = -__builtin_inff();
  int best_idx = 0;
  int done = 0;
  int dbg_chunk = 0;
  const int dbg = rvv_argmax_dbg_on();

  if (dbg) {
    print_uart("[rvv_argmax] begin n=");
    rvv_argmax_dbg_i32(n);
    print_uart(" vlmax=");
    rvv_argmax_dbg_i32((int32_t)vlmax);
    print_uart("\r\n");
    rvv_argmax_dbg_input_head(x, n);
  }

  while (done + (int)vlmax <= n) {
    const int base = done;
    vfloat32m1_t v = __riscv_vle32_v_f32m1(x + done, vlmax);
    vfloat32m1_t vred = __riscv_vfredmax_vs_f32m1_f32m1(v, vneg_inf, vlmax);
    float chunk_max = rvv_reduce_extract_lane0_f32(vred);
    const int dbg_this = dbg && dbg_chunk < RVV_ARGMAX_DEBUG_MAX_CHUNKS;
    if (dbg_this) {
      print_uart("[rvv_argmax] chunk base=");
      rvv_argmax_dbg_i32(base);
      print_uart(" vl=");
      rvv_argmax_dbg_i32((int32_t)vlmax);
      print_uart(" step=max chunk_max=");
      rvv_argmax_dbg_f32(chunk_max);
      print_uart(" old_best_idx=");
      rvv_argmax_dbg_i32(best_idx);
      print_uart(" old_best=");
      rvv_argmax_dbg_f32(best_val);
      print_uart("\r\n");
    }
    if (chunk_max == chunk_max && chunk_max > best_val) {
      vfloat32m1_t vmax = rvv_reduce_broadcast_u32_to_f32vec(rvv_reduce_f32_to_bits(chunk_max), vlmax);
      vbool32_t mask = __riscv_vmfeq_vv_f32m1_b32(v, vmax, vlmax);
      int first = __riscv_vfirst_m_b32(mask, vlmax);
      if (dbg_this) {
        print_uart("  step=compare_equal_and_vfirst first=");
        rvv_argmax_dbg_i32(first);
        print_uart("\r\n");
        rvv_argmax_dbg_mask_head(x, base, (int)vlmax, chunk_max);
      }
      if (first >= 0) {
        best_val = chunk_max;
        best_idx = base + first;
        if (dbg_this) {
          print_uart("  step=update best_idx=");
          rvv_argmax_dbg_i32(best_idx);
          print_uart(" best=");
          rvv_argmax_dbg_f32(best_val);
          print_uart("\r\n");
        }
      }
    } else if (dbg_this) {
      print_uart("  step=skip_update reason=");
      print_uart((chunk_max == chunk_max) ? "not_greater" : "nan");
      print_uart("\r\n");
    }
    done += (int)vlmax;
    dbg_chunk++;
  }

  while (done < n) {
    const int chunk = rvv_reduce_min_int(n - done, (int)vlmax);
    const size_t vl = __riscv_vsetvl_e32m1((size_t)chunk);
    vfloat32m1_t tail = __riscv_vle32_v_f32m1(x + done, vl);
    vfloat32m1_t vred = __riscv_vfredmax_vs_f32m1_f32m1(tail, vneg_inf, vl);
    float chunk_max = rvv_reduce_extract_lane0_f32(vred);
    const int dbg_this = dbg && dbg_chunk < RVV_ARGMAX_DEBUG_MAX_CHUNKS;
    if (dbg_this) {
      print_uart("[rvv_argmax] tail base=");
      rvv_argmax_dbg_i32(done);
      print_uart(" vl=");
      rvv_argmax_dbg_i32((int32_t)vl);
      print_uart(" step=max chunk_max=");
      rvv_argmax_dbg_f32(chunk_max);
      print_uart(" old_best_idx=");
      rvv_argmax_dbg_i32(best_idx);
      print_uart(" old_best=");
      rvv_argmax_dbg_f32(best_val);
      print_uart("\r\n");
    }
    if (chunk_max == chunk_max && chunk_max > best_val) {
      vfloat32m1_t vmax = rvv_reduce_broadcast_u32_to_f32vec(rvv_reduce_f32_to_bits(chunk_max), vl);
      vbool32_t mask = __riscv_vmfeq_vv_f32m1_b32(tail, vmax, vl);
      int first = __riscv_vfirst_m_b32(mask, vl);
      if (dbg_this) {
        print_uart("  step=compare_equal_and_vfirst first=");
        rvv_argmax_dbg_i32(first);
        print_uart("\r\n");
        rvv_argmax_dbg_mask_head(x, done, (int)vl, chunk_max);
      }
      if (first >= 0) {
        best_val = chunk_max;
        best_idx = done + first;
        if (dbg_this) {
          print_uart("  step=update best_idx=");
          rvv_argmax_dbg_i32(best_idx);
          print_uart(" best=");
          rvv_argmax_dbg_f32(best_val);
          print_uart("\r\n");
        }
      }
    } else if (dbg_this) {
      print_uart("  step=skip_update reason=");
      print_uart((chunk_max == chunk_max) ? "not_greater" : "nan");
      print_uart("\r\n");
    }
    done += chunk;
    dbg_chunk++;
  }

  if (dbg && dbg_chunk >= RVV_ARGMAX_DEBUG_MAX_CHUNKS) {
    print_uart("[rvv_argmax] debug chunk print limit reached\r\n");
  }
  if (best_val == -__builtin_inff()) {
    if (&print_uart) {
      print_uart("[rvv_argmax_f32_i32] WARNING: NaN/invalid max result\\r\\n");
    }
  }
  if (dbg) {
    print_uart("[rvv_argmax] end best_idx=");
    rvv_argmax_dbg_i32(best_idx);
    print_uart(" best=");
    rvv_argmax_dbg_f32(best_val);
    print_uart("\r\n");
  }
  return best_idx;
}

RVV_NO_AUTOVEC
void rvv_quantize_f32_i8_f32(int8_t *q, float *scales, const float *x,
                             int n, int gs) {
  if (!q || !scales || !x || n <= 0 || gs <= 0) return;
  const int groups = (n + gs - 1) / gs;
  for (int g = 0; g < groups; ++g) {
    const int base = g * gs;
    const int cnt = (base + gs <= n) ? gs : (n - base);
    float m = rvv_max_abs_f32(x + base, cnt);
    float s = (m > 0.0f) ? (m / 127.0f) : 1.0f;
    scales[g] = s;

    if (m == 0.0f) {
      for (int i = 0; i < cnt; ++i) q[base + i] = 0;
      continue;
    }

    int done = 0;
    const float inv = 1.0f / s;
    while (done < cnt) {
      size_t vl = rvv_setvl_e32m1_tama((size_t)(cnt - done));
      rvv_mul_f32_block(rvv_tmp_f32_a, x + base + done, vl, inv);
      for (size_t i = 0; i < vl; ++i) {
        q[base + done + (int)i] = rvv_round_clamp_i8(rvv_tmp_f32_a[i]);
      }
      done += (int)vl;
    }
  }
}

RVV_NO_AUTOVEC
void rvv_dequantize_i8_f32(float *x, const int8_t *q, const float *scales,
                           int n, int gs) {
  if (!x || !q || !scales || n <= 0 || gs <= 0) return;
  const int groups = (n + gs - 1) / gs;
  for (int g = 0; g < groups; ++g) {
    const int base = g * gs;
    const int cnt = (base + gs <= n) ? gs : (n - base);
    int done = 0;
    while (done < cnt) {
      size_t vl = rvv_setvl_e32m1_tama((size_t)(cnt - done));
      for (size_t i = 0; i < vl; ++i) {
        rvv_tmp_f32_a[i] = (float)q[base + done + (int)i];
      }
      rvv_mul_f32_block(rvv_tmp_f32_b, rvv_tmp_f32_a, vl, scales[g]);
      for (size_t i = 0; i < vl; ++i) {
        x[base + done + (int)i] = rvv_tmp_f32_b[i];
      }
      done += (int)vl;
    }
  }
}
