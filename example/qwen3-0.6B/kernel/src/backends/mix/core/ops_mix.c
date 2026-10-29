#include "backends/mix/core/ops_mix.h"

void mix_accum_ws_xs_i32_f32(float *dst,
                                const int32_t *src_i32,
                                const float *ws_base,
                                ptrdiff_t ws_stride_bytes,
                                float xs,
                                int n) {
#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_v))
  rvv_accum_ws_xs_i32_f32(dst, src_i32, ws_base, ws_stride_bytes, xs, n);
#else
  scalar_accum_ws_xs_i32_f32(dst, src_i32, ws_base, ws_stride_bytes, xs, n);
#endif
}

void mix_zero_f32(float *dst, int n) {
#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_v))
  rvv_zero_f32(dst, n);
#else
  scalar_zero_f32(dst, n);
#endif
}

void mix_zero_i32(int32_t *dst, int n) {
#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_v))
  rvv_zero_i32(dst, n);
#else
  scalar_zero_i32(dst, n);
#endif
}
