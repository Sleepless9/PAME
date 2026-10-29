#ifndef OPS_MIX_CORE_H
#define OPS_MIX_CORE_H

#include "backends/ops_lib.h"

void mix_accum_ws_xs_i32_f32(float *dst,
                                const int32_t *src_i32,
                                const float *ws_base,
                                ptrdiff_t ws_stride_bytes,
                                float xs,
                                int n);
void mix_zero_f32(float *dst, int n);
void mix_zero_i32(int32_t *dst, int n);

#endif
