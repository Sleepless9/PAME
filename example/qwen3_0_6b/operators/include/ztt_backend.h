#ifndef ZTT_QWEN3_W4A8_BACKEND_H
#define ZTT_QWEN3_W4A8_BACKEND_H

#include "ztt_w4a8.h"

void ame_matmul_quant_noblk_i8_i8_f32(float *out, const int8_t *xq,
                                      const float *xs, const int8_t *weights,
                                      const float *scales, int n, int d,
                                      int gs, int stride);
void ame_matmul_quant_batch_i8_i8_f32(float *out, const int8_t *xq,
                                      const float *xs, const int8_t *weights,
                                      const float *scales, int tokens, int n,
                                      int d, int gs, int stride);
void ame_matmul_quant_noblk_pair_i8_i8_f32(
    float *out0, float *out1, const int8_t *xq, const float *xs,
    const int8_t *weights0, const float *scales0, const int8_t *weights1,
    const float *scales1, int n, int d, int gs, int stride);
void ame_matmul_quant_batch_pair_i8_i8_f32(
    float *out0, float *out1, const int8_t *xq, const float *xs,
    const int8_t *weights0, const float *scales0, const int8_t *weights1,
    const float *scales1, int tokens, int n, int d, int gs, int stride);

void ame_matmul_stats_reset(void);

#endif
