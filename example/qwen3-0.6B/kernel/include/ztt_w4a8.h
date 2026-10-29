#ifndef ZTT_W4A8_H
#define ZTT_W4A8_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Qwen W4A8 API: xq is [n], weights are v6 tile-major [d/64][n/gs][64][gs],
 * and scales are group-major [n/gs][d]. */
void ztt_matmul_quant_i8_i8_f32(float *xout, const int8_t *xq,
                                const float *xs, const int8_t *wq,
                                const float *ws, int n, int d, int gs,
                                int w_row_stride);
void ztt_matmul_quant_batch_i8_i8_f32(float *xout, const int8_t *xq,
                                      const float *xs, const int8_t *wq,
                                      const float *ws, int T, int n, int d,
                                      int gs, int w_row_stride);
void ztt_matmul_quant_pair_i8_i8_f32(float *xout0, float *xout1,
                                     const int8_t *xq, const float *xs,
                                     const int8_t *wq0, const float *ws0,
                                     const int8_t *wq1, const float *ws1,
                                     int n, int d, int gs, int w_row_stride);
void ztt_matmul_quant_batch_pair_i8_i8_f32(
    float *xout0, float *xout1, const int8_t *xq, const float *xs,
    const int8_t *wq0, const float *ws0, const int8_t *wq1,
    const float *ws1, int T, int n, int d, int gs, int w_row_stride);

uint64_t ztt_matmul_stats_calls(void);
uint64_t ztt_matmul_stats_tiles(void);
void ztt_matmul_stats_reset(void);

#ifdef __cplusplus
}
#endif

#endif
