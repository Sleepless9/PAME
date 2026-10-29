#ifndef MATMUL_QUANT_RVV_TILE_H
#define MATMUL_QUANT_RVV_TILE_H

#include <stdint.h>

void rvv_matmul_quant_tile_i8_i8_f32(float *xout,
                                     const int8_t *xq,
                                     const float *xs,
                                     const int8_t *wq,
                                     const float *ws,
                                     int n,
                                     int d,
                                     int gs);

void rvv_tile_matmul_stats_reset(void);
uint64_t rvv_tile_matmul_stats_calls(void);
uint64_t rvv_tile_matmul_stats_macs(void);
uint64_t rvv_tile_matmul_stats_row_groups(void);
uint64_t rvv_tile_matmul_stats_vector_chunks(void);

#endif
