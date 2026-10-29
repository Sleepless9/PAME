#ifndef MATMUL_QUANT_SCALAR_W4_H
#define MATMUL_QUANT_SCALAR_W4_H

#include <stdint.h>

void scalar_matmul_quant_i8_i4_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const uint8_t *wq4,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs);

#endif
