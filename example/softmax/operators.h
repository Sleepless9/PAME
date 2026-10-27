#ifndef ZTT_SOFTMAX_OPERATORS_H
#define ZTT_SOFTMAX_OPERATORS_H

#include <stdint.h>

/* input is a row-major 4x4 tile, padded with masked values by the caller.
 * output is a compact row-major rows x cols matrix. */
uint64_t ztt_softmax_fp32(const float *input, unsigned rows, unsigned cols,
                          float *output);

#endif
