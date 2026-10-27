#ifndef ZTT_RMSNORM_OPERATORS_H
#define ZTT_RMSNORM_OPERATORS_H

#include <stdint.h>

/* input and gamma are row-major 4x4 tiles. gamma repeats its per-column
 * weights on each row. output is compact row-major rows x cols. */
uint64_t ztt_rmsnorm_fp32(const float *input, const float *gamma,
                          unsigned rows, unsigned cols, float epsilon,
                          float *output);

#endif
