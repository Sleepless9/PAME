#ifndef ZTT_FFN_OPERATORS_H
#define ZTT_FFN_OPERATORS_H

#include <stdint.h>

/* input, w1 and w2 are zero-padded row-major 4x4 tiles. The activation is
 * x / (1 + 2^(-2x)), matching the AME base-2 sigmoid approximation. */
uint64_t ztt_ffn_fp32(const float *input, const float *w1, const float *w2,
                      unsigned tokens, unsigned input_dim,
                      unsigned hidden_dim, unsigned output_dim,
                      float *output);

#endif
