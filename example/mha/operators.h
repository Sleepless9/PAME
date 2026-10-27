#ifndef ZTT_MHA_OPERATORS_H
#define ZTT_MHA_OPERATORS_H

#include <stdint.h>

/* input and each per-head weight are zero-padded row-major 4x4 tiles.
 * output is compact and ordered [head][sequence][head_dim]. */
uint64_t ztt_mha_fp32(const float *input, const float *const wq[2],
                      const float *const wk[2], const float *const wv[2],
                      unsigned sequence, unsigned model_dim,
                      unsigned head_dim, unsigned heads,
                      float softmax_scale, float *output);

#endif
