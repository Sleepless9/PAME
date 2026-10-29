#ifndef QWEN_HYBRID_FLASH_ATTENTION_H
#define QWEN_HYBRID_FLASH_ATTENTION_H

#include <stdint.h>

void hybrid_flash_attention(
    float *output, const float *query, const float *key_cache,
    const float *value_cache, int layer, int position, int query_heads,
    int kv_heads, int max_sequence, int head_dim, float head_dim_sqrt,
    uint64_t *qk_cycles, uint64_t *softmax_cycles, uint64_t *av_cycles);

#endif
