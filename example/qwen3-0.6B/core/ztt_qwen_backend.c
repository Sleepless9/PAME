#include "ztt_backend.h"

#include "ame_mix_offline_profile.h"

static uint64_t g_single_calls;
static uint64_t g_batch_calls;
static uint64_t g_single_tiles;
static uint64_t g_batch_tiles;

void ame_matmul_quant_noblk_i8_i8_f32(float *out, const int8_t *xq,
                                      const float *xs, const int8_t *weights,
                                      const float *weight_scales, int inputs,
                                      int outputs, int group_size, int stride) {
    const uint64_t tiles_before = ztt_matmul_stats_tiles();
    ztt_matmul_quant_i8_i8_f32(out, xq, xs, (const int8_t *)weights,
                                weight_scales, inputs, outputs, group_size,
                                stride);
    ++g_single_calls;
    g_single_tiles += ztt_matmul_stats_tiles() - tiles_before;
}

void ame_matmul_quant_batch_i8_i8_f32(float *out, const int8_t *xq,
                                      const float *xs, const int8_t *weights,
                                      const float *weight_scales, int tokens,
                                      int inputs, int outputs, int group_size,
                                      int stride) {
    const uint64_t tiles_before = ztt_matmul_stats_tiles();
    ztt_matmul_quant_batch_i8_i8_f32(out, xq, xs, (const int8_t *)weights,
                                      weight_scales, tokens, inputs, outputs,
                                      group_size, stride);
    ++g_batch_calls;
    g_batch_tiles += ztt_matmul_stats_tiles() - tiles_before;
}

void ame_matmul_quant_noblk_pair_i8_i8_f32(
    float *out0, float *out1, const int8_t *xq, const float *xs,
    const int8_t *weights0, const float *scales0, const int8_t *weights1,
    const float *scales1, int inputs, int outputs, int group_size,
    int stride) {
    const uint64_t tiles_before = ztt_matmul_stats_tiles();
    ztt_matmul_quant_pair_i8_i8_f32(out0, out1, xq, xs,
                                     (const int8_t *)weights0, scales0,
                                     (const int8_t *)weights1, scales1, inputs,
                                     outputs, group_size, stride);
    ++g_single_calls;
    g_single_tiles += ztt_matmul_stats_tiles() - tiles_before;
}

void ame_matmul_quant_batch_pair_i8_i8_f32(
    float *out0, float *out1, const int8_t *xq, const float *xs,
    const int8_t *weights0, const float *scales0, const int8_t *weights1,
    const float *scales1, int tokens, int inputs, int outputs, int group_size,
    int stride) {
    const uint64_t tiles_before = ztt_matmul_stats_tiles();
    ztt_matmul_quant_batch_pair_i8_i8_f32(
        out0, out1, xq, xs, (const int8_t *)weights0, scales0,
        (const int8_t *)weights1, scales1, tokens, inputs, outputs,
        group_size, stride);
    ++g_batch_calls;
    g_batch_tiles += ztt_matmul_stats_tiles() - tiles_before;
}

void ame_matmul_stats_reset(void) {
    ztt_matmul_stats_reset();
    g_single_calls = 0;
    g_batch_calls = 0;
    g_single_tiles = 0;
    g_batch_tiles = 0;
}

/* The shared Qwen front-end keeps the AME-shaped profiling contract.  Ztt
 * exposes compact call/tile counters, so cycle/instruction fields remain
 * zero while the common report keeps its single/batch breakdown. */
void ame_matmul_profile_snapshot(AmeMixOfflineProfile *out) {
    if (!out) return;
    *out = (AmeMixOfflineProfile){0};
    out->single_calls = g_single_calls;
    out->batch_calls = g_batch_calls;
    out->single_tiles = g_single_tiles;
    out->batch_tiles = g_batch_tiles;
    out->calls = g_single_calls + g_batch_calls;
    out->tiles = g_single_tiles + g_batch_tiles;
}
