#include "ztt_w4a8.h"
#include "ztt_report.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    MAX_N = 256,
    MAX_D = 256,
    MAX_TOKENS = 3,
    MAX_GS = 128,
    MAX_GROUPS = 4,
    MAX_TILED_WEIGHTS = 4 * MAX_GROUPS * 64 * MAX_GS,
};

static int8_t xq[MAX_TOKENS * MAX_N];
static float xs[MAX_TOKENS * MAX_GROUPS];
static int8_t logical_w0[MAX_D * MAX_N];
static int8_t logical_w1[MAX_D * MAX_N];
static int8_t tiled_w0[MAX_TILED_WEIGHTS];
static int8_t tiled_w1[MAX_TILED_WEIGHTS];
static float ws0[MAX_GROUPS * MAX_D];
static float ws1[MAX_GROUPS * MAX_D];
static float actual0[MAX_TOKENS * MAX_D];
static float actual1[MAX_TOKENS * MAX_D];
static float expected0[MAX_TOKENS * MAX_D];
static float expected1[MAX_TOKENS * MAX_D];

static int groups_for(int n, int gs) { return (n + gs - 1) / gs; }
static int blocks_for(int d, int tile) { return (d + tile - 1) / tile; }

static int8_t activation_value(unsigned seed, unsigned index) {
    return (int8_t)((int)((seed * 11u + index * 5u + 3u) % 31u) - 15);
}

static int8_t weight_value(unsigned seed, unsigned row, unsigned column) {
    return (int8_t)((int)((seed * 17u + row * 7u + column * 3u + 1u) % 15u) - 7);
}

static float x_scale_value(unsigned seed, unsigned group) {
    static const float values[] = {0.125f, 0.25f, 0.5f, 0.0625f};
    return values[(seed + group) % (sizeof(values) / sizeof(values[0]))];
}

static float weight_scale_value(unsigned seed, unsigned group, unsigned row) {
    static const float values[] = {0.0625f, 0.125f, 0.25f, 0.03125f};
    return values[(seed + group * 3u + row) % (sizeof(values) / sizeof(values[0]))];
}

static void fill_inputs(int T, int n, int gs, unsigned seed) {
    const int groups = groups_for(n, gs);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i)
            xq[t * n + i] = activation_value(seed + (unsigned)t, (unsigned)i);
        for (int group = 0; group < groups; ++group)
            xs[t * groups + group] =
                x_scale_value(seed + (unsigned)t, (unsigned)group);
    }
}

static void fill_logical_weights(int8_t *logical, int d, int n, unsigned seed) {
    for (int row = 0; row < d; ++row)
        for (int column = 0; column < n; ++column)
            logical[row * n + column] =
                weight_value(seed, (unsigned)row, (unsigned)column);
}

static void pack_qwen_v6_weights(int8_t *tiled, const int8_t *logical,
                                 int d, int n, int gs, int stride) {
    const int groups = groups_for(stride, gs);
    const int blocks = blocks_for(d, 64);
    const int tile_bytes = blocks * groups * 64 * gs;
    for (int i = 0; i < tile_bytes; ++i)
        tiled[i] = 0;

    for (int row = 0; row < d; ++row) {
        const int block = row / 64;
        const int block_row = row % 64;
        for (int group = 0; group < groups; ++group)
            for (int k = 0; k < gs; ++k) {
                const int column = group * gs + k;
                if (column < n) {
                    const int offset =
                        (((block * groups + group) * 64 + block_row) * gs) + k;
                    tiled[offset] = logical[row * n + column];
                }
            }
    }
}

static void fill_group_major_scales(float *scales, int d, int n, int gs,
                                    unsigned seed) {
    const int groups = groups_for(n, gs);
    for (int group = 0; group < groups; ++group)
        for (int row = 0; row < d; ++row)
            scales[group * d + row] =
                weight_scale_value(seed, (unsigned)group, (unsigned)row);
}

static void reference_one(float *out, const int8_t *input,
                          const float *input_scales, const int8_t *logical,
                          const float *weight_scales, int n, int d, int gs) {
    const int groups = groups_for(n, gs);
    for (int row = 0; row < d; ++row) {
        float sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            int32_t dot = 0;
            for (int k = 0; k < gs; ++k) {
                const int column = group * gs + k;
                if (column < n)
                    dot += (int32_t)input[column] * logical[row * n + column];
            }
            sum += (float)dot * input_scales[group] *
                   weight_scales[group * d + row];
        }
        out[row] = sum;
    }
}

static int close_enough(float actual, float expected) {
    float delta = actual - expected;
    if (delta < 0.0f)
        delta = -delta;
    float magnitude = expected < 0.0f ? -expected : expected;
    return delta <= 0.00001f * (1.0f + magnitude);
}

static int check_outputs(const char *name, const float *actual,
                         const float *expected, int values) {
    for (int i = 0; i < values; ++i) {
        if (!close_enough(actual[i], expected[i])) {
            fprintf(stderr, "%s mismatch at %d: actual=%g expected=%g\n", name,
                    i, (double)actual[i], (double)expected[i]);
            return 0;
        }
    }
    return 1;
}

static uint64_t expected_tiles(int T, int n, int d, int gs, int projections) {
    return (uint64_t)T * (uint64_t)projections *
           (uint64_t)blocks_for(d, 4) * (uint64_t)groups_for(n, gs) *
           (uint64_t)blocks_for(gs, 4);
}

static int check_stats(const char *name, uint64_t calls, uint64_t tiles) {
    if (ztt_matmul_stats_calls() != calls || ztt_matmul_stats_tiles() != tiles) {
        fprintf(stderr,
                "%s dispatch counters mismatch: calls=%llu/%llu tiles=%llu/%llu\n",
                name, (unsigned long long)ztt_matmul_stats_calls(),
                (unsigned long long)calls,
                (unsigned long long)ztt_matmul_stats_tiles(),
                (unsigned long long)tiles);
        return 0;
    }
    return 1;
}

static void record_pair(struct ztt_gemm_report *report, unsigned test,
                        const float *first, const float *second, int values) {
    ztt_gemm_report_bits(report, test, first, (unsigned)values,
                         sizeof(first[0]));
    for (int i = 0; i < values; ++i) {
        uint32_t bits;
        memcpy(&bits, &second[i], sizeof(bits));
        report->values[test][values + i] = bits;
    }
    report->case_values[test] = (unsigned)(2 * values);
}

static int run_decode(struct ztt_gemm_report *report) {
    const int n = 128, d = 64, gs = 128;
    fill_inputs(1, n, gs, 1);
    fill_logical_weights(logical_w0, d, n, 2);
    pack_qwen_v6_weights(tiled_w0, logical_w0, d, n, gs, n);
    fill_group_major_scales(ws0, d, n, gs, 3);
    reference_one(expected0, xq, xs, logical_w0, ws0, n, d, gs);

    ztt_matmul_stats_reset();
    ztt_matmul_quant_i8_i8_f32(actual0, xq, xs, tiled_w0, ws0, n, d, gs, n);
    if (!check_outputs("decode", actual0, expected0, d) ||
        !check_stats("decode", 1, expected_tiles(1, n, d, gs, 1)))
        return 0;
    ztt_gemm_report_bits(report, 0, actual0, d, sizeof(actual0[0]));
    return 1;
}

static int run_batch_prefill(struct ztt_gemm_report *report) {
    const int T = 2, n = 128, d = 64, gs = 128;
    fill_inputs(T, n, gs, 4);
    fill_logical_weights(logical_w0, d, n, 5);
    pack_qwen_v6_weights(tiled_w0, logical_w0, d, n, gs, n);
    fill_group_major_scales(ws0, d, n, gs, 6);
    for (int t = 0; t < T; ++t)
        reference_one(expected0 + t * d, xq + t * n, xs + t * groups_for(n, gs),
                      logical_w0, ws0, n, d, gs);

    ztt_matmul_stats_reset();
    ztt_matmul_quant_batch_i8_i8_f32(actual0, xq, xs, tiled_w0, ws0, T, n, d,
                                      gs, n);
    if (!check_outputs("batch-prefill", actual0, expected0, T * d) ||
        !check_stats("batch-prefill", T, expected_tiles(T, n, d, gs, 1)))
        return 0;
    ztt_gemm_report_bits(report, 1, actual0, T * d, sizeof(actual0[0]));
    return 1;
}

static int run_pair(struct ztt_gemm_report *report) {
    const int n = 128, d = 64, gs = 128;
    fill_inputs(1, n, gs, 7);
    fill_logical_weights(logical_w0, d, n, 8);
    fill_logical_weights(logical_w1, d, n, 9);
    pack_qwen_v6_weights(tiled_w0, logical_w0, d, n, gs, n);
    pack_qwen_v6_weights(tiled_w1, logical_w1, d, n, gs, n);
    fill_group_major_scales(ws0, d, n, gs, 10);
    fill_group_major_scales(ws1, d, n, gs, 11);
    reference_one(expected0, xq, xs, logical_w0, ws0, n, d, gs);
    reference_one(expected1, xq, xs, logical_w1, ws1, n, d, gs);

    ztt_matmul_stats_reset();
    ztt_matmul_quant_pair_i8_i8_f32(actual0, actual1, xq, xs, tiled_w0, ws0,
                                     tiled_w1, ws1, n, d, gs, n);
    if (!check_outputs("pair-w1", actual0, expected0, d) ||
        !check_outputs("pair-w3", actual1, expected1, d) ||
        !check_stats("pair", 2, expected_tiles(1, n, d, gs, 2)))
        return 0;
    record_pair(report, 2, actual0, actual1, d);
    return 1;
}

static int run_batch_pair(struct ztt_gemm_report *report) {
    const int T = 1, n = 256, d = 64, gs = 128;
    fill_inputs(T, n, gs, 12);
    fill_logical_weights(logical_w0, d, n, 13);
    fill_logical_weights(logical_w1, d, n, 14);
    pack_qwen_v6_weights(tiled_w0, logical_w0, d, n, gs, n);
    pack_qwen_v6_weights(tiled_w1, logical_w1, d, n, gs, n);
    fill_group_major_scales(ws0, d, n, gs, 15);
    fill_group_major_scales(ws1, d, n, gs, 16);
    for (int t = 0; t < T; ++t) {
        reference_one(expected0 + t * d, xq + t * n,
                      xs + t * groups_for(n, gs), logical_w0, ws0, n, d, gs);
        reference_one(expected1 + t * d, xq + t * n,
                      xs + t * groups_for(n, gs), logical_w1, ws1, n, d, gs);
    }

    ztt_matmul_stats_reset();
    ztt_matmul_quant_batch_pair_i8_i8_f32(
        actual0, actual1, xq, xs, tiled_w0, ws0, tiled_w1, ws1, T, n, d, gs,
        n);
    if (!check_outputs("batch-pair-w1", actual0, expected0, T * d) ||
        !check_outputs("batch-pair-w3", actual1, expected1, T * d) ||
        !check_stats("batch-pair", 2 * T, expected_tiles(T, n, d, gs, 2)))
        return 0;
    record_pair(report, 3, actual0, actual1, T * d);
    return 1;
}

static int run_tail(struct ztt_gemm_report *report) {
    const int T = 1, n = 130, d = 66, gs = 64;
    fill_inputs(T, n, gs, 17);
    fill_logical_weights(logical_w0, d, n, 18);
    pack_qwen_v6_weights(tiled_w0, logical_w0, d, n, gs, n);
    fill_group_major_scales(ws0, d, n, gs, 19);
    for (int t = 0; t < T; ++t)
        reference_one(expected0 + t * d, xq + t * n, xs + t * groups_for(n, gs),
                      logical_w0, ws0, n, d, gs);

    ztt_matmul_stats_reset();
    ztt_matmul_quant_batch_i8_i8_f32(actual0, xq, xs, tiled_w0, ws0, T, n, d,
                                      gs, n);
    if (!check_outputs("tail", actual0, expected0, T * d) ||
        !check_stats("tail", T, expected_tiles(T, n, d, gs, 1)))
        return 0;
    ztt_gemm_report_bits(report, 4, actual0, T * d, sizeof(actual0[0]));
    return 1;
}

int main(int argc, char **argv) {
    static const char *const case_names[ZTT_GEMM_MAX_CASES] = {
        "Qwen v6 decode: N=128 D=64 GS=128",
        "Qwen v6 prefill batch: T=2 N=128 D=64 GS=128",
        "Qwen v6 paired W1/W3: N=128 D=64 GS=128",
        "Qwen v6 paired batch + two groups: T=1 N=256 D=64 GS=128",
        "generic tails + batch scale offset: T=1 N=130 D=66 GS=64",
    };
    struct ztt_gemm_report report;

    if (argc != 2) {
        fprintf(stderr, "usage: %s REPORT_PATH\n", argv[0]);
        return 2;
    }

    ztt_gemm_report_init(&report, UINT64_C(0x713357344138), 5);
    if (!run_decode(&report) || !run_batch_prefill(&report) || !run_pair(&report) ||
        !run_batch_pair(&report) || !run_tail(&report))
        return 1;
    if (!ztt_gemm_report_write(argv[1], &report, case_names)) {
        fprintf(stderr, "cannot write report: %s\n", argv[1]);
        return 1;
    }
    return 0;
}
