#include "ztt_w4a8.h"

#include <stddef.h>
#ifdef QWEN_LOCAL_SPM
#include "gem5_local_spm.h"
#endif

#define ZTT_TILE 4
#define ZTT_OUTBLK 64
#define ZTT_TRANSFER_BYTES (ZTT_TILE * ZTT_TILE * 32 / 8)
#ifndef QWEN_PACKED_K16
#define QWEN_PACKED_K16 0
#endif
#ifndef QWEN_GROUP_ACC
#define QWEN_GROUP_ACC 0
#endif
#ifndef QWEN_OFFLINE_AME_LAYOUT
#define QWEN_OFFLINE_AME_LAYOUT 0
#endif

#if defined(__riscv) && !defined(ZTT_SCALAR_FALLBACK)
extern void ztt_i8_i8_i32_4x4(int32_t *c, const int8_t *a, const int8_t *b);
#endif

static uint64_t g_calls;
static uint64_t g_tiles;

void ztt_matmul_stats_reset(void) { g_calls = 0; g_tiles = 0; }
uint64_t ztt_matmul_stats_calls(void) { return g_calls; }
uint64_t ztt_matmul_stats_tiles(void) { return g_tiles; }

#if !defined(__riscv) || defined(ZTT_SCALAR_FALLBACK)
static void scalar_tile(int32_t *c, const int8_t *a, const int8_t *b) {
    for (int i = 0; i < ZTT_TILE; ++i)
        for (int j = 0; j < ZTT_TILE; ++j) {
            int32_t sum = 0;
            for (int s = 0; s < (QWEN_PACKED_K16 ? 4 : 1); ++s)
                for (int k = 0; k < ZTT_TILE; ++k)
                    sum += (int32_t)a[s * 16 + i * ZTT_TILE + k] *
                           b[s * 16 + k * ZTT_TILE + j];
            c[i * ZTT_TILE + j] = sum;
        }
}
#endif

static void ztt_tile(int32_t *c, const int8_t *a, const int8_t *b) {
#if defined(QWEN_LOCAL_SPM) && !defined(ZTT_SCALAR_FALLBACK)
    qwen_dma_tile(c, a, b);
#elif defined(__riscv) && !defined(ZTT_SCALAR_FALLBACK)
    ztt_i8_i8_i32_4x4(c, a, b);
#else
    scalar_tile(c, a, b);
#endif
    ++g_tiles;
}

static const int8_t *weight_row(const int8_t *wq, int out, int group,
                                int gs, int stride_groups) {
    const int block = out / ZTT_OUTBLK;
    const int row = out % ZTT_OUTBLK;
    return wq + ((size_t)block * (size_t)stride_groups + (size_t)group) *
                    (size_t)ZTT_OUTBLK * (size_t)gs +
           (size_t)row * (size_t)gs;
}

static int8_t weight_value(const int8_t *wq, int out, int group, int k,
                           int gs, int stride_groups) {
#if QWEN_OFFLINE_AME_LAYOUT
    if (gs == 128) {
        const size_t block4 = (size_t)(out / ZTT_OUTBLK) * 16u +
                              (size_t)((out % ZTT_OUTBLK) / 4);
        const size_t offset =
            (block4 * (size_t)stride_groups + (size_t)group) * 512u +
            (size_t)k * 4u + (size_t)(out % 4);
        return wq[offset];
    }
#endif
    return weight_row(wq, out, group, gs, stride_groups)[k];
}

#if QWEN_PACKED_K16
#if QWEN_DMA_OVERLAP && defined(QWEN_LOCAL_SPM) && !defined(ZTT_SCALAR_FALLBACK)
#if !QWEN_OFFLINE_AME_LAYOUT
static void pack_group_batch(int8_t *data, const int8_t *xq, const int8_t *wq,
    int n, int d, int row0, int first_group, int count, int stride_groups) {
    for (int i = 0; i < count * 1024; ++i) data[i] = 0;
    for (int g = 0; g < count; ++g) {
        const int group = first_group + g;
        const int group_base = group * 128;
        const int8_t *wr[4];
        for (int j = 0; j < 4; ++j)
            wr[j] = row0+j < d ?
                weight_row(wq,row0+j,group,128,stride_groups) : NULL;
        int8_t *step_data = data + g * 1024;
        for (int step = 0; step < 8; ++step, step_data += 128) {
            const int step_base = step * 16;
            for (int square = 0; square < 4; ++square) {
                int8_t *a_dst = step_data + square * 16;
                int8_t *b_dst = step_data + 64 + square * 16;
                const int k_base = step_base + square * 4;
                for (int lane = 0; lane < 4; ++lane) {
                    const int k = k_base + lane;
                    if (group_base + k >= n) continue;
                    a_dst[lane] = xq[group_base + k];
                    for (int j = 0; j < 4; ++j)
                        if (wr[j]) b_dst[lane * 4 + j] = wr[j][k];
                }
            }
        }
    }
}
#endif

#if QWEN_OFFLINE_AME_LAYOUT
static const int8_t *ame_v7_weight_packet(const int8_t *wq, int row0,
    int group, int stride_groups) {
    const size_t block4 = (size_t)(row0 / ZTT_OUTBLK) * 16u +
                          (size_t)((row0 % ZTT_OUTBLK) / 4);
    return wq + (block4 * (size_t)stride_groups + (size_t)group) * 512u;
}
#endif

static void dma_pipelined_matmul_one(float *out, const int8_t *xq, const float *xs,
    const int8_t *wq, const float *ws, int n, int d, int stride) {
    const int groups = (n+127)/128, stride_groups = (stride+127)/128;
#if QWEN_OFFLINE_AME_LAYOUT
    const uintptr_t activation_spm = qwen_spm_pack_activation_k128(xq, n);
#else
    int8_t data[4][4096] __attribute__((aligned(64)));
#endif
    int32_t tiles[2][4][16] __attribute__((aligned(64)));
    for (int row0 = 0; row0 < d; row0 += 4) {
        float accum[4] = {0};
        /* Two K128 groups per descriptor makes the common hidden=1024
         * projection expose exactly four independent weight descriptors.
         * Larger FFN projections retain the same four-batch lookahead. */
        const int groups_per_descriptor = 2;
        const int batches =
            (groups + groups_per_descriptor - 1) / groups_per_descriptor;
        const int primed = batches < 4 ? batches : 4;
        int count = 0;
        for (int batch = 0; batch < primed; ++batch) {
            const int first = batch * groups_per_descriptor;
            count = groups-first < groups_per_descriptor ?
                groups-first : groups_per_descriptor;
#if QWEN_OFFLINE_AME_LAYOUT
            qwen_dma_blocks_begin(
                ame_v7_weight_packet(wq,row0,first,stride_groups),
                0xc0010000+(uintptr_t)batch*4096u,count*512);
#else
            pack_group_batch(data[batch],xq,wq,n,d,row0,first,count,
                             stride_groups);
            qwen_dma_blocks_begin(data[batch],
                0xc0010000+(uintptr_t)batch*4096u,count*1024);
#endif
        }
        uint64_t pending_ticket = 0;
        int pending_first = 0, pending_count = 0, pending_slot = 0;
        for (int batch = 0, first = 0; first < groups;
             ++batch, first += groups_per_descriptor) {
            const int input_slot = batch & 3;
            const int output_slot = batch & 1;
            count = groups-first < groups_per_descriptor ?
                groups-first : groups_per_descriptor;
            const int next_first = first + 4 * groups_per_descriptor;
            int next_count = groups-next_first;
            if (next_count < 0) next_count = 0;
            if (next_count > groups_per_descriptor)
                next_count = groups_per_descriptor;
            if (next_count) {
#if !QWEN_OFFLINE_AME_LAYOUT
                pack_group_batch(data[input_slot],xq,wq,n,d,row0,next_first,
                                 next_count,stride_groups);
#endif
            }
#if QWEN_OFFLINE_AME_LAYOUT
            const int8_t *next = next_count ?
                ame_v7_weight_packet(wq,row0,next_first,stride_groups) : NULL;
            const uint64_t ticket = qwen_dma_blocks_compute(
                &tiles[output_slot][0][0],
                activation_spm+(uintptr_t)first*512u,
                0xc0010000+(uintptr_t)input_slot*4096u,count,next,
                0xc0010000+(uintptr_t)input_slot*4096u,next_count*512,
                output_slot);
#else
            const uint64_t ticket = qwen_dma_blocks_compute(
                &tiles[output_slot][0][0],0,
                0xc0010000+(uintptr_t)input_slot*4096u,count,
                data[input_slot],0xc0010000+(uintptr_t)input_slot*4096u,
                next_count*1024,output_slot);
#endif
            g_tiles += count*8;
            if (pending_ticket) {
                qwen_dma_blocks_wait(pending_ticket, pending_slot);
                for (int g = 0; g < pending_count; ++g)
                    for (int j = 0; j < 4 && row0+j < d; ++j)
                        accum[j] += (float)tiles[pending_slot][g][j] *
                            xs[pending_first+g] *
                            ws[(size_t)(pending_first+g)*d+row0+j];
            }
            pending_ticket = ticket;
            pending_first = first;
            pending_count = count;
            pending_slot = output_slot;
        }
        if (pending_ticket) {
            qwen_dma_blocks_wait(pending_ticket, pending_slot);
            for (int g = 0; g < pending_count; ++g)
                for (int j = 0; j < 4 && row0+j < d; ++j)
                    accum[j] += (float)tiles[pending_slot][g][j] *
                        xs[pending_first+g] *
                        ws[(size_t)(pending_first+g)*d+row0+j];
        }
        for (int j = 0; j < 4 && row0+j < d; ++j) out[row0+j] = accum[j];
    }
}
#endif

static void packed_matmul_one(float *out, const int8_t *xq, const float *xs,
                              const int8_t *wq, const float *ws,
                              int n, int d, int gs, int stride) {
    ++g_calls;
#if QWEN_DMA_OVERLAP && defined(QWEN_LOCAL_SPM) && !defined(ZTT_SCALAR_FALLBACK)
    if (gs == 128) {
        dma_pipelined_matmul_one(out,xq,xs,wq,ws,n,d,stride);
        return;
    }
#endif
    const int groups = (n + gs - 1) / gs;
    const int stride_groups = (stride + gs - 1) / gs;
    /* Fixed bounded workspace: current quantization groups are <=128.
     * Larger groups are split into chunks while keeping integer accumulation. */
    for (int row0 = 0; row0 < d; row0 += 4) {
        float accum[4] = {0};
        for (int g = 0; g < groups; ++g) {
            int32_t group_sum[4] = {0};
            for (int chunk = 0; chunk < gs; chunk += 128) {
                int8_t at[8][64] __attribute__((aligned(64))) = {{0}};
                int8_t bt[8][64] __attribute__((aligned(64))) = {{0}};
                const int end = gs - chunk < 128 ? gs - chunk : 128;
                const int steps = (end + 15) / 16;
                for (int step = 0; step < steps; ++step)
                    for (int k = 0; k < 16; ++k) {
                        int off = chunk + step * 16 + k;
                        if (off >= gs || g * gs + off >= n) continue;
                        int square = k / 4, lane = k % 4;
                        at[step][square * 16 + lane] = xq[g * gs + off];
                        for (int j = 0; j < 4; ++j)
                            if (row0 + j < d)
                                bt[step][square * 16 + lane * 4 + j] =
                                    weight_value(wq, row0 + j, g, off, gs,
                                                 stride_groups);
                    }
#if QWEN_GROUP_ACC && defined(QWEN_LOCAL_SPM) && !defined(ZTT_SCALAR_FALLBACK)
                int32_t tile[16] __attribute__((aligned(64)));
                qwen_dma_group(tile, &at[0][0], &bt[0][0], steps);
                g_tiles += steps; // Count mmulacc instructions, not group drains.
                for (int j = 0; j < 4; ++j) group_sum[j] += tile[j];
#else
                for (int step = 0; step < steps; ++step) {
                    int32_t tile[16];
                    ztt_tile(tile, at[step], bt[step]);
                    for (int j = 0; j < 4 && row0+j < d; ++j) {
#if QWEN_GROUP_ACC
                        group_sum[j] += tile[j];
#else
                        accum[j] += (float)tile[j] * xs[g] * ws[(size_t)g*d+row0+j];
#endif
                    }
                }
#endif
            }
#if QWEN_GROUP_ACC
            for (int j = 0; j < 4 && row0+j < d; ++j)
                accum[j] += (float)group_sum[j] * xs[g] * ws[(size_t)g*d+row0+j];
#endif
        }
        for (int j = 0; j < 4 && row0+j < d; ++j) out[row0+j] = accum[j];
    }
}
#endif

static void matmul_one(float *out, const int8_t *xq, const float *xs,
                       const int8_t *wq, const float *ws, int n, int d,
                       int gs, int stride) {
#if QWEN_PACKED_K16
    packed_matmul_one(out, xq, xs, wq, ws, n, d, gs, stride);
    return;
#endif
    ++g_calls;
    /* The public API uses one-dimensional output. Keep a row-major scratch
     * so the tile kernel can process four output rows at a time. */
    const int groups = (n + gs - 1) / gs;
    const int stride_groups = (stride + gs - 1) / gs;
    for (int row0 = 0; row0 < d; row0 += ZTT_TILE) {
        float accum[ZTT_TILE] = {0.0f};
        for (int group = 0; group < groups; ++group) {
            for (int k0 = 0; k0 < gs; k0 += ZTT_TILE) {
                /* AMEUDSZ defaults to 32, so an int8 tile load transfers
                 * four 4x4 squares (64 bytes), even when only square 0 is
                 * populated for this kernel. */
                int8_t at[ZTT_TRANSFER_BYTES] = {0};
                int8_t bt[ZTT_TRANSFER_BYTES] = {0};
                int32_t tile[ZTT_TILE * ZTT_TILE];
                for (int i = 0; i < ZTT_TILE; ++i)
                    for (int k = 0; k < ZTT_TILE; ++k)
                        if (i == 0 && group * gs + k0 + k < n)
                            at[k] = xq[group * gs + k0 + k];
                for (int k = 0; k < ZTT_TILE; ++k)
                    for (int j = 0; j < ZTT_TILE; ++j) {
                        const int oj = row0 + j;
                        if (oj < d && k0 + k < gs)
                            bt[k * ZTT_TILE + j] =
                                weight_value(wq, oj, group, k0 + k, gs,
                                             stride_groups);
                    }
                ztt_tile(tile, at, bt);
                for (int j = 0; j < ZTT_TILE && row0 + j < d; ++j)
                    accum[j] += (float)tile[j] * xs[group] *
                                ws[(size_t)group * (size_t)d + row0 + j];
            }
        }
        for (int j = 0; j < ZTT_TILE && row0 + j < d; ++j)
            out[row0 + j] = accum[j];
    }
}

void ztt_matmul_quant_i8_i8_f32(float *out, const int8_t *xq,
                                const float *xs, const int8_t *wq,
                                const float *ws, int n, int d, int gs,
                                int stride) {
    matmul_one(out, xq, xs, wq, ws, n, d, gs, stride);
}

void ztt_matmul_quant_batch_i8_i8_f32(float *out, const int8_t *xq,
                                      const float *xs, const int8_t *wq,
                                      const float *ws, int T, int n, int d,
                                      int gs, int stride) {
    for (int t = 0; t < T; ++t)
        matmul_one(out + (size_t)t * d, xq + (size_t)t * n,
                   xs + (size_t)t * ((n + gs - 1) / gs), wq, ws, n, d, gs,
                   stride);
}

void ztt_matmul_quant_pair_i8_i8_f32(float *o0, float *o1,
                                     const int8_t *xq, const float *xs,
                                     const int8_t *wq0, const float *ws0,
                                     const int8_t *wq1, const float *ws1,
                                     int n, int d, int gs, int stride) {
    matmul_one(o0, xq, xs, wq0, ws0, n, d, gs, stride);
    matmul_one(o1, xq, xs, wq1, ws1, n, d, gs, stride);
}

void ztt_matmul_quant_batch_pair_i8_i8_f32(
    float *o0, float *o1, const int8_t *xq, const float *xs,
    const int8_t *wq0, const float *ws0, const int8_t *wq1,
    const float *ws1, int T, int n, int d, int gs, int stride) {
    for (int t = 0; t < T; ++t) {
        const int8_t *xqt = xq + (size_t)t * n;
        const float *xst = xs + (size_t)t * ((n + gs - 1) / gs);
        matmul_one(o0 + (size_t)t * d, xqt, xst, wq0, ws0, n, d, gs, stride);
        matmul_one(o1 + (size_t)t * d, xqt, xst, wq1, ws1, n, d, gs, stride);
    }
}
