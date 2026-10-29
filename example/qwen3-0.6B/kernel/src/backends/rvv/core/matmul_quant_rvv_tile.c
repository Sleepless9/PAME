#include "backends/rvv/core/matmul_quant_rvv_tile.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#define RVV_TILE_OUTBLK 64

static uint64_t calls;
static uint64_t macs;
static uint64_t row_groups;
static uint64_t vector_chunks;

void
rvv_tile_matmul_stats_reset(void)
{
    calls = 0;
    macs = 0;
    row_groups = 0;
    vector_chunks = 0;
}

uint64_t rvv_tile_matmul_stats_calls(void) { return calls; }
uint64_t rvv_tile_matmul_stats_macs(void) { return macs; }
uint64_t rvv_tile_matmul_stats_row_groups(void) { return row_groups; }
uint64_t rvv_tile_matmul_stats_vector_chunks(void) { return vector_chunks; }

static inline int32_t
dot_i8_rvv(const int8_t *lhs, const int8_t *rhs, int count)
{
    int32_t sum = 0;
    int done = 0;

    while (done < count) {
        const size_t vl = __riscv_vsetvl_e8m2((size_t)(count - done));
        const vint8m2_t a = __riscv_vle8_v_i8m2(lhs + done, vl);
        const vint8m2_t b = __riscv_vle8_v_i8m2(rhs + done, vl);
        const vint16m4_t product = __riscv_vwmul_vv_i16m4(a, b, vl);
        const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, 1);
        const vint32m1_t reduced =
            __riscv_vwredsum_vs_i16m4_i32m1(product, zero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(reduced);
        done += (int)vl;
        vector_chunks++;
    }
    return sum;
}

void
rvv_matmul_quant_tile_i8_i8_f32(float *xout,
                                const int8_t *xq,
                                const float *xs,
                                const int8_t *wq,
                                const float *ws,
                                int n,
                                int d,
                                int gs)
{
    if (!xout || !xq || !xs || !wq || !ws ||
        n <= 0 || d <= 0 || gs <= 0 ||
        n % gs != 0 || d % RVV_TILE_OUTBLK != 0) {
        return;
    }

    const int groups = n / gs;
    const size_t tile_bytes = (size_t)RVV_TILE_OUTBLK * (size_t)gs;
    calls++;
    macs += (uint64_t)n * (uint64_t)d;

    for (int block = 0; block < d / RVV_TILE_OUTBLK; ++block) {
        float accum[RVV_TILE_OUTBLK] = {0};
        const int row_base = block * RVV_TILE_OUTBLK;

        for (int group = 0; group < groups; ++group) {
            const int8_t *tile = wq +
                ((size_t)block * (size_t)groups + (size_t)group) *
                tile_bytes;
            const int8_t *x_group = xq + (size_t)group * (size_t)gs;
            const float x_scale = xs[group];
            const float *w_scale = ws +
                (size_t)group * (size_t)d + (size_t)row_base;

            for (int row = 0; row < RVV_TILE_OUTBLK; ++row) {
                const int32_t dot = dot_i8_rvv(
                    tile + (size_t)row * (size_t)gs, x_group, gs);
                accum[row] += (float)dot * w_scale[row] * x_scale;
            }
            row_groups += RVV_TILE_OUTBLK;
        }

        for (int row = 0; row < RVV_TILE_OUTBLK; ++row)
            xout[row_base + row] = accum[row];
    }
}
