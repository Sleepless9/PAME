/* v6 W4A8 layout [output/64][group][64][group_size]. Uses every packed
 * square and accumulates one full quantization group before FP32 scaling.
 * All dot products execute as guest Ztt instructions with DMA/SPM supply.
 */
#include "gem5_kernel_abi.h"
#include "gem5_m5ops.h"

extern void qwen_ztt_group(int32_t *, const int8_t *, const int8_t *, int);
extern void qwen_dma_copy(uintptr_t, uintptr_t, unsigned, unsigned);
static int8_t packed_a[4 * 4096] __attribute__((aligned(64)));
static int8_t packed_b[4 * 4096] __attribute__((aligned(64)));
static int32_t tile_c[16] __attribute__((aligned(64)));
static const uintptr_t spm_a = 0xc0000000;
static const uintptr_t spm_b = 0xc0010040;
static const uintptr_t spm_c = 0xc0020000;

/* Explicit RVV unit loads and strided stores avoid unsupported segmented
 * instructions. A fence makes the packed data visible before DMA starts. */
static void pack_column(int8_t *dst, const int8_t *src, size_t count) {
    const size_t stride = 4;
    while (count) {
        size_t vl;
        __asm__ volatile(
            "vsetvli %0, %3, e8, m1, ta, ma\n\t"
            "vle8.v v8, (%2)\n\t"
            "vsse8.v v8, (%1), %4"
            : "=&r"(vl) : "r"(dst), "r"(src), "r"(count), "r"(stride)
            : "v8", "memory");
        count -= vl; src += vl; dst += 4 * vl;
    }
}

static void pack_activation(int8_t *dst, const int8_t *src, size_t count) {
    const size_t stride = 16;
    count /= 4;
    while (count) {
        size_t vl;
        __asm__ volatile(
            "vsetvli %0, %3, e32, m1, ta, ma\n\t"
            "vle32.v v8, (%2)\n\t"
            "vsse32.v v8, (%1), %4"
            : "=&r"(vl) : "r"(dst), "r"(src), "r"(count), "r"(stride)
            : "v8", "memory");
        count -= vl; src += 4 * vl; dst += 16 * vl;
    }
}

void ame_matmul_quant_noblk_i8_i8_f32(float *out, const int8_t *xq,
    const float *xs, const int8_t *weights, const float *scales,
    int n, int d, int gs, int stride) {
    if (gs <= 0 || gs > 4096 || gs % 16 || n % gs || d % 64 || stride != n)
        m5_fail(0, 40);
    const int groups = n / gs;
    for (int j = 0; j < d; ++j) out[j] = 0;
    for (int i = 0; i < 4 * gs; ++i) packed_a[i] = 0;
    for (int g = 0; g < groups; ++g) {
        // One active row, four consecutive K blocks per matrix register.
        pack_activation(packed_a, xq + g * gs, gs);
        qwen_dma_copy((uintptr_t)packed_a, spm_a, 4 * gs, 0);
        for (int j = 0; j < d; j += 4) {
            const int8_t *base = weights +
                ((size_t)(j / 64) * groups + g) * 64 * gs + (j % 64) * gs;
            for (int col = 0; col < 4; ++col)
                pack_column(packed_b + col, base + col * gs, gs);
            qwen_dma_copy((uintptr_t)packed_b, spm_b, 4 * gs, 0);
            qwen_ztt_group((int32_t *)spm_c, (const int8_t *)spm_a,
                           (const int8_t *)spm_b, gs / 16);
            qwen_dma_copy((uintptr_t)tile_c, spm_c, 64, 1);
            for (int col = 0; col < 4; ++col)
                out[j + col] += (float)tile_c[col] * xs[g] * scales[g * d + j + col];
        }
    }
}

void ame_matmul_quant_batch_i8_i8_f32(float *out, const int8_t *xq,
    const float *xs, const int8_t *w, const float *ws,
    int tokens, int n, int d, int gs, int stride) {
    for (int t = 0; t < tokens; ++t)
        ame_matmul_quant_noblk_i8_i8_f32(out + t * d, xq + t * n,
            xs + t * (n / gs), w, ws, n, d, gs, stride);
}

void ame_matmul_quant_noblk_pair_i8_i8_f32(float *o0, float *o1,
    const int8_t *xq, const float *xs, const int8_t *w0, const float *s0,
    const int8_t *w1, const float *s1, int n, int d, int gs, int stride) {
    ame_matmul_quant_noblk_i8_i8_f32(o0, xq, xs, w0, s0, n, d, gs, stride);
    ame_matmul_quant_noblk_i8_i8_f32(o1, xq, xs, w1, s1, n, d, gs, stride);
}

void ame_matmul_quant_batch_pair_i8_i8_f32(float *o0, float *o1,
    const int8_t *xq, const float *xs, const int8_t *w0, const float *s0,
    const int8_t *w1, const float *s1, int tokens, int n, int d, int gs, int stride) {
    ame_matmul_quant_batch_i8_i8_f32(o0, xq, xs, w0, s0, tokens, n, d, gs, stride);
    ame_matmul_quant_batch_i8_i8_f32(o1, xq, xs, w1, s1, tokens, n, d, gs, stride);
}
