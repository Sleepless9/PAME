#include "ztt_w4a8.h"

#include <stddef.h>
#include <stdint.h>

#include "spm_virt_dma.h"

#define ZTT_TILE 4
#define ZTT_OUTPUT_BLOCK 64
#define ZTT_TRANSFER_BYTES 64
#define ZTT_MAX_GROUP 128

#define SPM_A_OFFSET 0x0000
#define SPM_B_OFFSET 0x0040
#define SPM_C_OFFSET 0x0080
#define SPM_X_OFFSET 0x0100
#define SPM_OUTPUT_OFFSET 0x0400
#define SPM_WEIGHT_OFFSET 0x1000

#if defined(__riscv) && !defined(ZTT_SCALAR_FALLBACK)
extern void ztt_i8_i8_i32_4x4(int32_t *c, const int8_t *a,
                               const int8_t *b);
#endif
extern void rvv_memcpy_u8(void *dst, const void *src, size_t bytes);

static uint64_t calls;
static uint64_t tiles;

void ztt_matmul_stats_reset(void) { calls = 0; tiles = 0; }
uint64_t ztt_matmul_stats_calls(void) { return calls; }
uint64_t ztt_matmul_stats_tiles(void) { return tiles; }

static inline void *
spm_ptr(size_t offset)
{
    return (void *)(uintptr_t)(SPM_BASE_ADDR + offset);
}

static void
zero_bytes(void *pointer, size_t bytes)
{
    volatile uint8_t *dst = (volatile uint8_t *)pointer;
    for (size_t index = 0; index < bytes; ++index)
        dst[index] = 0;
}

static void
run_tile(int32_t *output, const int8_t *lhs, const int8_t *rhs)
{
#if defined(__riscv) && !defined(ZTT_SCALAR_FALLBACK)
    ztt_i8_i8_i32_4x4(output, lhs, rhs);
#else
    for (int row = 0; row < ZTT_TILE; ++row) {
        for (int col = 0; col < ZTT_TILE; ++col) {
            int32_t sum = 0;
            for (int k = 0; k < ZTT_TILE; ++k)
                sum += lhs[row * ZTT_TILE + k] *
                       rhs[k * ZTT_TILE + col];
            output[row * ZTT_TILE + col] = sum;
        }
    }
#endif
    ++tiles;
}

static const int8_t *
weight_at(const int8_t *weights, int output, int group, int k,
          int group_size, int stride)
{
    const int block = output / ZTT_OUTPUT_BLOCK;
    const int row = output % ZTT_OUTPUT_BLOCK;
    const int stride_groups = (stride + group_size - 1) / group_size;
    return weights +
        ((size_t)block * stride_groups + group) *
            ZTT_OUTPUT_BLOCK * group_size +
        (size_t)row * group_size + k;
}

static void
matmul_direct(float *output, const int8_t *input, const float *input_scales,
              const int8_t *weights, const float *weight_scales, int tokens,
              int inputs, int outputs, int group_size, int stride)
{
    const int groups = (inputs + group_size - 1) / group_size;
    for (int token_base = 0; token_base < tokens; token_base += ZTT_TILE) {
        const int token_count = tokens - token_base < ZTT_TILE ?
            tokens - token_base : ZTT_TILE;
        for (int output_base = 0; output_base < outputs;
             output_base += ZTT_TILE) {
            float accum[ZTT_TILE][ZTT_TILE] = {{0.0f}};
            for (int group = 0; group < groups; ++group) {
                for (int k0 = 0; k0 < group_size; k0 += ZTT_TILE) {
                    int8_t lhs[ZTT_TRANSFER_BYTES] = {0};
                    int8_t rhs[ZTT_TRANSFER_BYTES] = {0};
                    int32_t tile[ZTT_TILE * ZTT_TILE];
                    for (int row = 0; row < token_count; ++row) {
                        for (int k = 0; k < ZTT_TILE; ++k) {
                            const int input_index = group * group_size + k0 + k;
                            if (input_index < inputs)
                                lhs[row * ZTT_TILE + k] =
                                    input[(size_t)(token_base + row) * inputs +
                                          input_index];
                        }
                    }
                    for (int k = 0; k < ZTT_TILE; ++k) {
                        for (int col = 0; col < ZTT_TILE; ++col) {
                            if (output_base + col < outputs &&
                                k0 + k < group_size) {
                                rhs[k * ZTT_TILE + col] = *weight_at(
                                    weights, output_base + col, group,
                                    k0 + k, group_size, stride);
                            }
                        }
                    }
                    run_tile(tile, lhs, rhs);
                    for (int row = 0; row < token_count; ++row) {
                        for (int col = 0; col < ZTT_TILE &&
                                          output_base + col < outputs; ++col) {
                            accum[row][col] +=
                                (float)tile[row * ZTT_TILE + col] *
                                input_scales[(size_t)(token_base + row) * groups +
                                             group] *
                                weight_scales[(size_t)group * outputs +
                                              output_base + col];
                        }
                    }
                }
            }
            for (int row = 0; row < token_count; ++row) {
                for (int col = 0; col < ZTT_TILE &&
                                  output_base + col < outputs; ++col) {
                    output[(size_t)(token_base + row) * outputs +
                           output_base + col] = accum[row][col];
                }
            }
        }
    }
}

static int
matmul_spm(float *output, const int8_t *input, const float *input_scales,
           const int8_t *weights, const float *weight_scales, int tokens,
           int inputs, int outputs, int group_size, int stride)
{
    if (tokens <= 0 || group_size <= 0 || group_size > ZTT_MAX_GROUP ||
        stride <= 0)
        return -1;

    int8_t *lhs = (int8_t *)spm_ptr(SPM_A_OFFSET);
    int8_t *rhs = (int8_t *)spm_ptr(SPM_B_OFFSET);
    int32_t *tile = (int32_t *)spm_ptr(SPM_C_OFFSET);
    int8_t *input_spm = (int8_t *)spm_ptr(SPM_X_OFFSET);
    float *output_spm = (float *)spm_ptr(SPM_OUTPUT_OFFSET);
    int8_t *weight_spm = (int8_t *)spm_ptr(SPM_WEIGHT_OFFSET);
    const int groups = (inputs + group_size - 1) / group_size;
    const int stride_groups = (stride + group_size - 1) / group_size;

    for (int token_base = 0; token_base < tokens; token_base += ZTT_TILE) {
        const int token_count = tokens - token_base < ZTT_TILE ?
            tokens - token_base : ZTT_TILE;
        for (int output_block = 0;
             output_block * ZTT_OUTPUT_BLOCK < outputs; ++output_block) {
            const int output_base = output_block * ZTT_OUTPUT_BLOCK;
            const int output_count = outputs - output_base < ZTT_OUTPUT_BLOCK ?
                outputs - output_base : ZTT_OUTPUT_BLOCK;
            float accum[ZTT_TILE][ZTT_OUTPUT_BLOCK] = {{0.0f}};

            for (int group = 0; group < groups; ++group) {
                const int input_base = group * group_size;
                const int input_count = inputs - input_base < group_size ?
                    inputs - input_base : group_size;
                const int8_t *weight_source = weights +
                    ((size_t)output_block * stride_groups + group) *
                    ZTT_OUTPUT_BLOCK * group_size;

                zero_bytes(input_spm, ZTT_TILE * ZTT_MAX_GROUP);
                if (spm_dma_copy_2d_to_spm(
                        input_spm,
                        input + (size_t)token_base * inputs + input_base,
                        input_count, token_count, inputs, group_size) != 0 ||
                    spm_dma_copy_to_spm(
                        weight_spm, weight_source,
                        ZTT_OUTPUT_BLOCK * group_size) != 0)
                    return -1;

                for (int row = 0; row < output_count; row += ZTT_TILE) {
                    for (int k0 = 0; k0 < group_size; k0 += ZTT_TILE) {
                        zero_bytes(lhs, ZTT_TRANSFER_BYTES);
                        zero_bytes(rhs, ZTT_TRANSFER_BYTES);
                        for (int token = 0; token < token_count; ++token) {
                            const int valid = input_count - k0 < ZTT_TILE ?
                                input_count - k0 : ZTT_TILE;
                            if (valid > 0)
                                rvv_memcpy_u8(
                                    lhs + token * ZTT_TILE,
                                    input_spm + token * group_size + k0,
                                    valid);
                        }
                        for (int k = 0; k < ZTT_TILE; ++k) {
                            for (int col = 0; col < ZTT_TILE; ++col) {
                                if (k0 + k < input_count &&
                                    row + col < output_count) {
                                    rhs[k * ZTT_TILE + col] =
                                        weight_spm[(size_t)(row + col) *
                                                   group_size + k0 + k];
                                }
                            }
                        }
                        run_tile(tile, lhs, rhs);
                        for (int token = 0; token < token_count; ++token) {
                            for (int col = 0; col < ZTT_TILE &&
                                              row + col < output_count; ++col) {
                                accum[token][row + col] +=
                                    (float)tile[token * ZTT_TILE + col] *
                                    input_scales[
                                        (size_t)(token_base + token) * groups +
                                        group] *
                                    weight_scales[(size_t)group * outputs +
                                                  output_base + row + col];
                            }
                        }
                    }
                }
            }

            for (int token = 0; token < token_count; ++token) {
                for (int row = 0; row < output_count; ++row)
                    output_spm[token * ZTT_OUTPUT_BLOCK + row] =
                        accum[token][row];
            }
            if (spm_dma_copy_2d_from_spm(
                    output + (size_t)token_base * outputs + output_base,
                    output_spm, output_count * sizeof(float), token_count,
                    outputs * sizeof(float),
                    ZTT_OUTPUT_BLOCK * sizeof(float)) != 0)
                return -1;
        }
    }
    return 0;
}

static void
matmul(float *output, const int8_t *input, const float *input_scales,
       const int8_t *weights, const float *weight_scales, int tokens,
       int inputs, int outputs, int group_size, int stride)
{
    calls += tokens;
    if (matmul_spm(output, input, input_scales, weights, weight_scales,
                   tokens, inputs, outputs, group_size, stride) != 0) {
        matmul_direct(output, input, input_scales, weights, weight_scales,
                      tokens, inputs, outputs, group_size, stride);
    }
}

void
ztt_matmul_quant_i8_i8_f32(float *output, const int8_t *input,
                            const float *input_scales, const int8_t *weights,
                            const float *weight_scales, int inputs,
                            int outputs, int group_size, int stride)
{
    matmul(output, input, input_scales, weights, weight_scales, 1, inputs,
           outputs, group_size, stride);
}

void
ztt_matmul_quant_batch_i8_i8_f32(
    float *output, const int8_t *input, const float *input_scales,
    const int8_t *weights, const float *weight_scales, int tokens, int inputs,
    int outputs, int group_size, int stride)
{
    matmul(output, input, input_scales, weights, weight_scales, tokens,
           inputs, outputs, group_size, stride);
}

void
ztt_matmul_quant_pair_i8_i8_f32(
    float *output0, float *output1, const int8_t *input,
    const float *input_scales, const int8_t *weights0,
    const float *weight_scales0, const int8_t *weights1,
    const float *weight_scales1, int inputs, int outputs, int group_size,
    int stride)
{
    matmul(output0, input, input_scales, weights0, weight_scales0, 1,
           inputs, outputs, group_size, stride);
    matmul(output1, input, input_scales, weights1, weight_scales1, 1,
           inputs, outputs, group_size, stride);
}

void
ztt_matmul_quant_batch_pair_i8_i8_f32(
    float *output0, float *output1, const int8_t *input,
    const float *input_scales, const int8_t *weights0,
    const float *weight_scales0, const int8_t *weights1,
    const float *weight_scales1, int tokens, int inputs, int outputs,
    int group_size, int stride)
{
    matmul(output0, input, input_scales, weights0, weight_scales0, tokens,
           inputs, outputs, group_size, stride);
    matmul(output1, input, input_scales, weights1, weight_scales1, tokens,
           inputs, outputs, group_size, stride);
}
