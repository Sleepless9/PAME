#include "hybrid_flash_attention.h"

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

#include "backends/ops_lib.h"
#include "spm_virt_dma.h"

#define KV_BLOCK_TOKENS 16
#define MAX_HEAD_DIM 256
#define MAX_QUERIES_PER_KV 8

#define SPM_ATTN_Q_OFFSET 0x08000
#define SPM_ATTN_K_OFFSET 0x0a000
#define SPM_ATTN_V_OFFSET 0x0e000
#define SPM_ATTN_OUTPUT_OFFSET 0x12000
#define SPM_ATTN_SCORE_OFFSET 0x14000
#define SPM_ATTN_PRODUCT_OFFSET 0x14400

static inline uint64_t
read_cycle(void)
{
    uint64_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline float *
spm_float(size_t offset)
{
    return (float *)(uintptr_t)(SPM_BASE_ADDR + offset);
}

static void
scale_vector(float *values, float scale, int count)
{
    int done = 0;
    while (done < count) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(count - done));
        vfloat32m1_t vector = __riscv_vle32_v_f32m1(values + done, vl);
        vector = __riscv_vfmul_vf_f32m1(vector, scale, vl);
        __riscv_vse32_v_f32m1(values + done, vector, vl);
        done += (int)vl;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static void
axpy_vector(float *output, const float *input, float scale, int count)
{
    int done = 0;
    while (done < count) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(count - done));
        vfloat32m1_t out = __riscv_vle32_v_f32m1(output + done, vl);
        vfloat32m1_t in = __riscv_vle32_v_f32m1(input + done, vl);
        in = __riscv_vfmul_vf_f32m1(in, scale, vl);
        out = __riscv_vfadd_vv_f32m1(out, in, vl);
        __riscv_vse32_v_f32m1(output + done, out, vl);
        done += (int)vl;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static float
dot_vector(const float *lhs, const float *rhs, int count)
{
    float *products = spm_float(SPM_ATTN_PRODUCT_OFFSET);
    int done = 0;
    while (done < count) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(count - done));
        vfloat32m1_t left = __riscv_vle32_v_f32m1(lhs + done, vl);
        vfloat32m1_t right = __riscv_vle32_v_f32m1(rhs + done, vl);
        vfloat32m1_t product = __riscv_vfmul_vv_f32m1(left, right, vl);
        __riscv_vse32_v_f32m1(products + done, product, vl);
        done += (int)vl;
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
    float sum = 0.0f;
    for (int index = 0; index < count; ++index)
        sum += products[index];
    return sum;
}

static float
exp_scalar(float value)
{
    float result = 0.0f;
    rvv_exp_vec_f32(&result, &value, 1);
    return result;
}

void
hybrid_flash_attention(
    float *output, const float *query, const float *key_cache,
    const float *value_cache, int layer, int position, int query_heads,
    int kv_heads, int max_sequence, int head_dim, float head_dim_sqrt,
    uint64_t *qk_cycles, uint64_t *softmax_cycles, uint64_t *av_cycles)
{
    uint64_t qk_total = 0;
    uint64_t softmax_total = 0;
    uint64_t av_total = 0;
    const int queries_per_kv = query_heads / kv_heads;
    const size_t layer_stride =
        (size_t)kv_heads * max_sequence * head_dim;
    float *query_spm = spm_float(SPM_ATTN_Q_OFFSET);
    float *key_spm = spm_float(SPM_ATTN_K_OFFSET);
    float *value_spm = spm_float(SPM_ATTN_V_OFFSET);
    float *output_spm = spm_float(SPM_ATTN_OUTPUT_OFFSET);
    float *scores = spm_float(SPM_ATTN_SCORE_OFFSET);

    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM || kv_heads <= 0 ||
        queries_per_kv <= 0 || queries_per_kv > MAX_QUERIES_PER_KV) {
        return;
    }

    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const int first_query_head = kv_head * queries_per_kv;
        const float *query_source =
            query + (size_t)first_query_head * head_dim;
        const float *query_work = query_spm;
        uint64_t start = read_cycle();
        if (spm_dma_copy_to_spm(
                query_spm, query_source,
                (size_t)queries_per_kv * head_dim * sizeof(float)) != 0) {
            query_work = query_source;
        }
        rvv_zero_f32(output_spm, queries_per_kv * head_dim);
        qk_total += read_cycle() - start;

        float running_max[MAX_QUERIES_PER_KV];
        float running_sum[MAX_QUERIES_PER_KV];
        for (int query_index = 0; query_index < queries_per_kv;
             ++query_index) {
            running_max[query_index] = -3.402823466e38f;
            running_sum[query_index] = 0.0f;
        }

        const float *key_head = key_cache + layer * layer_stride +
            (size_t)kv_head * max_sequence * head_dim;
        const float *value_head = value_cache + layer * layer_stride +
            (size_t)kv_head * max_sequence * head_dim;

        for (int token_base = 0; token_base <= position;
             token_base += KV_BLOCK_TOKENS) {
            const int token_count = position + 1 - token_base < KV_BLOCK_TOKENS ?
                position + 1 - token_base : KV_BLOCK_TOKENS;
            const size_t block_bytes =
                (size_t)token_count * head_dim * sizeof(float);
            const float *key_work = key_spm;
            const float *value_work = value_spm;

            start = read_cycle();
            if (spm_dma_copy_to_spm(
                    key_spm, key_head + (size_t)token_base * head_dim,
                    block_bytes) != 0) {
                key_work = key_head + (size_t)token_base * head_dim;
            }
            for (int query_index = 0; query_index < queries_per_kv;
                 ++query_index) {
                const float *query_row =
                    query_work + (size_t)query_index * head_dim;
                float *score_row = scores + query_index * KV_BLOCK_TOKENS;
                for (int token = 0; token < token_count; ++token) {
                    score_row[token] = dot_vector(
                        query_row, key_work + (size_t)token * head_dim,
                        head_dim) / head_dim_sqrt;
                }
            }
            qk_total += read_cycle() - start;

            start = read_cycle();
            if (spm_dma_copy_to_spm(
                    value_spm,
                    value_head + (size_t)token_base * head_dim,
                    block_bytes) != 0) {
                value_work = value_head + (size_t)token_base * head_dim;
            }
            av_total += read_cycle() - start;

            for (int query_index = 0; query_index < queries_per_kv;
                 ++query_index) {
                float *score_row = scores + query_index * KV_BLOCK_TOKENS;
                float *output_row =
                    output_spm + (size_t)query_index * head_dim;
                start = read_cycle();
                float block_max = score_row[0];
                for (int token = 1; token < token_count; ++token) {
                    if (score_row[token] > block_max)
                        block_max = score_row[token];
                }
                const float old_max = running_max[query_index];
                const float new_max = old_max > block_max ? old_max : block_max;
                const float old_scale = running_sum[query_index] == 0.0f ?
                    0.0f : exp_scalar(old_max - new_max);
                for (int token = 0; token < token_count; ++token)
                    score_row[token] -= new_max;
                rvv_exp_vec_f32(score_row, score_row, token_count);
                float block_sum = 0.0f;
                for (int token = 0; token < token_count; ++token)
                    block_sum += score_row[token];
                scale_vector(output_row, old_scale, head_dim);
                running_sum[query_index] =
                    running_sum[query_index] * old_scale + block_sum;
                running_max[query_index] = new_max;
                softmax_total += read_cycle() - start;

                start = read_cycle();
                for (int token = 0; token < token_count; ++token) {
                    axpy_vector(output_row,
                                value_work + (size_t)token * head_dim,
                                score_row[token], head_dim);
                }
                av_total += read_cycle() - start;
            }
        }

        start = read_cycle();
        for (int query_index = 0; query_index < queries_per_kv;
             ++query_index) {
            float *output_row = output_spm + (size_t)query_index * head_dim;
            scale_vector(output_row, 1.0f / running_sum[query_index],
                         head_dim);
        }
        const size_t output_bytes =
            (size_t)queries_per_kv * head_dim * sizeof(float);
        if (spm_dma_copy_from_spm(
                output + (size_t)first_query_head * head_dim, output_spm,
                output_bytes) != 0) {
            rvv_memcpy_f32(
                output + (size_t)first_query_head * head_dim, output_spm,
                queries_per_kv * head_dim);
        }
        av_total += read_cycle() - start;
    }

    if (qk_cycles)
        *qk_cycles += qk_total;
    if (softmax_cycles)
        *softmax_cycles += softmax_total;
    if (av_cycles)
        *av_cycles += av_total;
}
