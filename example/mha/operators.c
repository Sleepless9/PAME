/* Copyright (c) 2026 BOSC & ICT, CAS. All rights reserved. */

#if !defined(__riscv)
#include <math.h>
#endif
#include <stdint.h>

#include "operators.h"
#include "ztt_operator_case.h"

static float ztt_operator_outputs[ZTT_OPERATOR_CASE_COUNT]
                                 [ZTT_OPERATOR_MAX_OUTPUT_ELEMENTS]
                                 ZTT_LLM_ALIGNED;

#if !defined(__riscv)
static void
ztt_host_matmul_tile(const float *a, const float *b, float *output,
                     int transpose_b)
{
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            float value = 0.0f;
            for (unsigned depth = 0; depth < ZTT_LLM_TILE; ++depth) {
                const float b_value = transpose_b ?
                    b[col * ZTT_LLM_TILE + depth] :
                    b[depth * ZTT_LLM_TILE + col];
                value += a[row * ZTT_LLM_TILE + depth] * b_value;
            }
            output[row * ZTT_LLM_TILE + col] = value;
        }
}
#endif

#if defined(__riscv)
static inline void
ztt_matmul_tile(const float *a, const float *b, float *output,
                int transpose_b)
{
    float zero[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED = {0};
    ztt_llm_init_fp32_acc(zero);
    if (transpose_b)
        ztt_llm_accumulate_fp32_bt(a, b);
    else
        ztt_llm_accumulate_fp32(a, b);
    ztt_llm_store_fp32_acc(output, 0);
}

static inline void
ztt_softmax_tile(const float *input, float *output)
{
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m1, %[fp32]\n\t"
        "msettyp m2, %[fp32]\n\t"
        "msettyp m3, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mreducemax.row m1, m0\n\t"
        "mmax.ew m1, m1, m0\n\t"
        "msub.ew m2, m1, m0\n\t"
        "mexp2.ew m2, m2\n\t"
        "mreduceadd.row m3, m2\n\t"
        "mrec.ew m3, m3\n\t"
        "mmul.ew m4, m2, m3\n\t"
        "mss.rm m4, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input), [output] "r"(output)
        : "memory");
}
#else
static void
ztt_softmax_tile(const float *input, float *output)
{
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
        float maximum = input[row * ZTT_LLM_TILE];
        for (unsigned col = 1; col < ZTT_LLM_TILE; ++col)
            if (input[row * ZTT_LLM_TILE + col] > maximum)
                maximum = input[row * ZTT_LLM_TILE + col];
        float sum = 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            output[row * ZTT_LLM_TILE + col] =
                exp2f(input[row * ZTT_LLM_TILE + col] - maximum);
            sum += output[row * ZTT_LLM_TILE + col];
        }
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            output[row * ZTT_LLM_TILE + col] /= sum;
    }
}
#endif

uint64_t
ztt_mha_fp32(const float *input, const float *const wq[2],
             const float *const wk[2], const float *const wv[2],
             unsigned sequence, unsigned model_dim, unsigned head_dim,
             unsigned heads, float softmax_scale, float *output)
{
    if (!input || !output || sequence == 0 || sequence > ZTT_LLM_TILE ||
        model_dim == 0 || model_dim > ZTT_LLM_TILE || head_dim == 0 ||
        head_dim > ZTT_LLM_TILE || heads == 0 || heads > 2)
        return ZTT_OPERATOR_INVALID_ARGUMENT;

    float q[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float k[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float v[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float scores[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float probabilities[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float context[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;

#if defined(__riscv)
    ztt_llm_acquire();
    ztt_llm_configure_fp32_gemm();
#endif
    for (unsigned head = 0; head < heads; ++head) {
        if (!wq[head] || !wk[head] || !wv[head]) {
#if defined(__riscv)
            (void)ztt_llm_release();
#endif
            return ZTT_OPERATOR_INVALID_ARGUMENT;
        }
#if defined(__riscv)
        ztt_matmul_tile(input, wq[head], q, 0);
        ztt_matmul_tile(input, wk[head], k, 0);
        ztt_matmul_tile(input, wv[head], v, 0);
        ztt_matmul_tile(q, k, scores, 1);
#else
        ztt_host_matmul_tile(input, wq[head], q, 0);
        ztt_host_matmul_tile(input, wk[head], k, 0);
        ztt_host_matmul_tile(input, wv[head], v, 0);
        ztt_host_matmul_tile(q, k, scores, 1);
#endif
        for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
            for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
                scores[row * ZTT_LLM_TILE + col] =
                    row < sequence && col < sequence ?
                    scores[row * ZTT_LLM_TILE + col] * softmax_scale :
                    (row < sequence ? -200.0f : 0.0f);
        ztt_softmax_tile(scores, probabilities);
#if defined(__riscv)
        ztt_matmul_tile(probabilities, v, context, 0);
#else
        ztt_host_matmul_tile(probabilities, v, context, 0);
#endif
        for (unsigned row = 0; row < sequence; ++row)
            for (unsigned col = 0; col < head_dim; ++col)
                output[(head * sequence + row) * head_dim + col] =
                    context[row * ZTT_LLM_TILE + col];
    }
#if defined(__riscv)
    return ztt_llm_release();
#else
    return 0;
#endif
}

uint64_t
ztt_operator_run_suite(struct ztt_operator_suite_result *result)
{
    uint64_t status = 0;
    result->case_count = ZTT_OPERATOR_CASE_COUNT;
    for (unsigned i = 0; i < ZTT_OPERATOR_CASE_COUNT; ++i) {
        const struct ztt_generated_mha_case *test = &ztt_operator_cases[i];
        const uint64_t case_status = ztt_mha_fp32(
            test->input, test->wq, test->wk, test->wv, test->sequence,
            test->model_dim, test->head_dim, test->heads,
            test->softmax_scale, ztt_operator_outputs[i]);
        result->cases[i] = (struct ztt_operator_case_result){
            test->name, ztt_operator_outputs[i],
            test->heads * test->sequence * test->head_dim,
            sizeof(float), case_status};
        status |= case_status;
    }
    return status;
}
