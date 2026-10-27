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
ztt_host_matmul_tile(const float *a, const float *b, float *output)
{
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row)
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            float value = 0.0f;
            for (unsigned depth = 0; depth < ZTT_LLM_TILE; ++depth)
                value += a[row * ZTT_LLM_TILE + depth] *
                         b[depth * ZTT_LLM_TILE + col];
            output[row * ZTT_LLM_TILE + col] = value;
        }
}
#endif

#if defined(__riscv)
static inline void
ztt_matmul_tile(const float *a, const float *b, float *output)
{
    float zero[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED = {0};
    ztt_llm_init_fp32_acc(zero);
    ztt_llm_accumulate_fp32(a, b);
    ztt_llm_store_fp32_acc(output, 0);
}

static inline void
ztt_gelu_tile(const float *input, float *output)
{
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    float four[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float zero[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED = {0};
    float one[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i) {
        four[i] = 4.0f;
        one[i] = 1.0f;
    }
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m1, %[fp32]\n\t"
        "msettyp m2, %[fp32]\n\t"
        "msettyp m3, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "msettyp m5, %[fp32]\n\t"
        "msettyp m6, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mls.rm m1, %[four]\n\t"
        "mls.rm m4, %[zero]\n\t"
        "mls.rm m5, %[one]\n\t"
        "msqrt.ew m2, m1\n\t"
        "mmul.ew m3, m0, m2\n\t"
        "msub.ew m3, m3, m4\n\t"
        "mexp2.ew m3, m3\n\t"
        "madd.ew m3, m3, m5\n\t"
        "mrec.ew m3, m3\n\t"
        "mmul.ew m6, m0, m3\n\t"
        "mss.rm m6, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input), [four] "r"(four),
           [zero] "r"(zero), [one] "r"(one), [output] "r"(output)
        : "memory");
}
#else
static void
ztt_gelu_tile(const float *input, float *output)
{
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i)
        output[i] = input[i] / (1.0f + exp2f(-2.0f * input[i]));
}
#endif

uint64_t
ztt_ffn_fp32(const float *input, const float *w1, const float *w2,
             unsigned tokens, unsigned input_dim, unsigned hidden_dim,
             unsigned output_dim, float *output)
{
    if (!input || !w1 || !w2 || !output || tokens == 0 ||
        tokens > ZTT_LLM_TILE || input_dim == 0 || input_dim > ZTT_LLM_TILE ||
        hidden_dim == 0 || hidden_dim > ZTT_LLM_TILE || output_dim == 0 ||
        output_dim > ZTT_LLM_TILE)
        return ZTT_OPERATOR_INVALID_ARGUMENT;

    float hidden[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float activated[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float result[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
#if defined(__riscv)
    ztt_llm_acquire();
    ztt_llm_configure_fp32_gemm();
    ztt_matmul_tile(input, w1, hidden);
    ztt_gelu_tile(hidden, activated);
    ztt_matmul_tile(activated, w2, result);
    const uint64_t status = ztt_llm_release();
#else
    ztt_host_matmul_tile(input, w1, hidden);
    ztt_gelu_tile(hidden, activated);
    ztt_host_matmul_tile(activated, w2, result);
    const uint64_t status = 0;
#endif
    for (unsigned row = 0; row < tokens; ++row)
        for (unsigned col = 0; col < output_dim; ++col)
            output[row * output_dim + col] =
                result[row * ZTT_LLM_TILE + col];
    return status;
}

uint64_t
ztt_operator_run_suite(struct ztt_operator_suite_result *result)
{
    uint64_t status = 0;
    result->case_count = ZTT_OPERATOR_CASE_COUNT;
    for (unsigned i = 0; i < ZTT_OPERATOR_CASE_COUNT; ++i) {
        const struct ztt_generated_ffn_case *test = &ztt_operator_cases[i];
        const uint64_t case_status = ztt_ffn_fp32(
            test->input, test->w1, test->w2, test->tokens,
            test->input_dim, test->hidden_dim, test->output_dim,
            ztt_operator_outputs[i]);
        result->cases[i] = (struct ztt_operator_case_result){
            test->name, ztt_operator_outputs[i],
            test->tokens * test->output_dim, sizeof(float), case_status};
        status |= case_status;
    }
    return status;
}
