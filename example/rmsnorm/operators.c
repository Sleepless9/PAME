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

uint64_t
ztt_rmsnorm_fp32(const float *input, const float *gamma, unsigned rows,
                 unsigned cols, float epsilon, float *output)
{
    if (!input || !gamma || !output || rows == 0 || rows > ZTT_LLM_TILE ||
        cols == 0 || cols > ZTT_LLM_TILE || epsilon <= 0.0f)
        return ZTT_OPERATOR_INVALID_ARGUMENT;

    float inverse_cols[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float epsilon_tile[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    float tile[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
    for (unsigned i = 0; i < ZTT_LLM_TILE_ELEMENTS; ++i) {
        inverse_cols[i] = 1.0f / (float)cols;
        epsilon_tile[i] = epsilon;
    }
#if !defined(__riscv)
    (void)inverse_cols;
    (void)epsilon_tile;
#endif

#if defined(__riscv)
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    ztt_llm_acquire();
    __asm__ volatile(
        "msettyp m0, %[fp32]\n\t"
        "msettyp m1, %[fp32]\n\t"
        "msettyp m2, %[fp32]\n\t"
        "msettyp m3, %[fp32]\n\t"
        "msettyp m4, %[fp32]\n\t"
        "msettyp m5, %[fp32]\n\t"
        "msettyp m6, %[fp32]\n\t"
        "mls.rm m0, %[input]\n\t"
        "mls.rm m3, %[inverse_cols]\n\t"
        "mls.rm m4, %[epsilon]\n\t"
        "mls.rm m5, %[gamma]\n\t"
        "mmul.ew m1, m0, m0\n\t"
        "mreduceadd.row m2, m1\n\t"
        "mmul.ew m2, m2, m3\n\t"
        "madd.ew m2, m2, m4\n\t"
        "msqrt.ew m2, m2\n\t"
        "mrec.ew m2, m2\n\t"
        "mmul.ew m6, m0, m2\n\t"
        "mmul.ew m6, m6, m5\n\t"
        "mss.rm m6, %[output]"
        :: [fp32] "r"(fp32), [input] "r"(input),
           [inverse_cols] "r"(inverse_cols), [epsilon] "r"(epsilon_tile),
           [gamma] "r"(gamma), [output] "r"(tile)
        : "memory");
    const uint64_t status = ztt_llm_release();
#else
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
        float sum = 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            const float value = input[row * ZTT_LLM_TILE + col];
            sum += value * value;
        }
        const float inverse = 1.0f / sqrtf(sum / (float)cols + epsilon);
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            tile[row * ZTT_LLM_TILE + col] =
                input[row * ZTT_LLM_TILE + col] * inverse *
                gamma[row * ZTT_LLM_TILE + col];
    }
    const uint64_t status = 0;
#endif

    for (unsigned row = 0; row < rows; ++row)
        for (unsigned col = 0; col < cols; ++col)
            output[row * cols + col] = tile[row * ZTT_LLM_TILE + col];
    return status;
}

uint64_t
ztt_operator_run_suite(struct ztt_operator_suite_result *result)
{
    uint64_t status = 0;
    result->case_count = ZTT_OPERATOR_CASE_COUNT;
    for (unsigned i = 0; i < ZTT_OPERATOR_CASE_COUNT; ++i) {
        const struct ztt_generated_rmsnorm_case *test = &ztt_operator_cases[i];
        const uint64_t case_status = ztt_rmsnorm_fp32(
            test->input, test->gamma, test->rows, test->cols,
            test->epsilon, ztt_operator_outputs[i]);
        result->cases[i] = (struct ztt_operator_case_result){
            test->name, ztt_operator_outputs[i], test->rows * test->cols,
            sizeof(float), case_status};
        status |= case_status;
    }
    return status;
}
