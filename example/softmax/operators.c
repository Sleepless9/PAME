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
ztt_softmax_fp32(const float *input, unsigned rows, unsigned cols,
                 float *output)
{
    if (!input || !output || rows == 0 || rows > ZTT_LLM_TILE ||
        cols == 0 || cols > ZTT_LLM_TILE)
        return ZTT_OPERATOR_INVALID_ARGUMENT;

    float tile[ZTT_LLM_TILE_ELEMENTS] ZTT_LLM_ALIGNED;
#if defined(__riscv)
    const uint64_t fp32 = ZTT_DTYPE_FP32_RNE;
    ztt_llm_acquire();
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
        :: [fp32] "r"(fp32), [input] "r"(input), [output] "r"(tile)
        : "memory");
    const uint64_t status = ztt_llm_release();
#else
    for (unsigned row = 0; row < ZTT_LLM_TILE; ++row) {
        float maximum = input[row * ZTT_LLM_TILE];
        for (unsigned col = 1; col < ZTT_LLM_TILE; ++col)
            if (input[row * ZTT_LLM_TILE + col] > maximum)
                maximum = input[row * ZTT_LLM_TILE + col];
        float sum = 0.0f;
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col) {
            tile[row * ZTT_LLM_TILE + col] =
                exp2f(input[row * ZTT_LLM_TILE + col] - maximum);
            sum += tile[row * ZTT_LLM_TILE + col];
        }
        for (unsigned col = 0; col < ZTT_LLM_TILE; ++col)
            tile[row * ZTT_LLM_TILE + col] /= sum;
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
        const struct ztt_generated_softmax_case *test = &ztt_operator_cases[i];
        const uint64_t case_status = ztt_softmax_fp32(
            test->input, test->rows, test->cols, ztt_operator_outputs[i]);
        result->cases[i] = (struct ztt_operator_case_result){
            test->name, ztt_operator_outputs[i], test->rows * test->cols,
            sizeof(float), case_status};
        status |= case_status;
    }
    return status;
}
