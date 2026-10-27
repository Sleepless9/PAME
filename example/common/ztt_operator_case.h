/*
 * Copyright (c) 2026 BOSC & ICT, CAS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ZTT_OPERATOR_CASE_H
#define ZTT_OPERATOR_CASE_H

#include <stdint.h>
#include <stdio.h>

#include "ztt_llm_case.h"

#define ZTT_OPERATOR_INVALID_ARGUMENT (UINT64_C(1) << 63)

struct ztt_operator_case_result
{
    const char *name;
    const void *output;
    unsigned element_count;
    unsigned element_bytes;
    uint64_t status;
};

struct ztt_operator_suite_result
{
    unsigned case_count;
    struct ztt_operator_case_result cases[ZTT_GEMM_MAX_CASES];
};

/* Implemented by each operator-specific operators.c. */
uint64_t ztt_operator_run_suite(struct ztt_operator_suite_result *result);

static inline int
ztt_operator_write_report(const char *path,
                          const struct ztt_operator_suite_result *result,
                          uint64_t suite)
{
    if (result->case_count > ZTT_GEMM_MAX_CASES)
        return 0;

    struct ztt_gemm_report report;
    ztt_gemm_report_init(&report, suite, result->case_count);
    const char *names[ZTT_GEMM_MAX_CASES] = {0};

    for (unsigned test = 0; test < result->case_count; ++test) {
        const struct ztt_operator_case_result *case_result =
            &result->cases[test];
        if (case_result->element_count + 2 > ZTT_GEMM_MAX_VALUES)
            return 0;
        names[test] = case_result->name;
        report.values[test][0] = case_result->status;
        report.values[test][1] = case_result->element_count;
        const uint8_t *bytes = case_result->output;
        for (unsigned i = 0; i < case_result->element_count; ++i) {
            uint64_t bits = 0;
            memcpy(&bits, bytes + (uint64_t)i * case_result->element_bytes,
                   case_result->element_bytes);
            report.values[test][2 + i] = bits;
        }
        report.case_values[test] = case_result->element_count + 2;
    }
    return ztt_gemm_report_write(path, &report, names);
}

static inline int
ztt_operator_case_main(int argc, char **argv, uint64_t suite,
                       const char *suite_name)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s REPORT_PATH\n", argv[0]);
        return 2;
    }

    struct ztt_operator_suite_result result;
    const uint64_t status = ztt_operator_run_suite(&result);
    if (!ztt_operator_write_report(argv[1], &result, suite)) {
        fprintf(stderr, "cannot write report: %s\n", argv[1]);
        return 1;
    }
    if (status != 0) {
        fprintf(stderr, "%s amestatus=0x%llx\n", suite_name,
                (unsigned long long)status);
        return 1;
    }
    return 0;
}

#endif /* ZTT_OPERATOR_CASE_H */
