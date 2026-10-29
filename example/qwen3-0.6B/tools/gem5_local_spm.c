#include "gem5_local_spm.h"
#include "gem5_m5ops.h"

void qwen_spm_stage_pair(const float *a, const float *b, unsigned elements)
{
    if (elements > QWEN_SPM_RVV_BYTES / sizeof(float)) m5_fail(0, 21);
    if (!elements) return;
    qwen_dma_copy((uintptr_t)a, QWEN_SPM_RVV0, elements * sizeof(float), 0);
    qwen_dma_copy((uintptr_t)b, QWEN_SPM_RVV1, elements * sizeof(float), 0);
}

void rvv_memcpy_f32_r4(float *dst, const float *src, int elements)
{
    while (elements > 0) {
        const unsigned count = elements < QWEN_SPM_RVV_BYTES / sizeof(float) ?
                               elements : QWEN_SPM_RVV_BYTES / sizeof(float);
        qwen_dma_copy((uintptr_t)src, QWEN_SPM_RVV0, count * sizeof(float), 0);
        uintptr_t from = QWEN_SPM_RVV0, to = QWEN_SPM_RVV1;
        unsigned remaining = count;
        while (remaining) {
            unsigned long vl;
            __asm__ volatile(
                "vsetvli %0,%3,e32,m1,ta,ma\n\t"
                "vle32.v v0,(%1)\n\t"
                "vse32.v v0,(%2)"
                : "=&r"(vl) : "r"(from), "r"(to), "r"(remaining)
                : "v0", "memory");
            from += vl * sizeof(float);
            to += vl * sizeof(float);
            remaining -= vl;
        }
        __asm__ volatile("fence rw,rw" ::: "memory");
        qwen_dma_copy((uintptr_t)dst, QWEN_SPM_RVV1, count * sizeof(float), 1);
        src += count; dst += count; elements -= count;
    }
}

uintptr_t qwen_spm_pack_activation_k128(const int8_t *source,
                                        unsigned elements)
{
    const unsigned groups = (elements + 127u) / 128u;
    const unsigned raw_bytes = groups * 128u;
    if (raw_bytes > 32768u || groups * 512u > 131072u) m5_fail(0, 21);
    qwen_dma_copy((uintptr_t)source, QWEN_SPM_ACTIVATION_RAW, raw_bytes, 0);

    unsigned long zero_vl;
    const unsigned zero_request = 512;
    __asm__ volatile(
        "vsetvli %0,%1,e8,m1,ta,ma\n\t"
        "vmv.v.i v0,0"
        : "=&r"(zero_vl) : "r"(zero_request) : "v0", "memory");
    uintptr_t zero_at = QWEN_SPM_ACTIVATION_PACKED;
    unsigned zero_remaining = groups * 512u;
    while (zero_remaining) {
        __asm__ volatile("vse8.v v0,(%0)" :: "r"(zero_at) : "memory");
        zero_at += zero_vl;
        zero_remaining -= zero_vl;
    }

    unsigned long word_vl;
    const unsigned word_request = 32;
    __asm__ volatile("vsetvli %0,%1,e32,m1,ta,ma"
                     : "=&r"(word_vl) : "r"(word_request) : "memory");
    for (unsigned group = 0; group < groups; ++group) {
        uintptr_t raw = QWEN_SPM_ACTIVATION_RAW + group * 128u;
        uintptr_t packed = QWEN_SPM_ACTIVATION_PACKED + group * 512u;
        unsigned remaining = 32; /* 128 int8 values viewed as 32 words. */
        while (remaining) {
            const long stride = 16;
            __asm__ volatile(
                "vle32.v v0,(%0)\n\t"
                "vsse32.v v0,(%1),%2"
                :: "r"(raw), "r"(packed), "r"(stride)
                : "v0", "memory");
            raw += word_vl * 4u;
            packed += word_vl * 16u;
            remaining -= word_vl;
        }
    }
    __asm__ volatile("fence rw,rw" ::: "memory");
    return QWEN_SPM_ACTIVATION_PACKED;
}
