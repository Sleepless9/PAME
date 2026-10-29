#ifndef QWEN_GEM5_LOCAL_SPM_H
#define QWEN_GEM5_LOCAL_SPM_H
#include <stdint.h>
#include <stddef.h>

// Blocking baseline workspace. AME owns [0,192); RVV scratch starts at 4KiB.
// All helpers run synchronously on the single host hart; no overlapping owners.
#define QWEN_SPM_RVV0 ((uintptr_t)0xc0001000)
#define QWEN_SPM_RVV1 ((uintptr_t)0xc0002000)
#define QWEN_SPM_RVV2 ((uintptr_t)0xc0003000)
#define QWEN_SPM_RVV_BYTES 4096u
#define QWEN_SPM_ACTIVATION_RAW ((uintptr_t)0xc0004000)
#define QWEN_SPM_ACTIVATION_PACKED ((uintptr_t)0xc0020000)
void qwen_dma_copy(uintptr_t system, uintptr_t spm, unsigned bytes, unsigned to_system);
/* Descriptor-ring runtime. submit is nonblocking; fence waits for the
 * requested descriptor and all older descriptors. */
uint64_t qwen_dma_submit(uintptr_t system, uintptr_t spm, unsigned bytes,
                         unsigned to_system);
void qwen_dma_fence(uint64_t descriptor_id);
unsigned qwen_dma_buffer_index(unsigned slot);
void qwen_dma_tile(int32_t *out, const int8_t *a, const int8_t *b);
void qwen_dma_group(int32_t *out, const int8_t *a, const int8_t *b, unsigned steps);
void qwen_dma_blocks_begin(const int8_t *data, uintptr_t spm, unsigned bytes);
void qwen_dma_blocks_session_begin(void);
void qwen_dma_blocks_session_end(void);
uint64_t qwen_dma_blocks_compute(int32_t *out, uintptr_t activation_spm,
    uintptr_t weight_spm, unsigned groups, const int8_t *next,
    uintptr_t next_spm, unsigned next_bytes, unsigned output_slot);
void qwen_dma_blocks_wait(uint64_t ticket, unsigned output_slot);
void qwen_spm_stage_pair(const float *a, const float *b, unsigned elements);
void rvv_memcpy_f32_r4(float *dst, const float *src, int elements);
uintptr_t qwen_spm_pack_activation_k128(const int8_t *source,
                                        unsigned elements);
#endif
