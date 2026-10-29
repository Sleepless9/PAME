#ifndef QWEN_HYBRID_SPM_VIRT_DMA_H
#define QWEN_HYBRID_SPM_VIRT_DMA_H

#include <stddef.h>
#include <stdint.h>

#ifndef SPM_BASE_ADDR
#define SPM_BASE_ADDR 0x70000000ULL
#endif

#ifndef SPM_DMA_BASE_ADDR
#define SPM_DMA_BASE_ADDR 0x40010000ULL
#endif

typedef struct {
    uint64_t descriptors;
    uint64_t bytes_to_spm;
    uint64_t bytes_from_spm;
    uint64_t errors;
} SpmDmaGuestStats;

int spm_dma_copy_2d_to_spm(void *spm_dst, const void *system_src,
                           size_t row_bytes, size_t rows,
                           size_t system_stride, size_t spm_stride);
int spm_dma_copy_2d_from_spm(void *system_dst, const void *spm_src,
                             size_t row_bytes, size_t rows,
                             size_t system_stride, size_t spm_stride);
int spm_dma_copy_to_spm(void *spm_dst, const void *system_src, size_t bytes);
int spm_dma_copy_from_spm(void *system_dst, const void *spm_src, size_t bytes);
void spm_dma_guest_stats_reset(void);
SpmDmaGuestStats spm_dma_guest_stats(void);

#endif
