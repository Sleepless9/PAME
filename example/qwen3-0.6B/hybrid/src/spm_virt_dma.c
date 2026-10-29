#include "spm_virt_dma.h"

enum {
    DMA_CONTROL = 0x00,
    DMA_STATUS = 0x08,
    DMA_SYSTEM_ADDR = 0x10,
    DMA_SPM_ADDR = 0x18,
    DMA_ROW_BYTES = 0x20,
    DMA_ROWS = 0x28,
    DMA_SYSTEM_STRIDE = 0x30,
    DMA_SPM_STRIDE = 0x38,
    DMA_DESCRIPTOR_ID = 0x50,
};

enum {
    DMA_START = 1u << 0,
    DMA_SPM_TO_SYSTEM = 1u << 1,
    DMA_CLEAR_STATUS = 1u << 2,
    DMA_BUSY = 1u << 0,
    DMA_DONE = 1u << 1,
    DMA_ERROR = 1u << 2,
};

static uint64_t next_descriptor_id = 1;
static SpmDmaGuestStats guest_stats;

static inline volatile uint64_t *
dma_reg(uintptr_t offset)
{
    return (volatile uint64_t *)(uintptr_t)(SPM_DMA_BASE_ADDR + offset);
}

static inline void
dma_fence(void)
{
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static int
spm_dma_copy_2d(void *system_ptr, void *spm_ptr, size_t row_bytes,
                size_t rows, size_t system_stride, size_t spm_stride,
                int spm_to_system)
{
    if (!system_ptr || !spm_ptr || row_bytes == 0 || rows == 0)
        return -1;

    *dma_reg(DMA_SYSTEM_ADDR) = (uint64_t)(uintptr_t)system_ptr;
    *dma_reg(DMA_SPM_ADDR) = (uint64_t)(uintptr_t)spm_ptr;
    *dma_reg(DMA_ROW_BYTES) = row_bytes;
    *dma_reg(DMA_ROWS) = rows;
    *dma_reg(DMA_SYSTEM_STRIDE) = system_stride;
    *dma_reg(DMA_SPM_STRIDE) = spm_stride;
    *dma_reg(DMA_DESCRIPTOR_ID) = next_descriptor_id++;
    dma_fence();
    *dma_reg(DMA_CONTROL) = DMA_START | DMA_CLEAR_STATUS |
        (spm_to_system ? DMA_SPM_TO_SYSTEM : 0);

    uint64_t status;
    do {
        dma_fence();
        status = *dma_reg(DMA_STATUS);
    } while (status & DMA_BUSY);

    ++guest_stats.descriptors;
    if ((status & DMA_ERROR) || !(status & DMA_DONE)) {
        ++guest_stats.errors;
        return -1;
    }
    if (spm_to_system)
        guest_stats.bytes_from_spm += row_bytes * rows;
    else
        guest_stats.bytes_to_spm += row_bytes * rows;
    dma_fence();
    return 0;
}

int
spm_dma_copy_2d_to_spm(void *spm_dst, const void *system_src,
                       size_t row_bytes, size_t rows,
                       size_t system_stride, size_t spm_stride)
{
    return spm_dma_copy_2d((void *)system_src, spm_dst, row_bytes, rows,
                           system_stride, spm_stride, 0);
}

int
spm_dma_copy_2d_from_spm(void *system_dst, const void *spm_src,
                         size_t row_bytes, size_t rows,
                         size_t system_stride, size_t spm_stride)
{
    return spm_dma_copy_2d(system_dst, (void *)spm_src, row_bytes, rows,
                           system_stride, spm_stride, 1);
}

int
spm_dma_copy_to_spm(void *spm_dst, const void *system_src, size_t bytes)
{
    return spm_dma_copy_2d_to_spm(spm_dst, system_src, bytes, 1, bytes, bytes);
}

int
spm_dma_copy_from_spm(void *system_dst, const void *spm_src, size_t bytes)
{
    return spm_dma_copy_2d_from_spm(system_dst, spm_src, bytes, 1,
                                    bytes, bytes);
}

void
spm_dma_guest_stats_reset(void)
{
    guest_stats = (SpmDmaGuestStats){0};
    next_descriptor_id = 1;
}

SpmDmaGuestStats
spm_dma_guest_stats(void)
{
    return guest_stats;
}
