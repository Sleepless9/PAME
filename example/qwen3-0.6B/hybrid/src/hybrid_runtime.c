#include "qwen_hybrid_runtime.h"

#include <stdint.h>

#include "spm_virt_dma.h"

extern void print_uart(const char *text);
extern void write_serial(char value);

static void
print_u64(uint64_t value)
{
    char digits[24];
    int count = 0;
    if (value == 0) {
        write_serial('0');
        return;
    }
    while (value != 0) {
        digits[count++] = (char)('0' + value % 10);
        value /= 10;
    }
    while (count != 0)
        write_serial(digits[--count]);
}

void
qwen_hybrid_stats_reset(void)
{
    spm_dma_guest_stats_reset();
}

void
qwen_hybrid_stats_report(void)
{
    const SpmDmaGuestStats stats = spm_dma_guest_stats();
    print_uart("\r\n=== Shared SPM DMA guest counters ===\r\n");
    print_uart("  descriptors   : ");
    print_u64(stats.descriptors);
    print_uart("\r\n  DDR -> SPM B  : ");
    print_u64(stats.bytes_to_spm);
    print_uart("\r\n  SPM -> DDR B  : ");
    print_u64(stats.bytes_from_spm);
    print_uart("\r\n  errors        : ");
    print_u64(stats.errors);
    print_uart("\r\n");
}
