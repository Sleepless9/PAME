/* Small integration guest for the hybrid MinorCPU/engine/DMA/SPM topology. */

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "gem5_m5ops.h"
#include "spm_virt_dma.h"

#define SPM_BASE_ADDR 0x70000000ULL
#define SPM_STRIDE_OFFSET 0x2000
#define SPM_RVV_OFFSET 0x2100
#define SPM_A_OFFSET 0x0000
#define SPM_B_OFFSET 0x0040
#define SPM_C_OFFSET 0x0080

extern void init_uart(uint32_t freq, uint32_t baud);
extern void print_uart(const char *text);
extern void ztt_i8_i8_i32_4x4(int32_t *c, const int8_t *a, const int8_t *b);

static const uint8_t stride_input[4][8] __attribute__((aligned(64))) = {
    {1, 2, 3, 4, 0, 0, 0, 0},
    {5, 6, 7, 8, 0, 0, 0, 0},
    {9, 10, 11, 12, 0, 0, 0, 0},
    {13, 14, 15, 16, 0, 0, 0, 0},
};
static uint8_t stride_output[4][8] __attribute__((aligned(64)));
static const uint8_t rvv_input[16] __attribute__((aligned(64))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static uint8_t rvv_output[16] __attribute__((aligned(64)));
static const int8_t matrix_a[16] __attribute__((aligned(64))) = {
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
};
static const int8_t matrix_b[16] __attribute__((aligned(64))) = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};
static int32_t matrix_output[16] __attribute__((aligned(64)));

static int
check_stride(void)
{
    const int rc = spm_dma_copy_2d_to_spm(
        (void *)(uintptr_t)(SPM_BASE_ADDR + SPM_STRIDE_OFFSET), stride_input,
        4, 4, 8, 16);
    if (rc != 0)
        return 1;
    return spm_dma_copy_2d_from_spm(
        stride_output,
        (const void *)(uintptr_t)(SPM_BASE_ADDR + SPM_STRIDE_OFFSET), 4, 4,
        8, 16);
}

static int
check_rvv(void)
{
    if (spm_dma_copy_to_spm(
            (void *)(uintptr_t)(SPM_BASE_ADDR + SPM_RVV_OFFSET), rvv_input,
            sizeof(rvv_input)) != 0)
        return 1;
    uint8_t *spm_data = (uint8_t *)(uintptr_t)(SPM_BASE_ADDR + SPM_RVV_OFFSET);
    const size_t vl = __riscv_vsetvl_e8m1(sizeof(rvv_input));
    vuint8m1_t value = __riscv_vle8_v_u8m1(spm_data, vl);
    value = __riscv_vadd_vx_u8m1(value, 1, vl);
    __riscv_vse8_v_u8m1(spm_data, value, vl);
    __asm__ volatile("fence rw, rw" ::: "memory");
    return spm_dma_copy_from_spm(rvv_output, spm_data, sizeof(rvv_output));
}

static int
check_ame(void)
{
    uint8_t *spm = (uint8_t *)(uintptr_t)SPM_BASE_ADDR;
    if (spm_dma_copy_to_spm(spm + SPM_A_OFFSET, matrix_a,
                            sizeof(matrix_a)) != 0 ||
        spm_dma_copy_to_spm(spm + SPM_B_OFFSET, matrix_b,
                            sizeof(matrix_b)) != 0)
        return 1;
    ztt_i8_i8_i32_4x4((int32_t *)(spm + SPM_C_OFFSET),
                      (const int8_t *)(spm + SPM_A_OFFSET),
                      (const int8_t *)(spm + SPM_B_OFFSET));
    __asm__ volatile("fence rw, rw" ::: "memory");
    return spm_dma_copy_from_spm(matrix_output, spm + SPM_C_OFFSET,
                                 sizeof(matrix_output));
}

int
main(void)
{
    init_uart(50000000, 115200);
    uintptr_t mstatus;
    __asm__ volatile("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9);  /* RVV */
    mstatus |= (uintptr_t)(3u << 13); /* FPU */
    mstatus |= (uintptr_t)(3u << 15); /* extensions */
    mstatus |= (uintptr_t)(3u << 23); /* AME/matrix */
    __asm__ volatile("csrw mstatus, %0" :: "r"(mstatus) : "memory");

    print_uart("hybrid smoke begin\r\n");
    if (check_stride() != 0)
        m5_fail(0, 1);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            if (stride_output[row][col] != stride_input[row][col])
                m5_fail(0, 2);
    if (check_rvv() != 0)
        m5_fail(0, 3);
    for (int i = 0; i < 16; ++i)
        if (rvv_output[i] != (uint8_t)(rvv_input[i] + 1))
            m5_fail(0, 4);
    if (check_ame() != 0)
        m5_fail(0, 5);
    for (int i = 0; i < 16; ++i)
        if (matrix_output[i] != matrix_b[i])
            m5_fail(0, 6);
    print_uart("hybrid smoke pass\r\n");
    m5_exit(0);
    return 0;
}
