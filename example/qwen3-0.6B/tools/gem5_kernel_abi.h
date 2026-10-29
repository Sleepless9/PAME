/* ABI of helpers retained at their original addresses in the supplied ELF. */
#include <stdint.h>
#include <stddef.h>
void init_uart(uint32_t clock, uint32_t baud);
void print_uart(const char *text);
void print_uart_int_dec(uint64_t value);
void print_uart_hex(unsigned long long value);
void rvv_memcpy_f32_r4(float *dst, const float *src, int elements);
void ame_matmul_stats_reset(void);
void ame_matmul_quant_noblk_i8_i8_f32(float *, const int8_t *, const float *,
    const int8_t *, const float *, int, int, int, int);
void ame_matmul_quant_batch_i8_i8_f32(float *, const int8_t *, const float *,
    const int8_t *, const float *, int, int, int, int, int);
