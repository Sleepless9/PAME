#ifndef QWEN3_0_6B_PLATFORM_H
#define QWEN3_0_6B_PLATFORM_H

#include <stdint.h>

void init_uart(uint32_t frequency, uint32_t baud);
void write_serial(char value);
void print_uart(const char *text);
void print_uart_int_dec(uint64_t value);
void print_uart_hex(unsigned long long value);
void print_float_fixed3(float value);

#endif
