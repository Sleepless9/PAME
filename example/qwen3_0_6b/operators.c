#include <stdint.h>
#include <stdio.h>

#include "platform/gem5_m5ops.h"
#include "platform/qwen_platform.h"

void
init_uart(uint32_t frequency, uint32_t baud)
{
    (void)frequency;
    (void)baud;
}

void
write_serial(char value)
{
    (void)fputc((unsigned char)value, stderr);
}

void
print_uart(const char *text)
{
    while (*text)
        write_serial(*text++);
}

void
print_uart_int_dec(uint64_t value)
{
    char buffer[32];
    unsigned used = 0;
    do {
        buffer[used++] = (char)('0' + value % 10);
        value /= 10;
    } while (value != 0);
    while (used != 0)
        write_serial(buffer[--used]);
}

void
print_uart_hex(unsigned long long value)
{
    static const char digits[] = "0123456789ABCDEF";
    for (int shift = 60; shift >= 0; shift -= 4)
        write_serial(digits[(value >> shift) & 15u]);
}

void
print_float_fixed3(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = {.f = value};
    const uint32_t exponent = bits.u & UINT32_C(0x7f800000);
    const uint32_t fraction = bits.u & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x7f800000)) {
        print_uart(fraction == 0 ? (bits.u >> 31 ? "-inf" : "inf") : "nan");
        return;
    }
    if (value < 0.0f) {
        write_serial('-');
        value = -value;
    }
    uint64_t scaled = (uint64_t)(value * 1000000.0f + 0.5f);
    print_uart_int_dec(scaled / 1000000u);
    write_serial('.');
    uint64_t remainder = scaled % 1000000u;
    for (uint64_t divisor = 100000u; divisor != 0; divisor /= 10u)
        write_serial((char)('0' + (remainder / divisor) % 10u));
}

#if !QWEN_BARE_METAL
void m5_exit(uint64_t delay) { (void)delay; }
void m5_fail(uint64_t delay, uint64_t code) { (void)delay; (void)code; }
void m5_checkpoint(uint64_t delay, uint64_t period)
{
    (void)delay;
    (void)period;
}
void m5_work_begin(uint64_t work_id, uint64_t thread_id)
{
    (void)work_id;
    (void)thread_id;
}
void m5_work_end(uint64_t work_id, uint64_t thread_id)
{
    (void)work_id;
    (void)thread_id;
}
#else
int
strcmp(const char *lhs, const char *rhs)
{
    while (*lhs && *lhs == *rhs) {
        ++lhs;
        ++rhs;
    }
    return (unsigned char)*lhs - (unsigned char)*rhs;
}

void __attribute__((noreturn))
exit(int status)
{
    if (status == 0)
        m5_exit(0);
    else
        m5_fail(0, (uint64_t)(unsigned)status);
    for (;;)
        __asm__ volatile("wfi");
}
#endif
