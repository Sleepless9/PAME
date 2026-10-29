// UART-friendly tiny libc/syscalls for FPGA/real hardware.
//
// Provides the minimal set of libc-like symbols used by the bare-metal llama
// harness, without any HTIF (tohost/fromhost) dependency.

#include <stdint.h>
#include <stddef.h>
#include "gem5_m5ops.h"

// These are provided by uart.c (typically via including uart_helper.c in the app).
void write_serial(char a);

// ------------------------------------------------------------
// Process termination
void __attribute__((noreturn)) exit(int code)
{
    if (code == 0)
        m5_exit(0);
    else
        m5_fail(0, (uint64_t)(unsigned int)code);

    /* Keep real-hardware behavior defined if gem5 does not consume the op. */
    for (;;) {
        __asm__ volatile ("wfi");
    }
}

void __attribute__((noreturn)) abort(void)
{
    exit(1);
}

// ------------------------------------------------------------
// Memory / string
void* memcpy(void* dest, const void* src, size_t len)
{
    const unsigned char* s = (const unsigned char*)src;
    unsigned char* d = (unsigned char*)dest;
    for (size_t i = 0; i < len; i++) d[i] = s[i];
    return dest;
}

void* memset(void* dest, int byte, size_t len)
{
    unsigned char* d = (unsigned char*)dest;
#if defined(QWEN_RVV_LIBC) && QWEN_RVV_LIBC && defined(__riscv_vector)
    while (len != 0) {
        size_t vl;
        __asm__ volatile (
            "vsetvli %0, %1, e8, m8, ta, ma"
            : "=r"(vl) : "r"(len));
        __asm__ volatile (
            "vmv.v.x v0, %0\n\t"
            "vse8.v v0, (%1)"
            : : "r"((uintptr_t)(unsigned char)byte), "r"(d) : "memory");
        d += vl;
        len -= vl;
    }
#else
    for (size_t i = 0; i < len; i++) d[i] = (unsigned char)byte;
#endif
    return dest;
}

static inline size_t my_strlen(const char *s)
{
    const char *p = s;
    while (*p) p++;
    return (size_t)(p - s);
}

size_t strlen(const char *s)
{
    return my_strlen(s);
}

size_t strnlen(const char *s, size_t n)
{
    const char *p = s;
    while (n-- && *p) p++;
    return (size_t)(p - s);
}

int strcmp(const char* s1, const char* s2)
{
    unsigned char c1, c2;
    do { c1 = (unsigned char)*s1++; c2 = (unsigned char)*s2++; } while (c1 != 0 && c1 == c2);
    return (int)c1 - (int)c2;
}

char* strcpy(char* dest, const char* src)
{
    char* d = dest;
    while ((*d++ = *src++)) ;
    return dest;
}

long atol(const char* str)
{
    long res = 0;
    int sign = 0;
    while (*str == ' ') str++;
    if (*str == '-' || *str == '+') { sign = (*str == '-'); str++; }
    while (*str) { res *= 10; res += *str++ - '0'; }
    return sign ? -res : res;
}

// ------------------------------------------------------------
// Minimal stdio-like helpers (optional)
int putchar(int ch)
{
    write_serial((char)ch);
    return ch;
}

int puts(const char *s)
{
    while (*s) putchar(*s++);
    putchar('\n');
    return 0;
}
