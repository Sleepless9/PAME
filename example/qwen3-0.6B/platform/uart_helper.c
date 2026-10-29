// A small print abstraction used by bare-metal tests.
//
// NOTE: This file is sometimes included as a .c ("#include \"uart_helper.c\"")
// from different headers / translation units. Guard it so accidental multiple
// inclusion within the same translation unit does not cause redefinition errors.
#ifndef UART_HELPER_C_INCLUDED
#define UART_HELPER_C_INCLUDED
//
// - Default (hardware): use memory-mapped UART via uart.c.
// - Spike simulation: define PLAT_PRINT_USE_PRINTF=1 to route output to printf
//   (implemented in bare_syscalls.c via tohost/fromhost).
//
// This file is intentionally included as a .c in some tests (e.g. matmul.c).
// To keep that workflow working, we conditionally include the UART implementation
// only when needed.

#include <stdint.h>

#if defined(PLAT_PRINT_USE_PRINTF) && (PLAT_PRINT_USE_PRINTF)
    #include <stdio.h>

    // Shims matching uart.c symbols, so existing code can keep calling them.
    static inline void init_uart(uint32_t freq, uint32_t baud) { (void)freq; (void)baud; }
    static inline void write_serial(char a) { (void)printf("%c", a); }
    static inline void print_uart(const char* str) { (void)printf("%s", str); }

#else
    // Hardware UART backend
    #include "uart.h"
#endif

// ---------------------------------------------------------------------------
// Helper for printing numbers (decimal)
static void  __attribute__((noinline)) print_uart_int_dec(uint64_t val) {
    char buf[32];
    int i = 0;
    if (val == 0) {
        print_uart("0");
        return;
    }
    while (val > 0) {
        buf[i++] = (val % 10) + '0';
        val /= 10;
    }
    for (int j = i - 1; j >= 0; j--) {
        write_serial(buf[j]);
    }
}


static inline float my_fabs(float x) {
    return x < 0.0f ? -x : x;
}
static void print_uart_hex(unsigned long long val) {
    char hex_chars[] = "0123456789ABCDEF";
    char buf[17];
    buf[16] = '\0';
    for(int i = 15; i >= 0; --i) {
        buf[i] = hex_chars[val & 0xF];
        val >>= 4;
    }
    print_uart(buf);
}

static void  __attribute__((noinline))  uart_putc(char c) { write_serial((uint8_t)c); }

static void  __attribute__((noinline)) print_dec32(int32_t v) {
    char buf[16]; int i = 0;
    if (v == 0) { uart_putc('0'); return; }
    if (v < 0) { uart_putc('-'); v = -v; }
    while (v && i < (int)sizeof(buf)) { buf[i++] = (char)('0' + (v % 10)); v /= 10; }
    while (i--) uart_putc(buf[i]);
}


static void  __attribute__((noinline)) print_float_fixed3(float x) {
    // 特殊值：NaN/Inf/-Inf 与 -0.0
    union { float f; uint32_t u; } xu = { .f = x };
    uint32_t u = xu.u, exp = u & 0x7F800000u, frac = u & 0x007FFFFFu, sign = u >> 31;
    if (exp == 0x7F800000u) {               // Inf/NaN
        if (frac == 0) { print_uart(sign ? "-inf" : "inf"); return; }
        print_uart("nan"); return;
    }
    if ((u & 0x7FFFFFFFu) == 0) {           // ±0 -> 0.000000
        print_uart("0.000000"); return;
    }

    int neg = (x < 0.0f);
    if (neg) x = -x;

    // 大数：改用科学计数法，保留 6 位小数
    if (x >= 1000000000.0f) {               // |x| ≥ 1e9 -> S.DDDDDD e±E
        if (neg) uart_putc('-');
        int e = 0;
        // 规范化到 [1,10)
        while (x >= 10.0f) { x *= 0.1f; ++e; }
        while (x < 1.0f)  { x *= 10.0f; --e; }
        uint32_t ip = (uint32_t)x;          // 1..9
        float fracf = x - (float)ip;
        uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
        if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; if (ip >= 10u) { ip = 1u; ++e; } }
        print_dec32((int32_t)ip);
        uart_putc('.');
        // 打印 6 位小数（补零）
        uart_putc((char)('0' + (frac6 / 100000) % 10));
        uart_putc((char)('0' + (frac6 / 10000)  % 10));
        uart_putc((char)('0' + (frac6 / 1000)   % 10));
        uart_putc((char)('0' + (frac6 / 100)    % 10));
        uart_putc((char)('0' + (frac6 / 10)     % 10));
        uart_putc((char)('0' + (frac6 % 10)));
        uart_putc('e');
        if (e >= 0) { uart_putc('+'); print_dec32(e); }
        else        { uart_putc('-'); print_dec32(-e); }
        return;
    }

    // 常规路径：[-]I.FFFFFF（6 位小数，四舍五入）
    uint32_t ip = (uint32_t)x;              // 此处 x < 1e9，安全
    float fracf = x - (float)ip;
    uint32_t frac6 = (uint32_t)(fracf * 1000000.0f + 0.5f);
    if (frac6 >= 1000000u) { ip += 1u; frac6 -= 1000000u; }
    if (neg) uart_putc('-');
    print_dec32((int32_t)ip);
    uart_putc('.');
    // 打印 6 位小数（补零）
    uart_putc((char)('0' + (frac6 / 100000) % 10));
    uart_putc((char)('0' + (frac6 / 10000)  % 10));
    uart_putc((char)('0' + (frac6 / 1000)   % 10));
    uart_putc((char)('0' + (frac6 / 100)    % 10));
    uart_putc((char)('0' + (frac6 / 10)     % 10));
    uart_putc((char)('0' + (frac6 % 10)));
}



static void  __attribute__((noinline, unused))  print_f32_scalar(const char* name, float v) {
    print_uart(name); 
    print_uart("="); 
    print_float_fixed3(v); 
    print_uart("\r\n");
}




static void __attribute__((noinline)) print_idx_prefix(const char* name, int i) {
    print_uart(name); 
    uart_putc('['); 
    print_dec32(i); 
    uart_putc(']'); 
    uart_putc('='); 
}

// UART打印数组与比较 
static void __attribute__((noinline, unused)) print_f32_array(const char* name, const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        print_idx_prefix(name, i);
        print_float_fixed3(a[i]);
        print_uart("\r\n");
    }
}

#endif // UART_HELPER_C_INCLUDED

