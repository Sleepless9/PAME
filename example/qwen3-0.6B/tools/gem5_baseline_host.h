/* Native numerical reference only: same K=4 scaling and RVV reduction order. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char *host_model;
static char host_heap[256 * 1024 * 1024];
static const char *host_output_prefix;
void init_uart(uint32_t hz, uint32_t baud) { (void)hz; (void)baud; }
void print_uart(const char *s) { fputs(s, stdout); fflush(stdout); }
static void print_uart_int_dec(uint64_t n) { printf("%llu", (unsigned long long)n); }
static void print_uart_hex(unsigned long long n) { printf("%016llX", n); }
void m5_exit(uint64_t delay) { (void)delay; exit(0); }
void m5_fail(uint64_t delay, uint64_t code) { (void)delay; exit(code ? code : 1); }
void m5_work_begin(uint64_t id, uint64_t tid) { (void)id; (void)tid; }
void m5_work_end(uint64_t id, uint64_t tid) { (void)id; (void)tid; }
void m5_checkpoint(uint64_t delay, uint64_t period) { (void)delay; (void)period; }
static void qwen_dma_blocks_session_begin(void) {}
static void qwen_dma_blocks_session_end(void) {}
void rvv_memcpy_f32_r4(float *dst, const float *src, int n) { memcpy(dst, src, n * sizeof(float)); }
static void save_reference(const char *phase, const float *values, int n) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s.%s.f32", host_output_prefix, phase) >= sizeof(path)) exit(6);
    FILE *file = fopen(path, "wb");
    if (!file || fwrite(values, sizeof(float), n, file) != n) exit(7);
    fclose(file);
}
