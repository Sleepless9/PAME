// Bare-metal Qwen3-0.6B W4A8 end-to-end inference for RISC-V.
// All data embedded in ELF: model weights (.incbin), tokenizer (.incbin), prompt (compile-time).
// Full pipeline: tokenize -> forward -> decode -> UART output. No OS dependencies.
//
// W4A8 per-group quantization: int4 packed weights, int8 activations at matmul, fp32 scales.
// Between layers activations remain fp32. AME uses the migrated Ztt path over
// the prepacked v6 int8 tiles; the wrapper keeps the shared ame_* API.
// Backend selection via compile flags: USE_RVV, USE_AME, or scalar W4.

#ifndef QKV_MATMUL_LOG_ENABLE
#define QKV_MATMUL_LOG_ENABLE 0
#endif

#ifndef QKV_MATMUL_LOG_ONCE
#define QKV_MATMUL_LOG_ONCE 1
#endif

#ifndef QKV_MATMUL_LOG_MAX_ELEMS
#define QKV_MATMUL_LOG_MAX_ELEMS 32
#endif

#include <stdint.h>
#include <stddef.h>
#include "../platform/uart_helper.c"
#include "gem5_m5ops.h"
#include "backends/ops_lib.h"
#include "w4_common.h"
#ifdef QWEN_LOCAL_SPM
#include "gem5_local_spm.h"
#endif
#include "w4a8/matmul_quant_scalar_w4.h"
#if defined(USE_AME) && USE_AME
#include "ame_mix_offline_profile.h"
void ame_matmul_quant_noblk_pair_i8_i8_f32(float *xout0,
                                           float *xout1,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq0,
                                           const float *ws0,
                                           const int8_t *wq1,
                                           const float *ws1,
                                           int n,
                                           int d,
                                           int gs,
                                           int w_row_stride);
void ame_matmul_quant_batch_pair_i8_i8_f32(float *xout0,
                                           float *xout1,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq0,
                                           const float *ws0,
                                           const int8_t *wq1,
                                           const float *ws1,
                                           int T,
                                           int n,
                                           int d,
                                           int gs,
                                           int w_row_stride);
#elif defined(USE_RVV) && USE_RVV
#include "backends/rvv/core/matmul_quant_rvv_tile.h"
#endif

#ifndef BUILD_TIMESTAMP
#define BUILD_TIMESTAMP "unknown"
#endif

#ifndef FPGA_CLOCK_HZ
#define FPGA_CLOCK_HZ 50000000ULL
#endif

// --- Cycle counter for timing logs (64-bit safe on RV32 via mcycleh) ---
static inline uint64_t rdcycle(void) {
#if __riscv_xlen == 32
    uint32_t lo, hi, hi2;
    do {
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi));
        __asm__ volatile ("csrr %0, mcycle"  : "=r"(lo));
        __asm__ volatile ("csrr %0, mcycleh" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    uint64_t c;
    __asm__ volatile ("rdcycle %0" : "=r"(c));
    return c;
#endif
}

__attribute__((unused))
static void print_cycles(const char *label, uint64_t cycles) {
    print_uart(label);
    print_uart_int_dec(cycles);
    print_uart(" cyc\r\n");
}

#if defined(USE_AME) && USE_AME
static uint64_t sat_sub_u64(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : 0;
}

static uint64_t ame_profile_matmul_other_cycles(const AmeMixOfflineProfile *p) {
    uint64_t accounted = p->pack_cycles + p->zero_cycles +
                         p->ame_kernel_cycles + p->fp32_accum_cycles;
    return sat_sub_u64(p->total_cycles, accounted);
}

__attribute__((unused))
static AmeMixOfflineProfile ame_profile_delta(const AmeMixOfflineProfile *now,
                                              const AmeMixOfflineProfile *base) {
    AmeMixOfflineProfile d = {0};
    d.calls             = sat_sub_u64(now->calls,             base->calls);
    d.single_calls      = sat_sub_u64(now->single_calls,      base->single_calls);
    d.batch_calls       = sat_sub_u64(now->batch_calls,       base->batch_calls);
    d.tiles             = sat_sub_u64(now->tiles,             base->tiles);
    d.single_tiles      = sat_sub_u64(now->single_tiles,      base->single_tiles);
    d.batch_tiles       = sat_sub_u64(now->batch_tiles,       base->batch_tiles);
    d.total_cycles      = sat_sub_u64(now->total_cycles,      base->total_cycles);
    d.single_cycles     = sat_sub_u64(now->single_cycles,     base->single_cycles);
    d.batch_cycles      = sat_sub_u64(now->batch_cycles,      base->batch_cycles);
    d.zero_cycles       = sat_sub_u64(now->zero_cycles,       base->zero_cycles);
    d.ame_kernel_cycles = sat_sub_u64(now->ame_kernel_cycles, base->ame_kernel_cycles);
    d.fp32_accum_cycles = sat_sub_u64(now->fp32_accum_cycles, base->fp32_accum_cycles);
    d.pack_cycles       = sat_sub_u64(now->pack_cycles,       base->pack_cycles);
    return d;
}

static void print_ame_matmul_l2_line(const char *label,
                                     uint64_t cycles,
                                     uint64_t total_cycles) {
    print_uart("  ");
    print_uart(label);
    print_uart_int_dec(cycles);
    print_uart(" cyc (");
    uint64_t pct = (total_cycles > 0) ? (cycles * 1000ULL / total_cycles) : 0;
    print_uart_int_dec(pct / 10);
    print_uart(".");
    print_uart_int_dec(pct % 10);
    print_uart("%)\r\n");
}

__attribute__((unused))
static void print_ame_matmul_l2_breakdown(const char *title,
                                          const AmeMixOfflineProfile *p) {
    uint64_t other = ame_profile_matmul_other_cycles(p);
    print_uart("\r\n=== AME_MATMUL_I8_I8_F32_ESTIMATE_");
    print_uart(title);
    print_uart(" ===\r\n");
    print_uart("  calls="); print_uart_int_dec(p->calls);
    print_uart(" single="); print_uart_int_dec(p->single_calls);
    print_uart(" batch="); print_uart_int_dec(p->batch_calls);
    print_uart(" tiles="); print_uart_int_dec(p->tiles);
    print_uart(" single_tiles="); print_uart_int_dec(p->single_tiles);
    print_uart(" batch_tiles="); print_uart_int_dec(p->batch_tiles);
    print_uart("\r\n");
    print_ame_matmul_l2_line("pack              : ", p->pack_cycles,       p->total_cycles);
    print_ame_matmul_l2_line("zero              : ", p->zero_cycles,       p->total_cycles);
    print_ame_matmul_l2_line("matmul_i8_i8_i32  : ", p->ame_kernel_cycles, p->total_cycles);
    print_ame_matmul_l2_line("accum             : ", p->fp32_accum_cycles, p->total_cycles);
    print_ame_matmul_l2_line("other             : ", other,                p->total_cycles);
    print_ame_matmul_l2_line("total             : ", p->total_cycles,      p->total_cycles);
    print_uart("  single_total     : "); print_uart_int_dec(p->single_cycles); print_uart(" cyc\r\n");
    print_uart("  batch_total      : "); print_uart_int_dec(p->batch_cycles); print_uart(" cyc\r\n");
}
#endif

#if QKV_MATMUL_LOG_ENABLE
static int g_qkv_matmul_log_count;

static void print_i8_array_limited(const char *name, const int8_t *a, int n, int limit) {
    int shown = (n < limit) ? n : limit;
    for (int i = 0; i < shown; i++) {
        print_idx_prefix(name, i);
        print_dec32((int32_t)a[i]);
        print_uart("\r\n");
    }
    if (shown < n) {
        print_uart(name);
        print_uart("[...] truncated, total=");
        print_uart_int_dec((uint64_t)n);
        print_uart("\r\n");
    }
}

static void print_f32_array_limited(const char *name, const float *a, int n, int limit) {
    int shown = (n < limit) ? n : limit;
    for (int i = 0; i < shown; i++) {
        print_idx_prefix(name, i);
        print_float_fixed3(a[i]);
        print_uart("\r\n");
    }
    if (shown < n) {
        print_uart(name);
        print_uart("[...] truncated, total=");
        print_uart_int_dec((uint64_t)n);
        print_uart("\r\n");
    }
}

static int should_log_qkv_matmul(void) {
#if QKV_MATMUL_LOG_ONCE
    if (g_qkv_matmul_log_count != 0) return 0;
#endif
    g_qkv_matmul_log_count++;
    return 1;
}

static void log_qkv_matmul_input(int layer,
                                 int pos,
                                 int token,
                                 int m,
                                 int n,
                                 int group_size,
                                 const float *x,
                                 const int8_t *w,
                                 const float *w_scales) {
    int n_groups = n / group_size;
    print_uart("[qkv_matmul] input layer=");
    print_uart_int_dec((uint64_t)layer);
    print_uart(" pos=");
    print_uart_int_dec((uint64_t)pos);
    print_uart(" token=");
    print_uart_int_dec((uint64_t)token);
    print_uart(" m=");
    print_uart_int_dec((uint64_t)m);
    print_uart(" n=");
    print_uart_int_dec((uint64_t)n);
    print_uart(" gs=");
    print_uart_int_dec((uint64_t)group_size);
    print_uart("\r\n");

    print_f32_array_limited("qkv_x", x, n, QKV_MATMUL_LOG_MAX_ELEMS);
    print_i8_array_limited("qkv_w_row0", w, n, QKV_MATMUL_LOG_MAX_ELEMS);
    print_f32_array_limited("qkv_ws_row0", w_scales, n_groups, QKV_MATMUL_LOG_MAX_ELEMS);
}

static void log_qkv_matmul_output(const float *o, int m) {
    print_uart("[qkv_matmul] output\r\n");
    print_f32_array_limited("qkv_o", o, m, QKV_MATMUL_LOG_MAX_ELEMS);
}
#endif

// --- Per-phase cycle accumulators (updated inside forward()) ---
typedef struct {
    uint64_t embed;       // embedding lookup
    uint64_t attn_norm;   // rmsnorm before attention (all layers)
    uint64_t qkv_matmul;  // wqkv matmul (all layers)
    uint64_t qkv_quant;   // activation quantize_x before qkv
    uint64_t qkv_ame;     // qkv AME backend call after quantization
    uint64_t qknorm_rope; // qk_norm + rope (all layers)
    uint64_t attn_score;  // QK score + softmax + AV accumulate (all layers)
    uint64_t attn_qk_dot; // scalar Q dot K score calculation inside attn_score
    uint64_t attn_softmax;// scalar softmax inside attn_score
    uint64_t attn_av;     // scalar attention probability dot V inside attn_score
    uint64_t wo_matmul;   // wo matmul (all layers)
    uint64_t wo_quant;    // activation quantize_x before wo
    uint64_t wo_ame;      // wo AME backend call after quantization
    uint64_t ffn_norm;    // rmsnorm before FFN (all layers)
    uint64_t ffn_gate;    // w1 + w3 matmul (all layers)
    uint64_t ffn_gate_quant; // shared activation quantize_x before w1+w3
    uint64_t ffn_gate_ame;   // paired w1+w3 AME backend call after quantization
    uint64_t ffn_silu;    // silu + elementwise mul (all layers)
    uint64_t ffn_down;    // w2 matmul (all layers)
    uint64_t ffn_down_quant; // activation quantize_x before w2
    uint64_t ffn_down_ame;   // w2 AME backend call after quantization
    uint64_t final_norm;  // final rmsnorm
    uint64_t lm_head;     // lm_head matmul
    uint64_t lm_head_quant; // activation quantize_x before lm_head
    uint64_t lm_head_ame;   // lm_head AME backend call after quantization
    uint64_t n_calls;     // total forward() calls
    uint64_t n_layers;    // total layer passes
} FwdTiming;
static FwdTiming g_timing;
static FwdTiming g_layer_sel;   /* single-layer profile: accumulates only layer==PROFILE_LAYER */

__attribute__((unused))
static uint64_t attn_score_other_cycles(uint64_t total,
                                        uint64_t qk_dot,
                                        uint64_t softmax_cycles,
                                        uint64_t av) {
    uint64_t children = qk_dot + softmax_cycles + av;
    return total > children ? total - children : 0;
}

__attribute__((unused))
static uint64_t two_child_other_cycles(uint64_t total,
                                       uint64_t a,
                                       uint64_t b) {
    uint64_t children = a + b;
    return total > children ? total - children : 0;
}

#ifndef PROFILE_LAYER
#define PROFILE_LAYER 0   /* isolate this transformer layer for per-layer profiling */
#endif
#define ADD_LAYER_TIMING(layer, field, cycles) do { \
    uint64_t _alt_c = (cycles); \
    g_timing.field += _alt_c; \
    if ((int)(layer) == PROFILE_LAYER) g_layer_sel.field += _alt_c; \
} while (0)

// Forward declarations for bare_syscalls_uart.c functions
void *memcpy(void *dest, const void *src, size_t len);
void *memset(void *dest, int byte, size_t len);
int strcmp(const char *s1, const char *s2);
size_t strlen(const char *s);
void exit(int code);

static inline void prefill_memcpy_f32_r4(float *dst, const float *src, int n) {
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
    rvv_memcpy_f32_r4(dst, src, n);
#else
    for (int i = 0; i < n; i++) dst[i] = src[i];
#endif
}

// --- Embedded blobs ---
extern const char _model_blob_start[];
extern const char _tokenizer_blob_start[];
extern const char __heap_start[];
extern const char __heap_end[];
#define MODEL_ADDR ((const char *)_model_blob_start)

// --- Fixed heap region ---
static char *heap_ptr;

static inline uintptr_t heap_start_addr(void) {
    return (uintptr_t)__heap_start;
}

static inline uintptr_t heap_end_addr(void) {
    return (uintptr_t)__heap_end;
}

void *malloc(size_t n) {
    uintptr_t cur = ((uintptr_t)heap_ptr + 63) & ~(uintptr_t)63;
    char *p = (char *)cur;
    if ((uintptr_t)(p + n) > heap_end_addr()) { print_uart("OOM\r\n"); exit(1); }
    heap_ptr = p + n;
    return p;
}
void *calloc(size_t count, size_t size) {
    size_t total = count * size;
    void *p = malloc(total);
    memset(p, 0, total);
    return p;
}
void free(void *p) { (void)p; }

// ============================================================
// Software math (no libm dependency)
// ============================================================
static float my_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
    union { float f; uint32_t i; } u = { .f = x };
    u.i = (u.i >> 1) + 0x1FC00000;
    u.f = 0.5f * (u.f + x / u.f);
    u.f = 0.5f * (u.f + x / u.f);
    u.f = 0.5f * (u.f + x / u.f);
    return u.f;
}

static inline float quake2_exp_from_affine(float l) {
    const uint32_t mant_mask = 0x007FFFFFu;
    const uint32_t exp_mask  = 0xFF800000u;
    const uint32_t one_bits  = 0x3F800000u;

    int32_t zi = (int32_t)l;
    uint32_t z = (uint32_t)zi;

    union { uint32_t u; float f; } am = { .u = (z & mant_mask) | one_bits };
    am.f = (am.f * am.f + 2.0f) * 0.3333333333333333f;

    union { uint32_t u; float f; } y;
    y.u = (z & exp_mask) + am.u - one_bits;
    return y.f;
}

static inline float quake2_exp_unchecked(float x) {
    return quake2_exp_from_affine(x * 12102203.161561486f + 1065353216.0f);
}

static inline float quake2_exp_neg_unchecked(float x) {
    return quake2_exp_from_affine(1065353216.0f - x * 12102203.161561486f);
}

static float my_expf(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;
    return quake2_exp_unchecked(x);
}

static float my_logf(float x) {
    union { float f; uint32_t i; } u = { .f = x };
    int e = (int)(u.i >> 23) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000;
    float m = u.f;
    float t = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float ln_m = 2.0f * t * (1.0f + t2 * (0.3333333f + t2 * (0.2f + t2 * 0.142857f)));
    return ln_m + (float)e * 0.6931471805599453f;
}

static float my_powf(float base, float exp) {
    if (base <= 0.0f) return 0.0f;
    return my_expf(exp * my_logf(base));
}

static float my_fmodf(float x, float y) {
    return x - (float)(int)(x / y) * y;
}

static float my_sinf_inner(float x) {
    float x2 = x * x;
    return x * (1.0f + x2 * (-0.16666667f + x2 * (0.008333333f + x2 * (-0.0001984127f + x2 * 2.7557319e-06f))));
}

static float my_sinf(float x) {
    float pi = 3.14159265358979f;
    x = my_fmodf(x, 2.0f * pi);
    if (x < 0) x += 2.0f * pi;
    if (x > pi) { x -= pi; return -my_sinf_inner(x); }
    return my_sinf_inner(x);
}

static float my_cosf(float x) {
    return my_sinf(x + 1.5707963267948966f);
}

// ============================================================
// Model structures (W4A8 quantized)
// ============================================================
typedef struct {
    int hidden, n_layers, vocab, max_seq;
    int q_heads, kv_heads, head_dim;
    int q_dim, kv_dim, intermediate;
    int has_qk_norm;
    int group_size;
    int weight_bits;
    int act_bits;
    int rowwise_scales;
    float rms_eps, rope_theta;
} Config;

typedef struct {
    float   *embed_scales;
    uint8_t *embed_data;

    float **rms_att_weight;
    float **q_norm;
    float **k_norm;
    float **rms_ffn_weight;

    float   **wqkv_scales;   uint8_t **wqkv_data;
    float   **wo_scales;     uint8_t **wo_data;
    float   **w1_scales;     uint8_t **w1_data;
    float   **w3_scales;     uint8_t **w3_data;
    float   **w2_scales;     uint8_t **w2_data;

    float *rms_final_weight;

    float   *lm_head_scales;
    uint8_t *lm_head_data;
} Weights;

typedef struct {
    float *x, *xb, *xb2;
    float *q, *k, *v;
    float *att;
    float *hb, *hb2;
    float *logits;
    float *key_cache, *value_cache;
    float *qkv;
} RunState;

// ============================================================
// W4A8 matmul backend (compile-time selection)
// ============================================================

static void dequant_embed_row(float *x, const uint8_t *packed, const float *sc,
                              int hidden, int gs) {
    const int ng = hidden / gs;
    for (int g = 0; g < ng; g++) {
        const float scale = sc[g];
        const int base = g * gs;
        for (int j = 0; j < gs; j += 2) {
            const uint8_t byte = packed[(base + j) / 2];
            x[base + j]     = scale * (float)w4_unpack_nibble(byte, 0);
            x[base + j + 1] = scale * (float)w4_unpack_nibble(byte, 1);
        }
    }
}

static inline float _fabsf(float x) { return x < 0.0f ? -x : x; }

static inline int8_t _round_clamp_i8(float x) {
    int qi = (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    return (int8_t)qi;
}

// Quantize fp32 vector x[n] to int8 with per-group scales.
// xq_out: int8 output, xs_out: per-group float scales.
static void quantize_x(int8_t *xq_out, float *xs_out,
                        const float *x, int n, int group_size) {
    int n_groups = n / group_size;
    for (int g = 0; g < n_groups; g++) {
        int base = g * group_size;
        float amax = 0.0f;
        for (int j = 0; j < group_size; j++) {
            float a = _fabsf(x[base + j]);
            if (a > amax) amax = a;
        }
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        xs_out[g] = scale;
        if (amax == 0.0f) {
            for (int j = 0; j < group_size; j++)
                xq_out[base + j] = 0;
        } else {
            float inv = 1.0f / scale;
            for (int j = 0; j < group_size; j++)
                xq_out[base + j] = _round_clamp_i8(x[base + j] * inv);
        }
    }
}

#if defined(USE_AME) && USE_AME
// Use kernel mix backend: AME i8xi8->i32 + RVV/scalar accumulation.
#define MK_Q8_MAX_N 4096
static int8_t _mk_xq_buf[MK_Q8_MAX_N] __attribute__((aligned(64)));
static float _mk_xs_buf[MK_Q8_MAX_N / 32 + 1] __attribute__((aligned(64)));

__attribute__((unused))
static void matmul_w4_mk(float *o, const uint8_t *w, const float *w_scales,
                         const float *x, int m, int n, int group_size) {
    if (n > MK_Q8_MAX_N) {
        print_uart("MK_Q8_MAX_N overflow\r\n");
        exit(1);
    }
    quantize_x(_mk_xq_buf, _mk_xs_buf, x, n, group_size);
    ame_matmul_quant_noblk_i8_i8_f32(o, _mk_xq_buf, _mk_xs_buf,
                                      (const int8_t *)w, w_scales,
                                      n, m, group_size, n);
}

static void matmul_w4_mk_prequant(float *o, const uint8_t *w, const float *w_scales,
                                  int m, int n, int group_size,
                                  const int8_t *xq, const float *xs) {
    ame_matmul_quant_noblk_i8_i8_f32(o, xq, xs, (const int8_t *)w, w_scales,
                                      n, m, group_size, n);
}

static void quantize_x_batch(int8_t *xq_out, float *xs_out,
                             const float *x, int T, int n, int group_size) {
    int n_groups = n / group_size;
    for (int t = 0; t < T; t++) {
        quantize_x(xq_out + (size_t)t * n,
                   xs_out + (size_t)t * n_groups,
                   x + (size_t)t * n,
                   n, group_size);
    }
}

__attribute__((unused))
static void matmul_w4_mk_batch(float *o, const uint8_t *w, const float *w_scales,
                               const float *x, int T, int m, int n,
                               int group_size, int8_t *xq, float *xs) {
    quantize_x_batch(xq, xs, x, T, n, group_size);
    ame_matmul_quant_batch_i8_i8_f32(o, xq, xs, (const int8_t *)w, w_scales,
                                      T, n, m, group_size, n);
}

static void matmul_w4_mk_batch_prequant(float *o, const uint8_t *w, const float *w_scales,
                                        int T, int m, int n, int group_size,
                                        const int8_t *xq, const float *xs) {
    ame_matmul_quant_batch_i8_i8_f32(o, xq, xs, (const int8_t *)w, w_scales,
                                      T, n, m, group_size, n);
}

static void matmul_w4_mk_pair_prequant(float *o0, float *o1,
                                       const uint8_t *w0, const float *w0_scales,
                                       const uint8_t *w1, const float *w1_scales,
                                       int m, int n, int group_size,
                                       const int8_t *xq, const float *xs) {
    ame_matmul_quant_noblk_pair_i8_i8_f32(o0, o1, xq, xs,
                                          (const int8_t *)w0, w0_scales,
                                          (const int8_t *)w1, w1_scales,
                                          n, m, group_size, n);
}

static void matmul_w4_mk_batch_pair_prequant(float *o0, float *o1,
                                             const uint8_t *w0, const float *w0_scales,
                                             const uint8_t *w1, const float *w1_scales,
                                             int T, int m, int n, int group_size,
                                             const int8_t *xq, const float *xs) {
    ame_matmul_quant_batch_pair_i8_i8_f32(o0, o1, xq, xs,
                                          (const int8_t *)w0, w0_scales,
                                          (const int8_t *)w1, w1_scales,
                                          T, n, m, group_size, n);
}

#define MATMUL_W4 matmul_w4_mk
#if defined(USE_ZTT) && USE_ZTT
#define MATMUL_NAME "ztt_w4"
#else
#define MATMUL_NAME "ame_w4"
#endif

#elif defined(USE_RVV) && USE_RVV
// Model v6 layout: [d/64][n/gs][64][gs], with scales [n/gs][d].
#define MK_Q8_MAX_N 4096
static int8_t _mk_xq_buf[MK_Q8_MAX_N] __attribute__((aligned(64)));
static float _mk_xs_buf[MK_Q8_MAX_N / 32 + 1] __attribute__((aligned(64)));

static void matmul_w4_mk(float *o, const uint8_t *w, const float *w_scales,
                         const float *x, int m, int n, int group_size) {
    if (n > MK_Q8_MAX_N) {
        print_uart("MK_Q8_MAX_N overflow\r\n");
        exit(1);
    }
    quantize_x(_mk_xq_buf, _mk_xs_buf, x, n, group_size);
    rvv_matmul_quant_tile_i8_i8_f32(o, _mk_xq_buf, _mk_xs_buf,
                                     (const int8_t *)w, w_scales,
                                     n, m, group_size);
}

static void matmul_w4_mk_prequant(float *o, const uint8_t *w,
                                  const float *w_scales, int m, int n,
                                  int group_size, const int8_t *xq,
                                  const float *xs) {
    rvv_matmul_quant_tile_i8_i8_f32(o, xq, xs, (const int8_t *)w,
                                     w_scales, n, m, group_size);
}

static void quantize_x_batch(int8_t *xq_out, float *xs_out,
                             const float *x, int T, int n, int group_size) {
    const int n_groups = n / group_size;
    for (int t = 0; t < T; ++t) {
        quantize_x(xq_out + (size_t)t * n,
                   xs_out + (size_t)t * n_groups,
                   x + (size_t)t * n, n, group_size);
    }
}

static void matmul_w4_mk_batch_prequant(float *o, const uint8_t *w,
                                        const float *w_scales, int T, int m,
                                        int n, int group_size,
                                        const int8_t *xq, const float *xs) {
    const int n_groups = n / group_size;
    for (int t = 0; t < T; ++t) {
        matmul_w4_mk_prequant(o + (size_t)t * m, w, w_scales, m, n,
                              group_size, xq + (size_t)t * n,
                              xs + (size_t)t * n_groups);
    }
}

static void matmul_w4_mk_pair_prequant(float *o0, float *o1,
                                       const uint8_t *w0,
                                       const float *w0_scales,
                                       const uint8_t *w1,
                                       const float *w1_scales,
                                       int m, int n, int group_size,
                                       const int8_t *xq, const float *xs) {
    matmul_w4_mk_prequant(o0, w0, w0_scales, m, n, group_size, xq, xs);
    matmul_w4_mk_prequant(o1, w1, w1_scales, m, n, group_size, xq, xs);
}

static void matmul_w4_mk_batch_pair_prequant(float *o0, float *o1,
                                             const uint8_t *w0,
                                             const float *w0_scales,
                                             const uint8_t *w1,
                                             const float *w1_scales,
                                             int T, int m, int n,
                                             int group_size,
                                             const int8_t *xq,
                                             const float *xs) {
    matmul_w4_mk_batch_prequant(o0, w0, w0_scales, T, m, n, group_size,
                                xq, xs);
    matmul_w4_mk_batch_prequant(o1, w1, w1_scales, T, m, n, group_size,
                                xq, xs);
}

#define MATMUL_W4 matmul_w4_mk
#define MATMUL_NAME "rvv_w4"

#else
#define SCALAR_W4_MAX_N 4096
static int8_t _scalar_xq_buf[SCALAR_W4_MAX_N] __attribute__((aligned(64)));
static float _scalar_xs_buf[SCALAR_W4_MAX_N / 32 + 1] __attribute__((aligned(64)));

static void matmul_w4_scalar(float *o, const uint8_t *w, const float *w_scales,
                              const float *x, int m, int n, int group_size) {
    if (n > SCALAR_W4_MAX_N) {
        print_uart("SCALAR_W4_MAX_N overflow\r\n");
        exit(1);
    }
    quantize_x(_scalar_xq_buf, _scalar_xs_buf, x, n, group_size);
    scalar_matmul_quant_i8_i4_f32(o, _scalar_xq_buf, _scalar_xs_buf,
                                  w, w_scales, n, m, group_size);
}
#define MATMUL_W4 matmul_w4_scalar
#define MATMUL_NAME "scalar_w4"
#endif

// ============================================================
// Embedded tokenizer
// ============================================================
#include "../tokenizer/tokenizer_bare.h"

enum {
    IM_START_ID = 151644,
    IM_END_ID = 151645,
    THINK_START_ID = 151667,
    THINK_END_ID = 151668,
    USER_ID = 872,
    SYSTEM_ID = 8948,
    ASSISTANT_ID = 77091,
    NEWLINE_ID = 198,
};

#ifndef IDENTITY_SYSTEM_PROMPT
#define IDENTITY_SYSTEM_PROMPT "Qwen3"
#endif
#ifndef IDENTITY_SYSTEM_CHAT_MESSAGE
#define IDENTITY_SYSTEM_CHAT_MESSAGE 1
#endif

static int is_thinking_wrapper_token(int token) {
    return token == THINK_START_ID || token == THINK_END_ID;
}

static int is_replacement_piece_token(BareTokenizer *tok, int token) {
    static const unsigned char repl[] = {0xc3, 0xaf, 0xc2, 0xbf, 0xc2, 0xbd};
    if (token < 0 || token >= tok->vocab_size || !tok->pieces_by_id[token]) return 0;
    const unsigned char *p = (const unsigned char *)tok->pieces_by_id[token];
    while (*p) {
        int match = 1;
        for (int i = 0; i < 6; i++) {
            if (p[i] != repl[i]) { match = 0; break; }
        }
        if (match) return 1;
        p++;
    }
    return 0;
}

static int is_hidden_output_token(BareTokenizer *tok, int token) {
    return is_thinking_wrapper_token(token) || is_replacement_piece_token(tok, token);
}

typedef struct {
    unsigned char bytes[4];
    int len;
    int need;
} SafeUtf8State;

static int safe_utf8_codepoint(const unsigned char *p, int n) {
    if (n == 1) return p[0];
    if (n == 2) return ((p[0] & 0x1f) << 6) | (p[1] & 0x3f);
    if (n == 3) return ((p[0] & 0x0f) << 12) | ((p[1] & 0x3f) << 6) | (p[2] & 0x3f);
    return ((p[0] & 0x07) << 18) | ((p[1] & 0x3f) << 12) |
           ((p[2] & 0x3f) << 6) | (p[3] & 0x3f);
}

static int safe_utf8_valid_complete(const unsigned char *p, int n) {
    if (n == 1) return p[0] < 0x80;
    if (n == 2) return p[0] >= 0xc2 && p[0] <= 0xdf &&
                       p[1] >= 0x80 && p[1] <= 0xbf;
    if (n == 3) {
        if (p[1] < 0x80 || p[1] > 0xbf || p[2] < 0x80 || p[2] > 0xbf) return 0;
        if (p[0] == 0xe0) return p[1] >= 0xa0;
        if (p[0] == 0xed) return p[1] <= 0x9f;
        return p[0] >= 0xe1 && p[0] <= 0xef;
    }
    if (n == 4) {
        if (p[1] < 0x80 || p[1] > 0xbf || p[2] < 0x80 || p[2] > 0xbf ||
            p[3] < 0x80 || p[3] > 0xbf) return 0;
        if (p[0] == 0xf0) return p[1] >= 0x90;
        if (p[0] == 0xf4) return p[1] <= 0x8f;
        return p[0] >= 0xf1 && p[0] <= 0xf3;
    }
    return 0;
}

static void safe_utf8_emit_complete(SafeUtf8State *st) {
    extern void write_serial(char a);
    if (!safe_utf8_valid_complete(st->bytes, st->len)) {
        st->len = 0;
        st->need = 0;
        return;
    }
    const int cp = safe_utf8_codepoint(st->bytes, st->len);
    if (cp == 0xfffd) {
        st->len = 0;
        st->need = 0;
        return;
    }
    for (int i = 0; i < st->len; i++) write_serial((char)st->bytes[i]);
    st->len = 0;
    st->need = 0;
}

static void safe_utf8_emit_byte(SafeUtf8State *st, unsigned char b) {
    extern void write_serial(char a);
    if (st->need == 0) {
        if (b < 0x80) {
            write_serial((char)b);
        } else if (b >= 0xc2 && b <= 0xdf) {
            st->bytes[0] = b;
            st->len = 1;
            st->need = 1;
        } else if (b >= 0xe0 && b <= 0xef) {
            st->bytes[0] = b;
            st->len = 1;
            st->need = 2;
        } else if (b >= 0xf0 && b <= 0xf4) {
            st->bytes[0] = b;
            st->len = 1;
            st->need = 3;
        }
        return;
    }

    if (b < 0x80 || b > 0xbf) {
        st->len = 0;
        st->need = 0;
        safe_utf8_emit_byte(st, b);
        return;
    }
    st->bytes[st->len++] = b;
    st->need--;
    if (st->need == 0) safe_utf8_emit_complete(st);
}

static void tokenizer_decode_uart_safe(BareTokenizer *tok, int token, SafeUtf8State *st) {
    if (token == -2147483647) tokenizer_decode_uart(tok, &token, 0);
    if (is_hidden_output_token(tok, token)) return;
    if (token < 0 || token >= tok->vocab_size || !tok->pieces_by_id[token]) return;
    const char *pp = tok->pieces_by_id[token];
    while (*pp) {
        int cp;
        const char *old = pp;
        if (!tok_utf8_next(&pp, &cp)) break;
        if (cp >= 0 && cp < 65536 && tok->unicode_to_byte_valid[cp]) {
            safe_utf8_emit_byte(st, tok->unicode_to_byte[cp]);
        } else {
            while (old < pp) {
                safe_utf8_emit_byte(st, (unsigned char)*old);
                old++;
            }
        }
    }
}

#define NO_THINKING_PROMPT_TOKENS 4
static int append_no_thinking_prompt_tokens(int *tokens, int n_tokens, int cap) {
    static const int no_thinking[NO_THINKING_PROMPT_TOKENS] = {
        THINK_START_ID, 271, THINK_END_ID, 271
    };
    for (int i = 0; i < NO_THINKING_PROMPT_TOKENS && n_tokens < cap; i++) {
        tokens[n_tokens++] = no_thinking[i];
    }
    return n_tokens;
}

__attribute__((unused))
static int append_chat_message_tokens(BareTokenizer *tok,
                                      int *tokens,
                                      int n_tokens,
                                      int cap,
                                      int role_token,
                                      const char *text) {
    if (n_tokens + 4 >= cap) return n_tokens;
    tokens[n_tokens++] = IM_START_ID;
    tokens[n_tokens++] = role_token;
    tokens[n_tokens++] = NEWLINE_ID;

    int budget = cap - n_tokens - 2;
    int enc = tokenizer_encode_bare(tok, text, tokens + n_tokens, budget);
    if (enc < 0) enc = 0;
    n_tokens += enc;

    if (n_tokens + 2 > cap) return n_tokens;
    tokens[n_tokens++] = IM_END_ID;
    tokens[n_tokens++] = NEWLINE_ID;
    return n_tokens;
}

static int append_identity_system_tokens(BareTokenizer *tok,
                                         int *tokens,
                                         int n_tokens,
                                         int cap) {
#if defined(IDENTITY_SYSTEM_CHAT_MESSAGE) && IDENTITY_SYSTEM_CHAT_MESSAGE
    return append_chat_message_tokens(tok, tokens, n_tokens, cap,
                                      SYSTEM_ID, IDENTITY_SYSTEM_PROMPT);
#else
    int budget = cap - n_tokens;
    int enc = tokenizer_encode_bare(tok, IDENTITY_SYSTEM_PROMPT,
                                    tokens + n_tokens, budget);
    if (enc < 0) enc = 0;
    return n_tokens + enc;
#endif
}

// ============================================================
// Core ops
// ============================================================
static float silu(float x) {
    if (x > 88.0f) return x;
    if (x < -88.0f) return 0.0f;
    return x / (1.0f + quake2_exp_neg_unchecked(x));
}

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) { double v = x[i]; ss += v * v; }
    float inv = 1.0f / my_sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = w[i] * (x[i] * inv);
}

static void rmsnorm_inplace(float *x, const float *w, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) { double v = x[i]; ss += v * v; }
    float inv = 1.0f / my_sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = w[i] * (x[i] * inv);
}

static void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { x[i] = my_expf(x[i] - maxv); sum += x[i]; }
    float inv = 1.0f / (float)sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
#ifndef QK_RVV_MAX_HEAD_DIM
#define QK_RVV_MAX_HEAD_DIM 256
#endif
#ifdef QWEN_LOCAL_SPM
#define g_qk_rvv_prod ((float *)QWEN_SPM_RVV2)
#define g_qk_rvv_acc ((float *)QWEN_SPM_RVV2)
/* The scalar host reduces a DDR copy; it never reads RVV-owned SPM. */
static float g_qk_rvv_host[QK_RVV_MAX_HEAD_DIM]
    __attribute__((aligned(64)));
_Static_assert(QK_RVV_MAX_HEAD_DIM * sizeof(float) <= QWEN_SPM_RVV_BYTES,
               "QK scratch exceeds its SPM window");
#else
static float g_qk_rvv_prod[QK_RVV_MAX_HEAD_DIM] __attribute__((aligned(64)));
static float g_qk_rvv_acc[128] __attribute__((aligned(64)));
#endif

static inline uint32_t rvv_f32_bits(float x) {
    union { float f; uint32_t u; } v = { .f = x };
    return v.u;
}

static inline long qk_vsetvl_e32m1(long n) {
    long vl;
    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +v\n\t"
        "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
        ".option pop\n\t"
        : [vl] "=r"(vl)
        : [n] "r"(n)
        : "memory");
    return vl;
}

static inline float qk_dot_vfmul_store_rvv(const float *q, const float *k, int n) {
    if (n > QK_RVV_MAX_HEAD_DIM) {
        float fallback = 0.0f;
        for (int i = 0; i < n; i++) fallback += q[i] * k[i];
        return fallback;
    }

    long remaining = n;
    const float *qp = q;
    const float *kp = k;
    float *pp = g_qk_rvv_prod;

    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +v\n\t"
        ".Lqk_vfmul_loop_%=:\n\t"
        "beqz %[remaining], .Lqk_vfmul_done_%=\n\t"
        "vsetvli t0, %[remaining], e32, m1, ta, ma\n\t"
        "vle32.v v0, (%[qp])\n\t"
        "vle32.v v1, (%[kp])\n\t"
        "vfmul.vv v0, v0, v1\n\t"
        "vse32.v v0, (%[pp])\n\t"
        "slli t1, t0, 2\n\t"
        "add %[qp], %[qp], t1\n\t"
        "add %[kp], %[kp], t1\n\t"
        "add %[pp], %[pp], t1\n\t"
        "sub %[remaining], %[remaining], t0\n\t"
        "j .Lqk_vfmul_loop_%=\n\t"
        ".Lqk_vfmul_done_%=:\n\t"
        "fence rw,rw\n\t"
        ".option pop\n\t"
        : [qp] "+r"(qp), [kp] "+r"(kp), [pp] "+r"(pp),
          [remaining] "+r"(remaining)
        :
        : "t0", "t1", "v0", "v1", "memory");

    const float *host_prod = g_qk_rvv_prod;
#ifdef QWEN_LOCAL_SPM
    qwen_dma_copy((uintptr_t)g_qk_rvv_host, QWEN_SPM_RVV2,
                  n * sizeof(float), 1);
    host_prod = g_qk_rvv_host;
#endif
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += host_prod[i];
    return sum;
}

static inline float qk_dot_vfadd_accum_128_rvv(const float *q, const float *k) {
    long acc_vl = qk_vsetvl_e32m1(128);
    if (acc_vl <= 0 || acc_vl > 128 || (128 % acc_vl) != 0) {
        return qk_dot_vfmul_store_rvv(q, k, 128);
    }

    long remaining = 128;
    const float *qp = q;
    const float *kp = k;
    float *ap = g_qk_rvv_acc;
    long step_bytes = acc_vl * 4;
    uint64_t zero_bits = 0;

    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +v\n\t"
        "vsetvli t0, %[acc_vl], e32, m1, ta, ma\n\t"
        "vmv.v.x v8, %[zero_bits]\n\t"
        ".Lqk_vfadd_accum128_loop_%=:\n\t"
        "beqz %[remaining], .Lqk_vfadd_accum128_done_%=\n\t"
        "vle32.v v0, (%[qp])\n\t"
        "vle32.v v1, (%[kp])\n\t"
        "vfmul.vv v0, v0, v1\n\t"
        "vfadd.vv v8, v8, v0\n\t"
        "fence rw,rw\n\t"
        "add %[qp], %[qp], %[step_bytes]\n\t"
        "add %[kp], %[kp], %[step_bytes]\n\t"
        "sub %[remaining], %[remaining], %[acc_vl]\n\t"
        "j .Lqk_vfadd_accum128_loop_%=\n\t"
        ".Lqk_vfadd_accum128_done_%=:\n\t"
        "vsetvli t0, %[acc_vl], e32, m1, ta, ma\n\t"
        "fence rw,rw\n\t"
        "vse32.v v8, (%[ap])\n\t"
        "fence rw,rw\n\t"
        ".option pop\n\t"
        : [qp] "+r"(qp), [kp] "+r"(kp), [remaining] "+r"(remaining)
        : [ap] "r"(ap), [acc_vl] "r"(acc_vl),
          [step_bytes] "r"(step_bytes), [zero_bits] "r"(zero_bits)
        : "t0", "v0", "v1", "v8", "memory");

    const float *host_acc = g_qk_rvv_acc;
#ifdef QWEN_LOCAL_SPM
    qwen_dma_copy((uintptr_t)g_qk_rvv_host, QWEN_SPM_RVV2,
                  acc_vl * sizeof(float), 1);
    host_acc = g_qk_rvv_host;
#endif
    float sum = 0.0f;
    for (int i = 0; i < acc_vl; i++) sum += host_acc[i];
    return sum;
}

static inline float qk_dot_vfmul_rvv(const float *q, const float *k, int n) {
#ifdef QWEN_LOCAL_SPM
    if (n > 0 && n <= QK_RVV_MAX_HEAD_DIM) {
        qwen_spm_stage_pair(q, k, n);
        q = (const float *)QWEN_SPM_RVV0;
        k = (const float *)QWEN_SPM_RVV1;
    }
#endif
    if (n == 128) return qk_dot_vfadd_accum_128_rvv(q, k);
    return qk_dot_vfmul_store_rvv(q, k, n);
}

#define QK_DOT(q, k, n) qk_dot_vfmul_rvv((q), (k), (n))

static inline void attn_av_axpy_rvv(float *oh, const float *vt, float a, int n) {
    if (n > QK_RVV_MAX_HEAD_DIM) {
        for (int i = 0; i < n; i++) oh[i] += a * vt[i];
        return;
    }

#ifdef QWEN_LOCAL_SPM
    float *ddr_output = oh;
    qwen_spm_stage_pair(oh, vt, n);
    oh = (float *)QWEN_SPM_RVV0;
    vt = (const float *)QWEN_SPM_RVV1;
#endif
    long remaining = n;
    float *op = oh;
    const float *vp = vt;
    uint64_t a_bits = (uint64_t)rvv_f32_bits(a);

    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +v\n\t"
        ".Lattn_av_axpy_loop_%=:\n\t"
        "beqz %[remaining], .Lattn_av_axpy_done_%=\n\t"
        "vsetvli t0, %[remaining], e32, m1, ta, ma\n\t"
        "vle32.v v8, (%[op])\n\t"
        "vle32.v v9, (%[vp])\n\t"
        "vmv.v.x v10, %[a_bits]\n\t"
        "vfmul.vv v9, v9, v10\n\t"
        "vfadd.vv v8, v8, v9\n\t"
        "fence rw,rw\n\t"
        "vse32.v v8, (%[op])\n\t"
        "fence rw,rw\n\t"
        "slli t1, t0, 2\n\t"
        "add %[op], %[op], t1\n\t"
        "add %[vp], %[vp], t1\n\t"
        "sub %[remaining], %[remaining], t0\n\t"
        "j .Lattn_av_axpy_loop_%=\n\t"
        ".Lattn_av_axpy_done_%=:\n\t"
        ".option pop\n\t"
        : [op] "+r"(op), [vp] "+r"(vp), [remaining] "+r"(remaining)
        : [a_bits] "r"(a_bits)
        : "t0", "t1", "v8", "v9", "v10", "memory");
#ifdef QWEN_LOCAL_SPM
    if (n > 0) qwen_dma_copy((uintptr_t)ddr_output, QWEN_SPM_RVV0, n * sizeof(float), 1);
#endif
}
#else
static inline float qk_dot_scalar(const float *q, const float *k, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += q[i] * k[i];
    return sum;
}
#define QK_DOT(q, k, n) qk_dot_scalar((q), (k), (n))
#endif

__attribute__((unused))
static inline void attn_av_axpy_scalar(float *oh, const float *vt, float a, int n) {
    for (int i = 0; i < n; i++) oh[i] += a * vt[i];
}
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
#define ATTN_AV_AXPY(oh, vt, a, n) attn_av_axpy_rvv((oh), (vt), (a), (n))
#else
#define ATTN_AV_AXPY(oh, vt, a, n) attn_av_axpy_scalar((oh), (vt), (a), (n))
#endif

// RoPE rotation with cached trig.
//
// freq[i] = theta^-(2i/head_dim) depends only on (head_dim, theta) -> built
// once for the whole run. cos/sin(freq[i]*pos) depend only on pos, and every
// head/layer of a token shares the same pos, so the 24*n_layers calls per token
// collapse to a single table rebuild per distinct position. The per-element
// arithmetic is unchanged, so output is bit-identical to the naive version.
#ifndef ROPE_MAX_HALF
#define ROPE_MAX_HALF 128        // supports head_dim up to 256 (qwen3 uses 128 -> half 64)
#endif
static float g_rope_freq[ROPE_MAX_HALF];
static float g_rope_cos[ROPE_MAX_HALF];
static float g_rope_sin[ROPE_MAX_HALF];
static int   g_rope_half  = 0;   // geometry for which g_rope_freq is valid
static float g_rope_theta = 0.0f;
static int   g_rope_pos   = -1;  // pos for which g_rope_cos/sin are valid

static void rope_rotate(float *v, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    if (half > ROPE_MAX_HALF) {
        // Geometry beyond cache: fall back to the per-element path.
        for (int i = 0; i < half; i++) {
            float exponent = (float)(2.0f * i) / (float)head_dim;
            float freq = my_powf(theta, -exponent);
            float ang = freq * (float)pos;
            float c = my_cosf(ang), s = my_sinf(ang);
            float v0 = v[i], v1 = v[i + half];
            v[i] = v0 * c - v1 * s;
            v[i + half] = v0 * s + v1 * c;
        }
        return;
    }
    if (half != g_rope_half || theta != g_rope_theta) {
        for (int i = 0; i < half; i++) {
            float exponent = (float)(2.0f * i) / (float)head_dim;
            g_rope_freq[i] = my_powf(theta, -exponent);
        }
        g_rope_half = half;
        g_rope_theta = theta;
        g_rope_pos = -1;
    }
    if (pos != g_rope_pos) {
        for (int i = 0; i < half; i++) {
            float ang = g_rope_freq[i] * (float)pos;
            g_rope_cos[i] = my_cosf(ang);
            g_rope_sin[i] = my_sinf(ang);
        }
        g_rope_pos = pos;
    }
    for (int i = 0; i < half; i++) {
        float c = g_rope_cos[i], s = g_rope_sin[i];
        float v0 = v[i], v1 = v[i + half];
        v[i] = v0 * c - v1 * s;
        v[i + half] = v0 * s + v1 * c;
    }
}

// ============================================================
// Load W4A8 model from embedded blob (zero-copy)
// ============================================================
static void load_model(const char *data, Config *cfg, Weights *w) {
    const char *p = data;

    uint64_t magic = *(const uint64_t *)p; p += 8;
    if (magic != W4_MAGIC) { print_uart("bad magic\r\n"); exit(1); }

    int32_t version = *(const int32_t *)p; p += 4;
    if (version < W4_FORMAT_VERSION_MIN || version > W4_FORMAT_VERSION) {
        print_uart("expected W4A8 blob v");
        print_uart_int_dec(W4_FORMAT_VERSION);
        print_uart("\r\n");
        exit(1);
    }

    cfg->hidden       = *(const int32_t *)p; p += 4;
    cfg->n_layers     = *(const int32_t *)p; p += 4;
    cfg->vocab        = *(const int32_t *)p; p += 4;
    cfg->max_seq      = *(const int32_t *)p; p += 4;
    cfg->q_heads      = *(const int32_t *)p; p += 4;
    cfg->kv_heads     = *(const int32_t *)p; p += 4;
    cfg->head_dim     = *(const int32_t *)p; p += 4;
    cfg->q_dim        = *(const int32_t *)p; p += 4;
    cfg->kv_dim       = *(const int32_t *)p; p += 4;
    cfg->intermediate = *(const int32_t *)p; p += 4;
    cfg->has_qk_norm  = *(const int32_t *)p; p += 4;
    cfg->group_size   = *(const int32_t *)p; p += 4;
    cfg->weight_bits  = *(const int32_t *)p; p += 4;
    cfg->act_bits     = *(const int32_t *)p; p += 4;
    cfg->rowwise_scales = 0;
    cfg->rms_eps      = *(const float *)p; p += 4;
    cfg->rope_theta   = *(const float *)p; p += 4;

    if (cfg->weight_bits != W4_WEIGHT_BITS || cfg->act_bits != W4_ACT_BITS) {
        print_uart("unsupported W4A8 bitwidth\r\n");
        exit(1);
    }
    if ((cfg->group_size % 2) != 0) {
        print_uart("group_size must be even for int4 packing\r\n");
        exit(1);
    }

    p = (const char *)(((uintptr_t)p + 63u) & ~(uintptr_t)63u);

    int h = cfg->hidden, L = cfg->n_layers, V = cfg->vocab;
    int qd = cfg->q_dim, kvd = cfg->kv_dim, hd = cfg->head_dim, inter = cfg->intermediate;
    int qkv_dim = qd + 2 * kvd;
    int gs = cfg->group_size;

    int ng_h     = h / gs;
    int ng_qd    = qd / gs;
    int ng_inter = inter / gs;
    const int tile_k_bytes = (version >= 6) ? 1 : 2; /* v6 int8 tile; v5 int4 packed */

    w->rms_att_weight = (float **)malloc((size_t)L * sizeof(float *));
    w->q_norm         = (float **)malloc((size_t)L * sizeof(float *));
    w->k_norm         = (float **)malloc((size_t)L * sizeof(float *));
    w->rms_ffn_weight = (float **)malloc((size_t)L * sizeof(float *));

    w->wqkv_scales = (float **)malloc((size_t)L * sizeof(float *));
    w->wqkv_data   = (uint8_t **)malloc((size_t)L * sizeof(uint8_t *));
    w->wo_scales   = (float **)malloc((size_t)L * sizeof(float *));
    w->wo_data     = (uint8_t **)malloc((size_t)L * sizeof(uint8_t *));
    w->w1_scales   = (float **)malloc((size_t)L * sizeof(float *));
    w->w1_data     = (uint8_t **)malloc((size_t)L * sizeof(uint8_t *));
    w->w3_scales   = (float **)malloc((size_t)L * sizeof(float *));
    w->w3_data     = (uint8_t **)malloc((size_t)L * sizeof(uint8_t *));
    w->w2_scales   = (float **)malloc((size_t)L * sizeof(float *));
    w->w2_data     = (uint8_t **)malloc((size_t)L * sizeof(uint8_t *));

    w->embed_scales = (float *)p;   p += (size_t)V * ng_h * 4;
    w->embed_data   = (uint8_t *)p; p += (size_t)V * (h / 2);

    for (int l = 0; l < L; l++) {
        w->rms_att_weight[l] = (float *)p; p += (size_t)h * 4;
        w->wqkv_scales[l] = (float *)p;    p += (size_t)qkv_dim * ng_h * 4;
        w->wqkv_data[l]   = (uint8_t *)p;  p += (size_t)qkv_dim * h / tile_k_bytes;
        if (cfg->has_qk_norm) {
            w->q_norm[l] = (float *)p; p += (size_t)hd * 4;
            w->k_norm[l] = (float *)p; p += (size_t)hd * 4;
        } else { w->q_norm[l] = NULL; w->k_norm[l] = NULL; }
        w->wo_scales[l] = (float *)p;  p += (size_t)h * ng_qd * 4;
        w->wo_data[l]   = (uint8_t *)p; p += (size_t)h * qd / tile_k_bytes;
        w->rms_ffn_weight[l] = (float *)p; p += (size_t)h * 4;
        w->w1_scales[l] = (float *)p;  p += (size_t)inter * ng_h * 4;
        w->w1_data[l]   = (uint8_t *)p; p += (size_t)inter * h / tile_k_bytes;
        w->w3_scales[l] = (float *)p;  p += (size_t)inter * ng_h * 4;
        w->w3_data[l]   = (uint8_t *)p; p += (size_t)inter * h / tile_k_bytes;
        w->w2_scales[l] = (float *)p;  p += (size_t)h * ng_inter * 4;
        w->w2_data[l]   = (uint8_t *)p; p += (size_t)h * inter / tile_k_bytes;
    }

    w->rms_final_weight = (float *)p; p += (size_t)h * 4;
    w->lm_head_scales = (float *)p;  p += (size_t)V * ng_h * 4;
    w->lm_head_data   = (uint8_t *)p;

#if !defined(CHAT_BUILD)
    print_uart("model: v"); print_uart_int_dec(version);
    print_uart(" W4A8 h="); print_uart_int_dec(h);
    print_uart(" L="); print_uart_int_dec(L);
    print_uart(" V="); print_uart_int_dec(V);
    print_uart(" gs="); print_uart_int_dec(cfg->group_size);
    print_uart(" qk="); print_uart_int_dec(cfg->has_qk_norm);
    if (version >= 7) print_uart(" ame4x4k128");
    else if (version >= 6) print_uart(" i8tile");
    print_uart("\r\n");
#endif

#if (!defined(USE_AME) || !USE_AME) && (!defined(USE_RVV) || !USE_RVV)
    if (version >= 6) {
        print_uart("scalar_w4 requires v5 int4-tile blob; rebuild with USE_RVV\r\n");
        exit(1);
    }
#endif
}

static void build_state(RunState *s, const Config *p) {
    s->x      = (float *)malloc((size_t)p->hidden * 4);
    s->xb     = (float *)malloc((size_t)p->hidden * 4);
    s->xb2    = (float *)malloc((size_t)p->hidden * 4);
    s->q      = (float *)malloc((size_t)p->q_dim * 4);
    s->k      = (float *)malloc((size_t)p->kv_dim * 4);
    s->v      = (float *)malloc((size_t)p->kv_dim * 4);
    s->qkv    = (float *)malloc((size_t)(p->q_dim + 2*p->kv_dim) * 4);
    s->att    = (float *)malloc((size_t)p->q_heads * p->max_seq * 4);
    s->hb     = (float *)malloc((size_t)p->intermediate * 4);
    s->hb2    = (float *)malloc((size_t)p->intermediate * 4);
    s->logits = (float *)malloc((size_t)p->vocab * 4);
    size_t kv = (size_t)p->n_layers * p->kv_heads * p->max_seq * p->head_dim;
    s->key_cache   = (float *)malloc(kv * 4);
    s->value_cache = (float *)malloc(kv * 4);
}

static inline float *kc_head(RunState *s, const Config *p, int l, int kvh, int pos) {
    return s->key_cache + (((size_t)l * p->kv_heads + kvh) * p->max_seq + pos) * p->head_dim;
}
static inline float *vc_head(RunState *s, const Config *p, int l, int kvh, int pos) {
    return s->value_cache + (((size_t)l * p->kv_heads + kvh) * p->max_seq + pos) * p->head_dim;
}

static void store_kv_cache(RunState *s, const Config *p, int l, int pos,
                           const float *k, const float *v) {
    for (int h = 0; h < p->kv_heads; h++) {
        const float *kh = k + (size_t)h * p->head_dim;
        const float *vh = v + (size_t)h * p->head_dim;
        prefill_memcpy_f32_r4(kc_head(s, p, l, h, pos), kh, p->head_dim);
        prefill_memcpy_f32_r4(vc_head(s, p, l, h, pos), vh, p->head_dim);
    }
}

// ============================================================
// Forward pass
// ============================================================
static void forward(const Config *p, const Weights *w, RunState *s, int token, int pos) {
    int gs_h = p->rowwise_scales ? p->hidden : p->group_size;
    int gs_qd = p->rowwise_scales ? p->q_dim : p->group_size;
    int gs_inter = p->rowwise_scales ? p->intermediate : p->group_size;
    int ng_h = p->hidden / gs_h;
    uint64_t _t0, _t1;

    _t0 = rdcycle();
    {
        const float *sc = w->embed_scales + (size_t)token * ng_h;
        const uint8_t *ed = w->embed_data + (size_t)token * (p->hidden / 2);
        dequant_embed_row(s->x, ed, sc, p->hidden, gs_h);
    }
    g_timing.embed += rdcycle() - _t0;

    int q_per_kv = p->q_heads / p->kv_heads;
    const float hd_sqrt = my_sqrtf((float)p->head_dim);  // constant; hoisted out of attn loop

    for (int l = 0; l < p->n_layers; l++) {
        _t0 = rdcycle();
        rmsnorm(s->xb, s->x, w->rms_att_weight[l], p->hidden, p->rms_eps);
        ADD_LAYER_TIMING(l, attn_norm, rdcycle() - _t0);

        int qkv_dim = p->q_dim + 2 * p->kv_dim;
        _t0 = rdcycle();
#if QKV_MATMUL_LOG_ENABLE
        int log_qkv_this_call = should_log_qkv_matmul();
        if (log_qkv_this_call) {
            log_qkv_matmul_input(l, pos, token, qkv_dim, p->hidden, gs_h,
                                 s->xb, w->wqkv_data[l], w->wqkv_scales[l]);
        }
#endif
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
        _t1 = rdcycle();
        quantize_x(_mk_xq_buf, _mk_xs_buf, s->xb, p->hidden, gs_h);
        ADD_LAYER_TIMING(l, qkv_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_prequant(s->qkv, w->wqkv_data[l], w->wqkv_scales[l],
                              qkv_dim, p->hidden, gs_h, _mk_xq_buf, _mk_xs_buf);
        ADD_LAYER_TIMING(l, qkv_ame, rdcycle() - _t1);
#else
        MATMUL_W4(s->qkv, w->wqkv_data[l], w->wqkv_scales[l], s->xb, qkv_dim, p->hidden, gs_h);
#endif
#if QKV_MATMUL_LOG_ENABLE
        if (log_qkv_this_call)
            log_qkv_matmul_output(s->qkv, qkv_dim);
#endif
        ADD_LAYER_TIMING(l, qkv_matmul, rdcycle() - _t0);

        prefill_memcpy_f32_r4(s->q, s->qkv, p->q_dim);
        prefill_memcpy_f32_r4(s->k, s->qkv + p->q_dim, p->kv_dim);
        prefill_memcpy_f32_r4(s->v, s->qkv + p->q_dim + p->kv_dim, p->kv_dim);

        _t0 = rdcycle();
        if (p->has_qk_norm) {
            for (int h = 0; h < p->q_heads; h++)
                rmsnorm_inplace(s->q + (size_t)h * p->head_dim, w->q_norm[l], p->head_dim, p->rms_eps);
            for (int h = 0; h < p->kv_heads; h++)
                rmsnorm_inplace(s->k + (size_t)h * p->head_dim, w->k_norm[l], p->head_dim, p->rms_eps);
        }
        for (int h = 0; h < p->q_heads; h++)
            rope_rotate(s->q + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
        for (int h = 0; h < p->kv_heads; h++)
            rope_rotate(s->k + (size_t)h * p->head_dim, p->head_dim, pos, p->rope_theta);
        ADD_LAYER_TIMING(l, qknorm_rope, rdcycle() - _t0);

        store_kv_cache(s, p, l, pos, s->k, s->v);

        uint64_t attn_t0 = rdcycle();
        float *ao = s->qkv;
        memset(ao, 0, (size_t)p->q_dim * 4);

        // Attention score is scalar code. It does not call MATMUL_Q8 or AME.
        _t0 = rdcycle();
        for (int h = 0; h < p->q_heads; h++) {
            int kvh = h / q_per_kv;
            const float *qh = s->q + (size_t)h * p->head_dim;
            float *as = s->att + (size_t)h * p->max_seq;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kc_head(s, p, l, kvh, t);
                float sc = QK_DOT(qh, kt, p->head_dim);
                as[t] = (float)(sc / hd_sqrt);
            }
        }
        ADD_LAYER_TIMING(l, attn_qk_dot, rdcycle() - _t0);

        _t0 = rdcycle();
        for (int h = 0; h < p->q_heads; h++) {
            float *as = s->att + (size_t)h * p->max_seq;
            softmax(as, pos + 1);
        }
        ADD_LAYER_TIMING(l, attn_softmax, rdcycle() - _t0);

        _t0 = rdcycle();
        for (int h = 0; h < p->q_heads; h++) {
            int kvh = h / q_per_kv;
            float *as = s->att + (size_t)h * p->max_seq;
            float *oh = ao + (size_t)h * p->head_dim;
            for (int t = 0; t <= pos; t++) {
                const float *vt = vc_head(s, p, l, kvh, t);
                float a = as[t];
                ATTN_AV_AXPY(oh, vt, a, p->head_dim);
            }
        }
        ADD_LAYER_TIMING(l, attn_av, rdcycle() - _t0);
        ADD_LAYER_TIMING(l, attn_score, rdcycle() - attn_t0);

        _t0 = rdcycle();
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
        _t1 = rdcycle();
        quantize_x(_mk_xq_buf, _mk_xs_buf, ao, p->q_dim, gs_qd);
        ADD_LAYER_TIMING(l, wo_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_prequant(s->xb2, w->wo_data[l], w->wo_scales[l],
                              p->hidden, p->q_dim, gs_qd, _mk_xq_buf, _mk_xs_buf);
        ADD_LAYER_TIMING(l, wo_ame, rdcycle() - _t1);
#else
        MATMUL_W4(s->xb2, w->wo_data[l], w->wo_scales[l], ao, p->hidden, p->q_dim, gs_qd);
#endif
        ADD_LAYER_TIMING(l, wo_matmul, rdcycle() - _t0);
        for (int i = 0; i < p->hidden; i++) s->x[i] += s->xb2[i];

        _t0 = rdcycle();
        rmsnorm(s->xb, s->x, w->rms_ffn_weight[l], p->hidden, p->rms_eps);
        ADD_LAYER_TIMING(l, ffn_norm, rdcycle() - _t0);

        _t0 = rdcycle();
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
        _t1 = rdcycle();
        quantize_x(_mk_xq_buf, _mk_xs_buf, s->xb, p->hidden, gs_h);
        ADD_LAYER_TIMING(l, ffn_gate_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_pair_prequant(s->hb, s->hb2,
                                   w->w1_data[l], w->w1_scales[l],
                                   w->w3_data[l], w->w3_scales[l],
                                   p->intermediate, p->hidden, gs_h,
                                   _mk_xq_buf, _mk_xs_buf);
        ADD_LAYER_TIMING(l, ffn_gate_ame, rdcycle() - _t1);
#else
        MATMUL_W4(s->hb, w->w1_data[l], w->w1_scales[l], s->xb, p->intermediate, p->hidden, gs_h);
        MATMUL_W4(s->hb2, w->w3_data[l], w->w3_scales[l], s->xb, p->intermediate, p->hidden, gs_h);
#endif
        _t1 = rdcycle();
        ADD_LAYER_TIMING(l, ffn_gate, _t1 - _t0);

        _t0 = rdcycle();
        for (int i = 0; i < p->intermediate; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        ADD_LAYER_TIMING(l, ffn_silu, rdcycle() - _t0);

        _t0 = rdcycle();
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
        _t1 = rdcycle();
        quantize_x(_mk_xq_buf, _mk_xs_buf, s->hb, p->intermediate, gs_inter);
        ADD_LAYER_TIMING(l, ffn_down_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_prequant(s->xb2, w->w2_data[l], w->w2_scales[l],
                              p->hidden, p->intermediate, gs_inter, _mk_xq_buf, _mk_xs_buf);
        ADD_LAYER_TIMING(l, ffn_down_ame, rdcycle() - _t1);
#else
        MATMUL_W4(s->xb2, w->w2_data[l], w->w2_scales[l], s->hb, p->hidden, p->intermediate, gs_inter);
#endif
        ADD_LAYER_TIMING(l, ffn_down, rdcycle() - _t0);
        for (int i = 0; i < p->hidden; i++) s->x[i] += s->xb2[i];

        g_timing.n_layers++;
    }

    _t0 = rdcycle();
    rmsnorm(s->x, s->x, w->rms_final_weight, p->hidden, p->rms_eps);
    g_timing.final_norm += rdcycle() - _t0;

    _t0 = rdcycle();
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
    _t1 = rdcycle();
    quantize_x(_mk_xq_buf, _mk_xs_buf, s->x, p->hidden, gs_h);
    g_timing.lm_head_quant += rdcycle() - _t1;
    _t1 = rdcycle();
    matmul_w4_mk_prequant(s->logits, w->lm_head_data, w->lm_head_scales,
                          p->vocab, p->hidden, gs_h, _mk_xq_buf, _mk_xs_buf);
    g_timing.lm_head_ame += rdcycle() - _t1;
#else
    MATMUL_W4(s->logits, w->lm_head_data, w->lm_head_scales, s->x, p->vocab, p->hidden, gs_h);
#endif
    g_timing.lm_head += rdcycle() - _t0;

    g_timing.n_calls++;
}

__attribute__((unused))
static int argmax(const float *x, int n) {
    int mi = 0; float mv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mv) { mv = x[i]; mi = i; }
    return mi;
}

#ifndef SAMPLE_TOP_K
#define SAMPLE_TOP_K 20
#endif
#ifndef SAMPLE_TEMPERATURE
#define SAMPLE_TEMPERATURE 0.6f
#endif
#ifndef SAMPLE_TOP_P
#define SAMPLE_TOP_P 0.95f
#endif
#ifndef SAMPLE_RNG_SEED
#define SAMPLE_RNG_SEED 0x12345678u
#endif
#ifndef SAMPLE_TOP_K_MAX
#define SAMPLE_TOP_K_MAX 64
#endif

static uint32_t xorshift32_next(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0u) x = SAMPLE_RNG_SEED ? SAMPLE_RNG_SEED : 0x6d2b79f5u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rng_uniform_0_1(uint32_t *state) {
    return (float)(xorshift32_next(state) >> 8) * (1.0f / 16777216.0f);
}

static int sample_topk(const float *logits,
                       int n,
                       int top_k,
                       float temperature,
                       float top_p,
                       uint32_t *rng_state) {
    if (n <= 0) return 0;
    if (temperature <= 0.0f || top_k <= 1) return argmax(logits, n);
    if (top_k > SAMPLE_TOP_K_MAX) top_k = SAMPLE_TOP_K_MAX;
    if (top_k > n) top_k = n;

    int top_ids[SAMPLE_TOP_K_MAX];
    float top_logits[SAMPLE_TOP_K_MAX];
    float probs[SAMPLE_TOP_K_MAX];
    int count = 0;

    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (count == top_k && v <= top_logits[count - 1]) continue;

        int pos = (count < top_k) ? count++ : (top_k - 1);
        while (pos > 0 && v > top_logits[pos - 1]) {
            top_logits[pos] = top_logits[pos - 1];
            top_ids[pos] = top_ids[pos - 1];
            pos--;
        }
        top_logits[pos] = v;
        top_ids[pos] = i;
    }

    float max_logit = top_logits[0];
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float p = my_expf((top_logits[i] - max_logit) / temperature);
        probs[i] = p;
        sum += p;
    }
    if (sum <= 0.0f) return top_ids[0];

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < count; i++) probs[i] *= inv_sum;

    int sample_count = count;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < count; i++) {
            cdf += probs[i];
            if (cdf >= top_p) {
                sample_count = i + 1;
                break;
            }
        }
        float kept_sum = 0.0f;
        for (int i = 0; i < sample_count; i++) kept_sum += probs[i];
        if (kept_sum > 0.0f) {
            float inv_kept = 1.0f / kept_sum;
            for (int i = 0; i < sample_count; i++) probs[i] *= inv_kept;
        }
    }

    float r = rng_uniform_0_1(rng_state);
    float cdf = 0.0f;
    for (int i = 0; i < sample_count; i++) {
        cdf += probs[i];
        if (r < cdf) return top_ids[i];
    }
    return top_ids[sample_count - 1];
}

// ============================================================
__attribute__((unused))
static inline void enable_accel_state(void) {
    uintptr_t mstatus;
    __asm__ volatile ("csrr %0, mstatus" : "=r"(mstatus));
    mstatus |= (uintptr_t)(3u << 9);   // VS[10:9]  = dirty (RVV)
    mstatus |= (uintptr_t)(3u << 13);  // FS[14:13] = dirty (FPU)
    mstatus |= (uintptr_t)(3u << 15);  // XS[16:15] = dirty (extensions)
    mstatus |= (uintptr_t)(3u << 23);  // MS[24:23] = dirty (Ztt gem5 CSR layout)
    __asm__ volatile ("csrw mstatus, %0" :: "r"(mstatus) : "memory");
}

uintptr_t handle_trap(uintptr_t mcause, uintptr_t mepc, uintptr_t *regs) {
    (void)regs;
    init_uart((uint32_t)FPGA_CLOCK_HZ, 115200);
    print_uart("TRAP mcause=0x");
    print_uart_hex((unsigned long long)mcause);
    print_uart(" mepc=0x");
    print_uart_hex((unsigned long long)mepc);
    print_uart(" cycle=");
    print_uart_int_dec(rdcycle());
    print_uart("\r\n");
    for (;;) { __asm__ volatile ("" ::: "memory"); }
    return mepc;
}

// ============================================================
// Prompt text (injected at compile time via -DPROMPT_TEXT=...)
// Default: "Hello"
// ============================================================
#ifndef PROMPT_TEXT
#define PROMPT_TEXT "Hello"
#endif
#ifndef GEN_STEPS
#define GEN_STEPS 30
#endif
#ifndef MAX_SEQ
#define MAX_SEQ 128
#endif
#ifndef GEM5_MINIMAL_PROMPT
#define GEM5_MINIMAL_PROMPT 0
#endif

#ifndef PREFILL_TRACE_TOKENS
#define PREFILL_TRACE_TOKENS 0
#endif

#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
typedef struct {
    float *x;
    float *xb;
    float *qkv;
    float *ao;
    float *xb2;
    float *hb;
    float *hb2;
    int8_t *xq;
    float *xs;
} PrefillBatchState;

static void build_prefill_batch_state(PrefillBatchState *bs,
                                      const Config *p,
                                      int T) {
    int qkv_dim = p->q_dim + 2 * p->kv_dim;
    int max_n = p->hidden;
    if (p->q_dim > max_n) max_n = p->q_dim;
    if (p->intermediate > max_n) max_n = p->intermediate;
    int min_gs = p->rowwise_scales ? max_n : p->group_size;
    int max_groups = (max_n + min_gs - 1) / min_gs;

    bs->x   = (float *)malloc((size_t)T * p->hidden * 4);
    bs->xb  = (float *)malloc((size_t)T * p->hidden * 4);
    bs->qkv = (float *)malloc((size_t)T * qkv_dim * 4);
    bs->ao  = (float *)malloc((size_t)T * p->q_dim * 4);
    bs->xb2 = (float *)malloc((size_t)T * p->hidden * 4);
    bs->hb  = (float *)malloc((size_t)T * p->intermediate * 4);
    bs->hb2 = (float *)malloc((size_t)T * p->intermediate * 4);
    bs->xq  = (int8_t *)malloc((size_t)T * max_n);
    bs->xs  = (float *)malloc((size_t)T * max_groups * 4);
}

static void prefill_all_tokens_batch_accel(const Config *cfg,
                                           const Weights *w,
                                           RunState *st,
                                           const int *tokens,
                                           int n_tokens,
                                           int start_pos) {
    int gs_h = cfg->rowwise_scales ? cfg->hidden : cfg->group_size;
    int gs_qd = cfg->rowwise_scales ? cfg->q_dim : cfg->group_size;
    int gs_inter = cfg->rowwise_scales ? cfg->intermediate : cfg->group_size;
    int ng_h = cfg->hidden / gs_h;
    int qkv_dim = cfg->q_dim + 2 * cfg->kv_dim;
    int q_per_kv = cfg->q_heads / cfg->kv_heads;
    const float hd_sqrt = my_sqrtf((float)cfg->head_dim);  // constant; hoisted out of attn loop
    uint64_t _t0, _t1;

    PrefillBatchState bs;
    build_prefill_batch_state(&bs, cfg, n_tokens);

    _t0 = rdcycle();
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        const float *sc = w->embed_scales + (size_t)token * ng_h;
        const uint8_t *ed = w->embed_data + (size_t)token * (cfg->hidden / 2);
        float *xrow = bs.x + (size_t)t * cfg->hidden;
        dequant_embed_row(xrow, ed, sc, cfg->hidden, gs_h);
    }
    g_timing.embed += rdcycle() - _t0;

    for (int l = 0; l < cfg->n_layers; l++) {
        _t0 = rdcycle();
        for (int t = 0; t < n_tokens; t++) {
            rmsnorm(bs.xb + (size_t)t * cfg->hidden,
                    bs.x + (size_t)t * cfg->hidden,
                    w->rms_att_weight[l], cfg->hidden, cfg->rms_eps);
        }
        ADD_LAYER_TIMING(l, attn_norm, rdcycle() - _t0);

        _t0 = rdcycle();
        _t1 = rdcycle();
        quantize_x_batch(bs.xq, bs.xs, bs.xb, n_tokens, cfg->hidden, gs_h);
        ADD_LAYER_TIMING(l, qkv_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_batch_prequant(bs.qkv, w->wqkv_data[l], w->wqkv_scales[l],
                                    n_tokens, qkv_dim, cfg->hidden, gs_h,
                                    bs.xq, bs.xs);
        ADD_LAYER_TIMING(l, qkv_ame, rdcycle() - _t1);
        ADD_LAYER_TIMING(l, qkv_matmul, rdcycle() - _t0);

        _t0 = rdcycle();
        for (int t = 0; t < n_tokens; t++) {
            int pos = start_pos + t;
            float *q = bs.qkv + (size_t)t * qkv_dim;
            float *k = q + cfg->q_dim;
            float *v = k + cfg->kv_dim;

            if (cfg->has_qk_norm) {
                for (int h = 0; h < cfg->q_heads; h++)
                    rmsnorm_inplace(q + (size_t)h * cfg->head_dim, w->q_norm[l], cfg->head_dim, cfg->rms_eps);
                for (int h = 0; h < cfg->kv_heads; h++)
                    rmsnorm_inplace(k + (size_t)h * cfg->head_dim, w->k_norm[l], cfg->head_dim, cfg->rms_eps);
            }
            for (int h = 0; h < cfg->q_heads; h++)
                rope_rotate(q + (size_t)h * cfg->head_dim, cfg->head_dim, pos, cfg->rope_theta);
            for (int h = 0; h < cfg->kv_heads; h++)
                rope_rotate(k + (size_t)h * cfg->head_dim, cfg->head_dim, pos, cfg->rope_theta);

            store_kv_cache(st, cfg, l, pos, k, v);
        }
        ADD_LAYER_TIMING(l, qknorm_rope, rdcycle() - _t0);

        uint64_t attn_t0 = rdcycle();
        memset(bs.ao, 0, (size_t)n_tokens * cfg->q_dim * 4);

        // Attention score is scalar code. It does not call MATMUL_Q8 or AME.
        uint64_t attn_qk_cycles = 0;
        uint64_t attn_softmax_cycles = 0;
        uint64_t attn_av_cycles = 0;
        for (int t = 0; t < n_tokens; t++) {
            int pos = start_pos + t;
            float *q = bs.qkv + (size_t)t * qkv_dim;
            float *ao = bs.ao + (size_t)t * cfg->q_dim;

            _t0 = rdcycle();
            for (int h = 0; h < cfg->q_heads; h++) {
                int kvh = h / q_per_kv;
                const float *qh = q + (size_t)h * cfg->head_dim;
                float *as = st->att + (size_t)h * cfg->max_seq;
                for (int past = 0; past <= pos; past++) {
                    const float *kt = kc_head(st, cfg, l, kvh, past);
                    float sc = QK_DOT(qh, kt, cfg->head_dim);
                    as[past] = (float)(sc / hd_sqrt);
                }
            }
            attn_qk_cycles += rdcycle() - _t0;

            _t0 = rdcycle();
            for (int h = 0; h < cfg->q_heads; h++) {
                float *as = st->att + (size_t)h * cfg->max_seq;
                softmax(as, pos + 1);
            }
            attn_softmax_cycles += rdcycle() - _t0;

            _t0 = rdcycle();
            for (int h = 0; h < cfg->q_heads; h++) {
                int kvh = h / q_per_kv;
                float *as = st->att + (size_t)h * cfg->max_seq;
                float *oh = ao + (size_t)h * cfg->head_dim;
                for (int past = 0; past <= pos; past++) {
                    const float *vt = vc_head(st, cfg, l, kvh, past);
                    float a = as[past];
                    ATTN_AV_AXPY(oh, vt, a, cfg->head_dim);
                }
            }
            attn_av_cycles += rdcycle() - _t0;
        }
        ADD_LAYER_TIMING(l, attn_qk_dot, attn_qk_cycles);
        ADD_LAYER_TIMING(l, attn_softmax, attn_softmax_cycles);
        ADD_LAYER_TIMING(l, attn_av, attn_av_cycles);
        ADD_LAYER_TIMING(l, attn_score, rdcycle() - attn_t0);

        _t0 = rdcycle();
        _t1 = rdcycle();
        quantize_x_batch(bs.xq, bs.xs, bs.ao, n_tokens, cfg->q_dim, gs_qd);
        ADD_LAYER_TIMING(l, wo_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_batch_prequant(bs.xb2, w->wo_data[l], w->wo_scales[l],
                                    n_tokens, cfg->hidden, cfg->q_dim, gs_qd,
                                    bs.xq, bs.xs);
        ADD_LAYER_TIMING(l, wo_ame, rdcycle() - _t1);
        ADD_LAYER_TIMING(l, wo_matmul, rdcycle() - _t0);
        for (int t = 0; t < n_tokens; t++) {
            float *xrow = bs.x + (size_t)t * cfg->hidden;
            float *rrow = bs.xb2 + (size_t)t * cfg->hidden;
            for (int i = 0; i < cfg->hidden; i++) xrow[i] += rrow[i];
        }

        _t0 = rdcycle();
        for (int t = 0; t < n_tokens; t++) {
            rmsnorm(bs.xb + (size_t)t * cfg->hidden,
                    bs.x + (size_t)t * cfg->hidden,
                    w->rms_ffn_weight[l], cfg->hidden, cfg->rms_eps);
        }
        ADD_LAYER_TIMING(l, ffn_norm, rdcycle() - _t0);

        _t0 = rdcycle();
        _t1 = rdcycle();
        quantize_x_batch(bs.xq, bs.xs, bs.xb, n_tokens, cfg->hidden, gs_h);
        ADD_LAYER_TIMING(l, ffn_gate_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_batch_pair_prequant(bs.hb, bs.hb2,
                                         w->w1_data[l], w->w1_scales[l],
                                         w->w3_data[l], w->w3_scales[l],
                                         n_tokens, cfg->intermediate, cfg->hidden, gs_h,
                                         bs.xq, bs.xs);
        ADD_LAYER_TIMING(l, ffn_gate_ame, rdcycle() - _t1);
        _t1 = rdcycle();
        ADD_LAYER_TIMING(l, ffn_gate, _t1 - _t0);

        _t0 = rdcycle();
        for (int i = 0; i < n_tokens * cfg->intermediate; i++)
            bs.hb[i] = silu(bs.hb[i]) * bs.hb2[i];
        ADD_LAYER_TIMING(l, ffn_silu, rdcycle() - _t0);

        _t0 = rdcycle();
        _t1 = rdcycle();
        quantize_x_batch(bs.xq, bs.xs, bs.hb, n_tokens, cfg->intermediate, gs_inter);
        ADD_LAYER_TIMING(l, ffn_down_quant, rdcycle() - _t1);
        _t1 = rdcycle();
        matmul_w4_mk_batch_prequant(bs.xb2, w->w2_data[l], w->w2_scales[l],
                                    n_tokens, cfg->hidden, cfg->intermediate, gs_inter,
                                    bs.xq, bs.xs);
        ADD_LAYER_TIMING(l, ffn_down_ame, rdcycle() - _t1);
        ADD_LAYER_TIMING(l, ffn_down, rdcycle() - _t0);
        for (int t = 0; t < n_tokens; t++) {
            float *xrow = bs.x + (size_t)t * cfg->hidden;
            float *rrow = bs.xb2 + (size_t)t * cfg->hidden;
            for (int i = 0; i < cfg->hidden; i++) xrow[i] += rrow[i];
        }

        g_timing.n_layers += n_tokens;
    }

    _t0 = rdcycle();
    float *last_x = bs.x + (size_t)(n_tokens - 1) * cfg->hidden;
    rmsnorm(last_x, last_x, w->rms_final_weight, cfg->hidden, cfg->rms_eps);
    prefill_memcpy_f32_r4(st->x, last_x, cfg->hidden);
    g_timing.final_norm += rdcycle() - _t0;

    _t0 = rdcycle();
    _t1 = rdcycle();
    quantize_x_batch(bs.xq, bs.xs, st->x, 1, cfg->hidden, gs_h);
    g_timing.lm_head_quant += rdcycle() - _t1;
    _t1 = rdcycle();
    matmul_w4_mk_batch_prequant(st->logits, w->lm_head_data, w->lm_head_scales,
                                1, cfg->vocab, cfg->hidden, gs_h,
                                bs.xq, bs.xs);
    g_timing.lm_head_ame += rdcycle() - _t1;
    g_timing.lm_head += rdcycle() - _t0;
    g_timing.n_calls += n_tokens;
}
#endif

__attribute__((unused))
static void prefill_tokens_at(const Config *cfg,
                              const Weights *w,
                              RunState *st,
                              const int *tokens,
                              int n_tokens,
                              int start_pos) {
    if (n_tokens <= 0) {
        print_uart("prefill_all: empty input\r\n");
        return;
    }
    if (start_pos < 0 || start_pos + n_tokens > cfg->max_seq) {
        print_uart("prefill_all: n_tokens exceeds max_seq\r\n");
        exit(1);
    }

#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
    prefill_all_tokens_batch_accel(cfg, w, st, tokens, n_tokens, start_pos);
#else
    for (int i = 0; i < n_tokens; i++) {
#if PREFILL_TRACE_TOKENS
        uint64_t t_tok = rdcycle();
        print_uart("  [prefill_all ");
        print_uart_int_dec(i + 1);
        print_uart("/");
        print_uart_int_dec(n_tokens);
        print_uart("] tok=");
        print_uart_int_dec(tokens[i]);
        print_uart("\r\n");
#endif
        forward(cfg, w, st, tokens[i], start_pos + i);
#if PREFILL_TRACE_TOKENS
        print_uart("  [prefill_all ");
        print_uart_int_dec(i + 1);
        print_uart("/");
        print_uart_int_dec(n_tokens);
        print_uart("] cyc=");
        print_uart_int_dec(rdcycle() - t_tok);
        print_uart("\r\n");
#endif
    }
#endif
}

__attribute__((unused))
static void prefill_all_tokens(const Config *cfg,
                               const Weights *w,
                               RunState *st,
                               const int *tokens,
                               int n_tokens) {
    prefill_tokens_at(cfg, w, st, tokens, n_tokens, 0);
}

#ifndef CHAT_BUILD
enum Gem5ProfilePhase {
    GEM5_PHASE_TOKENIZER_LOAD = 1,
    GEM5_PHASE_PROMPT_TOKENIZE = 2,
    GEM5_PHASE_MODEL_LOAD = 3,
    GEM5_PHASE_STATE_BUILD = 4,
    GEM5_PHASE_PREFILL = 5,
    GEM5_PHASE_GENERATE = 6,
    GEM5_PHASE_OUTPUT = 7,
};

static inline void
gem5_phase_begin(enum Gem5ProfilePhase phase)
{
    m5_work_begin((uint64_t)phase, 0);
}

static inline void
gem5_phase_end(enum Gem5ProfilePhase phase)
{
    m5_work_end((uint64_t)phase, 0);
}

int main(void) {
    uint64_t t_total_start = rdcycle();
    uint64_t t_stage = 0;
    init_uart((uint32_t)FPGA_CLOCK_HZ, 115200);
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
    enable_accel_state();
#endif
#if defined(USE_AME) && USE_AME
    ame_matmul_stats_reset();
#elif defined(USE_RVV) && USE_RVV
    rvv_tile_matmul_stats_reset();
#endif
    print_uart("build: "); print_uart(BUILD_TIMESTAMP); print_uart("\r\n");
    print_uart("=== Qwen3-0.6B W4A8+"); print_uart(MATMUL_NAME); print_uart(" ===\r\n");

    heap_ptr = (char *)heap_start_addr();

    // [1] Load tokenizer from embedded blob
    print_uart("[1] load tokenizer...\r\n");
    gem5_phase_begin(GEM5_PHASE_TOKENIZER_LOAD);
    t_stage = rdcycle();
    BareTokenizer *tok = tokenizer_load_from_mem(_tokenizer_blob_start);
    if (!tok) { print_uart("tokenizer load failed\r\n"); exit(1); }
    gem5_phase_end(GEM5_PHASE_TOKENIZER_LOAD);
    print_uart("    vocab_size="); print_uart_int_dec(tok->vocab_size); print_uart("\r\n");
    print_cycles("    tokenizer load: ", rdcycle() - t_stage);

    // [2] Tokenize prompt with chat template
    gem5_phase_begin(GEM5_PHASE_PROMPT_TOKENIZE);
#ifdef QUICK_LOGIT
    // Single-token mode: just BOS, one forward pass, print logits, exit.
    // Use for fast cross-backend numerical comparison.
    (void)tok;
    #define MAX_TOKENS 1
    int tokens[MAX_TOKENS];
    int n_tokens = 1;
    tokens[0] = 151643;  // BOS only
    print_uart("[2] QUICK_LOGIT: single token 151643\r\n");
#else
    print_uart("[2] tokenize: \""); print_uart(PROMPT_TEXT); print_uart("\"\r\n");
    t_stage = rdcycle();
#if GEM5_MINIMAL_PROMPT
    // Cycle-accurate validation mode: tokenize the raw prompt without adding
    // the 21-token chat template. All model layers and generation still run.
    #define MAX_TOKENS 1024
    int tokens[MAX_TOKENS];
    int n_tokens = tokenizer_encode_bare(tok, PROMPT_TEXT, tokens, MAX_TOKENS);
#else
    // BOS + system identity + user prompt + assistant header + empty thinking span
    #define MAX_TOKENS 1024
    int tokens[MAX_TOKENS];
    int n_tokens = 0;
    tokens[n_tokens++] = 151643;  // BOS
    n_tokens = append_identity_system_tokens(tok, tokens, n_tokens, MAX_TOKENS);
    tokens[n_tokens++] = IM_START_ID;
    tokens[n_tokens++] = USER_ID;
    tokens[n_tokens++] = NEWLINE_ID;
    int enc_count = tokenizer_encode_bare(tok, PROMPT_TEXT, tokens + n_tokens,
                                          MAX_TOKENS - n_tokens - 5 - NO_THINKING_PROMPT_TOKENS);//将PROMPT_TEXT编码为token id并存储到tokens数组中
    n_tokens += enc_count;
    tokens[n_tokens++] = IM_END_ID;
    tokens[n_tokens++] = NEWLINE_ID;
    tokens[n_tokens++] = IM_START_ID;
    tokens[n_tokens++] = ASSISTANT_ID;
    tokens[n_tokens++] = NEWLINE_ID;
    n_tokens = append_no_thinking_prompt_tokens(tokens, n_tokens, MAX_TOKENS);
#endif

    print_uart("    "); print_uart_int_dec(n_tokens); print_uart(" tokens:");
    for (int i = 0; i < n_tokens; i++) {
        print_uart(" "); 
        print_uart_int_dec(tokens[i]);
    }
    print_uart("\r\n");
    print_cycles("    tokenize: ", rdcycle() - t_stage);
#endif /* QUICK_LOGIT */
    gem5_phase_end(GEM5_PHASE_PROMPT_TOKENIZE);

    // [3] Load model
    print_uart("[3] load model...\r\n");
    Config cfg;
    Weights w;
    gem5_phase_begin(GEM5_PHASE_MODEL_LOAD);
    t_stage = rdcycle();
    load_model(MODEL_ADDR, &cfg, &w);//从内嵌 W4A8 blob header 解析，给 Weights 赋指针（零拷贝）
    gem5_phase_end(GEM5_PHASE_MODEL_LOAD);
    print_cycles("    model load: ", rdcycle() - t_stage);
    cfg.max_seq = MAX_SEQ;

    RunState st;
    gem5_phase_begin(GEM5_PHASE_STATE_BUILD);
    t_stage = rdcycle();
    build_state(&st, &cfg);
    gem5_phase_end(GEM5_PHASE_STATE_BUILD);
    print_cycles("    state build: ", rdcycle() - t_stage);

    // [4] Prefill the full prompt in one entrypoint. The current forward path
    // remains causal and sequential internally because every position consumes
    // the KV cache produced by previous positions.
    print_uart("[4] prefill_all "); print_uart_int_dec(n_tokens); print_uart(" tokens...\r\n");
    gem5_phase_begin(GEM5_PHASE_PREFILL);
    uint64_t t_prefill_start = rdcycle();
    prefill_all_tokens(&cfg, &w, &st, tokens, n_tokens);
    gem5_phase_end(GEM5_PHASE_PREFILL);
    uint64_t pf_wall = rdcycle() - t_prefill_start;
    print_cycles("    prefill_all done: ", pf_wall);

    // --- Profiling snapshot at the prefill/decode boundary (L1+L2) ---
    // Per-phase timing and backend sub-counters are accumulated
    // across both stages; snapshot here so decode = final - prefill.
    FwdTiming pf_timing = g_timing;
    FwdTiming pf_layer_sel = g_layer_sel;   /* single-layer prefill snapshot */
#if defined(USE_AME) && USE_AME
    AmeMixOfflineProfile pf_ame_profile;
    ame_matmul_profile_snapshot(&pf_ame_profile);
#elif defined(USE_RVV) && USE_RVV
    const uint64_t pf_rvv_calls = rvv_tile_matmul_stats_calls();
    const uint64_t pf_rvv_macs = rvv_tile_matmul_stats_macs();
    const uint64_t pf_rvv_row_groups = rvv_tile_matmul_stats_row_groups();
    const uint64_t pf_rvv_chunks = rvv_tile_matmul_stats_vector_chunks();
#endif

    // [5] Generate
#ifdef QUICK_LOGIT
    int steps = 0;   // skip generation entirely
#else
    int steps = GEN_STEPS;
#endif
    int EOS_ID = 151645;
    int IM_END_ID = 151645;
    int pos = n_tokens - 1;
    int gen_ids[1024];
    int n_gen = 0;
    int n_out = 0;
    uint32_t rng_state = SAMPLE_RNG_SEED ^ (uint32_t)n_tokens;

    print_uart("[5] generate "); print_uart_int_dec(steps); print_uart(" tokens...\r\n");
    gem5_phase_begin(GEM5_PHASE_GENERATE);
    uint64_t t_gen_start = rdcycle();
    for (int s = 0; s < steps; s++) {
        uint64_t t_tok = rdcycle();
        int next = sample_topk(st.logits, cfg.vocab,
                               SAMPLE_TOP_K, SAMPLE_TEMPERATURE,
                               SAMPLE_TOP_P, &rng_state);
        if (next == EOS_ID || next == IM_END_ID) break;
        if (!is_hidden_output_token(tok, next)) gen_ids[n_out++] = next;
        n_gen++;
        pos++;
        if (pos >= cfg.max_seq) break;
        if (s + 1 < steps)
            forward(&cfg, &w, &st, next, pos);
        uint64_t t_tok_cyc = rdcycle() - t_tok;
        print_uart("  [gen "); print_uart_int_dec(s + 1);
        print_uart("/"); print_uart_int_dec(steps);
        print_uart("] tok="); print_uart_int_dec(next);
        print_uart(" pos="); print_uart_int_dec(pos);
        print_uart(" cyc="); print_uart_int_dec(t_tok_cyc);
        print_uart("\r\n");
    }
    uint64_t gen_wall = rdcycle() - t_gen_start;
    print_cycles("    generate done: ", gen_wall);

    // [verify] top-5 logits from last forward (for cross-backend comparison)
    {
        print_uart("[verify] top5 logits:\r\n");
        float masked[5];
        int   top[5];
        // partial selection: each pass finds the current max, then masks it
        for (int t = 0; t < 5; t++) {
            int best = 0;
            for (int i = 1; i < cfg.vocab; i++) {
                if (st.logits[i] > st.logits[best]) best = i;
            }
            top[t]    = best;
            masked[t] = st.logits[best];
            st.logits[best] = -1e38f;   // mask for next pass
        }
        // restore and print
        for (int t = 0; t < 5; t++) {
            st.logits[top[t]] = masked[t];
            print_uart("  rank"); print_uart_int_dec(t + 1);
            print_uart(" tok="); print_uart_int_dec(top[t]);
            print_uart(" logit="); print_float_fixed3(masked[t]);
            print_uart("\r\n");
        }
    }
    gem5_phase_end(GEM5_PHASE_GENERATE);

    // [6] Decode and output
    print_uart("[6] output ("); print_uart_int_dec(n_out); print_uart(" tokens):\r\n");
    gem5_phase_begin(GEM5_PHASE_OUTPUT);
    t_stage = rdcycle();
    SafeUtf8State safe_out = {{0, 0, 0, 0}, 0, 0};
    for (int i = 0; i < n_out; i++) {
        tokenizer_decode_uart_safe(tok, gen_ids[i], &safe_out);
        if (((i + 1) & 15) == 0) print_uart("\r\n");
    }
    print_uart("\r\n");
    gem5_phase_end(GEM5_PHASE_OUTPUT);
    print_cycles("    decode: ", rdcycle() - t_stage);

    // Also print raw token IDs
    print_uart("[ids]");
    for (int i = 0; i < n_out; i++) {
        print_uart(" "); print_uart_int_dec(gen_ids[i]);
    }
    print_uart("\r\n");

    // --------------------------------------------------------
    // Timing breakdown summary
    // --------------------------------------------------------
    uint64_t total_cyc = rdcycle() - t_total_start;
    print_uart("\r\n=== Timing Breakdown ===\r\n");
    print_uart("calls="); print_uart_int_dec(g_timing.n_calls);
    print_uart("  layer_passes="); print_uart_int_dec(g_timing.n_layers);
    print_uart("\r\n");

    // Helper macro: print label, cycles, percent
    #define PRINT_PHASE(label, cyc) do { \
        uint64_t _c = (cyc); \
        print_uart("  "); print_uart(label); \
        print_uart_int_dec(_c); \
        print_uart(" cyc ("); \
        /* integer percent with one decimal: (c*1000)/total */ \
        uint64_t _pct = (_c <= total_cyc && total_cyc > 0) ? (_c * 1000 / total_cyc) : 0; \
        print_uart_int_dec(_pct / 10); print_uart("."); print_uart_int_dec(_pct % 10); \
        print_uart("%)\r\n"); \
    } while(0)

    PRINT_PHASE("embed        : ", g_timing.embed);
    PRINT_PHASE("attn_norm    : ", g_timing.attn_norm);
    PRINT_PHASE("qkv_matmul   : ", g_timing.qkv_matmul);
    PRINT_PHASE("  quantize_x : ", g_timing.qkv_quant);
    PRINT_PHASE("  backend_call: ", g_timing.qkv_ame);
    PRINT_PHASE("  wrap+ovhd  : ", two_child_other_cycles(g_timing.qkv_matmul, g_timing.qkv_quant, g_timing.qkv_ame));
    PRINT_PHASE("qknorm+rope  : ", g_timing.qknorm_rope);
    PRINT_PHASE("attn_score   : ", g_timing.attn_score);
    print_uart("    detail: RVV QK/AV + scalar softmax\r\n");
    PRINT_PHASE("  attn_qk_dot: ", g_timing.attn_qk_dot);
    PRINT_PHASE("  softmax    : ", g_timing.attn_softmax);
    PRINT_PHASE("  attn_av_dot: ", g_timing.attn_av);
    PRINT_PHASE("  clear+ovhd : ",
                attn_score_other_cycles(g_timing.attn_score,
                                        g_timing.attn_qk_dot,
                                        g_timing.attn_softmax,
                                        g_timing.attn_av));
    PRINT_PHASE("wo_matmul    : ", g_timing.wo_matmul);
    PRINT_PHASE("  quantize_x : ", g_timing.wo_quant);
    PRINT_PHASE("  backend_call: ", g_timing.wo_ame);
    PRINT_PHASE("  wrap+ovhd  : ", two_child_other_cycles(g_timing.wo_matmul, g_timing.wo_quant, g_timing.wo_ame));
    PRINT_PHASE("ffn_norm     : ", g_timing.ffn_norm);
    PRINT_PHASE("ffn_gate(w1w3): ", g_timing.ffn_gate);
    PRINT_PHASE("  quantize_x : ", g_timing.ffn_gate_quant);
    PRINT_PHASE("  backend_call: ", g_timing.ffn_gate_ame);
    PRINT_PHASE("  wrap+ovhd  : ", two_child_other_cycles(g_timing.ffn_gate, g_timing.ffn_gate_quant, g_timing.ffn_gate_ame));
    PRINT_PHASE("ffn_silu     : ", g_timing.ffn_silu);
    PRINT_PHASE("ffn_down(w2) : ", g_timing.ffn_down);
    PRINT_PHASE("  quantize_x : ", g_timing.ffn_down_quant);
    PRINT_PHASE("  backend_call: ", g_timing.ffn_down_ame);
    PRINT_PHASE("  wrap+ovhd  : ", two_child_other_cycles(g_timing.ffn_down, g_timing.ffn_down_quant, g_timing.ffn_down_ame));
    PRINT_PHASE("final_norm   : ", g_timing.final_norm);
    PRINT_PHASE("lm_head      : ", g_timing.lm_head);
    PRINT_PHASE("  quantize_x : ", g_timing.lm_head_quant);
    PRINT_PHASE("  backend_call: ", g_timing.lm_head_ame);
    PRINT_PHASE("  wrap+ovhd  : ", two_child_other_cycles(g_timing.lm_head, g_timing.lm_head_quant, g_timing.lm_head_ame));
    uint64_t accounted = g_timing.embed + g_timing.attn_norm + g_timing.qkv_matmul +
        g_timing.qknorm_rope + g_timing.attn_score + g_timing.wo_matmul +
        g_timing.ffn_norm + g_timing.ffn_gate + g_timing.ffn_silu + g_timing.ffn_down +
        g_timing.final_norm + g_timing.lm_head;
    PRINT_PHASE("(other/load) : ", total_cyc > accounted ? total_cyc - accounted : 0);
    print_uart("  total      : "); print_uart_int_dec(total_cyc); print_uart(" cyc\r\n");

#if defined(USE_AME) && USE_AME
    AmeMixOfflineProfile total_ame_profile;
    ame_matmul_profile_snapshot(&total_ame_profile);
    AmeMixOfflineProfile dec_ame_profile = ame_profile_delta(&total_ame_profile, &pf_ame_profile);
#elif defined(USE_RVV) && USE_RVV
    const uint64_t total_rvv_calls = rvv_tile_matmul_stats_calls();
    const uint64_t total_rvv_macs = rvv_tile_matmul_stats_macs();
    const uint64_t total_rvv_row_groups = rvv_tile_matmul_stats_row_groups();
    const uint64_t total_rvv_chunks = rvv_tile_matmul_stats_vector_chunks();
#endif

    // ========================================================
    // Profiling split: PREFILL vs DECODE (L1) + backend phases (L2)
    //   prefill = snapshot taken at the boundary
    //   decode  = final - snapshot
    // ========================================================
    #define PRINT_PH2(label, cyc, tot) do { \
        uint64_t _c = (cyc); uint64_t _t = (tot); \
        print_uart("  "); print_uart(label); print_uart_int_dec(_c); print_uart(" cyc ("); \
        uint64_t _p = (_t > 0) ? (_c * 1000ULL / _t) : 0; \
        print_uart_int_dec(_p / 10); print_uart("."); print_uart_int_dec(_p % 10); \
        print_uart("%)\r\n"); \
    } while (0)
    #define DECODE_DELTA(field) (g_timing.field - pf_timing.field)

    print_uart("\r\n=== Profiling PREFILL (wall="); print_uart_int_dec(pf_wall);
    print_uart(" cyc, tokens="); print_uart_int_dec(n_tokens); print_uart(") ===\r\n");
    PRINT_PH2("embed        : ", pf_timing.embed,       pf_wall);
    PRINT_PH2("attn_norm    : ", pf_timing.attn_norm,   pf_wall);
    PRINT_PH2("qkv_matmul   : ", pf_timing.qkv_matmul,  pf_wall);
    PRINT_PH2("  quantize_x : ", pf_timing.qkv_quant,   pf_wall);
    PRINT_PH2("  backend_call: ", pf_timing.qkv_ame,     pf_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(pf_timing.qkv_matmul, pf_timing.qkv_quant, pf_timing.qkv_ame), pf_wall);
    PRINT_PH2("qknorm+rope  : ", pf_timing.qknorm_rope, pf_wall);
    PRINT_PH2("attn_score   : ", pf_timing.attn_score,  pf_wall);
    PRINT_PH2("  attn_qk_dot: ", pf_timing.attn_qk_dot, pf_wall);
    PRINT_PH2("  softmax    : ", pf_timing.attn_softmax,pf_wall);
    PRINT_PH2("  attn_av_dot: ", pf_timing.attn_av,     pf_wall);
    PRINT_PH2("  clear+ovhd : ",
              attn_score_other_cycles(pf_timing.attn_score,
                                      pf_timing.attn_qk_dot,
                                      pf_timing.attn_softmax,
                                      pf_timing.attn_av),
              pf_wall);
    PRINT_PH2("wo_matmul    : ", pf_timing.wo_matmul,   pf_wall);
    PRINT_PH2("  quantize_x : ", pf_timing.wo_quant,    pf_wall);
    PRINT_PH2("  backend_call: ", pf_timing.wo_ame,      pf_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(pf_timing.wo_matmul, pf_timing.wo_quant, pf_timing.wo_ame), pf_wall);
    PRINT_PH2("ffn_norm     : ", pf_timing.ffn_norm,    pf_wall);
    PRINT_PH2("ffn_gate     : ", pf_timing.ffn_gate,    pf_wall);
    PRINT_PH2("  quantize_x : ", pf_timing.ffn_gate_quant, pf_wall);
    PRINT_PH2("  backend_call: ", pf_timing.ffn_gate_ame,   pf_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(pf_timing.ffn_gate, pf_timing.ffn_gate_quant, pf_timing.ffn_gate_ame), pf_wall);
    PRINT_PH2("ffn_silu     : ", pf_timing.ffn_silu,    pf_wall);
    PRINT_PH2("ffn_down     : ", pf_timing.ffn_down,    pf_wall);
    PRINT_PH2("  quantize_x : ", pf_timing.ffn_down_quant, pf_wall);
    PRINT_PH2("  backend_call: ", pf_timing.ffn_down_ame,   pf_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(pf_timing.ffn_down, pf_timing.ffn_down_quant, pf_timing.ffn_down_ame), pf_wall);
    PRINT_PH2("final_norm   : ", pf_timing.final_norm,  pf_wall);
    PRINT_PH2("lm_head      : ", pf_timing.lm_head,     pf_wall);
    PRINT_PH2("  quantize_x : ", pf_timing.lm_head_quant, pf_wall);
    PRINT_PH2("  backend_call: ", pf_timing.lm_head_ame,   pf_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(pf_timing.lm_head, pf_timing.lm_head_quant, pf_timing.lm_head_ame), pf_wall);

    print_uart("\r\n=== Profiling DECODE (wall="); print_uart_int_dec(gen_wall);
    print_uart(" cyc, tokens="); print_uart_int_dec((uint64_t)n_gen); print_uart(") ===\r\n");
    PRINT_PH2("embed        : ", DECODE_DELTA(embed),       gen_wall);
    PRINT_PH2("attn_norm    : ", DECODE_DELTA(attn_norm),   gen_wall);
    PRINT_PH2("qkv_matmul   : ", DECODE_DELTA(qkv_matmul),  gen_wall);
    PRINT_PH2("  quantize_x : ", DECODE_DELTA(qkv_quant),   gen_wall);
    PRINT_PH2("  backend_call: ", DECODE_DELTA(qkv_ame),     gen_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(DECODE_DELTA(qkv_matmul), DECODE_DELTA(qkv_quant), DECODE_DELTA(qkv_ame)), gen_wall);
    PRINT_PH2("qknorm+rope  : ", DECODE_DELTA(qknorm_rope), gen_wall);
    PRINT_PH2("attn_score   : ", DECODE_DELTA(attn_score),  gen_wall);
    PRINT_PH2("  attn_qk_dot: ", DECODE_DELTA(attn_qk_dot), gen_wall);
    PRINT_PH2("  softmax    : ", DECODE_DELTA(attn_softmax),gen_wall);
    PRINT_PH2("  attn_av_dot: ", DECODE_DELTA(attn_av),     gen_wall);
    PRINT_PH2("  clear+ovhd : ",
              attn_score_other_cycles(DECODE_DELTA(attn_score),
                                      DECODE_DELTA(attn_qk_dot),
                                      DECODE_DELTA(attn_softmax),
                                      DECODE_DELTA(attn_av)),
              gen_wall);
    PRINT_PH2("wo_matmul    : ", DECODE_DELTA(wo_matmul),   gen_wall);
    PRINT_PH2("  quantize_x : ", DECODE_DELTA(wo_quant),    gen_wall);
    PRINT_PH2("  backend_call: ", DECODE_DELTA(wo_ame),      gen_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(DECODE_DELTA(wo_matmul), DECODE_DELTA(wo_quant), DECODE_DELTA(wo_ame)), gen_wall);
    PRINT_PH2("ffn_norm     : ", DECODE_DELTA(ffn_norm),    gen_wall);
    PRINT_PH2("ffn_gate     : ", DECODE_DELTA(ffn_gate),    gen_wall);
    PRINT_PH2("  quantize_x : ", DECODE_DELTA(ffn_gate_quant), gen_wall);
    PRINT_PH2("  backend_call: ", DECODE_DELTA(ffn_gate_ame),   gen_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(DECODE_DELTA(ffn_gate), DECODE_DELTA(ffn_gate_quant), DECODE_DELTA(ffn_gate_ame)), gen_wall);
    PRINT_PH2("ffn_silu     : ", DECODE_DELTA(ffn_silu),    gen_wall);
    PRINT_PH2("ffn_down     : ", DECODE_DELTA(ffn_down),    gen_wall);
    PRINT_PH2("  quantize_x : ", DECODE_DELTA(ffn_down_quant), gen_wall);
    PRINT_PH2("  backend_call: ", DECODE_DELTA(ffn_down_ame),   gen_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(DECODE_DELTA(ffn_down), DECODE_DELTA(ffn_down_quant), DECODE_DELTA(ffn_down_ame)), gen_wall);
    PRINT_PH2("final_norm   : ", DECODE_DELTA(final_norm),  gen_wall);
    PRINT_PH2("lm_head      : ", DECODE_DELTA(lm_head),     gen_wall);
    PRINT_PH2("  quantize_x : ", DECODE_DELTA(lm_head_quant), gen_wall);
    PRINT_PH2("  backend_call: ", DECODE_DELTA(lm_head_ame),   gen_wall);
    PRINT_PH2("  wrap+ovhd  : ", two_child_other_cycles(DECODE_DELTA(lm_head), DECODE_DELTA(lm_head_quant), DECODE_DELTA(lm_head_ame)), gen_wall);
    #undef PRINT_PH2
    #undef DECODE_DELTA

#if defined(USE_AME) && USE_AME
    print_ame_matmul_l2_breakdown("PREFILL", &pf_ame_profile);
    print_ame_matmul_l2_breakdown("DECODE", &dec_ame_profile);
    print_ame_matmul_l2_breakdown("OVERALL", &total_ame_profile);
#elif defined(USE_RVV) && USE_RVV
    print_uart("\r\n=== RVV matmul counters PREFILL ===\r\n");
    print_uart("  calls         : "); print_uart_int_dec(pf_rvv_calls); print_uart("\r\n");
    print_uart("  MACs          : "); print_uart_int_dec(pf_rvv_macs); print_uart("\r\n");
    print_uart("  row-groups    : "); print_uart_int_dec(pf_rvv_row_groups); print_uart("\r\n");
    print_uart("  vector chunks : "); print_uart_int_dec(pf_rvv_chunks); print_uart("\r\n");
    print_uart("=== RVV matmul counters DECODE ===\r\n");
    print_uart("  calls         : "); print_uart_int_dec(total_rvv_calls - pf_rvv_calls); print_uart("\r\n");
    print_uart("  MACs          : "); print_uart_int_dec(total_rvv_macs - pf_rvv_macs); print_uart("\r\n");
    print_uart("  row-groups    : "); print_uart_int_dec(total_rvv_row_groups - pf_rvv_row_groups); print_uart("\r\n");
    print_uart("  vector chunks : "); print_uart_int_dec(total_rvv_chunks - pf_rvv_chunks); print_uart("\r\n");
    print_uart("=== RVV matmul counters OVERALL ===\r\n");
    print_uart("  calls         : "); print_uart_int_dec(total_rvv_calls); print_uart("\r\n");
    print_uart("  MACs          : "); print_uart_int_dec(total_rvv_macs); print_uart("\r\n");
    print_uart("  row-groups    : "); print_uart_int_dec(total_rvv_row_groups); print_uart("\r\n");
    print_uart("  vector chunks : "); print_uart_int_dec(total_rvv_chunks); print_uart("\r\n");
#endif

    // --------------------------------------------------------
    // Single transformer layer [PROFILE_LAYER] internal breakdown.
    // pf_layer_sel = that layer during prefill (one batch of n_tokens);
    // g_layer_sel - pf_layer_sel = that layer summed over the decode tokens.
    // Percentages are within the layer (denominator = layer component sum),
    // so each line is the share of that one layer's time.
    // --------------------------------------------------------
    #define LPF(f)  (pf_layer_sel.f)
    #define LDEC(f) (g_layer_sel.f - pf_layer_sel.f)
    #define LSUM(M) (M(attn_norm)+M(qkv_matmul)+M(qknorm_rope)+M(attn_score)+ \
                     M(wo_matmul)+M(ffn_norm)+M(ffn_gate)+M(ffn_silu)+M(ffn_down))
    #define PRINT_L(label, cyc, tot) do { \
        uint64_t _c=(cyc), _t=(tot); \
        print_uart("  "); print_uart(label); print_uart_int_dec(_c); print_uart(" cyc ("); \
        uint64_t _p=(_t>0)?(_c*1000ULL/_t):0; \
        print_uart_int_dec(_p/10); print_uart("."); print_uart_int_dec(_p%10); print_uart("%)\r\n"); \
    } while (0)
    {
        uint64_t lpf = LSUM(LPF), ldec = LSUM(LDEC);
        print_uart("\r\n=== Profiling PREFILL layer["); print_uart_int_dec(PROFILE_LAYER);
        print_uart("] (layer_total="); print_uart_int_dec(lpf); print_uart(" cyc, "); print_uart_int_dec(n_tokens); print_uart(" tok) ===\r\n");
        PRINT_L("attn_norm  : ", LPF(attn_norm),   lpf);
        PRINT_L("qkv_matmul : ", LPF(qkv_matmul),  lpf);
        PRINT_L("  quantize : ", LPF(qkv_quant),   lpf);
        PRINT_L("  backend  : ", LPF(qkv_ame),     lpf);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LPF(qkv_matmul), LPF(qkv_quant), LPF(qkv_ame)), lpf);
        PRINT_L("qknorm+rope: ", LPF(qknorm_rope), lpf);
        PRINT_L("attn_score : ", LPF(attn_score),  lpf);
        PRINT_L("wo_matmul  : ", LPF(wo_matmul),   lpf);
        PRINT_L("  quantize : ", LPF(wo_quant),    lpf);
        PRINT_L("  backend  : ", LPF(wo_ame),      lpf);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LPF(wo_matmul), LPF(wo_quant), LPF(wo_ame)), lpf);
        PRINT_L("ffn_norm   : ", LPF(ffn_norm),    lpf);
        PRINT_L("ffn_gate   : ", LPF(ffn_gate),    lpf);
        PRINT_L("  quantize : ", LPF(ffn_gate_quant), lpf);
        PRINT_L("  backend  : ", LPF(ffn_gate_ame),   lpf);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LPF(ffn_gate), LPF(ffn_gate_quant), LPF(ffn_gate_ame)), lpf);
        PRINT_L("ffn_silu   : ", LPF(ffn_silu),    lpf);
        PRINT_L("ffn_down   : ", LPF(ffn_down),    lpf);
        PRINT_L("  quantize : ", LPF(ffn_down_quant), lpf);
        PRINT_L("  backend  : ", LPF(ffn_down_ame),   lpf);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LPF(ffn_down), LPF(ffn_down_quant), LPF(ffn_down_ame)), lpf);
        print_uart("=== Profiling DECODE layer["); print_uart_int_dec(PROFILE_LAYER);
        print_uart("] (layer_total="); print_uart_int_dec(ldec); print_uart(" cyc over "); print_uart_int_dec((uint64_t)n_gen); print_uart(" tok) ===\r\n");
        PRINT_L("attn_norm  : ", LDEC(attn_norm),   ldec);
        PRINT_L("qkv_matmul : ", LDEC(qkv_matmul),  ldec);
        PRINT_L("  quantize : ", LDEC(qkv_quant),   ldec);
        PRINT_L("  backend  : ", LDEC(qkv_ame),     ldec);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LDEC(qkv_matmul), LDEC(qkv_quant), LDEC(qkv_ame)), ldec);
        PRINT_L("qknorm+rope: ", LDEC(qknorm_rope), ldec);
        PRINT_L("attn_score : ", LDEC(attn_score),  ldec);
        PRINT_L("wo_matmul  : ", LDEC(wo_matmul),   ldec);
        PRINT_L("  quantize : ", LDEC(wo_quant),    ldec);
        PRINT_L("  backend  : ", LDEC(wo_ame),      ldec);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LDEC(wo_matmul), LDEC(wo_quant), LDEC(wo_ame)), ldec);
        PRINT_L("ffn_norm   : ", LDEC(ffn_norm),    ldec);
        PRINT_L("ffn_gate   : ", LDEC(ffn_gate),    ldec);
        PRINT_L("  quantize : ", LDEC(ffn_gate_quant), ldec);
        PRINT_L("  backend  : ", LDEC(ffn_gate_ame),   ldec);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LDEC(ffn_gate), LDEC(ffn_gate_quant), LDEC(ffn_gate_ame)), ldec);
        PRINT_L("ffn_silu   : ", LDEC(ffn_silu),    ldec);
        PRINT_L("ffn_down   : ", LDEC(ffn_down),    ldec);
        PRINT_L("  quantize : ", LDEC(ffn_down_quant), ldec);
        PRINT_L("  backend  : ", LDEC(ffn_down_ame),   ldec);
        PRINT_L("  ovhd     : ", two_child_other_cycles(LDEC(ffn_down), LDEC(ffn_down_quant), LDEC(ffn_down_ame)), ldec);
    }
    #undef PRINT_L
    #undef LPF
    #undef LDEC
    #undef LSUM


    // --------------------------------------------------------
    // Compute analysis (quantized MACs as 2 FLOPs each)
    // Geometry: hidden=1024, n_layers=28, q_dim=2048, kv_dim=1024,
    //           intermediate=3072, head_dim=128, q_heads=16, vocab=151936
    // avg_pos = average sequence position across all forward calls
    // --------------------------------------------------------
    {
        int H = cfg.hidden;           // 1024
        int L = cfg.n_layers;         // 28
        int QD = cfg.q_dim;           // 2048
        int KD = cfg.kv_dim;          // 1024
        int QKV = QD + 2 * KD;        // 4096
        int I = cfg.intermediate;     // 3072
        int V = cfg.vocab;            // 151936
        int Qh = cfg.q_heads;         // 16
        int HD = cfg.head_dim;        // 128
        int total_calls = (int)g_timing.n_calls;
        // avg_pos: prefill tokens have increasing pos 0..n_tokens-1; gen tokens are n_tokens..n_tokens+n_gen-1
        // simple estimate: n_tokens/2 for prefill + n_tokens + n_gen/2 for gen, weighted by call count
        int avg_pos_prefill = n_tokens / 2;
        int avg_pos_gen = n_tokens + n_gen / 2;
        // total attention FLOPs = sum over all calls of 2 * 4 * pos * Qh * HD
        // = L * (n_tokens * 2*4*avg_pos_prefill*Qh*HD + n_gen * 2*4*avg_pos_gen*Qh*HD)
        // We compute per-call FLOPs:
        //   matmul layers: 2*L*(QKV*H + H*QD + 2*I*H + H*I) per call
        //   attn score:    4 * L * Qh * HD * (pos+1) per call  (QK dot + AV accumulate)
        //   lm_head:       2*V*H per call
        uint64_t flop_matmul_per_call = 2ULL * L * ((uint64_t)QKV*H + (uint64_t)H*QD + 2ULL*I*H + (uint64_t)H*I);
        uint64_t flop_lmhead_per_call = 2ULL * V * H;
        // attention: use avg_pos per phase
        uint64_t flop_attn_prefill = 4ULL * L * Qh * HD * avg_pos_prefill * n_tokens;
        uint64_t flop_attn_gen     = 4ULL * L * Qh * HD * avg_pos_gen     * n_gen;
        uint64_t total_flops = flop_matmul_per_call * (uint64_t)total_calls
                             + flop_lmhead_per_call  * (uint64_t)total_calls
                             + flop_attn_prefill + flop_attn_gen;

        print_uart("\r\n=== Compute Analysis ===\r\n");
        print_uart("  forward calls  : "); print_uart_int_dec(total_calls); print_uart("\r\n");
        print_uart("  layer passes   : "); print_uart_int_dec((int)g_timing.n_layers); print_uart("\r\n");
        print_uart("  matmul/call    : "); print_uart_int_dec((uint32_t)(flop_matmul_per_call/1000000)); print_uart(" MFLOPs\r\n");
        print_uart("  lm_head/call   : "); print_uart_int_dec((uint32_t)(flop_lmhead_per_call/1000000)); print_uart(" MFLOPs\r\n");
        print_uart("  attn_prefill   : "); print_uart_int_dec((uint32_t)(flop_attn_prefill/1000000)); print_uart(" MFLOPs\r\n");
        print_uart("  attn_gen       : "); print_uart_int_dec((uint32_t)(flop_attn_gen/1000000)); print_uart(" MFLOPs\r\n");
        print_uart("  total FLOPs    : "); print_uart_int_dec((uint32_t)(total_flops/1000000)); print_uart(" MFLOPs\r\n");

        // GFLOPS/s = total_flops / (total_cyc / freq)
        // freq = target clock, e.g. 50MHz on FPGA
        // print as MFLOPS/kcycle to be hardware-neutral
        if (total_cyc > 0) {
            // FLOPs/cycle = total_flops / total_cyc
            // At 1GHz:  GFLOPS/s = FLOPs/cycle
            // At 50MHz: MFLOPS/s = FLOPs/cycle * 50
            uint64_t flops_per_cyc_x1000 = total_flops * 1000 / total_cyc;
            print_uart("  FLOPs/cycle    : ");
            print_uart_int_dec((uint32_t)(flops_per_cyc_x1000 / 1000)); print_uart(".");
            print_uart_int_dec((uint32_t)(flops_per_cyc_x1000 % 1000)); print_uart("\r\n");
            // @1GHz: FLOPs/cycle == GFLOPS, but scalar<1 so print in MFLOPS (x1000)
            print_uart("  @1GHz  MFLOPS  : "); print_uart_int_dec((uint32_t)flops_per_cyc_x1000); print_uart("\r\n");
            // @50MHz: MFLOPS = FLOPs/cycle * 50, show one decimal
            uint64_t mflops50_x10 = flops_per_cyc_x1000 * 50 / 100; // *50/1000*10
            print_uart("  @50MHz MFLOPS  : ");
            print_uart_int_dec((uint32_t)(mflops50_x10 / 10)); print_uart(".");
            print_uart_int_dec((uint32_t)(mflops50_x10 % 10)); print_uart("\r\n");
        }
    }

    print_cycles("end-to-end: ", total_cyc);
    print_uart("=== done ===\r\n");
    exit(0);
    return 0;
}
#else /* CHAT_BUILD: multi-turn UART REPL */

#ifndef CHAT_MAX_SEQ
#define CHAT_MAX_SEQ 1024
#endif
#ifndef CHAT_GEN_MAX
#define CHAT_GEN_MAX 256
#endif
#ifndef CHAT_INPUT_BUF
#define CHAT_INPUT_BUF 512
#endif
#ifndef CHAT_PROMPT_TOKENS
#define CHAT_PROMPT_TOKENS 1024
#endif
#ifndef CHAT_CYCLE_HZ
#define CHAT_CYCLE_HZ FPGA_CLOCK_HZ
#endif
#ifndef CHAT_ESC_DRAIN_CYCLES
#define CHAT_ESC_DRAIN_CYCLES (CHAT_CYCLE_HZ / 500ULL) /* 2 ms */
#endif

// Firmware-side input echo. Default OFF: most interactive terminals already
// echo locally, and echoing again here shows every keystroke twice. Build with
// -DCHAT_ECHO=1 for raw terminals with no local echo (e.g. default minicom),
// where the firmware must echo so typed input is visible.
#ifndef CHAT_ECHO
#define CHAT_ECHO 0
#endif

// Blocking line reader. Handles BS/DEL, returns on CR/LF. CR is treated as
// end-of-line; a following LF is swallowed on the next iteration via the
// "(c<0x20) continue" path. Echoes input only when CHAT_ECHO is set. Returns
// line length (0 if the user just hit Enter).
static int chat_read_line(char *buf, int cap) {
    int n = 0;
    int csi_state = 0;
    while (1) {
        uint8_t c;
        while (!read_serial(&c)) { /* spin */ }

        // Filter terminal ANSI CSI sequences, including bracketed-paste
        // markers ESC[200~ and ESC[201~, instead of feeding them to commands
        // or the tokenizer.
        if (csi_state == 1) {
            csi_state = (c == '[') ? 2 : 0;
            if (csi_state != 0) continue;
        } else if (csi_state == 2) {
            if (c >= 0x40 && c <= 0x7E) csi_state = 0;
            continue;
        }
        if (c == 0x1B) {
            csi_state = 1;
            continue;
        }

        if (c == '\r' || c == '\n') {
#if CHAT_ECHO
            write_serial('\r');
            write_serial('\n');
#endif
            buf[n] = '\0';
            return n;
        }
        if (c == 0x08 || c == 0x7F) {           // BS / DEL
            if (n > 0) {
                n--;
#if CHAT_ECHO
                write_serial(0x08); write_serial(' '); write_serial(0x08);
#endif
            }
            continue;
        }
        if (c < 0x20) continue;                  // ignore other control bytes
        if (n < cap - 1) {
            buf[n++] = (char)c;
#if CHAT_ECHO
            write_serial((char)c);               // echo (off by default)
#endif
        }
    }
}

static char *chat_trim_line(char *line) {
    while (*line == ' ' || *line == '\t') line++;

    char *end = line;
    while (*end) end++;
    while (end > line && (end[-1] == ' ' || end[-1] == '\t')) end--;
    *end = '\0';
    return line;
}

static int chat_str_eq(const char *a, const char *b) {
    while (*a && (*a == *b)) { a++; b++; }
    return *a == *b;
}

__attribute__((noreturn))
static void chat_jump_to_boot(void) {
    print_uart("\r\nJumping to 0x80000000...\r\n");
    __asm__ volatile(
        "fence rw, rw\n\t"
        "fence.i\n\t"
        "li t0, 0x80000000\n\t"
        "jr t0\n\t"
        :
        :
        : "t0", "memory");
    __builtin_unreachable();
}

// Poll UART between generated tokens. After ESC, keep draining briefly so a
// terminal escape sequence whose bytes arrive separately cannot leak into the
// next command line.
static int chat_generation_interrupted(void) {
    uint8_t c;
    int interrupted = 0;
    while (read_serial(&c)) {
        if (c == 0x1B) interrupted = 1;
    }
    if (interrupted) {
        uint64_t drain_start = rdcycle();
        while (rdcycle() - drain_start < CHAT_ESC_DRAIN_CYCLES) {
            while (read_serial(&c)) { /* discard ESC sequence tail */ }
        }
    }
    return interrupted;
}

static void chat_print_fixed2(uint64_t value_x100) {
    print_uart_int_dec(value_x100 / 100ULL);
    print_uart(".");
    uint64_t frac = value_x100 % 100ULL;
    if (frac < 10ULL) print_uart("0");
    print_uart_int_dec(frac);
}

static uint64_t chat_tokens_per_second_x100(int tokens, uint64_t cycles) {
    if (tokens <= 0 || cycles == 0) return 0;
    return ((uint64_t)tokens * CHAT_CYCLE_HZ * 100ULL + cycles / 2ULL) / cycles;
}

static void chat_print_metrics(uint64_t ttft_cycles,
                               uint64_t prefill_cycles,
                               int prefill_tokens,
                               uint64_t decode_cycles,
                               int output_tokens,
                               const Config *cfg,
                               int kv_tokens) {
    uint64_t ttft_s_x100 =
        (ttft_cycles * 100ULL + CHAT_CYCLE_HZ / 2ULL) / CHAT_CYCLE_HZ;
    uint64_t prefill_tps_x100 =
        chat_tokens_per_second_x100(prefill_tokens, prefill_cycles);
    uint64_t decode_tps_x100 =
        chat_tokens_per_second_x100(output_tokens, decode_cycles);
    uint64_t kv_bytes =
        2ULL * (uint64_t)cfg->n_layers * (uint64_t)cfg->kv_heads *
        (uint64_t)cfg->head_dim * (uint64_t)kv_tokens * sizeof(float);
    uint64_t kv_mib_x100 =
        (kv_bytes * 100ULL + (1ULL << 19)) >> 20;

    print_uart("=== Metrics ===\r\n");
    print_uart("TTFT                : ");
    chat_print_fixed2(ttft_s_x100);
    print_uart(" s\r\n");

    print_uart("Prefill throughput  : ");
    chat_print_fixed2(prefill_tps_x100);
    print_uart(" tokens/s\r\n");

    print_uart("Decode throughput   : ");
    chat_print_fixed2(decode_tps_x100);
    print_uart(" tokens/s\r\n");

    print_uart("Prefill tokens      : ");
    print_uart_int_dec(prefill_tokens);
    print_uart("\r\n");

    print_uart("Output tokens       : ");
    print_uart_int_dec(output_tokens);
    print_uart("\r\n");

    print_uart("KV cache used       : ");
    chat_print_fixed2(kv_mib_x100);
    print_uart(" MiB (");
    print_uart_int_dec(kv_tokens);
    print_uart(" tokens)\r\n");
}

static int chat_prefill_identity_system_prompt(const Config *cfg,
                                               const Weights *w,
                                               RunState *st,
                                               BareTokenizer *tok) {
    int sys_tokens[CHAT_PROMPT_TOKENS];
    int n_sys = append_identity_system_tokens(tok, sys_tokens, 0, CHAT_PROMPT_TOKENS);
    if (n_sys <= 0 || n_sys >= cfg->max_seq) {
        print_uart("[chat] identity system prompt too long\r\n");
        return -1;
    }
    prefill_tokens_at(cfg, w, st, sys_tokens, n_sys, 0);
    return n_sys - 1;
}

int main(void) {
    init_uart((uint32_t)FPGA_CLOCK_HZ, 115200);
#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)
    enable_accel_state();
#endif
#if defined(USE_AME) && USE_AME
    ame_matmul_stats_reset();
#elif defined(USE_RVV) && USE_RVV
    rvv_tile_matmul_stats_reset();
#endif
    print_uart("Qwen3-0.6B W4A8+"); print_uart(MATMUL_NAME);
    print_uart(" chat-repl\r\n");

    heap_ptr = (char *)heap_start_addr();

    BareTokenizer *tok = tokenizer_load_from_mem(_tokenizer_blob_start);
    if (!tok) { print_uart("tokenizer load failed\r\n"); exit(1); }

    Config cfg; Weights w;
    load_model(MODEL_ADDR, &cfg, &w);
    cfg.max_seq = CHAT_MAX_SEQ;

    RunState st;
    build_state(&st, &cfg);

    print_uart("Hello! I am Qwen3-0.6B, an INT4-quantized language model\r\n");
    print_uart("running locally on this FPGA-based platform.\r\n\r\n");
    print_uart("Press ESC to stop generation. Commands: /clear, /quit\r\n");
    print_uart("How can I help you today?\r\n");

    const int EOS_ID    = IM_END_ID;
    const int CHAT_HEADER_TAIL = 5 + NO_THINKING_PROMPT_TOKENS; // im_end + \n + im_start + assistant + \n + empty thinking span

    int abs_pos = chat_prefill_identity_system_prompt(&cfg, &w, &st, tok);
    uint32_t rng_state = SAMPLE_RNG_SEED;
    int prompt_tokens[CHAT_PROMPT_TOKENS];
    char input_buf[CHAT_INPUT_BUF];

    while (1) {
        print_uart("\r\nUser> ");
        int line_len = chat_read_line(input_buf, sizeof(input_buf));
        if (line_len == 0) continue;
        char *input = chat_trim_line(input_buf);
        if (*input == '\0') continue;
        if (chat_str_eq(input, "/quit") || chat_str_eq(input, "/exit"))
            chat_jump_to_boot();
        if (chat_str_eq(input, "/clear") || chat_str_eq(input, "/reset")) {
            abs_pos = chat_prefill_identity_system_prompt(&cfg, &w, &st, tok);
            print_uart("Context cleared.\r\n");
            continue;
        }

        // Build the new tokens for this turn under the Qwen chat template.
        int n_new = 0;
        prompt_tokens[n_new++] = IM_START_ID;
        prompt_tokens[n_new++] = USER_ID;
        prompt_tokens[n_new++] = NEWLINE_ID;
        int budget = CHAT_PROMPT_TOKENS - n_new - CHAT_HEADER_TAIL;
        int enc = tokenizer_encode_bare(tok, input, prompt_tokens + n_new, budget);
        if (enc < 0) enc = 0;
        n_new += enc;
        prompt_tokens[n_new++] = IM_END_ID;
        prompt_tokens[n_new++] = NEWLINE_ID;
        prompt_tokens[n_new++] = IM_START_ID;
        prompt_tokens[n_new++] = ASSISTANT_ID;
        prompt_tokens[n_new++] = NEWLINE_ID;
        n_new = append_no_thinking_prompt_tokens(prompt_tokens, n_new, CHAT_PROMPT_TOKENS);

        // Reserve at least one slot for the first generated token.
        if (abs_pos + n_new + 1 >= cfg.max_seq) {
            print_uart("[chat] context window full; use /clear.\r\n");
            continue;
        }

        uint64_t t_turn = rdcycle();

        int prompt_start_pos = abs_pos + 1;
        uint64_t t_prefill = rdcycle();
        prefill_tokens_at(&cfg, &w, &st, prompt_tokens, n_new, prompt_start_pos);
        uint64_t prefill_cyc = rdcycle() - t_prefill;
        abs_pos = prompt_start_pos + n_new - 1;

        // Stream-generate assistant reply.
        print_uart("\r\nAssistant> ");
        int n_gen = 0;
        int n_out = 0;
        uint64_t ttft_cyc = 0;
        uint64_t t_gen = rdcycle();
        int interrupted = 0;
        SafeUtf8State safe_chat = {{0, 0, 0, 0}, 0, 0};
        while (n_gen < CHAT_GEN_MAX && (abs_pos + 1) < cfg.max_seq) {
            if (chat_generation_interrupted()) {
                interrupted = 1;
                break;
            }
            int next = sample_topk(st.logits, cfg.vocab,
                                   SAMPLE_TOP_K, SAMPLE_TEMPERATURE,
                                   SAMPLE_TOP_P, &rng_state);
            if (next == EOS_ID) break;
            if (!is_hidden_output_token(tok, next)) {
                if (n_out == 0) ttft_cyc = rdcycle() - t_turn;
                tokenizer_decode_uart_safe(tok, next, &safe_chat);
                n_out++;
                if ((n_out & 15) == 0) print_uart("\r\n");
            }
            n_gen++;
            abs_pos++;
            forward(&cfg, &w, &st, next, abs_pos);
        }
        uint64_t gen_cyc = rdcycle() - t_gen;
        if (ttft_cyc == 0) ttft_cyc = rdcycle() - t_turn;

        if (interrupted) print_uart("\r\nGeneration stopped by ESC.");
        print_uart("\r\n\r\n");
        chat_print_metrics(ttft_cyc, prefill_cyc, n_new, gen_cyc, n_out,
                           &cfg, abs_pos + 1);
        print_uart("\r\n");
    }

    chat_jump_to_boot();
}
#endif /* CHAT_BUILD */
