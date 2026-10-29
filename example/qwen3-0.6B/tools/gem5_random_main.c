/* Included after the unchanged model/forward routines generated from core/.
 * Token IDs are the input interface; no tokenizer is needed for this run.
 */
#ifndef RANDOM_SEED
#define RANDOM_SEED 20260907u
#endif

void exit(int code) {
    if (code) m5_fail(0, (uint64_t)code);
    else m5_exit(0);
    for (;;) __asm__ volatile("wfi");
}

static void report_forward(const char *phase, const Config *cfg,
                           const RunState *st, uint64_t start) {
    for (int i = 0; i < cfg->vocab; ++i) {
        union { float f; uint32_t u; } v = { .f = st->logits[i] };
        if ((v.u & 0x7f800000u) == 0x7f800000u) {
            print_uart("ERROR nonfinite logits\r\n");
            m5_fail(0, 2);
        }
    }
    print_uart(phase);
    print_uart(" complete layers_total="); print_uart_int_dec(g_timing.n_layers);
    print_uart(" cycles="); print_uart_int_dec(rdcycle() - start);
    print_uart(" argmax="); print_uart_int_dec(argmax(st->logits, cfg->vocab));
    print_uart("\r\n");
}

#ifdef QWEN_BACKEND_TEST
int main(void) {
    init_uart(1000000000, 115200);
    enable_accel_state();
    print_uart("PACKED backend validation begin\r\n");
    enum { N = 64, D = 64, G = 32, T = 2, NG = N / G };
    static int8_t x[T * N], w[D * N];
    static float xs[T * NG], ws[NG * D], out[T * D];
    for (int i = 0; i < T * N; ++i) x[i] = (i * 17 % 255) - 127;
    for (int i = 0; i < D * N; ++i) w[i] = (i * 7 % 15) - 7;
    for (int i = 0; i < T * NG; ++i) xs[i] = (i + 1) * 0.125f;
    for (int i = 0; i < NG * D; ++i) ws[i] = (i % 7 + 1) * 0.0625f;
    ame_matmul_quant_batch_i8_i8_f32(out, x, xs, w, ws, T, N, D, G, N);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < D; ++j) {
            float reference = 0;
            for (int g = 0; g < NG; ++g) {
                int sum = 0;
                for (int k = 0; k < G; ++k)
                    sum += x[t * N + g * G + k] * w[(g * D + j) * G + k];
                reference += sum * xs[t * NG + g] * ws[g * D + j];
            }
            if (reference != out[t * D + j]) {
                print_uart("FAIL packed output index="); print_uart_int_dec(t * D + j);
                print_uart("\r\n");
                m5_fail(0, 41);
            }
        }
    }
    print_uart("PACKED backend validation PASS outputs=128\r\n");
    m5_exit(0);
    for (;;) __asm__ volatile("wfi");
}
#elif defined(QWEN_TILE_TEST)
extern void ztt_i8_i8_i32_4x4(int32_t *, const int8_t *, const int8_t *);
int main(void) {
    init_uart(1000000000, 115200);
    enable_accel_state();
    print_uart("ZTT tile validation begin\r\n");
    int8_t a[64] __attribute__((aligned(64)));
    int8_t b[64] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));
    for (int i = 0; i < 64; ++i) { a[i] = 0; b[i] = 0; }
    for (int i = 0; i < 16; ++i) { a[i] = 2; b[i] = 3; out[i] = -1; }
    ztt_i8_i8_i32_4x4(out, a, b);
    for (int i = 0; i < 16; ++i) {
        if (out[i] != 24) {
            print_uart("FAIL constant tile index="); print_uart_int_dec(i);
            print_uart(" value="); print_uart_hex(out[i]); print_uart("\r\n");
            m5_fail(0, 30);
        }
    }
    for (int i = 0; i < 16; ++i) {
        a[i] = i % 5 == 0 ? (i / 5 - 2) : 0;
        b[i] = i % 5 == 0 ? 1 : 0;
        out[i] = -99;
    }
    ztt_i8_i8_i32_4x4(out, a, b);
    for (int i = 0; i < 16; ++i) {
        if (out[i] != a[i]) m5_fail(0, 31);
    }
    print_uart("ZTT tile validation PASS\r\n");
    m5_exit(0);
    for (;;) __asm__ volatile("wfi");
}
#else
int main(void) {
    init_uart(1000000000, 115200);
    enable_accel_state();
    ame_matmul_stats_reset();
    heap_ptr = (char *)heap_start_addr();
    print_uart("QWEN_FULL_RANDOM prefill=1 decode=1 seed=");
    print_uart_int_dec(RANDOM_SEED); print_uart("\r\n");
    Config cfg;
    Weights w;
    RunState st;
    load_model(MODEL_ADDR, &cfg, &w);
    cfg.max_seq = 2;
    if (cfg.n_layers != 28 || cfg.hidden != 1024) {
        print_uart("ERROR unexpected model dimensions\r\n");
        m5_fail(0, 3);
    }
    build_state(&st, &cfg);
    uint32_t rng = RANDOM_SEED;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    const int token = rng % 151643u;
    print_uart("INPUT token="); print_uart_int_dec(token); print_uart("\r\n");
    m5_work_begin(5, 0);
    uint64_t start = rdcycle();
    prefill_all_tokens(&cfg, &w, &st, &token, 1);
    m5_work_end(5, 0);
    report_forward("PREFILL", &cfg, &st, start);
    if (g_timing.n_layers != 28) m5_fail(0, 4);
    const int next = argmax(st.logits, cfg.vocab);
    print_uart("DECODE input_token="); print_uart_int_dec(next); print_uart("\r\n");
    m5_work_begin(6, 0);
    start = rdcycle();
    forward(&cfg, &w, &st, next, 1);
    m5_work_end(6, 0);
    report_forward("DECODE", &cfg, &st, start);
    if (g_timing.n_layers != 56) m5_fail(0, 5);
    print_uart("QWEN_FULL_RANDOM PASS prefill_layers=28 decode_layers=28\r\n");
    m5_exit(0);
    for (;;) __asm__ volatile("wfi");
}
#endif
