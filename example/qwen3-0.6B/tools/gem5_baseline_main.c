#ifndef PREFILL_TOKENS
#define PREFILL_TOKENS 4
#endif
#ifndef DECODE_STEPS
#define DECODE_STEPS 1
#endif
#define BASELINE_SEED 20260907u
volatile uintptr_t baseline_logits_address;
volatile uint64_t baseline_vocab_size;

static void baseline_fence(void) {
#ifndef BASELINE_HOST_REFERENCE
    __asm__ volatile("fence rw,rw" ::: "memory");
#endif
}

static uint64_t logits_hash(const float *values, int count) {
    uint64_t hash = 14695981039346656037ull;
    int nonzero = 0;
    for (int i = 0; i < count; ++i) {
        union { float f; uint32_t u; } value = { .f = values[i] };
        nonzero += values[i] != 0.0f;
        if ((value.u & 0x7f800000u) == 0x7f800000u) {
            print_uart("FAIL nonfinite logits\n"); exit(42);
        }
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (value.u >> (8 * byte)) & 255;
            hash *= 1099511628211ull;
        }
    }
    if (!nonzero) { print_uart("FAIL all-zero logits\n"); exit(46); }
    return hash;
}

#ifdef BASELINE_OPERATOR_TEST
#include "gem5_baseline_operator_test.c"
#else
int main(
#ifdef BASELINE_HOST_REFERENCE
    int argc, char **argv
#else
    void
#endif
) {
#ifdef BASELINE_HOST_REFERENCE
    if (argc != 3) return 2;
    FILE *model = fopen(argv[1], "rb");
    if (!model) return 3;
    fseek(model, 0, SEEK_END);
    size_t bytes = ftell(model);
    rewind(model);
    // The blob loader aligns tensor offsets using the absolute base address.
    host_model = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    if (!host_model || fread(host_model, 1, bytes, model) != bytes) return 4;
    fclose(model);
    host_output_prefix = argv[2];
#endif
    init_uart(1000000000, 115200);
    enable_accel_state();
    ame_matmul_stats_reset();
    heap_ptr = (char *)heap_start_addr();
    Config cfg; Weights weights; RunState state;
    load_model(MODEL_ADDR, &cfg, &weights);
    if (cfg.n_layers != 28 || cfg.hidden != 1024 || cfg.group_size != 128) exit(43);
    cfg.max_seq = PREFILL_TOKENS + DECODE_STEPS;
    build_state(&state, &cfg);
    baseline_logits_address = (uintptr_t)state.logits;
    baseline_vocab_size = cfg.vocab;
    static int tokens[PREFILL_TOKENS];
    uint32_t rng = BASELINE_SEED;
    for (int i = 0; i < PREFILL_TOKENS; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        tokens[i] = rng % 151643u;
    }
    print_uart("QWEN_BASELINE begin prefill_tokens="); print_uart_int_dec(PREFILL_TOKENS);
    print_uart(" decode_steps="); print_uart_int_dec(DECODE_STEPS);
    print_uart(" seed="); print_uart_int_dec(BASELINE_SEED); print_uart("\n");
#if QWEN_DMA_OVERLAP && QWEN_GROUP_ACC
    /* One architectural AME ownership epoch covers prefill and decode. */
    qwen_dma_blocks_session_begin();
#endif
    baseline_fence();
    m5_work_begin(5, 0);
    prefill_all_tokens(&cfg, &weights, &state, tokens, PREFILL_TOKENS);
    baseline_fence();
    m5_work_end(5, 0);
    if (g_timing.n_layers != 28 * PREFILL_TOKENS) exit(44);
    print_uart("PREFILL hash="); print_uart_hex(logits_hash(state.logits, cfg.vocab));
    print_uart(" argmax="); print_uart_int_dec(argmax(state.logits, cfg.vocab)); print_uart("\n");
#ifdef BASELINE_HOST_REFERENCE
    save_reference("prefill", state.logits, cfg.vocab);
#endif
    int next = argmax(state.logits, cfg.vocab);
#ifdef QWEN_DECODE_TOKENS
    static const int decode_inputs[DECODE_STEPS] = QWEN_DECODE_TOKENS;
#endif
    baseline_fence();
    m5_work_begin(6, 0);
    for (int step = 0; step < DECODE_STEPS; ++step) {
#ifdef QWEN_DECODE_TOKENS
        next = decode_inputs[step];
#endif
        forward(&cfg, &weights, &state, next, PREFILL_TOKENS + step);
        next = argmax(state.logits, cfg.vocab);
    }
    baseline_fence();
    m5_work_end(6, 0);
#if QWEN_DMA_OVERLAP && QWEN_GROUP_ACC
    qwen_dma_blocks_session_end();
#endif
    if (g_timing.n_layers != 28 * (PREFILL_TOKENS + DECODE_STEPS)) exit(45);
    print_uart("DECODE hash="); print_uart_hex(logits_hash(state.logits, cfg.vocab));
    print_uart(" argmax="); print_uart_int_dec(next); print_uart("\n");
#ifdef BASELINE_HOST_REFERENCE
    save_reference("decode", state.logits, cfg.vocab);
#endif
    print_uart("QWEN_BASELINE PASS layers_total="); print_uart_int_dec(g_timing.n_layers);
    print_uart("\n");
    return 0;
}
#endif
