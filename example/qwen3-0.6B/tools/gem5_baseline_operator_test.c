static void check_projection(int n, int d, int gs, int stride) {
    enum { T = 2, MAX_N = 1024, MAX_D = 68, MAX_G = 8, MAX_W = 2 * 8 * 64 * 128 };
    static int8_t x[T * MAX_N], weights[MAX_W];
    static float scales[T * MAX_G], ws[MAX_G * MAX_D], output[T * MAX_D];
    int groups = (n + gs - 1) / gs, stride_groups = (stride + gs - 1) / gs;
    for (int i = 0; i < T * n; ++i) x[i] = (i * 17 % 255) - 127;
    for (int i = 0; i < MAX_W; ++i) weights[i] = 0;
#if QWEN_OFFLINE_AME_LAYOUT
    /* Match model v7: [out/64][out-within-block/4][group][K][lane]. */
    for (int row = 0; row < d; ++row)
        for (int group = 0; group < groups; ++group)
            for (int k = 0; k < 128; ++k) {
                const int column = group * 128 + k;
                const size_t offset =
                    (((size_t)(row / 64) * 16 + (size_t)((row % 64) / 4)) *
                         (size_t)stride_groups + (size_t)group) * 512 +
                    (size_t)k * 4 + (size_t)(row % 4);
                if (column < n)
                    weights[offset] =
                        (int8_t)(((row * n + column) * 7 % 15) - 7);
            }
#else
    for (int i = 0; i < MAX_W; ++i) weights[i] = (i * 7 % 15) - 7;
#endif
    for (int i = 0; i < T * groups; ++i) scales[i] = (i + 1) * 0.0137f;
    for (int i = 0; i < groups * d; ++i) ws[i] = (i % 7 + 1) * 0.0713f;
    ame_matmul_quant_batch_i8_i8_f32(output, x, scales, weights, ws, T, n, d, gs, stride);
    for (int t = 0; t < T; ++t) for (int j = 0; j < d; ++j) {
        float reference = 0;
        const int width = QWEN_GROUP_ACC ? gs : (QWEN_PACKED_K16 ? 16 : 4);
        for (int g = 0; g < groups; ++g) for (int k0 = 0; k0 < gs; k0 += width) {
            int sum = 0;
            for (int k = 0; k < width && k0+k < gs && g * gs + k0 + k < n; ++k) {
#if QWEN_OFFLINE_AME_LAYOUT
                const int column = g * gs + k0 + k;
                const int8_t weight =
                    (int8_t)(((j * n + column) * 7 % 15) - 7);
#else
                size_t index = ((size_t)(j / 64) * stride_groups + g) * 64 * gs + (j % 64) * gs + k0 + k;
                const int8_t weight = weights[index];
#endif
                sum += x[t * n + g * gs + k0 + k] * weight;
            }
            reference += (float)sum * scales[t * groups + g] * ws[g * d + j];
        }
        if (reference != output[t * d + j]) {
            print_uart("FAIL projection index="); print_uart_int_dec(t * d + j); print_uart("\n");
            exit(51);
        }
    }
}

int main(void) {
    init_uart(1000000000, 115200); enable_accel_state();
    print_uart("BASELINE_OPERATOR_TEST begin\n");
    static float a[1100], b[1100], c[1100];
    for (int i = 0; i < 1100; ++i) { a[i] = (i % 17 - 8) * 0.125f; b[i] = (i % 13 - 6) * 0.25f; }
    rvv_memcpy_f32_r4(c, a, 1100);  // Both multi-VL and workspace-chunk boundaries.
    for (int i = 0; i < 1100; ++i) if (c[i] != a[i]) exit(52);
    for (int n = 127; n <= 129; ++n) {
        float reference = 0;
        for (int i = 0; i < n; ++i) reference += a[i] * b[i];
        if (qk_dot_vfmul_rvv(a, b, n) != reference) exit(53);
        attn_av_axpy_rvv(c, b, 0.5f, n);
        for (int i = 0; i < n; ++i) {
            a[i] += 0.5f * b[i];
            if (c[i] != a[i]) exit(54);
        }
    }
#if QWEN_OFFLINE_AME_LAYOUT
    qwen_dma_blocks_session_begin();
    check_projection(128, 8, 128, 128);
    check_projection(1024, 8, 128, 1024); // Multiple alternating DMA group batches.
    qwen_dma_blocks_session_end();
#else
    check_projection(128, 8, 128, 128);
    check_projection(65, 66, 32, 96); // K tail, output-block crossing, padded stride.
    check_projection(13, 7, 7, 21); // Non-multiple-of-four group and padded stride.
    check_projection(129, 8, 129, 129); // Bounded group workspace chunk boundary.
    check_projection(1024, 8, 128, 1024); // Multiple alternating DMA group batches.
#endif
    print_uart("BASELINE_OPERATOR_TEST PASS rvv_copy_qk_av_and_projection\n");
    return 0;
}
