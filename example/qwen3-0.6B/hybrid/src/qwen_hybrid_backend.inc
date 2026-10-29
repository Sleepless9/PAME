/* Included by the shared Qwen front-end after quantize_x() is defined. */
#define MK_Q8_MAX_N 4096
static int8_t _mk_xq_buf[MK_Q8_MAX_N] __attribute__((aligned(64)));
static float _mk_xs_buf[MK_Q8_MAX_N / 32 + 1] __attribute__((aligned(64)));

static void
matmul_w4_mk(float *out, const uint8_t *weights, const float *weight_scales,
             const float *input, int outputs, int inputs, int group_size)
{
    if (inputs > MK_Q8_MAX_N) {
        print_uart("MK_Q8_MAX_N overflow\r\n");
        exit(1);
    }
    quantize_x(_mk_xq_buf, _mk_xs_buf, input, inputs, group_size);
    rvv_matmul_quant_tile_i8_i8_f32(
        out, _mk_xq_buf, _mk_xs_buf, (const int8_t *)weights,
        weight_scales, inputs, outputs, group_size);
}

static void
matmul_w4_mk_prequant(float *out, const uint8_t *weights,
                      const float *weight_scales, int outputs, int inputs,
                      int group_size, const int8_t *input,
                      const float *input_scales)
{
    /* Every single-row call is decode GEMV and belongs on RVV. */
    rvv_matmul_quant_tile_i8_i8_f32(
        out, input, input_scales, (const int8_t *)weights, weight_scales,
        inputs, outputs, group_size);
}

static void
quantize_x_batch(int8_t *output, float *scales, const float *input,
                 int tokens, int inputs, int group_size)
{
    const int groups = (inputs + group_size - 1) / group_size;
    for (int token = 0; token < tokens; ++token) {
        quantize_x(output + (size_t)token * inputs,
                   scales + (size_t)token * groups,
                   input + (size_t)token * inputs, inputs, group_size);
    }
}

static void
matmul_w4_mk_batch_prequant(float *out, const uint8_t *weights,
                            const float *weight_scales, int tokens,
                            int outputs, int inputs, int group_size,
                            const int8_t *input, const float *input_scales)
{
    /* Batch entrypoints are prefill and use four-row Ztt tiles. */
    ame_matmul_quant_batch_i8_i8_f32(
        out, input, input_scales, (const int8_t *)weights, weight_scales,
        tokens, inputs, outputs, group_size, inputs);
}

static void
matmul_w4_mk_pair_prequant(float *out0, float *out1,
                           const uint8_t *weights0,
                           const float *weight_scales0,
                           const uint8_t *weights1,
                           const float *weight_scales1, int outputs,
                           int inputs, int group_size, const int8_t *input,
                           const float *input_scales)
{
    matmul_w4_mk_prequant(out0, weights0, weight_scales0, outputs, inputs,
                          group_size, input, input_scales);
    matmul_w4_mk_prequant(out1, weights1, weight_scales1, outputs, inputs,
                          group_size, input, input_scales);
}

static void
matmul_w4_mk_batch_pair_prequant(
    float *out0, float *out1, const uint8_t *weights0,
    const float *weight_scales0, const uint8_t *weights1,
    const float *weight_scales1, int tokens, int outputs, int inputs,
    int group_size, const int8_t *input, const float *input_scales)
{
    ame_matmul_quant_batch_pair_i8_i8_f32(
        out0, out1, input, input_scales, (const int8_t *)weights0,
        weight_scales0, (const int8_t *)weights1, weight_scales1, tokens,
        inputs, outputs, group_size, inputs);
}

#define MATMUL_W4 matmul_w4_mk
#define MATMUL_NAME "rvv+ztt_spm_w4"
