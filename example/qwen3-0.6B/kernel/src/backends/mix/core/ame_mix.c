#include "backends/mix/core/ops_mix.h"

/* ----------------------------------------------------------------------
 * AME mix backend - quantized W8A16 matmul.
 *
 * Layout: ABT with M=1, N=64 (64 distinct output rows).
 *
 *   A[1, count]       - direct pointer to xq[base..base+count).
 *   B_pack[64, count] - row-major ABT layout:
 *                       B[j, t] = wq[i0+j, base+t].
 *   C[1, 64]          - row 0 contains the 64 partial dot products.
 *
 * The AME probe verifies that msettilem(1) is honored, row 0 is not polluted
 * by phantom rows, and multi-K ABT accumulation is correct. This removes the
 * previous replicated-activation workaround and its M-side waste.
 * --------------------------------------------------------------------*/

#define AME_MIX_OUTBLK 64      /* 4 acc lanes x tileN=16 */
#define AME_MIX_M_TILE 16      /* hardware M tile cap */
#define AME_MIX_MAX_GS 128     /* group size cap for static buffers */

static int8_t  ame_mix_b_pack[AME_MIX_MAX_GS * AME_MIX_OUTBLK] __attribute__((aligned(64)));
static float ame_mix_c_pack[AME_MIX_M_TILE * AME_MIX_OUTBLK] __attribute__((aligned(64)));

/* ----- Backend kernel-call statistics ----- */
static uint64_t g_ame_mm_calls;
static uint64_t g_ame_mm_macs;
static uint64_t g_ame_mm_tile_invocations;
/* Keep these symbols for existing tests; this implementation does not split
 * AME/non-AME cycles internally. */
static uint64_t g_ame_noblk_cycles_total;
static uint64_t g_ame_noblk_cycles_ame;

void ame_matmul_stats_reset(void) {
  g_ame_mm_calls = 0;
  g_ame_mm_macs = 0;
  g_ame_mm_tile_invocations = 0;
  g_ame_noblk_cycles_total = 0;
  g_ame_noblk_cycles_ame = 0;
}
uint64_t ame_matmul_stats_calls(void)      { return g_ame_mm_calls; }
uint64_t ame_matmul_stats_macs(void)       { return g_ame_mm_macs; }
uint64_t ame_matmul_stats_tile_calls(void) { return g_ame_mm_tile_invocations; }
uint64_t ame_matmul_noblk_cycles_total(void) { return g_ame_noblk_cycles_total; }
uint64_t ame_matmul_noblk_cycles_ame(void)   { return g_ame_noblk_cycles_ame; }
uint64_t ame_matmul_noblk_cycles_noname(void) {
  return (g_ame_noblk_cycles_total >= g_ame_noblk_cycles_ame)
             ? (g_ame_noblk_cycles_total - g_ame_noblk_cycles_ame)
             : 0;
}

/* Weak hooks into the platform layer so the kernel stays portable. */
extern void print_uart(const char *str) __attribute__((weak));

static void ame_mix_fail(const char *msg) {
  if (&print_uart) print_uart(msg);
  __builtin_trap();
}

void ame_matmul_quant_noblk_i8_i8_f32(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride) {
  if (!xout || !xq || !xs || !wq || !ws || n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS) {
    ame_mix_fail("[ame_mix] gs > AME_MIX_MAX_GS, halting\r\n");
  }
  g_ame_mm_calls++;
  g_ame_mm_macs += (uint64_t)n * (uint64_t)d;

  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  const int groups = (n + gs - 1) / gs;
  /* ws row stride follows wq's physical row stride, not the view length n. */
  const int row_groups = (stride_w + gs - 1) / gs;

  mix_zero_f32(xout, d);

  for (int g = 0; g < groups; g++) {
    const int base  = g * gs;
    const int count = (base + gs <= n) ? gs : (n - base);

    const int8_t *xseg = xq + base;
    const float xsg = xs[g];

    for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
      const int oblk = (d - i0 < AME_MIX_OUTBLK) ? (d - i0) : AME_MIX_OUTBLK;

      /* Pack ABT B[j, t] = wq[i0+j, base+t]. Each output row owns one
       * contiguous count-element weight slice. */
      for (int j = 0; j < oblk; j++) {
        const int8_t *wrow = wq + (ptrdiff_t)(i0 + j) * (ptrdiff_t)stride_w + (ptrdiff_t)base;
        int8_t *brow = ame_mix_b_pack + (size_t)j * (size_t)count;
        rvv_memcpy_u8_r4(brow, wrow, (size_t)count);
      }
      /* Zero unused N lanes when oblk < AME_MIX_OUTBLK. */
      if (oblk < AME_MIX_OUTBLK) {
        for (int j = oblk; j < AME_MIX_OUTBLK; j++) {
          int8_t *brow = ame_mix_b_pack + (size_t)j * (size_t)count;
          for (int t = 0; t < count; t++) {
            brow[t] = 0;
          }
        }
      }

      /* mqma accumulates into C; zero row 0 before each call. */
      mix_zero_f32(ame_mix_c_pack, AME_MIX_OUTBLK);
      asm volatile("fence rw,rw" ::: "memory");
      (void)ame_matmul_quadacc_abt_i8_i8_f32(xseg,
                                             ame_mix_b_pack,
                                             ame_mix_c_pack,
                                             1,                /* m */
                                             AME_MIX_OUTBLK,   /* n */
                                             count);           /* k */
      asm volatile("fence rw,rw" ::: "memory");
      g_ame_mm_tile_invocations++;

      /* C[0, j] holds the fp32 dot product for output row (i0 + j). */
      const float *crow0 = ame_mix_c_pack;
      for (int j = 0; j < oblk; j++) {
        const float wsr = ws[(ptrdiff_t)(i0 + j) * (ptrdiff_t)row_groups + (ptrdiff_t)g];
        xout[i0 + j] += crow0[j] * wsr * xsg;
      }
    }
  }
}

void ame_matmul_quant_blocked_i8_i8_f32(float *xout,
                                        const int8_t *xq,
                                        const float *xs,
                                        const int8_t *wq,
                                        const float *ws,
                                        int n,
                                        int d,
                                        int gs,
                                        int w_row_stride) {
  ame_matmul_quant_noblk_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs, w_row_stride);
}

/* ----------------------------------------------------------------------
 * Batched quantized matmul (true T-token prefill path).
 *
 * ABT layout with T tokens packed along M and 64 outputs along N. This path
 * uses the ABT 4-acc kernel for prefill.
 * For T==1 we delegate to the ABT noblk path.
 *
 * Math:
 *   xout[t, j] += C[t, j] * ws[i0+j, g] * xs[t, g]
 *   where C[t, j] = sum_k xq[t, base+k] * wq[i0+j, base+k]   (i8 -> fp32)
 * --------------------------------------------------------------------*/

#define AME_MIX_BATCH_T_MAX  128

static int8_t  ame_mix_batch_a_pack[AME_MIX_BATCH_T_MAX * AME_MIX_MAX_GS] __attribute__((aligned(64)));
static int8_t  ame_mix_batch_b_pack[AME_MIX_MAX_GS * AME_MIX_OUTBLK]      __attribute__((aligned(64)));
static float ame_mix_batch_c_pack[AME_MIX_BATCH_T_MAX * AME_MIX_OUTBLK] __attribute__((aligned(64)));

void ame_matmul_quant_batch_i8_i8_f32(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const int8_t *wq,
                                      const float *ws,
                                      int T,
                                      int n,
                                      int d,
                                      int gs,
                                      int w_row_stride) {
  if (!xout || !xq || !xs || !wq || !ws) return;
  if (T <= 0 || n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS) {
    ame_mix_fail("[ame_mix] batch: gs > AME_MIX_MAX_GS, halting\r\n");
  }

  if (T == 1) {
    ame_matmul_quant_noblk_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs,
                                     (w_row_stride > 0) ? w_row_stride : n);
    return;
  }

  if (T > AME_MIX_BATCH_T_MAX) {
    /* Fall back to T sequential single-token matmuls when batch dim too large. */
    for (int t = 0; t < T; t++) {
      ame_matmul_quant_noblk_i8_i8_f32(
          xout + (size_t)t * (size_t)d,
          xq   + (size_t)t * (size_t)n,
          xs   + (size_t)t * (size_t)((n + gs - 1) / gs),
          wq, ws, n, d, gs,
          (w_row_stride > 0) ? w_row_stride : n);
    }
    return;
  }
  g_ame_mm_calls++;
  g_ame_mm_macs += (uint64_t)T * (uint64_t)n * (uint64_t)d;

  const int stride_w   = (w_row_stride > 0) ? w_row_stride : n;
  const int groups     = (n + gs - 1) / gs;
  const int row_groups = (stride_w + gs - 1) / gs;
  const int x_groups   = groups;

  /* Round T up to AME_MIX_M_TILE; padded rows of A get zero so their
   * mqma contribution is zero (their C rows are not read). */
  int Tpad = T;
  if (Tpad % AME_MIX_M_TILE) Tpad += AME_MIX_M_TILE - (Tpad % AME_MIX_M_TILE);

  mix_zero_f32(xout, T * d);

  for (int g = 0; g < groups; g++) {
    const int base  = g * gs;
    const int count = (base + gs <= n) ? gs : (n - base);

    /* Pack A[t, k] = xq[t, base+k] for t<T, zero for t>=T. */
    for (int t = 0; t < T; t++) {
      const int8_t *src = xq + (size_t)t * (size_t)n + (size_t)base;
      int8_t *dst       = ame_mix_batch_a_pack + (size_t)t * (size_t)count;
      rvv_memcpy_u8_r4(dst, src, (size_t)count);
    }
    for (int t = T; t < Tpad; t++) {
      int8_t *dst = ame_mix_batch_a_pack + (size_t)t * (size_t)count;
      for (int c = 0; c < count; c++) dst[c] = 0;
    }

    for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
      const int oblk = (d - i0 < AME_MIX_OUTBLK) ? (d - i0) : AME_MIX_OUTBLK;

      /* Pack ABT B[j, k] = wq[i0+j, base+k] in row-major [64, count]. */
      for (int j = 0; j < oblk; j++) {
        const int8_t *wrow = wq + (ptrdiff_t)(i0 + j) * (ptrdiff_t)stride_w + (ptrdiff_t)base;
        int8_t *brow = ame_mix_batch_b_pack + (size_t)j * (size_t)count;
        rvv_memcpy_u8_r4(brow, wrow, (size_t)count);
      }
      if (oblk < AME_MIX_OUTBLK) {
        for (int j = oblk; j < AME_MIX_OUTBLK; j++) {
          int8_t *brow = ame_mix_batch_b_pack + (size_t)j * (size_t)count;
          for (int kk = 0; kk < count; kk++) {
            brow[kk] = 0;
          }
        }
      }

      mix_zero_f32(ame_mix_batch_c_pack, Tpad * AME_MIX_OUTBLK);
      asm volatile("fence rw,rw" ::: "memory");
      (void)ame_matmul_quadacc_abt_i8_i8_f32(ame_mix_batch_a_pack,
                                             ame_mix_batch_b_pack,
                                             ame_mix_batch_c_pack,
                                             Tpad,
                                             AME_MIX_OUTBLK,
                                             count);
      asm volatile("fence rw,rw" ::: "memory");
      g_ame_mm_tile_invocations++;

      for (int t = 0; t < T; t++) {
        const float *crow = ame_mix_batch_c_pack + (size_t)t * AME_MIX_OUTBLK;
        const float xst = xs[(size_t)t * (size_t)x_groups + (size_t)g];
        float *xrow = xout + (size_t)t * (size_t)d + (size_t)i0;
        for (int j = 0; j < oblk; j++) {
          const float wsr = ws[(ptrdiff_t)(i0 + j) * (ptrdiff_t)row_groups + (ptrdiff_t)g];
          xrow[j] += crow[j] * wsr * xst;
        }
      }
    }
  }
}

void ame_matmul_quant_i8_i8_f32(float *xout,
                                const int8_t *xq,
                                const float *xs,
                                const int8_t *wq,
                                const float *ws,
                                int n,
                                int d,
                                int gs) {
  ame_matmul_quant_noblk_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs, n);
}

void ame_gemv_quant_i8_i8_f32(float *xout,
                              const int8_t *xq,
                              const float *xs,
                              const int8_t *wq,
                              const float *ws,
                              int n,
                              int d,
                              int gs,
                              int w_row_stride) {
  ame_matmul_quant_noblk_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs, w_row_stride);
}
