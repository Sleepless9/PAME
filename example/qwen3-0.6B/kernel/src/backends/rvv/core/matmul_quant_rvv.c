#include "backends/rvv/core/matmul_quant_rvv.h"

#include <stddef.h>
#include <stdint.h>

void rvv_matmul_ref_i8_f32(float *o,
                   const int8_t *w,
                   const float *w_scales,
                   const float *x,
                   int m,
                   int n,
                   int group_size) {
  /* Quantized matmul helper (fp32 activation, int8 weight, per-group scale).
   *
   * For each output row i:
   *   o[i] = sum_g [ w_scales[i,g] * (sum_{j in group g} w[i,j] * x[j]) ]
   */
  if (m <= 0 || n <= 0 || group_size <= 0) return;
  const int n_groups = n / group_size;

  for (int i = 0; i < m; i++) {
    const int8_t *wi = w + (size_t)i * (size_t)n;
    const float *si = w_scales + (size_t)i * (size_t)n_groups;
    double total = 0.0;

    for (int g = 0; g < n_groups; g++) {
      const int8_t *wg = wi + (size_t)g * (size_t)group_size;
      const float *xg = x + (size_t)g * (size_t)group_size;
      double gsum = 0.0;
      for (int j = 0; j < group_size; ++j) {
        gsum += (double)wg[j] * (double)xg[j];
      }
      total += (double)si[g] * gsum;
    }
    o[i] = (float)total;
  }
}

int32_t rvv_group_dot_i8(const int8_t *wg, const int8_t *xg, int cnt) {
  int32_t s = 0;
  for (int j = 0; j < cnt; ++j) s += (int32_t)wg[j] * (int32_t)xg[j];
  return s;
}

void rvv_group_dot_i8_x4(const int8_t *wg_base,
                         int w_stride_bytes,
                         const int8_t *xg,
                         int cnt,
                         int32_t *out) {
  for (int r = 0; r < 4; ++r) {
    out[r] = rvv_group_dot_i8(wg_base + (ptrdiff_t)r * (ptrdiff_t)w_stride_bytes,
                              xg, cnt);
  }
}

/* ----- Backend kernel-call statistics ----- */
static uint64_t g_rvv_mm_calls;       /* rvv_matmul_quant_i8_i8_f32 calls */
static uint64_t g_rvv_mm_macs;        /* total int8 MACs (n*d per call) */
static uint64_t g_rvv_mm_dots_x4;     /* invocations of the 4-row fast path */
static uint64_t g_rvv_mm_dots_x1;     /* invocations of the single-row tail path */

void rvv_matmul_stats_reset(void) {
  g_rvv_mm_calls = 0;
  g_rvv_mm_macs = 0;
  g_rvv_mm_dots_x4 = 0;
  g_rvv_mm_dots_x1 = 0;
}

uint64_t rvv_matmul_stats_calls(void)    { return g_rvv_mm_calls; }
uint64_t rvv_matmul_stats_macs(void)     { return g_rvv_mm_macs; }
uint64_t rvv_matmul_stats_dots_x4(void)  { return g_rvv_mm_dots_x4; }
uint64_t rvv_matmul_stats_dots_x1(void)  { return g_rvv_mm_dots_x1; }

void rvv_matmul_quant_i8_i8_f32(float *xout,
                                const int8_t *xq,
                                const float *xs,
                                const int8_t *wq,
                                const float *ws,
                                int n,
                                int d,
                                int gs) {
  /* Quantized matmul (RVV backend)
   *
   * For each output row i:
   *   xout[i] = sum_g [ (sum_{j in group g} w_q[i,j] * x_q[j]) * w_s[i,g] * x_s[g] ]
   *
   * Inner i8xi8 dot: RVV widening mul + widening reduction. Rows are
   * processed 4 at a time to share the activation load and pipeline
   * products through ping-ponged register groups.
   */
  if (!xout || !xq || !xs || !wq || !ws || n <= 0 || d <= 0 || gs <= 0) return;
  g_rvv_mm_calls++;
  g_rvv_mm_macs += (uint64_t)n * (uint64_t)d;

  const int groups_per_row = (n + gs - 1) / gs;
  uint64_t local_x4 = 0;
  uint64_t local_x1 = 0;
  int i = 0;

  for (; i + 4 <= d; i += 4) {
    const int8_t *wrow0 = wq + (size_t)i * (size_t)n;
    const float *wsrow0 = ws + (size_t)i * (size_t)groups_per_row;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (int g = 0; g < groups_per_row; ++g) {
      const int base = g * gs;
      const int cnt = (base + gs <= n) ? gs : (n - base);
      int32_t gs4[4];
      rvv_group_dot_i8_x4(wrow0 + base, n, xq + base, cnt, gs4);
      const float xsg = xs[g];
      const float s0 = wsrow0[g]                          * xsg;
      const float s1 = wsrow0[g + groups_per_row]         * xsg;
      const float s2 = wsrow0[g + 2 * groups_per_row]     * xsg;
      const float s3 = wsrow0[g + 3 * groups_per_row]     * xsg;
      t0 += (float)gs4[0] * s0;
      t1 += (float)gs4[1] * s1;
      t2 += (float)gs4[2] * s2;
      t3 += (float)gs4[3] * s3;
      local_x4++;
    }
    xout[i]     = t0;
    xout[i + 1] = t1;
    xout[i + 2] = t2;
    xout[i + 3] = t3;
  }

  for (; i < d; ++i) {
    const int8_t *wrow = wq + (size_t)i * (size_t)n;
    const float *wsrow = ws + (size_t)i * (size_t)groups_per_row;
    float total = 0.0f;
    for (int g = 0; g < groups_per_row; ++g) {
      const int base = g * gs;
      const int cnt = (base + gs <= n) ? gs : (n - base);
      const int32_t gsum = rvv_group_dot_i8(wrow + base, xq + base, cnt);
      total += (float)gsum * wsrow[g] * xs[g];
      local_x1++;
    }
    xout[i] = total;
  }

  g_rvv_mm_dots_x4 += local_x4;
  g_rvv_mm_dots_x1 += local_x1;
}
