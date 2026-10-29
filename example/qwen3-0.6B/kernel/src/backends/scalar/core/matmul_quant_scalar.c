#include "backends/scalar/core/matmul_quant_scalar.h"

#include <stddef.h>
#include <stdint.h>

/* ----- Backend kernel-call statistics ----- */
static uint64_t g_scalar_mm_calls;
static uint64_t g_scalar_mm_macs;

void scalar_matmul_stats_reset(void) {
  g_scalar_mm_calls = 0;
  g_scalar_mm_macs = 0;
}

uint64_t scalar_matmul_stats_calls(void) { return g_scalar_mm_calls; }
uint64_t scalar_matmul_stats_macs(void)  { return g_scalar_mm_macs; }

void scalar_matmul_quant_i8_i8_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const int8_t *wq,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs) {
  g_scalar_mm_calls++;
  g_scalar_mm_macs += (uint64_t)n * (uint64_t)d;
  /* Quantized matmul (scalar backend)
   *
   * For each output row i:
   *   xout[i] = sum_{g} { dot_g(i) * w_s[i,g] * x_s[g] }
   *
   * where:
   *   dot_g(i) = sum_{j in group g} (w_q[i,j] * x_q[j])
   *
   * Equivalent expanded form:
   *   xout[i] = sum_g [ (sum_{j in g} w_q[i,j] * x_q[j]) * w_s[i,g] * x_s[g] ]
   */
  const int groups_per_row = (n + gs - 1) / gs;
  for (int i = 0; i < d; i++) {
    const int8_t *wrow = wq + (size_t)i * (size_t)n;
    double sum = 0.0;
    for (int g = 0; g < groups_per_row; g++) {
      const int base = g * gs;
      const int count = (base + gs <= n) ? gs : (n - base);
      const double wsg = (double)ws[(size_t)i * (size_t)groups_per_row + (size_t)g];
      const double xsg = (double)xs[g];
      for (int j = 0; j < count; j++) {
        const int idx = base + j;
        sum += (double)wrow[idx] * (double)xq[idx] * wsg * xsg;
      }
    }
    xout[i] = (float)sum;
  }
}
