#include <stddef.h>
#include <stdint.h>

#include "../../../core/w4_common.h"

void scalar_matmul_quant_i8_i4_f32(float *xout,
                                   const int8_t *xq,
                                   const float *xs,
                                   const uint8_t *wq4,
                                   const float *ws,
                                   int n,
                                   int d,
                                   int gs) {
  const int groups_per_row = (n + gs - 1) / gs;
  for (int i = 0; i < d; i++) {
    const uint8_t *wrow_packed = wq4 + (size_t)i * (size_t)(n / 2);
    double sum = 0.0;
    for (int g = 0; g < groups_per_row; g++) {
      const int base = g * gs;
      const int count = (base + gs <= n) ? gs : (n - base);
      const double wsg = (double)ws[(size_t)i * (size_t)groups_per_row + (size_t)g];
      const double xsg = (double)xs[g];
      for (int j = 0; j < count; j++) {
        const int idx = base + j;
        const int8_t wq = w4_unpack_i4_at(wrow_packed, idx);
        sum += (double)wq * (double)xq[idx] * wsg * xsg;
      }
    }
    xout[i] = (float)sum;
  }
}
