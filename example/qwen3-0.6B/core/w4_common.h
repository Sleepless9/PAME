#ifndef W4_COMMON_H
#define W4_COMMON_H

#include <stdint.h>

#define W4_I4_MIN (-7)
#define W4_I4_MAX 7
#define W4_MAGIC 0x716F776E3362696EULL
#define W4_FORMAT_VERSION 7
#define W4_FORMAT_VERSION_MIN 6
#define W4_WEIGHT_BITS 4
#define W4_ACT_BITS 8
#define W4_TILE_OUTBLK 64
/* v6 tiled matmul weights: int8 tile-major (offline unpack). Embed stays int4. */
#define W4_TILE_WEIGHTS_INT8 1

static inline int8_t w4_unpack_nibble(uint8_t byte, int hi) {
  if (hi) {
    return (int8_t)((int8_t)byte >> 4);
  }
  return (int8_t)((int8_t)(byte << 4) >> 4);
}

static inline void w4_unpack_i4_tile(const uint8_t *packed,
                                     int8_t *out,
                                     int rows,
                                     int gs) {
  const int pack_cols = gs / 2;
  for (int r = 0; r < rows; r++) {
    const uint8_t *row_packed = packed + (size_t)r * (size_t)pack_cols;
    int8_t *row_out = out + (size_t)r * (size_t)gs;
    for (int k = 0; k < gs; k += 2) {
      const uint8_t byte = row_packed[k / 2];
      row_out[k] = w4_unpack_nibble(byte, 0);
      row_out[k + 1] = w4_unpack_nibble(byte, 1);
    }
  }
}

static inline int8_t w4_unpack_i4_at(const uint8_t *packed_row, int k) {
  const uint8_t byte = packed_row[k / 2];
  return w4_unpack_nibble(byte, k & 1);
}

#endif
