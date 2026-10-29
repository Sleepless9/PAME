#include "backends/ame/core/matmul_quant_ame.h"

#include <stddef.h>
#include <stdint.h>

/* mma[16]=1, mint8[4]=1, msew=0(e8) */
#define AME_MTYPE_INT8 ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
/* mma[16]=1, mint32[6]=1, msew=2(e32) */
#define AME_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)

#define AME_I8_TILE_M_MAX 16
#define AME_I8_TILE_N 16
#define AME_I8_TILE_K_MAX 64

static float ame_i8_sync_cell[1] __attribute__((aligned(64))) = {0};
static int8_t ame_i8_sync_a[1] __attribute__((aligned(64))) = {1};
static int8_t ame_i8_sync_b[1] __attribute__((aligned(64))) = {1};

static inline int ame_i8_msettilem(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("a6");
  asm volatile(".word 0x04055877\n\t# msettilem (aligned to matmul-AB.bin)"
               : "+r"(in), "=r"(out)
               :
               : "memory");
  return (int)out;
}

static inline int ame_i8_msettilen(int rem) {
  register size_t in asm("a0") = (size_t)rem;
  register size_t out asm("t2");
  asm volatile(".word 0x040543f7\n\t# msettilen (aligned to matmul-AB.bin)"
               : "+r"(in), "=r"(out)
               :
               : "memory");
  return (int)out;
}

static inline int ame_i8_msettilek(int rem) {
  /* Encoding 0x0406e6f7 decodes to: rs1=a3, rd=a3.
   * Both input and output MUST be a3 - hardware ignores a0. */
  register size_t inout asm("a3") = (size_t)rem;
  asm volatile(".word 0x0406e6f7\n\t# msettilek a3,a3 (aligned to matmul-AB.bin)"
               : "+r"(inout)
               :
               : "memory");
  return (int)inout;
}

static inline void ame_i8_msettype(uint64_t mtype) {
  register uint64_t a0 asm("a0") = mtype;
  asm volatile(".word 0x00054077\n\t# msettype x0,a0" :: "r"(a0) : "memory");
  /* Match matrix_intrinsic.h SET_MBA0_I32 safety behavior:
   * prevent the compiler from reusing caller-saved regs across msettype. */
  asm volatile("" ::: "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "t0",
               "t1", "t2", "t3", "t4", "t5", "t6", "memory");
}

static inline void ame_i8_mlce32(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e2077\n\t# mlce32.m (aligned to matmul-AB.bin)" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc1(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e20f7\n\t# mlce32.m acc1,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc2(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e2177\n\t# mlce32.m acc2,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc3(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e21f7\n\t# mlce32.m acc3,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc4(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e2277\n\t# mlce32.m acc4,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc5(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e22f7\n\t# mlce32.m acc5,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc6(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e2377\n\t# mlce32.m acc6,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlce32_acc7(const float *base, int stride_bytes) {
  register const float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x005e23f7\n\t# mlce32.m acc7,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e2077\n\t# msce32.m (aligned to matmul-AB.bin)" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc1(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e20f7\n\t# msce32.m acc1,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc2(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e2177\n\t# msce32.m acc2,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc3(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e21f7\n\t# msce32.m acc3,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc4(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e2277\n\t# msce32.m acc4,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc5(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e22f7\n\t# msce32.m acc5,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc6(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e2377\n\t# msce32.m acc6,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_msce32_acc7(float *base, int stride_bytes) {
  register float *rs1 asm("t3") = base;
  register size_t rs2 asm("t0") = (size_t)stride_bytes;
  asm volatile(".word 0x025e23f7\n\t# msce32.m acc7,(t3),t0" :: "r"(rs1), "r"(rs2)
               : "memory");
}

static inline void ame_i8_mlae8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50077\n\t# mlae8.m tr0,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlae8_tr2(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50177\n\t# mlae8.m tr2,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlae8_tr4(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50277\n\t# mlae8.m tr4,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlae8_tr6(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x04b50377\n\t# mlae8.m tr6,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbe8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b500f7\n\t# mlbe8/mlb path" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbe8_tr3(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b501f7\n\t# mlbe8.m tr3,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbe8_tr5(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b502f7\n\t# mlbe8.m tr5,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbe8_tr7(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b503f7\n\t# mlbe8.m tr7,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbt8(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b508f7\n\t# mlbt8/mlbt path aligned to matmul-AB.bin" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbt8_tr3(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b509f7\n\t# mlbte8.m tr3,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbt8_tr5(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b50af7\n\t# mlbte8.m tr5,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mlbt8_tr7(const int8_t *base, int stride_bytes) {
  register const int8_t *a0 asm("a0") = base;
  register size_t a1 asm("a1") = (size_t)stride_bytes;
  asm volatile(".word 0x08b50bf7\n\t# mlbte8.m tr7,(a0),a1" :: "r"(a0), "r"(a1)
               : "memory");
}

static inline void ame_i8_mqma_b(void) {
  asm volatile(".word 0x28180877\n\t# mqma.b.mm acc0,tr0,tr1" ::: "memory");
}

static inline void ame_i8_mqma_b_acc1(void) {
  asm volatile(".word 0x283908f7\n\t# mqma.b.mm acc1,tr2,tr3" ::: "memory");
}

static inline void ame_i8_mqma_b_acc2(void) {
  asm volatile(".word 0x285a0977\n\t# mqma.b.mm acc2,tr4,tr5" ::: "memory");
}

static inline void ame_i8_mqma_b_acc3(void) {
  asm volatile(".word 0x287b09f7\n\t# mqma.b.mm acc3,tr6,tr7" ::: "memory");
}

/* Non-CSR state resync:
 * Run a minimal legal matrix MMA sequence first, then load/store sequence.
 * This avoids relying only on mlce/msce when mstart is unexpectedly non-zero. */
static inline void ame_i8_resync_state(void) {
  (void)ame_i8_msettilem(1);
  (void)ame_i8_msettilen(1);
  (void)ame_i8_msettilek(1);

  ame_i8_msettype(AME_MTYPE_INT8);
  ame_i8_mlae8(ame_i8_sync_a, (int)sizeof(int8_t));
  ame_i8_mlbt8(ame_i8_sync_b, (int)sizeof(int8_t));
  ame_i8_mqma_b();

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_mlce32(ame_i8_sync_cell, (int)sizeof(float));
  ame_i8_msce32(ame_i8_sync_cell, (int)sizeof(float));
}

static inline void ame_memcpy_scalar_u8(void *dst, const void *src, size_t nbytes) {
  uint8_t *d = (uint8_t *)dst;
  const uint8_t *s = (const uint8_t *)src;
  for (size_t i = 0; i < nbytes; ++i) d[i] = s[i];
}

#define AME_MEMCPY_LDST_ACC2(SRC32, DST32, STRIDE, LANE_WORDS) \
  do { \
    ame_i8_mlce32((SRC32), (STRIDE)); \
    ame_i8_mlce32_acc1((SRC32) + (LANE_WORDS), (STRIDE)); \
    ame_i8_msce32((DST32), (STRIDE)); \
    ame_i8_msce32_acc1((DST32) + (LANE_WORDS), (STRIDE)); \
  } while (0)

#define AME_MEMCPY_LDST_ACC4(SRC32, DST32, STRIDE, LANE_WORDS) \
  do { \
    AME_MEMCPY_LDST_ACC2((SRC32), (DST32), (STRIDE), (LANE_WORDS)); \
    ame_i8_mlce32_acc2((SRC32) + (LANE_WORDS) * 2u, (STRIDE)); \
    ame_i8_mlce32_acc3((SRC32) + (LANE_WORDS) * 3u, (STRIDE)); \
    ame_i8_msce32_acc2((DST32) + (LANE_WORDS) * 2u, (STRIDE)); \
    ame_i8_msce32_acc3((DST32) + (LANE_WORDS) * 3u, (STRIDE)); \
  } while (0)

#define AME_MEMCPY_LDST_ACC6(SRC32, DST32, STRIDE, LANE_WORDS) \
  do { \
    AME_MEMCPY_LDST_ACC4((SRC32), (DST32), (STRIDE), (LANE_WORDS)); \
    ame_i8_mlce32_acc4((SRC32) + (LANE_WORDS) * 4u, (STRIDE)); \
    ame_i8_mlce32_acc5((SRC32) + (LANE_WORDS) * 5u, (STRIDE)); \
    ame_i8_msce32_acc4((DST32) + (LANE_WORDS) * 4u, (STRIDE)); \
    ame_i8_msce32_acc5((DST32) + (LANE_WORDS) * 5u, (STRIDE)); \
  } while (0)

#define AME_MEMCPY_LDST_ACC8(SRC32, DST32, STRIDE, LANE_WORDS) \
  do { \
    AME_MEMCPY_LDST_ACC6((SRC32), (DST32), (STRIDE), (LANE_WORDS)); \
    ame_i8_mlce32_acc6((SRC32) + (LANE_WORDS) * 6u, (STRIDE)); \
    ame_i8_mlce32_acc7((SRC32) + (LANE_WORDS) * 7u, (STRIDE)); \
    ame_i8_msce32_acc6((DST32) + (LANE_WORDS) * 6u, (STRIDE)); \
    ame_i8_msce32_acc7((DST32) + (LANE_WORDS) * 7u, (STRIDE)); \
  } while (0)

#define AME_DEFINE_MEMCPY_TYPED(PREFIX, CTYPE, ELEM_BYTES, REQ_M, REQ_N, ACC, LDST_SEQ) \
  void ame_memcpy_##PREFIX##_acc##ACC(CTYPE *dst, const CTYPE *src, size_t count) { \
    if (!dst || !src || count == 0) return; \
    if (count > (SIZE_MAX / (size_t)(ELEM_BYTES))) return; \
    const size_t nbytes = count * (size_t)(ELEM_BYTES); \
    uint8_t *d = (uint8_t *)dst; \
    const uint8_t *s = (const uint8_t *)src; \
    if ((((uintptr_t)d) & 3u) || (((uintptr_t)s) & 3u)) { \
      ame_memcpy_scalar_u8(d, s, nbytes); \
      return; \
    } \
    size_t words = nbytes / sizeof(int32_t); \
    const size_t tail = nbytes % sizeof(int32_t); \
    if (words != 0) { \
      ame_i8_resync_state(); \
      int tile_m = ame_i8_msettilem((REQ_M)); \
      int tile_n = ame_i8_msettilen((REQ_N)); \
      if (tile_m > (REQ_M)) tile_m = (REQ_M); \
      if (tile_n > (REQ_N)) tile_n = (REQ_N); \
      if (tile_m <= 0 || tile_n <= 0) { \
        ame_memcpy_scalar_u8(d, s, nbytes); \
        return; \
      } \
      ame_i8_msettype(AME_MTYPE_INT32); \
      const int stride_bytes = tile_n * (int)sizeof(float); \
      const size_t lane_words = (size_t)tile_m * (size_t)tile_n; \
      const size_t block_words = lane_words * (size_t)(ACC); \
      while (words >= block_words) { \
        const int32_t *src32 = (const int32_t *)s; \
        int32_t *dst32 = (int32_t *)d; \
        LDST_SEQ(src32, dst32, stride_bytes, lane_words); \
        const size_t step_bytes = block_words * sizeof(int32_t); \
        s += step_bytes; \
        d += step_bytes; \
        words -= block_words; \
      } \
      asm volatile("fence rw,rw" ::: "memory"); \
      ame_i8_resync_state(); \
      asm volatile("fence rw,rw" ::: "memory"); \
    } \
    { \
      const size_t rem_bytes = words * sizeof(int32_t) + tail; \
      if (rem_bytes) ame_memcpy_scalar_u8(d, s, rem_bytes); \
    } \
  }

AME_DEFINE_MEMCPY_TYPED(i8, int8_t, 1, 4, 4, 2, AME_MEMCPY_LDST_ACC2)
AME_DEFINE_MEMCPY_TYPED(i8, int8_t, 1, 4, 4, 4, AME_MEMCPY_LDST_ACC4)
AME_DEFINE_MEMCPY_TYPED(i8, int8_t, 1, 4, 4, 6, AME_MEMCPY_LDST_ACC6)
AME_DEFINE_MEMCPY_TYPED(i8, int8_t, 1, 4, 4, 8, AME_MEMCPY_LDST_ACC8)

AME_DEFINE_MEMCPY_TYPED(i32, int32_t, sizeof(int32_t), 8, 8, 2, AME_MEMCPY_LDST_ACC2)
AME_DEFINE_MEMCPY_TYPED(i32, int32_t, sizeof(int32_t), 8, 8, 4, AME_MEMCPY_LDST_ACC4)
AME_DEFINE_MEMCPY_TYPED(i32, int32_t, sizeof(int32_t), 8, 8, 6, AME_MEMCPY_LDST_ACC6)
AME_DEFINE_MEMCPY_TYPED(i32, int32_t, sizeof(int32_t), 8, 8, 8, AME_MEMCPY_LDST_ACC8)

AME_DEFINE_MEMCPY_TYPED(f32, float, sizeof(float), 8, 8, 2, AME_MEMCPY_LDST_ACC2)
AME_DEFINE_MEMCPY_TYPED(f32, float, sizeof(float), 8, 8, 4, AME_MEMCPY_LDST_ACC4)
AME_DEFINE_MEMCPY_TYPED(f32, float, sizeof(float), 8, 8, 6, AME_MEMCPY_LDST_ACC6)
AME_DEFINE_MEMCPY_TYPED(f32, float, sizeof(float), 8, 8, 8, AME_MEMCPY_LDST_ACC8)

void ame_memcpy_acc2(void *dst, const void *src, size_t nbytes) {
  ame_memcpy_i8_acc2((int8_t *)dst, (const int8_t *)src, nbytes);
}

void ame_memcpy_acc4(void *dst, const void *src, size_t nbytes) {
  ame_memcpy_i8_acc4((int8_t *)dst, (const int8_t *)src, nbytes);
}

void ame_memcpy_acc6(void *dst, const void *src, size_t nbytes) {
  ame_memcpy_i8_acc6((int8_t *)dst, (const int8_t *)src, nbytes);
}

void ame_memcpy_acc8(void *dst, const void *src, size_t nbytes) {
  ame_memcpy_i8_acc8((int8_t *)dst, (const int8_t *)src, nbytes);
}

__attribute__((noipa, noinline))
int ame_matmul_singleacc_ab_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k) {
  /* Integer GEMM (AB layout)
   *
   * Dimensions:
   *   A: [m x k], B: [k x n], C: [m x n]
   *
   * Formula:
   *   C[i,j] <- C[i,j] + sum_{t=0..k-1} A[i,t] * B[t,j]
   */
  ame_i8_resync_state();

  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = n * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(float);
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    int req_m = m - i;
    if (req_m > AME_I8_TILE_M_MAX) req_m = AME_I8_TILE_M_MAX;
    tile_m = ame_i8_msettilem(req_m);
    if (tile_m <= 0) break;
    if (tile_m > AME_I8_TILE_M_MAX) tile_m = AME_I8_TILE_M_MAX;
    for (int j = 0; j < n; j += tile_n) {
      int req_n = n - j;
      if (req_n > AME_I8_TILE_N) req_n = AME_I8_TILE_N;
      tile_n = ame_i8_msettilen(req_n);
      if (tile_n <= 0) break;
      if (tile_n > AME_I8_TILE_N) tile_n = AME_I8_TILE_N;
      float *cptr = C + i * n + j;
      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_mlce32(cptr, ldc_bytes);

      for (int kk = 0; kk < k; kk += tile_k) {
        int req_k = k - kk;
        if (req_k > AME_I8_TILE_K_MAX) req_k = AME_I8_TILE_K_MAX;
        tile_k = ame_i8_msettilek(req_k);
        if (tile_k <= 0) break;
        if (tile_k > AME_I8_TILE_K_MAX) tile_k = AME_I8_TILE_K_MAX;

        const int8_t *aptr = A + i * k + kk;
        const int8_t *bptr = B + (size_t)kk * (size_t)n + (size_t)j;

        ame_i8_msettype(AME_MTYPE_INT8);
        ame_i8_mlae8(aptr, lda_bytes);
        ame_i8_mlbt8(bptr, ldb_bytes);
        ame_i8_mqma_b();
      }

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_msce32(cptr, ldc_bytes);
    }
  }

  /* Symmetric epilogue: fence + resync. Resync clears any residual
   * mstart/mtype state so the next AME call starts from a known baseline. */
  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_matmul_quadacc_ab_i8_i8_f32(const int8_t *A,
                                    const int8_t *B,
                                    float *C,
                                    int m,
                                    int n,
                                    int k) {
  /* 4-acc AB kernel.
   * A: [m x k], B: [k x n], C: [m x n]
   * Use 4 N-lanes of width 16 each:
   *   acc0 <- tr0/tr1
   *   acc1 <- tr2/tr3
   *   acc2 <- tr4/tr5
   *   acc3 <- tr6/tr7
   */
  ame_i8_resync_state();

  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = n * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(float);
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    int req_m = m - i;
    if (req_m > AME_I8_TILE_M_MAX) req_m = AME_I8_TILE_M_MAX;
    tile_m = ame_i8_msettilem(req_m);
    if (tile_m <= 0) break;
    if (tile_m > AME_I8_TILE_M_MAX) tile_m = AME_I8_TILE_M_MAX;
    for (int j = 0; j < n; ) {
      int req_n = n - j;
      if (req_n > AME_I8_TILE_N) req_n = AME_I8_TILE_N;
      tile_n = ame_i8_msettilen(req_n);
      if (tile_n <= 0) break;
      if (tile_n > AME_I8_TILE_N) tile_n = AME_I8_TILE_N;

      const int j1 = j + tile_n;
      const int j2 = j1 + tile_n;
      const int j3 = j2 + tile_n;
      const int lane1 = (j1 + tile_n <= n) ? 1 : 0;
      const int lane2 = (j2 + tile_n <= n) ? 1 : 0;
      const int lane3 = (j3 + tile_n <= n) ? 1 : 0;
      const int lanes = 1 + lane1 + lane2 + lane3;

      float *cptr0 = C + i * n + j;
      float *cptr1 = 0;
      float *cptr2 = 0;
      float *cptr3 = 0;

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_mlce32(cptr0, ldc_bytes);
      if (lane1) {
        cptr1 = C + i * n + j1;
        ame_i8_mlce32_acc1(cptr1, ldc_bytes);
      }
      if (lane2) {
        cptr2 = C + i * n + j2;
        ame_i8_mlce32_acc2(cptr2, ldc_bytes);
      }
      if (lane3) {
        cptr3 = C + i * n + j3;
        ame_i8_mlce32_acc3(cptr3, ldc_bytes);
      }

      for (int kk = 0; kk < k; kk += tile_k) {
        int req_k = k - kk;
        if (req_k > AME_I8_TILE_K_MAX) req_k = AME_I8_TILE_K_MAX;
        tile_k = ame_i8_msettilek(req_k);
        if (tile_k <= 0) break;
        if (tile_k > AME_I8_TILE_K_MAX) tile_k = AME_I8_TILE_K_MAX;
        const int8_t *aptr = A + i * k + kk;

        ame_i8_msettype(AME_MTYPE_INT8);
        const int8_t *bptr0 = B + (size_t)kk * (size_t)n + (size_t)j;
        ame_i8_mlae8(aptr, lda_bytes);
        ame_i8_mlbt8(bptr0, ldb_bytes);
        ame_i8_mqma_b();

        if (lane1) {
          const int8_t *bptr1 = B + (size_t)kk * (size_t)n + (size_t)j1;
          ame_i8_mlae8_tr2(aptr, lda_bytes);
          ame_i8_mlbt8_tr3(bptr1, ldb_bytes);
          ame_i8_mqma_b_acc1();
        }
        if (lane2) {
          const int8_t *bptr2 = B + (size_t)kk * (size_t)n + (size_t)j2;
          ame_i8_mlae8_tr4(aptr, lda_bytes);
          ame_i8_mlbt8_tr5(bptr2, ldb_bytes);
          ame_i8_mqma_b_acc2();
        }
        if (lane3) {
          const int8_t *bptr3 = B + (size_t)kk * (size_t)n + (size_t)j3;
          ame_i8_mlae8_tr6(aptr, lda_bytes);
          ame_i8_mlbt8_tr7(bptr3, ldb_bytes);
          ame_i8_mqma_b_acc3();
        }
      }

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_msce32(cptr0, ldc_bytes);
      if (lane1) ame_i8_msce32_acc1(cptr1, ldc_bytes);
      if (lane2) ame_i8_msce32_acc2(cptr2, ldc_bytes);
      if (lane3) ame_i8_msce32_acc3(cptr3, ldc_bytes);

      j += tile_n * lanes;
    }
  }

  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_matmul_ab_i8_i8_f32(const int8_t *A,
                            const int8_t *B,
                            float *C,
                            int m,
                            int n,
                            int k) {
  return ame_matmul_quadacc_ab_i8_i8_f32(A, B, C, m, n, k);
}

__attribute__((noipa, noinline))
int ame_matmul_dualacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k) {
  /* Integer GEMM (ABT layout)
   *
   * Dimensions:
   *   A: [m x k], B: [n x k], C: [m x n]
   *
   * Formula:
   *   C[i,j] <- C[i,j] + sum_{t=0..k-1} A[i,t] * B[j,t]
   *
   * Backend contract:
   *   ABT loads B through the mlbe8 tr1 path directly from B_nk row-major.
   */
  ame_i8_resync_state();

  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = k * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(float);
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    int req_m = m - i;
    if (req_m > AME_I8_TILE_M_MAX) req_m = AME_I8_TILE_M_MAX;
    tile_m = ame_i8_msettilem(req_m);
    if (tile_m <= 0) break;
    if (tile_m > AME_I8_TILE_M_MAX) tile_m = AME_I8_TILE_M_MAX;
    for (int j = 0; j < n; ) {
      int req_n = n - j;
      if (req_n > AME_I8_TILE_N) req_n = AME_I8_TILE_N;
      tile_n = ame_i8_msettilen(req_n);
      if (tile_n <= 0) break;
      if (tile_n > AME_I8_TILE_N) tile_n = AME_I8_TILE_N;
      float *cptr0 = C + i * n + j;
      const int j1 = j + tile_n;
      const int dual_n = (j1 + tile_n <= n) ? 1 : 0;
      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_mlce32(cptr0, ldc_bytes);
      float *cptr1 = 0;
      if (dual_n) {
        cptr1 = C + i * n + j1;
        ame_i8_mlce32_acc1(cptr1, ldc_bytes);
      }

      for (int kk = 0; kk < k; kk += tile_k) {
        int req_k = k - kk;
        if (req_k > AME_I8_TILE_K_MAX) req_k = AME_I8_TILE_K_MAX;
        tile_k = ame_i8_msettilek(req_k);
        if (tile_k <= 0) break;
        if (tile_k > AME_I8_TILE_K_MAX) tile_k = AME_I8_TILE_K_MAX;
        const int8_t *aptr0 = A + i * k + kk;

        ame_i8_msettype(AME_MTYPE_INT8);
        ame_i8_mlae8(aptr0, lda_bytes);
        const int8_t *bptr0 = B + (size_t)j * (size_t)k + (size_t)kk;
        ame_i8_mlbe8(bptr0, ldb_bytes);
        ame_i8_mqma_b();
        if (dual_n) {
          const int8_t *aptr1 = A + i * k + kk;
          const int8_t *bptr1 = B + (size_t)j1 * (size_t)k + (size_t)kk;
          ame_i8_mlae8_tr2(aptr1, lda_bytes);
          ame_i8_mlbe8_tr3(bptr1, ldb_bytes);
          ame_i8_mqma_b_acc1();
        }
      }

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_msce32(cptr0, ldc_bytes);
      if (dual_n) {
        ame_i8_msce32_acc1(cptr1, ldc_bytes);
        j += tile_n + tile_n;
      } else {
        j += tile_n;
      }
    }
  }

  /* Symmetric epilogue: fence + resync (mirror of AB above). */
  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");

  return 0;
}

static int ame_matmul_abt_n64k64_fastpath(const int8_t *A,
                                          const int8_t *B,
                                          float *C,
                                          int m) {
  if (m <= 0 || m > AME_I8_TILE_M_MAX) return -1;

  ame_i8_resync_state();

  const int tile_m = ame_i8_msettilem(m);
  const int tile_n = ame_i8_msettilen(AME_I8_TILE_N);
  const int tile_k = ame_i8_msettilek(AME_I8_TILE_K_MAX);
  if (tile_m != m || tile_n != AME_I8_TILE_N || tile_k != AME_I8_TILE_K_MAX) {
    return -1;
  }

  const int lda_bytes = AME_I8_TILE_K_MAX * (int)sizeof(int8_t);
  const int ldb_bytes = AME_I8_TILE_K_MAX * (int)sizeof(int8_t);
  const int ldc_bytes = (AME_I8_TILE_N * 4) * (int)sizeof(float);

  float *cptr0 = C;
  float *cptr1 = C + AME_I8_TILE_N;
  float *cptr2 = C + AME_I8_TILE_N * 2;
  float *cptr3 = C + AME_I8_TILE_N * 3;

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_mlce32(cptr0, ldc_bytes);
  ame_i8_mlce32_acc1(cptr1, ldc_bytes);
  ame_i8_mlce32_acc2(cptr2, ldc_bytes);
  ame_i8_mlce32_acc3(cptr3, ldc_bytes);

  ame_i8_msettype(AME_MTYPE_INT8);
  ame_i8_mlae8(A, lda_bytes);
  ame_i8_mlbe8(B, ldb_bytes);
  ame_i8_mqma_b();

  ame_i8_mlae8_tr2(A, lda_bytes);
  ame_i8_mlbe8_tr3(B + (size_t)AME_I8_TILE_N * (size_t)AME_I8_TILE_K_MAX,
                   ldb_bytes);
  ame_i8_mqma_b_acc1();

  ame_i8_mlae8_tr4(A, lda_bytes);
  ame_i8_mlbe8_tr5(B + (size_t)AME_I8_TILE_N * 2u * (size_t)AME_I8_TILE_K_MAX,
                   ldb_bytes);
  ame_i8_mqma_b_acc2();

  ame_i8_mlae8_tr6(A, lda_bytes);
  ame_i8_mlbe8_tr7(B + (size_t)AME_I8_TILE_N * 3u * (size_t)AME_I8_TILE_K_MAX,
                   ldb_bytes);
  ame_i8_mqma_b_acc3();

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_msce32(cptr0, ldc_bytes);
  ame_i8_msce32_acc1(cptr1, ldc_bytes);
  ame_i8_msce32_acc2(cptr2, ldc_bytes);
  ame_i8_msce32_acc3(cptr3, ldc_bytes);

  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_matmul_quadacc_abt_i8_i8_f32(const int8_t *A,
                                      const int8_t *B,
                                      float *C,
                                      int m,
                                      int n,
                                      int k) {
  /* 4-acc ABT kernel.
   * Uses 8 tile regs: (tr0,tr1), (tr2,tr3), (tr4,tr5), (tr6,tr7)
   * and 4 accumulators: acc0..acc3.
   *
   * Opcode basis was verified from a clang-assembled probe:
   *   acc2 load/store: 0x005e2177 / 0x025e2177
   *   acc3 load/store: 0x005e21f7 / 0x025e21f7
   *   tr4/tr6 loads   : 0x04b50277 / 0x04b50377
   *   tr5/tr7 loads   : 0x08b502f7 / 0x08b503f7
   *   acc2/acc3 mqma  : 0x285a0977 / 0x287b09f7
   */
  if (n == AME_I8_TILE_N * 4 && k == AME_I8_TILE_K_MAX &&
      m > 0 && m <= AME_I8_TILE_M_MAX) {
    if (ame_matmul_abt_n64k64_fastpath(A, B, C, m) == 0) return 0;
  }

  ame_i8_resync_state();

  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = k * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(float);
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    int req_m = m - i;
    if (req_m > AME_I8_TILE_M_MAX) req_m = AME_I8_TILE_M_MAX;
    tile_m = ame_i8_msettilem(req_m);
    if (tile_m <= 0) break;
    if (tile_m > AME_I8_TILE_M_MAX) tile_m = AME_I8_TILE_M_MAX;
    for (int j = 0; j < n; ) {
      int req_n = n - j;
      if (req_n > AME_I8_TILE_N) req_n = AME_I8_TILE_N;
      tile_n = ame_i8_msettilen(req_n);
      if (tile_n <= 0) break;
      if (tile_n > AME_I8_TILE_N) tile_n = AME_I8_TILE_N;

      const int j1 = j + tile_n;
      const int j2 = j1 + tile_n;
      const int j3 = j2 + tile_n;
      const int lane1 = (j1 + tile_n <= n) ? 1 : 0;
      const int lane2 = (j2 + tile_n <= n) ? 1 : 0;
      const int lane3 = (j3 + tile_n <= n) ? 1 : 0;
      const int lanes = 1 + lane1 + lane2 + lane3;

      float *cptr0 = C + i * n + j;
      float *cptr1 = 0;
      float *cptr2 = 0;
      float *cptr3 = 0;

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_mlce32(cptr0, ldc_bytes);
      if (lane1) {
        cptr1 = C + i * n + j1;
        ame_i8_mlce32_acc1(cptr1, ldc_bytes);
      }
      if (lane2) {
        cptr2 = C + i * n + j2;
        ame_i8_mlce32_acc2(cptr2, ldc_bytes);
      }
      if (lane3) {
        cptr3 = C + i * n + j3;
        ame_i8_mlce32_acc3(cptr3, ldc_bytes);
      }

      for (int kk = 0; kk < k; kk += tile_k) {
        int req_k = k - kk;
        if (req_k > AME_I8_TILE_K_MAX) req_k = AME_I8_TILE_K_MAX;
        tile_k = ame_i8_msettilek(req_k);
        if (tile_k <= 0) break;
        if (tile_k > AME_I8_TILE_K_MAX) tile_k = AME_I8_TILE_K_MAX;
        const int8_t *aptr = A + i * k + kk;

        ame_i8_msettype(AME_MTYPE_INT8);
        const int8_t *bptr0 = B + (size_t)j * (size_t)k + (size_t)kk;
        ame_i8_mlae8(aptr, lda_bytes);
        ame_i8_mlbe8(bptr0, ldb_bytes);
        ame_i8_mqma_b();

        if (lane1) {
          const int8_t *bptr1 = B + (size_t)j1 * (size_t)k + (size_t)kk;
          ame_i8_mlae8_tr2(aptr, lda_bytes);
          ame_i8_mlbe8_tr3(bptr1, ldb_bytes);
          ame_i8_mqma_b_acc1();
        }
        if (lane2) {
          const int8_t *bptr2 = B + (size_t)j2 * (size_t)k + (size_t)kk;
          ame_i8_mlae8_tr4(aptr, lda_bytes);
          ame_i8_mlbe8_tr5(bptr2, ldb_bytes);
          ame_i8_mqma_b_acc2();
        }
        if (lane3) {
          const int8_t *bptr3 = B + (size_t)j3 * (size_t)k + (size_t)kk;
          ame_i8_mlae8_tr6(aptr, lda_bytes);
          ame_i8_mlbe8_tr7(bptr3, ldb_bytes);
          ame_i8_mqma_b_acc3();
        }
      }

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_msce32(cptr0, ldc_bytes);
      if (lane1) ame_i8_msce32_acc1(cptr1, ldc_bytes);
      if (lane2) ame_i8_msce32_acc2(cptr2, ldc_bytes);
      if (lane3) ame_i8_msce32_acc3(cptr3, ldc_bytes);

      j += tile_n * lanes;
    }
  }

  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_matmul_singleacc_abt_i8_i8_f32(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int m,
                                        int n,
                                        int k) {
  ame_i8_resync_state();

  const int lda_bytes = k * (int)sizeof(int8_t);
  const int ldb_bytes = k * (int)sizeof(int8_t);
  const int ldc_bytes = n * (int)sizeof(float);
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;

  for (int i = 0; i < m; i += tile_m) {
    int req_m = m - i;
    if (req_m > AME_I8_TILE_M_MAX) req_m = AME_I8_TILE_M_MAX;
    tile_m = ame_i8_msettilem(req_m);
    if (tile_m <= 0) break;
    if (tile_m > AME_I8_TILE_M_MAX) tile_m = AME_I8_TILE_M_MAX;
    for (int j = 0; j < n; j += tile_n) {
      int req_n = n - j;
      if (req_n > AME_I8_TILE_N) req_n = AME_I8_TILE_N;
      tile_n = ame_i8_msettilen(req_n);
      if (tile_n <= 0) break;
      if (tile_n > AME_I8_TILE_N) tile_n = AME_I8_TILE_N;
      float *cptr = C + i * n + j;
      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_mlce32(cptr, ldc_bytes);

      for (int kk = 0; kk < k; kk += tile_k) {
        int req_k = k - kk;
        if (req_k > AME_I8_TILE_K_MAX) req_k = AME_I8_TILE_K_MAX;
        tile_k = ame_i8_msettilek(req_k);
        if (tile_k <= 0) break;
        if (tile_k > AME_I8_TILE_K_MAX) tile_k = AME_I8_TILE_K_MAX;
        const int8_t *aptr = A + i * k + kk;

        ame_i8_msettype(AME_MTYPE_INT8);
        ame_i8_mlae8(aptr, lda_bytes);
        const int8_t *bptr = B + (size_t)j * (size_t)k + (size_t)kk;
        ame_i8_mlbe8(bptr, ldb_bytes);
        ame_i8_mqma_b();
      }

      ame_i8_msettype(AME_MTYPE_INT32);
      ame_i8_msce32(cptr, ldc_bytes);
    }
  }

  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_matmul_abt_i8_i8_f32(const int8_t *A,
                             const int8_t *B,
                             float *C,
                             int m,
                             int n,
                             int k) {
  return ame_matmul_quadacc_abt_i8_i8_f32(A, B, C, m, n, k);
}

__attribute__((noipa, noinline))
int ame_probe_matmul_ab_i8_i8_f32_once(const int8_t *A,
                                       const int8_t *B,
                                       float *C,
                                       int req_m,
                                       int req_n,
                                       int req_k,
                                       int *got_m,
                                       int *got_n,
                                       int *got_k) {
  if (!A || !B || !C || req_m <= 0 || req_n <= 0 || req_k <= 0) return -1;

  ame_i8_resync_state();

  const int tm = ame_i8_msettilem(req_m);
  const int tn = ame_i8_msettilen(req_n);
  const int tk = ame_i8_msettilek(req_k);

  if (got_m) *got_m = tm;
  if (got_n) *got_n = tn;
  if (got_k) *got_k = tk;

  if (tm <= 0 || tn <= 0 || tk <= 0) return -2;
  if (tm > AME_I8_TILE_M_MAX || tn > AME_I8_TILE_N || tk > AME_I8_TILE_K_MAX) return -3;

  const int lda_bytes = tk * (int)sizeof(int8_t);
  const int ldb_bytes = tn * (int)sizeof(int8_t);
  const int ldc_bytes = tn * (int)sizeof(float);

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_mlce32(C, ldc_bytes);

  ame_i8_msettype(AME_MTYPE_INT8);
  ame_i8_mlae8(A, lda_bytes);
  ame_i8_mlbt8(B, ldb_bytes);
  ame_i8_mqma_b();

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_msce32(C, ldc_bytes);

  /* Symmetric epilogue: fence + resync. */
  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}

__attribute__((noipa, noinline))
int ame_probe_matmul_abt_i8_i8_f32_once(const int8_t *A,
                                        const int8_t *B,
                                        float *C,
                                        int req_m,
                                        int req_n,
                                        int req_k,
                                        int *got_m,
                                        int *got_n,
                                        int *got_k) {
  if (!A || !B || !C || req_m <= 0 || req_n <= 0 || req_k <= 0) return -1;

  ame_i8_resync_state();

  const int tm = ame_i8_msettilem(req_m);
  const int tn = ame_i8_msettilen(req_n);
  const int tk = ame_i8_msettilek(req_k);

  if (got_m) *got_m = tm;
  if (got_n) *got_n = tn;
  if (got_k) *got_k = tk;

  if (tm <= 0 || tn <= 0 || tk <= 0) return -2;
  if (tm > AME_I8_TILE_M_MAX || tn > AME_I8_TILE_N || tk > AME_I8_TILE_K_MAX) return -3;

  const int lda_bytes = tk * (int)sizeof(int8_t);
  const int ldb_bytes = tk * (int)sizeof(int8_t);
  const int ldc_bytes = tn * (int)sizeof(float);

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_mlce32(C, ldc_bytes);

  ame_i8_msettype(AME_MTYPE_INT8);
  ame_i8_mlae8(A, lda_bytes);
  ame_i8_mlbe8(B, ldb_bytes);
  ame_i8_mqma_b();

  ame_i8_msettype(AME_MTYPE_INT32);
  ame_i8_msce32(C, ldc_bytes);

  asm volatile("fence rw,rw" ::: "memory");
  ame_i8_resync_state();
  asm volatile("fence rw,rw" ::: "memory");
  return 0;
}
