#include "backends/mix/core/ops_mix.h"
#include "ame_mix_offline_profile.h"
#include "w4_common.h"

/* ----------------------------------------------------------------------
 * AME mix backend — W4A8 offline tile-pack variant.
 *
 * Same symbol set and call-site contract as the INT8 offline backend:
 *
 *   ame_matmul_quant_noblk_i8_i8_f32(..., n, d, gs, w_row_stride);
 *   ame_matmul_quant_blocked_i8_i8_f32(...);
 *   ame_matmul_quant_batch_i8_i8_f32(..., T, n, d, gs, w_row_stride);
 *   ame_matmul_quant_i8_i8_f32(..., n, d, gs);
 *
 * Weights `wq` are int8 in tile-major layout from
 * convert_tools/repack_to_tile_w4.py (blob v6, offline int4→int8 unpack):
 *
 *   wq_tiled[d / OUTBLK][stride_w / gs][OUTBLK][gs]
 *
 * mlbe8 loads B tiles directly from the blob; no runtime unpack scratch.
 *
 * Geometry contract (verified by convert_to_int8.py):
 *   d % OUTBLK    == 0
 *   stride_w % gs == 0
 *   gs <= AME_MIX_MAX_GS
 *
 * The runtime caller still passes the logical [d, n] shape and
 * w_row_stride (= n in current Qwen3 paths). The caller does not need
 * to know about the tile layout.
 * --------------------------------------------------------------------*/

#define AME_MIX_OUTBLK 64      /* must match TILE_OUTBLK in convert_to_int8.py */
#define AME_MIX_M_TILE 16      /* hardware M tile cap */
#define AME_MIX_MAX_GS 128     /* group-size cap for static buffers */
#define AME_MIX_HW_N 16
#define AME_MIX_HW_K 64
#define AME_MIX_BREUSE_M 64    /* 4 A Tr blocks x 16 rows */

#define AME_MIX_MTYPE_INT8  ((1ULL << 16) | (1ULL << 4) | 0x0ULL)
#define AME_MIX_MTYPE_INT32 ((1ULL << 16) | (1ULL << 6) | 0x2ULL)

#ifndef AME_MIX_PROFILE
#define AME_MIX_PROFILE 0
#endif

static float ame_mix_c_pack[AME_MIX_M_TILE * AME_MIX_OUTBLK] __attribute__((aligned(64)));
static volatile AmeMixOfflineProfile g_ame_profile_detail;

#if AME_MIX_PROFILE
#define AME_PROFILE_ADD(field, value) \
  do { g_ame_profile_detail.field += (uint64_t)(value); } while (0)
#define AME_STAT_INC(field) \
  do { (field)++; } while (0)
#define AME_STAT_ADD(field, value) \
  do { (field) += (uint64_t)(value); } while (0)
#else
#define AME_PROFILE_ADD(field, value) do { (void)(value); } while (0)
#define AME_STAT_INC(field) do { } while (0)
#define AME_STAT_ADD(field, value) do { (void)sizeof(value); } while (0)
#endif

static inline void ame_profile_add_breuse_k64(int iblocks) {
  const uint64_t outer = (uint64_t)iblocks;
  const uint64_t nblocks = outer * (AME_MIX_OUTBLK / AME_MIX_HW_N);
  AME_PROFILE_ADD(msettilem, outer + 3 * nblocks);
  AME_PROFILE_ADD(msettilen, outer + 3 * nblocks);
  AME_PROFILE_ADD(msettilek, outer + nblocks);
  AME_PROFILE_ADD(msettype, outer + 3 * nblocks);
  AME_PROFILE_ADD(mlae8, 4 * outer);
  AME_PROFILE_ADD(mlbe8, nblocks);
  AME_PROFILE_ADD(mlce32, 4 * nblocks);
  AME_PROFILE_ADD(mqma_b, 4 * nblocks);
  AME_PROFILE_ADD(msce32, 4 * nblocks);
}

static inline void ame_profile_add_2a4b(int pairs, int tails, int ksteps) {
  const uint64_t pair_count = (uint64_t)pairs;
  const uint64_t tail_count = (uint64_t)tails;
  const uint64_t step_count = (uint64_t)ksteps;
  AME_PROFILE_ADD(msettilem, pair_count + tail_count);
  AME_PROFILE_ADD(msettilen, pair_count + tail_count);
  AME_PROFILE_ADD(msettilek, step_count * (pair_count + tail_count));
  AME_PROFILE_ADD(msettype, (2 + step_count) * (pair_count + tail_count));
  AME_PROFILE_ADD(mlae8, step_count * (2 * pair_count + tail_count));
  AME_PROFILE_ADD(mlbe8, step_count * (4 * pair_count + 4 * tail_count));
  AME_PROFILE_ADD(mlce32, 8 * pair_count + 4 * tail_count);
  AME_PROFILE_ADD(mqma_b, step_count * (8 * pair_count + 4 * tail_count));
  AME_PROFILE_ADD(msce32, 8 * pair_count + 4 * tail_count);
  AME_PROFILE_ADD(fence_rw_rw, pair_count + tail_count);
}

static inline void ame_profile_add_decode_1a4b(int ksteps, int msettilek_count) {
  const uint64_t step_count = (uint64_t)ksteps;
  AME_PROFILE_ADD(msettilek, msettilek_count);
  AME_PROFILE_ADD(msettype, 2 + step_count);
  AME_PROFILE_ADD(mlae8, step_count);
  AME_PROFILE_ADD(mlbe8, 4 * step_count);
  AME_PROFILE_ADD(mlce32, 4);
  AME_PROFILE_ADD(mqma_b, 4 * step_count);
  AME_PROFILE_ADD(msce32, 4);
}

static inline void ame_profile_add_decode_1a8b2out(int ksteps, int msettilek_count) {
  const uint64_t step_count = (uint64_t)ksteps;
  AME_PROFILE_ADD(msettilek, msettilek_count);
  AME_PROFILE_ADD(msettype, 2 + step_count);
  AME_PROFILE_ADD(mlae8, step_count);
  AME_PROFILE_ADD(mlbe8, 8 * step_count);
  AME_PROFILE_ADD(mlce32, 8);
  AME_PROFILE_ADD(mqma_b, 8 * step_count);
  AME_PROFILE_ADD(msce32, 8);
}

static inline void ame_profile_add_aresident_k128_load_a(void) {
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);
  AME_PROFILE_ADD(msettilek, 1);
  AME_PROFILE_ADD(msettype, 1);
  AME_PROFILE_ADD(mlae8, 4);
}

static inline void ame_profile_add_aresident_k128_n64(void) {
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);
  AME_PROFILE_ADD(msettilek, 1);
  AME_PROFILE_ADD(msettype, 3);
  AME_PROFILE_ADD(mlbe8, 8);
  AME_PROFILE_ADD(mlce32, 8);
  AME_PROFILE_ADD(mqma_b, 16);
  AME_PROFILE_ADD(msce32, 8);
}

/* ----------------------------------------------------------------------
 * Prefill B-register reuse fast path for K=64, N=64, M in 64-row chunks.
 *
 *   MLA/A tr0..tr3 = four M16 A tiles
 *   MLB/B tr4      = one N16 B tile
 *   acc0..3         = four M16xN16 output tiles
 *
 * The register assignment follows the current MLA=tr0..3 / MLB=tr4..7
 * convention; single-live B uses even tr4.
 * The blocking shape matches the K64 B-reuse kernel; revalidate on FPGA after
 * changing the tile-register assignment.
 *
 * The generic AME matmul remains the fallback for tails and non-K64 shapes.
 * --------------------------------------------------------------------*/

/* Single-token decode core. The caller programs the fixed M=1/N=16 shape once
 * per matmul and passes a persistent zero tile for accumulator initialisation.
 * Only K needs to be re-issued when a non-64 tail changes the active width. */
static void amex_decode_1a4b_n64_i8_i8_f32(const int8_t *A,
                                           const int8_t *B,
                                           float *C,
                                           const float *acc_zero,
                                           int k,
                                           int lda,
                                           int *active_k) {
  if (!A || !B || !C || !acc_zero || !active_k || k <= 0) return;

  const int ldb = k;
  const int ldc = AME_MIX_OUTBLK * (int)sizeof(float);
  const int8_t *b0base = B + (size_t)0 * AME_MIX_HW_N * k;
  const int8_t *b1base = B + (size_t)1 * AME_MIX_HW_N * k;
  const int8_t *b2base = B + (size_t)2 * AME_MIX_HW_N * k;
  const int8_t *b3base = B + (size_t)3 * AME_MIX_HW_N * k;
  int msettilek_count = 0;

  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

  for (int kk = 0; kk < k; kk += AME_MIX_HW_K) {
    int tk = k - kk;
    if (tk > AME_MIX_HW_K) tk = AME_MIX_HW_K;
    if (*active_k != tk) {
      asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(tk)) : "a3", "memory");
      *active_k = tk;
      msettilek_count++;
    }
    asm volatile(
        "msettype x0, %[type_i8] # 0x00054077\n\t"
        "mlae8.m tr0, (%[ap]), %[lda]\n\t"
        "mlbe8.m tr4, (%[b0]), %[ldb]\n\t"
        "mlbe8.m tr5, (%[b1]), %[ldb]\n\t"
        "mqma.b.mm acc0, tr0, tr4\n\t"
        "mlbe8.m tr6, (%[b2]), %[ldb]\n\t"
        "mqma.b.mm acc1, tr0, tr5\n\t"
        "mlbe8.m tr7, (%[b3]), %[ldb]\n\t"
        "mqma.b.mm acc2, tr0, tr6\n\t"
        "mqma.b.mm acc3, tr0, tr7\n\t"
        :: [type_i8] "r"((uint64_t)AME_MIX_MTYPE_INT8),
           [ap] "r"(A + kk), [lda] "r"((size_t)lda),
           [b0] "r"(b0base + kk), [b1] "r"(b1base + kk),
           [b2] "r"(b2base + kk), [b3] "r"(b3base + kk),
           [ldb] "r"((size_t)ldb)
        : "memory");
  }

  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(C + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(C + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(C + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(C + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

  ame_profile_add_decode_1a4b((k + AME_MIX_HW_K - 1) / AME_MIX_HW_K,
                              msettilek_count);
}

static void amex_decode_1a8b2out_n64_i8_i8_f32(const int8_t *A,
                                               const int8_t *B0,
                                               const int8_t *B1,
                                               float *C0,
                                               float *C1,
                                               const float *acc_zero,
                                               int k,
                                               int lda,
                                               int *active_k) {
  if (!A || !B0 || !B1 || !C0 || !C1 || !acc_zero || !active_k || k <= 0) return;

  const int ldb = k;
  const int ldc = AME_MIX_OUTBLK * (int)sizeof(float);
  const int8_t *b0_0 = B0 + (size_t)0 * AME_MIX_HW_N * k;
  const int8_t *b0_1 = B0 + (size_t)1 * AME_MIX_HW_N * k;
  const int8_t *b0_2 = B0 + (size_t)2 * AME_MIX_HW_N * k;
  const int8_t *b0_3 = B0 + (size_t)3 * AME_MIX_HW_N * k;
  const int8_t *b1_0 = B1 + (size_t)0 * AME_MIX_HW_N * k;
  const int8_t *b1_1 = B1 + (size_t)1 * AME_MIX_HW_N * k;
  const int8_t *b1_2 = B1 + (size_t)2 * AME_MIX_HW_N * k;
  const int8_t *b1_3 = B1 + (size_t)3 * AME_MIX_HW_N * k;
  int msettilek_count = 0;

  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(acc_zero + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

  for (int kk = 0; kk < k; kk += AME_MIX_HW_K) {
    int tk = k - kk;
    if (tk > AME_MIX_HW_K) tk = AME_MIX_HW_K;
    if (*active_k != tk) {
      asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(tk)) : "a3", "memory");
      *active_k = tk;
      msettilek_count++;
    }
    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");

    asm volatile(
        "mlae8.m tr0, (%[p]), %[s]\n\t"
        :: [p] "r"(A + kk), [s] "r"((size_t)(lda)) : "memory");

    asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b0_0 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b0_1 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc0, tr0, tr4" ::: "memory");
    asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b0_2 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc1, tr0, tr5" ::: "memory");
    asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b0_3 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc2, tr0, tr6" ::: "memory");
    asm volatile("mqma.b.mm acc3, tr0, tr7" ::: "memory");

    asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b1_0 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b1_1 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc4, tr0, tr4" ::: "memory");
    asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b1_2 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc5, tr0, tr5" ::: "memory");
    asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b1_3 + kk), [s] "r"((size_t)(ldb)) : "memory");
    asm volatile("mqma.b.mm acc6, tr0, tr6" ::: "memory");
    asm volatile("mqma.b.mm acc7, tr0, tr7" ::: "memory");
  }

  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(C0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(C0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(C0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(C0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(C1 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(C1 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(C1 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(C1 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

  ame_profile_add_decode_1a8b2out((k + AME_MIX_HW_K - 1) / AME_MIX_HW_K,
                                  msettilek_count);
}

/* lda = A row stride in bytes-of-int8 (n for direct-from-xq; gs for repacked).
 * Resync is hoisted to the matmul entry (caller), not done per kernel call. */
static int amex_prefill_breuse_k64_n64_i8_i8_f32(const int8_t *A,
                                                 const int8_t *B,
                                                 float *C,
                                                 const float *acc_zero,
                                                 int m,
                                                 int lda) {
  if (!A || !B || !C || !acc_zero || m < AME_MIX_BREUSE_M) return 0;

  const int rows = (m / AME_MIX_BREUSE_M) * AME_MIX_BREUSE_M;
  const int ldb = AME_MIX_HW_K;
  const int ldc = AME_MIX_OUTBLK * (int)sizeof(float);

  for (int ib = 0; ib < rows; ib += AME_MIX_BREUSE_M) {
    const int8_t *ap0 = A + (size_t)(ib + 0 * AME_MIX_M_TILE) * lda;
    const int8_t *ap1 = A + (size_t)(ib + 1 * AME_MIX_M_TILE) * lda;
    const int8_t *ap2 = A + (size_t)(ib + 2 * AME_MIX_M_TILE) * lda;
    const int8_t *ap3 = A + (size_t)(ib + 3 * AME_MIX_M_TILE) * lda;

    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");
    asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
    asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
    asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_K)) : "a3", "memory");
    asm volatile(
        "mlae8.m tr0, (%[p]), %[s]\n\t"
        :: [p] "r"(ap0), [s] "r"((size_t)(lda)) : "memory");
    asm volatile(
        "mlae8.m tr1, (%[p]), %[s]\n\t"
        :: [p] "r"(ap1), [s] "r"((size_t)(lda)) : "memory");
    asm volatile(
        "mlae8.m tr2, (%[p]), %[s]\n\t"
        :: [p] "r"(ap2), [s] "r"((size_t)(lda)) : "memory");
    asm volatile(
        "mlae8.m tr3, (%[p]), %[s]\n\t"
        :: [p] "r"(ap3), [s] "r"((size_t)(lda)) : "memory");

    for (int nb = 0; nb < AME_MIX_OUTBLK; nb += AME_MIX_HW_N) {
      float *cp0 = C + (size_t)(ib + 0 * AME_MIX_M_TILE) * AME_MIX_OUTBLK + nb;
      float *cp1 = C + (size_t)(ib + 1 * AME_MIX_M_TILE) * AME_MIX_OUTBLK + nb;
      float *cp2 = C + (size_t)(ib + 2 * AME_MIX_M_TILE) * AME_MIX_OUTBLK + nb;
      float *cp3 = C + (size_t)(ib + 3 * AME_MIX_M_TILE) * AME_MIX_OUTBLK + nb;
      const float *zp0 = acc_zero + (size_t)0 * AME_MIX_M_TILE * AME_MIX_OUTBLK + nb;
      const float *zp1 = acc_zero + (size_t)1 * AME_MIX_M_TILE * AME_MIX_OUTBLK + nb;
      const float *zp2 = acc_zero + (size_t)2 * AME_MIX_M_TILE * AME_MIX_OUTBLK + nb;
      const float *zp3 = acc_zero + (size_t)3 * AME_MIX_M_TILE * AME_MIX_OUTBLK + nb;
      const int8_t *bp = B + (size_t)nb * AME_MIX_HW_K;

      asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
      asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
      asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
      asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(zp0), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(zp1), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(zp2), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(zp3), [s] "r"((size_t)(ldc)) : "memory");

      asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");
      asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
      asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
      asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_K)) : "a3", "memory");
      asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(bp), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc0, tr0, tr4" ::: "memory");
      asm volatile("mqma.b.mm acc1, tr1, tr4" ::: "memory");
      asm volatile("mqma.b.mm acc2, tr2, tr4" ::: "memory");
      asm volatile("mqma.b.mm acc3, tr3, tr4" ::: "memory");

      asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
      asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
      asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
      asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(cp0), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(cp1), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(cp2), [s] "r"((size_t)(ldc)) : "memory");
      asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(cp3), [s] "r"((size_t)(ldc)) : "memory");
    }
  }

  ame_profile_add_breuse_k64(rows / AME_MIX_BREUSE_M);
  return rows;
}

/* ----------------------------------------------------------------------
 * 8-accumulator 2A x 4B prefill kernel for N = OUTBLK (=64), arbitrary K.
 *
 *   MLA/A tr0 / tr2     = two M16 A tiles (rows mb, mb+16)
 *   MLB/B tr4..tr7      = the four N16 B tiles of the OUTBLK column group
 *   acc0..acc7         = 2 (M) x 4 (N) output tiles
 *
 * Per K step: 2 A loads + 4 B loads + 8 mqma = 1.33 MAC/load. Each B (weight)
 * tile is loaded once per M-pair instead of once per M-tile, so weight DDR
 * traffic is halved versus the A-replicating generic ABT fallback (0.5
 * MAC/load). Works for any group size (count = gs), including gs > 64 where the
 * K=64 fast path never engages. A trailing odd M16 batch tile is handled by a
 * 1A x 4B (acc0..3) pass. Decode uses the trimmed dedicated 1A x 4B core above.
 *
 *   A: [m x k] row-major (lda=k)   B: [OUTBLK x k] N-major tile (ldb=k)
 *   C: [m x OUTBLK] row-major fp32 (ldc=OUTBLK*4). mlce32 reads a persistent
 *   zero tile, so scratch C does not need to be cleared for every output block.
 *   mqma accumulates int8×int8 products in fp32; msce32 stores fp32 to C.
 *
 * The blocking shape matches the previous prefill_max_2a4b/decode_base_1a4b
 * kernels; revalidate on FPGA after changing the tile-register assignment.
 * --------------------------------------------------------------------*/
/* lda = A row stride (n for direct-from-xq). k = group width (B k-stride).
 * Resync is hoisted to the matmul entry (caller). */
static void amex_prefill_2a4b_n64_i8_i8_f32(const int8_t *A,
                                            const int8_t *B,
                                            float *C,
                                            const float *acc_zero,
                                            int m,
                                            int k,
                                            int lda) {
  if (!A || !B || !C || !acc_zero || m <= 0 || k <= 0) return;

  const int ldb = k;
  const int ldc = AME_MIX_OUTBLK * (int)sizeof(float);
  const int8_t *b0base = B + (size_t)0 * AME_MIX_HW_N * k;
  const int8_t *b1base = B + (size_t)1 * AME_MIX_HW_N * k;
  const int8_t *b2base = B + (size_t)2 * AME_MIX_HW_N * k;
  const int8_t *b3base = B + (size_t)3 * AME_MIX_HW_N * k;

  int mb = 0;

  /* ---- M pairs: 2 M16 tiles x 4 N16 tiles, eight accumulators ---- */
  for (; mb + 2 * AME_MIX_M_TILE <= m; mb += 2 * AME_MIX_M_TILE) {
    float *r0 = C + (size_t)(mb + 0 * AME_MIX_M_TILE) * AME_MIX_OUTBLK;
    float *r1 = C + (size_t)(mb + 1 * AME_MIX_M_TILE) * AME_MIX_OUTBLK;
    const float *z0 = acc_zero;
    const float *z1 = acc_zero + (size_t)AME_MIX_M_TILE * AME_MIX_OUTBLK;

    asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
    asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");

    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
    asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

    for (int kk = 0; kk < k; kk += AME_MIX_HW_K) {
      int tk = k - kk;
      if (tk > AME_MIX_HW_K) tk = AME_MIX_HW_K;
      asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(tk)) : "a3", "memory");
      asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");

      asm volatile(
        "mlae8.m tr0, (%[p]), %[s]\n\t"
        :: [p] "r"(A + (size_t)(mb + 0 * AME_MIX_M_TILE) * lda + kk), [s] "r"((size_t)(lda)) : "memory");
      asm volatile(
        "mlae8.m tr2, (%[p]), %[s]\n\t"
        :: [p] "r"(A + (size_t)(mb + 1 * AME_MIX_M_TILE) * lda + kk), [s] "r"((size_t)(lda)) : "memory");
      asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b0base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b1base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc0, tr0, tr4" ::: "memory");
      asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b2base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc4, tr2, tr4" ::: "memory");
      asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b3base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc1, tr0, tr5" ::: "memory");
      asm volatile("mqma.b.mm acc5, tr2, tr5" ::: "memory");
      asm volatile("mqma.b.mm acc2, tr0, tr6" ::: "memory");
      asm volatile("mqma.b.mm acc6, tr2, tr6" ::: "memory");
      asm volatile("mqma.b.mm acc3, tr0, tr7" ::: "memory");
      asm volatile("mqma.b.mm acc7, tr2, tr7" ::: "memory");
    }

    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
    asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile("fence rw,rw" ::: "memory");
  }

  /* ---- trailing M tile (tile_m up to 16): 1A x 4B, four accumulators ---- */
  for (; mb < m; mb += AME_MIX_M_TILE) {
    int tm = m - mb;
    if (tm > AME_MIX_M_TILE) tm = AME_MIX_M_TILE;
    float *r0 = C + (size_t)mb * AME_MIX_OUTBLK;
    const float *z0 = acc_zero;

    asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(tm)) : "a6", "memory");
    asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");

    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
    asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

    for (int kk = 0; kk < k; kk += AME_MIX_HW_K) {
      int tk = k - kk;
      if (tk > AME_MIX_HW_K) tk = AME_MIX_HW_K;
      asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(tk)) : "a3", "memory");
      asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");

      asm volatile(
        "mlae8.m tr0, (%[p]), %[s]\n\t"
        :: [p] "r"(A + (size_t)mb * lda + kk), [s] "r"((size_t)(lda)) : "memory");
      asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b0base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b1base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc0, tr0, tr4" ::: "memory");
      asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b2base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc1, tr0, tr5" ::: "memory");
      asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b3base + kk), [s] "r"((size_t)(ldb)) : "memory");
      asm volatile("mqma.b.mm acc2, tr0, tr6" ::: "memory");
      asm volatile("mqma.b.mm acc3, tr0, tr7" ::: "memory");
    }

    asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
    asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
    asm volatile("fence rw,rw" ::: "memory");
  }

  ame_profile_add_2a4b(m / (2 * AME_MIX_M_TILE),
                       ((m % (2 * AME_MIX_M_TILE)) + AME_MIX_M_TILE - 1) /
                           AME_MIX_M_TILE,
                       (k + AME_MIX_HW_K - 1) / AME_MIX_HW_K);
}

/* K=128 prefill specialization. The blocking schedule matches
 * test/ame_prefill_restructure_fpga.c; revalidate on FPGA after changing the
 * tile-register assignment.
 * Four A tiles remain resident while the caller walks all N64 output blocks:
 *   MLA/A: tr0/tr1 = first M16 rows,  K0/K1
 *          tr2/tr3 = second M16 rows, K0/K1
 *   MLB/B: tr4..tr7 = four N16 B tiles
 * This removes A reloads from the output-block loop. */
static void amex_prefill_aresident_k128_load_a(const int8_t *A, int lda) {
  if (!A) return;

  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_K)) : "a3", "memory");
  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");
  asm volatile(
        "mlae8.m tr0, (%[p]), %[s]\n\t"
        :: [p] "r"(A), [s] "r"((size_t)(lda)) : "memory");
  asm volatile(
        "mlae8.m tr1, (%[p]), %[s]\n\t"
        :: [p] "r"(A + AME_MIX_HW_K), [s] "r"((size_t)(lda)) : "memory");
  asm volatile(
        "mlae8.m tr2, (%[p]), %[s]\n\t"
        :: [p] "r"(A + (size_t)AME_MIX_M_TILE * lda), [s] "r"((size_t)(lda)) : "memory");
  asm volatile(
        "mlae8.m tr3, (%[p]), %[s]\n\t"
        :: [p] "r"(A + (size_t)AME_MIX_M_TILE * lda + AME_MIX_HW_K), [s] "r"((size_t)(lda)) : "memory");
  ame_profile_add_aresident_k128_load_a();
}

static void amex_prefill_aresident_k128_n64_pair_i8_i8_f32(
    const int8_t *B,
    float *C,
    const float *acc_zero) {
  if (!B || !C || !acc_zero) return;

  const int ldb = 2 * AME_MIX_HW_K;
  const int ldc = AME_MIX_OUTBLK * (int)sizeof(float);
  const int8_t *b0 = B + (size_t)0 * AME_MIX_HW_N * ldb;
  const int8_t *b1 = B + (size_t)1 * AME_MIX_HW_N * ldb;
  const int8_t *b2 = B + (size_t)2 * AME_MIX_HW_N * ldb;
  const int8_t *b3 = B + (size_t)3 * AME_MIX_HW_N * ldb;
  float *r0 = C;
  float *r1 = C + (size_t)AME_MIX_M_TILE * AME_MIX_OUTBLK;
  const float *z0 = acc_zero;
  const float *z1 = acc_zero + (size_t)AME_MIX_M_TILE * AME_MIX_OUTBLK;

  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(AME_MIX_M_TILE)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "mlce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(z0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "mlce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(z1 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");

  asm volatile(
        "msettilek a3, %[v] # 0x0406e6f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_K)) : "a3", "memory");
  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT8)) : "memory");
  asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b0), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b1), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc0, tr0, tr4" ::: "memory");
  asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b2), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc4, tr2, tr4" ::: "memory");
  asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b3), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc1, tr0, tr5" ::: "memory");
  asm volatile("mqma.b.mm acc5, tr2, tr5" ::: "memory");
  asm volatile("mqma.b.mm acc2, tr0, tr6" ::: "memory");
  asm volatile("mqma.b.mm acc6, tr2, tr6" ::: "memory");
  asm volatile("mqma.b.mm acc3, tr0, tr7" ::: "memory");
  asm volatile("mqma.b.mm acc7, tr2, tr7" ::: "memory");

  asm volatile(
        "mlbe8.m tr4, (%[p]), %[s]\n\t"
        :: [p] "r"(b0 + AME_MIX_HW_K), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile(
        "mlbe8.m tr5, (%[p]), %[s]\n\t"
        :: [p] "r"(b1 + AME_MIX_HW_K), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc0, tr1, tr4" ::: "memory");
  asm volatile(
        "mlbe8.m tr6, (%[p]), %[s]\n\t"
        :: [p] "r"(b2 + AME_MIX_HW_K), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc4, tr3, tr4" ::: "memory");
  asm volatile(
        "mlbe8.m tr7, (%[p]), %[s]\n\t"
        :: [p] "r"(b3 + AME_MIX_HW_K), [s] "r"((size_t)(ldb)) : "memory");
  asm volatile("mqma.b.mm acc1, tr1, tr5" ::: "memory");
  asm volatile("mqma.b.mm acc5, tr3, tr5" ::: "memory");
  asm volatile("mqma.b.mm acc2, tr1, tr6" ::: "memory");
  asm volatile("mqma.b.mm acc6, tr3, tr6" ::: "memory");
  asm volatile("mqma.b.mm acc3, tr1, tr7" ::: "memory");
  asm volatile("mqma.b.mm acc7, tr3, tr7" ::: "memory");

  asm volatile(
        "msettype x0, %[t] # 0x00054077\n\t"
        :: [t] "r"((uint64_t)(AME_MIX_MTYPE_INT32)) : "memory");
  asm volatile(
        "msce32.m acc0, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc1, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc2, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc3, (%[p]), %[s]\n\t"
        :: [p] "r"(r0 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc4, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 0 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc5, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 1 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc6, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 2 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  asm volatile(
        "msce32.m acc7, (%[p]), %[s]\n\t"
        :: [p] "r"(r1 + 3 * AME_MIX_HW_N), [s] "r"((size_t)(ldc)) : "memory");
  ame_profile_add_aresident_k128_n64();
}

/* ----- Backend kernel-call statistics (same names as shared kernel) ----- */
static volatile uint64_t g_ame_mm_calls;
static volatile uint64_t g_ame_mm_macs;
static volatile uint64_t g_ame_mm_tile_invocations;
static volatile uint64_t g_ame_mm_total_cycles;
static volatile uint64_t g_ame_mm_pack_cycles;     /* always 0 in offline_pack */
static volatile uint64_t g_ame_mm_zero_cycles;
static volatile uint64_t g_ame_mm_mqma_cycles;
static volatile uint64_t g_ame_mm_accum_cycles;
static volatile uint64_t g_ame_mm_single_calls;
static volatile uint64_t g_ame_mm_single_tiles;
static volatile uint64_t g_ame_mm_single_cycles;
static volatile uint64_t g_ame_mm_single_kernel_cycles;
static volatile uint64_t g_ame_mm_batch_calls;
static volatile uint64_t g_ame_mm_batch_tiles;
static volatile uint64_t g_ame_mm_batch_cycles;
static volatile uint64_t g_ame_mm_batch_kernel_cycles;
static float ame_mix_decode_zero_tile[AME_MIX_OUTBLK] __attribute__((aligned(64)));

static inline uint64_t ame_mix_rdcycle(void) {
#if AME_MIX_PROFILE
  uint64_t c;
  asm volatile("rdcycle %0" : "=r"(c));
  return c;
#else
  return 0;
#endif
}

void ame_matmul_stats_reset(void) {
  g_ame_mm_calls = 0;
  g_ame_mm_macs = 0;
  g_ame_mm_tile_invocations = 0;
  g_ame_mm_total_cycles = 0;
  g_ame_mm_pack_cycles = 0;
  g_ame_mm_zero_cycles = 0;
  g_ame_mm_mqma_cycles = 0;
  g_ame_mm_accum_cycles = 0;
  g_ame_mm_single_calls = 0;
  g_ame_mm_single_tiles = 0;
  g_ame_mm_single_cycles = 0;
  g_ame_mm_single_kernel_cycles = 0;
  g_ame_mm_batch_calls = 0;
  g_ame_mm_batch_tiles = 0;
  g_ame_mm_batch_cycles = 0;
  g_ame_mm_batch_kernel_cycles = 0;
  g_ame_profile_detail.calls = 0;
  g_ame_profile_detail.single_calls = 0;
  g_ame_profile_detail.batch_calls = 0;
  g_ame_profile_detail.tiles = 0;
  g_ame_profile_detail.single_tiles = 0;
  g_ame_profile_detail.batch_tiles = 0;
  g_ame_profile_detail.total_cycles = 0;
  g_ame_profile_detail.single_cycles = 0;
  g_ame_profile_detail.batch_cycles = 0;
  g_ame_profile_detail.zero_cycles = 0;
  g_ame_profile_detail.ame_kernel_cycles = 0;
  g_ame_profile_detail.fp32_accum_cycles = 0;
  g_ame_profile_detail.pack_cycles = 0;
  g_ame_profile_detail.resync_calls = 0;
  g_ame_profile_detail.msettilem = 0;
  g_ame_profile_detail.msettilen = 0;
  g_ame_profile_detail.msettilek = 0;
  g_ame_profile_detail.msettype = 0;
  g_ame_profile_detail.mlae8 = 0;
  g_ame_profile_detail.mlbe8 = 0;
  g_ame_profile_detail.mlbt8 = 0;
  g_ame_profile_detail.mlce32 = 0;
  g_ame_profile_detail.mqma_b = 0;
  g_ame_profile_detail.msce32 = 0;
  g_ame_profile_detail.fence_rw_rw = 0;
  g_ame_profile_detail.rvv_accum_chunks = 0;
  g_ame_profile_detail.zero_f32_elements = 0;
}
uint64_t ame_matmul_stats_calls(void)        { return g_ame_mm_calls; }
uint64_t ame_matmul_stats_macs(void)         { return g_ame_mm_macs; }
uint64_t ame_matmul_stats_tile_calls(void)   { return g_ame_mm_tile_invocations; }
uint64_t ame_matmul_stats_total_cycles(void) { return g_ame_mm_total_cycles; }
uint64_t ame_matmul_stats_pack_cycles(void)  { return g_ame_mm_pack_cycles; }
uint64_t ame_matmul_stats_zero_cycles(void)  { return g_ame_mm_zero_cycles; }
uint64_t ame_matmul_stats_mqma_cycles(void)  { return g_ame_mm_mqma_cycles; }
uint64_t ame_matmul_stats_accum_cycles(void) { return g_ame_mm_accum_cycles; }

static uint64_t ame_profile_total_ame_instrs(void) {
  return g_ame_profile_detail.msettilem + g_ame_profile_detail.msettilen +
         g_ame_profile_detail.msettilek + g_ame_profile_detail.msettype +
         g_ame_profile_detail.mlae8 + g_ame_profile_detail.mlbe8 +
         g_ame_profile_detail.mlbt8 + g_ame_profile_detail.mlce32 +
         g_ame_profile_detail.mqma_b + g_ame_profile_detail.msce32;
}

void ame_matmul_profile_snapshot(AmeMixOfflineProfile *out) {
  if (!out) return;
  out->resync_calls = g_ame_profile_detail.resync_calls;
  out->msettilem = g_ame_profile_detail.msettilem;
  out->msettilen = g_ame_profile_detail.msettilen;
  out->msettilek = g_ame_profile_detail.msettilek;
  out->msettype = g_ame_profile_detail.msettype;
  out->mlae8 = g_ame_profile_detail.mlae8;
  out->mlbe8 = g_ame_profile_detail.mlbe8;
  out->mlbt8 = g_ame_profile_detail.mlbt8;
  out->mlce32 = g_ame_profile_detail.mlce32;
  out->mqma_b = g_ame_profile_detail.mqma_b;
  out->msce32 = g_ame_profile_detail.msce32;
  out->fence_rw_rw = g_ame_profile_detail.fence_rw_rw;
  out->rvv_accum_chunks = g_ame_profile_detail.rvv_accum_chunks;
  out->zero_f32_elements = g_ame_profile_detail.zero_f32_elements;
  out->calls = g_ame_mm_calls;
  out->single_calls = g_ame_mm_single_calls;
  out->batch_calls = g_ame_mm_batch_calls;
  out->tiles = g_ame_mm_tile_invocations;
  out->single_tiles = g_ame_mm_single_tiles;
  out->batch_tiles = g_ame_mm_batch_tiles;
  out->total_cycles = g_ame_mm_total_cycles;
  out->single_cycles = g_ame_mm_single_cycles;
  out->batch_cycles = g_ame_mm_batch_cycles;
  out->zero_cycles = g_ame_mm_zero_cycles;
  out->ame_kernel_cycles = g_ame_mm_mqma_cycles;
  out->fp32_accum_cycles = g_ame_mm_accum_cycles;
  out->pack_cycles = g_ame_mm_pack_cycles;
}

/* Compatibility exports retained for existing profiling callers. */
uint64_t ame_matmul_stats_total_instrs(void)  { return ame_profile_total_ame_instrs(); }
uint64_t ame_matmul_stats_pack_instrs(void)   { return 0; }
uint64_t ame_matmul_stats_zero_instrs(void)   { return 0; }
uint64_t ame_matmul_stats_mqma_instrs(void)   { return g_ame_profile_detail.mqma_b; }
uint64_t ame_matmul_stats_accum_instrs(void)  { return 9 * g_ame_profile_detail.rvv_accum_chunks; }
uint64_t ame_matmul_noblk_cycles_total(void)  { return g_ame_mm_single_cycles; }
uint64_t ame_matmul_noblk_cycles_ame(void)    { return g_ame_mm_single_kernel_cycles; }
uint64_t ame_matmul_noblk_cycles_noname(void) {
  return g_ame_mm_single_cycles >= g_ame_mm_single_kernel_cycles
             ? g_ame_mm_single_cycles - g_ame_mm_single_kernel_cycles
             : 0;
}
uint64_t ame_matmul_noblk_instrs_total(void)  { return ame_profile_total_ame_instrs(); }
uint64_t ame_matmul_noblk_instrs_ame(void)    { return g_ame_profile_detail.mqma_b; }
uint64_t ame_matmul_noblk_instrs_noname(void) { return 0; }
uint64_t ame_matmul_single_stats_calls(void)      { return g_ame_mm_single_calls; }
uint64_t ame_matmul_single_stats_tile_calls(void) { return g_ame_mm_single_tiles; }
uint64_t ame_matmul_single_cycles_total(void)     { return g_ame_mm_single_cycles; }
uint64_t ame_matmul_single_cycles_quadacc(void)   { return g_ame_mm_single_kernel_cycles; }
uint64_t ame_matmul_single_cycles_noname(void)    {
  return g_ame_mm_single_cycles >= g_ame_mm_single_kernel_cycles
             ? g_ame_mm_single_cycles - g_ame_mm_single_kernel_cycles
             : 0;
}
uint64_t ame_matmul_batch_stats_calls(void)       { return g_ame_mm_batch_calls; }
uint64_t ame_matmul_batch_stats_tile_calls(void)  { return g_ame_mm_batch_tiles; }
uint64_t ame_matmul_batch_cycles_total(void)      { return g_ame_mm_batch_cycles; }
uint64_t ame_matmul_batch_cycles_quadacc(void)    { return g_ame_mm_batch_kernel_cycles; }
uint64_t ame_matmul_batch_cycles_noname(void)     {
  return g_ame_mm_batch_cycles >= g_ame_mm_batch_kernel_cycles
             ? g_ame_mm_batch_cycles - g_ame_mm_batch_kernel_cycles
             : 0;
}

extern void print_uart(const char *str) __attribute__((weak));

static void ame_mix_fail(const char *msg) {
  if (&print_uart) print_uart(msg);
  __builtin_trap();
}

/* Pointer to int8 tile: OUTBLK * gs bytes per (i_block, k_group). */
static inline const int8_t *ame_mix_tile_i8_ptr(const int8_t *wq8,
                                                int i0,
                                                int g,
                                                int gs,
                                                int n_tile_groups) {
  const int i_block = i0 / AME_MIX_OUTBLK;
  const size_t tile_idx = (size_t)i_block * (size_t)n_tile_groups + (size_t)g;
  const size_t tile_bytes = (size_t)AME_MIX_OUTBLK * (size_t)gs;
  return wq8 + tile_idx * tile_bytes;
}

/* Return B tile pointer for mlbe8 (offline int8 tile layout, blob v6). */
static inline const int8_t *ame_mix_btile_i8(const int8_t *wq8,
                                             int i0,
                                             int g,
                                             int gs,
                                             int n_tile_groups) {
  return ame_mix_tile_i8_ptr(wq8, i0, g, gs, n_tile_groups);
}

/* RVV dequant-accumulate (replaces the scalar per-element loop).
 * msce32 stores mqma fp32 partial sums to C; RVV then does:
 *   xout[j] += crow[j] * ws_row[j] * xscale,  j = 0..n-1
 * ws_row is the group-major scale row (stride 1 over output index), so it loads
 * contiguously with vle32 (no strided gather).  The xscale scalar is broadcast
 * with integer vmv.v.x using its fp32 bit pattern; this avoids the problematic
 * float broadcast path on the current hardware. */

static inline void __attribute__((unused)) ame_mix_accum_f32_decode(float *xout,
                                                                    const float *crow,
                                                                    const float *ws_row,
                                                                    float xscale,
                                                                    int n) {
  union {
    float f;
    uint32_t u;
  } xscale_bits = { xscale };
  int done = 0;
  int chunks = 0;

  while (done < n) {
    size_t vl;
    float *xout_ptr = xout + done;
    const float *acc_ptr = crow + done;
    const float *ws_ptr = ws_row + done;
    const size_t avl = (size_t)(n - done);

    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "vle32.v v9, (%[acc])\n\t"
        "vle32.v v8, (%[xout])\n\t"
        "vmv.v.x v11, %[xs_bits]\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout])\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [xout] "r"(xout_ptr), [acc] "r"(acc_ptr),
          [ws] "r"(ws_ptr), [xs_bits] "r"((unsigned long)xscale_bits.u)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

    if (vl == 0) break;
    done += (int)vl;
    chunks++;
  }
  AME_PROFILE_ADD(rvv_accum_chunks, chunks);
}

static inline void ame_mix_accum_f32_decode_n64(float *xout,
                                                const float *crow,
                                                const float *ws_row,
                                                float xscale) {
  union { float f; uint32_t u; } xscale_bits = { xscale };
  int done = 0;
  int chunks = 0;

  while (done < AME_MIX_OUTBLK) {
    size_t vl;
    float *xout_ptr = xout + done;
    const float *acc_ptr = crow + done;
    const float *ws_ptr = ws_row + done;
    const size_t avl = (size_t)(AME_MIX_OUTBLK - done);

    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "vle32.v v9, (%[acc])\n\t"
        "vle32.v v8, (%[xout])\n\t"
        "vmv.v.x v11, %[xs_bits]\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout])\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [xout] "r"(xout_ptr), [acc] "r"(acc_ptr),
          [ws] "r"(ws_ptr), [xs_bits] "r"((unsigned long)xscale_bits.u)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");

    if (vl == 0) break;
    done += (int)vl;
    chunks++;
  }
  AME_PROFILE_ADD(rvv_accum_chunks, chunks);
}

static inline void ame_mix_accum_f32_decode2_n64(float *xout0,
                                                 float *xout1,
                                                 const float *crow0,
                                                 const float *crow1,
                                                 const float *ws_row0,
                                                 const float *ws_row1,
                                                 float xscale) {
  ame_mix_accum_f32_decode_n64(xout0, crow0, ws_row0, xscale);
  ame_mix_accum_f32_decode_n64(xout1, crow1, ws_row1, xscale);
}

static inline void ame_mix_accum_f32_prefill(float *xout,
                                             const float *crow,
                                             const float *ws_row,
                                             float xscale,
                                             int n) {
  union {
    float f;
    uint32_t u;
  } xscale_bits = { xscale };
  int done = 0;
  int chunks = 0;

  while (n - done >= 2) {
    size_t vl;
    float *xout_ptr = xout + done;
    const float *acc_ptr = crow + done;
    const float *ws_ptr = ws_row + done;
    const size_t avl = (size_t)(n - done) / 2u;

    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "slli t0, %[vl], 2\n\t"
        "vmv.v.x v11, %[xs_bits]\n\t"
        "vle32.v v8, (%[xout])\n\t"
        "vle32.v v9, (%[acc])\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout])\n\t"
        "vsetvli zero, %[vl], e32, m1, ta, ma\n\t"
        "add t1, %[xout], t0\n\t"
        "add t2, %[acc], t0\n\t"
        "add t3, %[ws], t0\n\t"
        "vmv.v.x v11, %[xs_bits]\n\t"
        "vle32.v v8, (t1)\n\t"
        "vle32.v v9, (t2)\n\t"
        "vle32.v v10, (t3)\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (t1)\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl), [xout] "r"(xout_ptr), [acc] "r"(acc_ptr),
          [ws] "r"(ws_ptr), [xs_bits] "r"((unsigned long)xscale_bits.u)
        : "t0", "t1", "t2", "t3", "v8", "v9", "v10", "v11", "v12", "v13", "memory");

    if (vl == 0) break;
    done += (int)(vl * 2u);
    chunks += 2;
  }

  while (done < n) {
    xout[done] += crow[done] * ws_row[done] * xscale;
    done++;
  }
  AME_PROFILE_ADD(rvv_accum_chunks, chunks);
}

static inline void ame_mix_accum_f32_prefill2(float *xout0,
                                              float *xout1,
                                              const float *crow0,
                                              const float *crow1,
                                              const float *ws_row,
                                              float xscale0,
                                              float xscale1,
                                              int n) {
  union { float f; uint32_t u; } xs0_bits = { xscale0 };
  union { float f; uint32_t u; } xs1_bits = { xscale1 };
  int done = 0;
  int chunks = 0;

  while (done < n) {
    size_t vl;
    const size_t avl = (size_t)(n - done);
    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "vmv.v.x v11, %[xs0]\n\t"
        "vle32.v v8, (%[xout0])\n\t"
        "vle32.v v9, (%[crow0])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout0])\n\t"
        "vmv.v.x v11, %[xs1]\n\t"
        "vle32.v v8, (%[xout1])\n\t"
        "vle32.v v9, (%[crow1])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout1])\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl),
          [xout0] "r"(xout0 + done), [xout1] "r"(xout1 + done),
          [crow0] "r"(crow0 + done), [crow1] "r"(crow1 + done),
          [ws] "r"(ws_row + done),
          [xs0] "r"((unsigned long)xs0_bits.u),
          [xs1] "r"((unsigned long)xs1_bits.u)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");
    if (vl == 0) break;
    done += (int)vl;
    chunks += 2;
  }
  AME_PROFILE_ADD(rvv_accum_chunks, chunks);
}

static inline void ame_mix_accum_f32_prefill4(float *xout0,
                                              float *xout1,
                                              float *xout2,
                                              float *xout3,
                                              const float *crow0,
                                              const float *crow1,
                                              const float *crow2,
                                              const float *crow3,
                                              const float *ws_row,
                                              float xscale0,
                                              float xscale1,
                                              float xscale2,
                                              float xscale3,
                                              int n) {
  union { float f; uint32_t u; } xs0_bits = { xscale0 };
  union { float f; uint32_t u; } xs1_bits = { xscale1 };
  union { float f; uint32_t u; } xs2_bits = { xscale2 };
  union { float f; uint32_t u; } xs3_bits = { xscale3 };
  int done = 0;
  int chunks = 0;

  while (done < n) {
    size_t vl;
    const size_t avl = (size_t)(n - done);
    asm volatile(
        "vsetvli %[vl], %[avl], e32, m1, ta, ma\n\t"
        "vle32.v v10, (%[ws])\n\t"
        "vmv.v.x v11, %[xs0]\n\t"
        "vle32.v v8, (%[xout0])\n\t"
        "vle32.v v9, (%[crow0])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout0])\n\t"
        "vmv.v.x v11, %[xs1]\n\t"
        "vle32.v v8, (%[xout1])\n\t"
        "vle32.v v9, (%[crow1])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout1])\n\t"
        "vmv.v.x v11, %[xs2]\n\t"
        "vle32.v v8, (%[xout2])\n\t"
        "vle32.v v9, (%[crow2])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout2])\n\t"
        "vmv.v.x v11, %[xs3]\n\t"
        "vle32.v v8, (%[xout3])\n\t"
        "vle32.v v9, (%[crow3])\n\t"
        "vfmul.vv v12, v9, v10\n\t"
        "vfmul.vv v13, v12, v11\n\t"
        "vfadd.vv v8, v8, v13\n\t"
        "vse32.v v8, (%[xout3])\n\t"
        : [vl] "=&r"(vl)
        : [avl] "r"(avl),
          [xout0] "r"(xout0 + done), [xout1] "r"(xout1 + done),
          [xout2] "r"(xout2 + done), [xout3] "r"(xout3 + done),
          [crow0] "r"(crow0 + done), [crow1] "r"(crow1 + done),
          [crow2] "r"(crow2 + done), [crow3] "r"(crow3 + done),
          [ws] "r"(ws_row + done),
          [xs0] "r"((unsigned long)xs0_bits.u),
          [xs1] "r"((unsigned long)xs1_bits.u),
          [xs2] "r"((unsigned long)xs2_bits.u),
          [xs3] "r"((unsigned long)xs3_bits.u)
        : "v8", "v9", "v10", "v11", "v12", "v13", "memory");
    if (vl == 0) break;
    done += (int)vl;
    chunks += 4;
  }
  AME_PROFILE_ADD(rvv_accum_chunks, chunks);
}

static inline void ame_mix_accum_f32_prefill8(float *xout0,
                                              float *xout1,
                                              float *xout2,
                                              float *xout3,
                                              float *xout4,
                                              float *xout5,
                                              float *xout6,
                                              float *xout7,
                                              const float *crow0,
                                              const float *crow1,
                                              const float *crow2,
                                              const float *crow3,
                                              const float *crow4,
                                              const float *crow5,
                                              const float *crow6,
                                              const float *crow7,
                                              const float *ws_row,
                                              float xscale0,
                                              float xscale1,
                                              float xscale2,
                                              float xscale3,
                                              float xscale4,
                                              float xscale5,
                                              float xscale6,
                                              float xscale7,
                                              int n) {
  ame_mix_accum_f32_prefill4(xout0, xout1, xout2, xout3,
                             crow0, crow1, crow2, crow3,
                             ws_row, xscale0, xscale1, xscale2, xscale3, n);
  ame_mix_accum_f32_prefill4(xout4, xout5, xout6, xout7,
                             crow4, crow5, crow6, crow7,
                             ws_row, xscale4, xscale5, xscale6, xscale7, n);
}

static int ame_mix_matmul_active_k;

static void ame_mix_matmul_noblk_segment(float *xout,
                                         const int8_t *xq,
                                         const float *xs,
                                         const uint8_t *wq4,
                                         const float *ws,
                                         int n,
                                         int d,
                                         int gs,
                                         int stride_w,
                                         int weight_row0,
                                         int scale_row0,
                                         int ws_d) {
  const int groups        = (n + gs - 1) / gs;
  const int n_tile_groups = stride_w / gs;
  const int wq_row_block  = weight_row0 / AME_MIX_OUTBLK;
  const uint8_t *wq_chunk = wq4 + (size_t)wq_row_block * (size_t)n_tile_groups *
                          (size_t)AME_MIX_OUTBLK * (size_t)gs;

  {
    const uint64_t t0 = ame_mix_rdcycle();
    mix_zero_f32(xout, d);
    AME_PROFILE_ADD(zero_f32_elements, d);
    AME_STAT_ADD(g_ame_mm_zero_cycles, ame_mix_rdcycle() - t0);
  }

  int i0 = 0;
  for (; i0 + 2 * AME_MIX_OUTBLK <= d; i0 += 2 * AME_MIX_OUTBLK) {
    float *c0 = ame_mix_c_pack;
    float *c1 = ame_mix_c_pack + AME_MIX_OUTBLK;

    for (int g = 0; g < groups; g++) {
      const int base  = g * gs;
      const int count = (base + gs <= n) ? gs : (n - base);
      if (count != gs) {
        ame_mix_fail("[ame_mix_offline] partial tail group not supported\r\n");
      }

      const int8_t *xseg = xq + base;
      const float xsg = xs[g];
      const int8_t *btile0 = ame_mix_btile_i8((const int8_t *)wq_chunk, i0, g, gs,
                                              n_tile_groups);
      const int8_t *btile1 = ame_mix_btile_i8((const int8_t *)wq_chunk,
                                              i0 + AME_MIX_OUTBLK, g, gs,
                                              n_tile_groups);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        amex_decode_1a8b2out_n64_i8_i8_f32(xseg, btile0, btile1,
                                            c0, c1,
                                            ame_mix_decode_zero_tile,
                                            count, n,
                                            &ame_mix_matmul_active_k);
        {
          const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
          AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
          AME_STAT_ADD(g_ame_mm_single_kernel_cycles, kernel_cycles);
        }
      }
      AME_STAT_ADD(g_ame_mm_tile_invocations, 2);
      AME_STAT_ADD(g_ame_mm_single_tiles, 2);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        const float *ws_row0 = ws + (ptrdiff_t)g * (ptrdiff_t)ws_d +
                               (ptrdiff_t)scale_row0 + (ptrdiff_t)i0;
        const float *ws_row1 = ws_row0 + AME_MIX_OUTBLK;
        ame_mix_accum_f32_decode2_n64(xout + i0, xout + i0 + AME_MIX_OUTBLK,
                                      c0, c1, ws_row0, ws_row1, xsg);
        AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
      }
    }
  }

  for (; i0 < d; i0 += AME_MIX_OUTBLK) {
    for (int g = 0; g < groups; g++) {
      const int base  = g * gs;
      const int count = (base + gs <= n) ? gs : (n - base);
      if (count != gs) {
        ame_mix_fail("[ame_mix_offline] partial tail group not supported\r\n");
      }

      const int8_t *xseg = xq + base;
      const float xsg = xs[g];
      const int8_t *btile = ame_mix_btile_i8((const int8_t *)wq_chunk, i0, g, gs,
                                             n_tile_groups);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        amex_decode_1a4b_n64_i8_i8_f32(xseg, btile, ame_mix_c_pack,
                                        ame_mix_decode_zero_tile,
                                        count, n, &ame_mix_matmul_active_k);
        {
          const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
          AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
          AME_STAT_ADD(g_ame_mm_single_kernel_cycles, kernel_cycles);
        }
      }
      AME_STAT_INC(g_ame_mm_tile_invocations);
      AME_STAT_INC(g_ame_mm_single_tiles);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        const float *ws_row0 = ws + (ptrdiff_t)g * (ptrdiff_t)ws_d +
                               (ptrdiff_t)scale_row0 + (ptrdiff_t)i0;
        ame_mix_accum_f32_decode_n64(xout + i0, ame_mix_c_pack, ws_row0, xsg);
        AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
      }
    }
  }
}

static void ame_mix_matmul_noblk_core(float *xout,
                                      const int8_t *xq,
                                      const float *xs,
                                      const uint8_t *wq4,
                                      const float *ws,
                                      int n,
                                      int d,
                                      int gs,
                                      int stride_w,
                                      int weight_row0,
                                      int scale_row0,
                                      int d_stride) {
  if (!xout || !xq || !xs || !wq4 || !ws || n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS || (gs % 2) != 0) {
    ame_mix_fail("[ame_mix_offline_w4] gs invalid, halting\r\n");
  }
  if (d % AME_MIX_OUTBLK) {
    ame_mix_fail("[ame_mix_offline] d % OUTBLK != 0, halting\r\n");
  }
  if (stride_w % gs) {
    ame_mix_fail("[ame_mix_offline] stride_w % gs != 0, halting\r\n");
  }
  if (weight_row0 != 0 && (weight_row0 % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] weight_row0 % OUTBLK != 0, halting\r\n");
  }
  if (scale_row0 != 0 && (scale_row0 % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] scale_row0 % OUTBLK != 0, halting\r\n");
  }

  const int ws_d = (d_stride > 0) ? d_stride : d;
  if (weight_row0 < 0 || scale_row0 < 0 || scale_row0 + d > ws_d) {
    ame_mix_fail("[ame_mix_offline] row window out of range, halting\r\n");
  }
  const uint64_t total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_single_calls);
  AME_STAT_ADD(g_ame_mm_macs, (uint64_t)n * (uint64_t)d);

  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(1)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  ame_mix_matmul_active_k = 0;
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);

  ame_mix_matmul_noblk_segment(xout, xq, xs, wq4, ws, n, d, gs, stride_w,
                               weight_row0, scale_row0, ws_d);

  {
    const uint64_t total_cycles = ame_mix_rdcycle() - total_t0;
    AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
    AME_STAT_ADD(g_ame_mm_single_cycles, total_cycles);
  }
}

static void ame_mix_matmul_noblk_pair_segment(float *xout0,
                                              float *xout1,
                                              const int8_t *xq,
                                              const float *xs,
                                              const uint8_t *wq4_0,
                                              const float *ws0,
                                              const uint8_t *wq4_1,
                                              const float *ws1,
                                              int n,
                                              int d,
                                              int gs,
                                              int stride_w,
                                              int weight_row0,
                                              int scale_row0,
                                              int ws_d) {
  const int groups        = (n + gs - 1) / gs;
  const int n_tile_groups = stride_w / gs;
  const int wq_row_block  = weight_row0 / AME_MIX_OUTBLK;
  const uint8_t *wq_chunk0 = wq4_0 + (size_t)wq_row_block * (size_t)n_tile_groups *
                             (size_t)AME_MIX_OUTBLK * (size_t)gs;
  const uint8_t *wq_chunk1 = wq4_1 + (size_t)wq_row_block * (size_t)n_tile_groups *
                             (size_t)AME_MIX_OUTBLK * (size_t)gs;

  {
    const uint64_t t0 = ame_mix_rdcycle();
    mix_zero_f32(xout0, d);
    mix_zero_f32(xout1, d);
    AME_PROFILE_ADD(zero_f32_elements, 2 * d);
    AME_STAT_ADD(g_ame_mm_zero_cycles, ame_mix_rdcycle() - t0);
  }

  for (int g = 0; g < groups; g++) {
    const int base  = g * gs;
    const int count = (base + gs <= n) ? gs : (n - base);
    if (count != gs) {
      ame_mix_fail("[ame_mix_offline] pair: partial tail group not supported\r\n");
    }

    const int8_t *xseg = xq + base;
    const float xsg = xs[g];

    for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
      const int8_t *btile0 = ame_mix_btile_i8((const int8_t *)wq_chunk0, i0, g, gs,
                                              n_tile_groups);
      const int8_t *btile1 = ame_mix_btile_i8((const int8_t *)wq_chunk1, i0, g, gs,
                                              n_tile_groups);
      float *c0 = ame_mix_c_pack;
      float *c1 = ame_mix_c_pack + AME_MIX_OUTBLK;

      {
        const uint64_t t0 = ame_mix_rdcycle();
        amex_decode_1a8b2out_n64_i8_i8_f32(xseg, btile0, btile1,
                                            c0, c1,
                                            ame_mix_decode_zero_tile,
                                            count, n,
                                            &ame_mix_matmul_active_k);
        {
          const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
          AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
          AME_STAT_ADD(g_ame_mm_single_kernel_cycles, kernel_cycles);
        }
      }
      AME_STAT_ADD(g_ame_mm_tile_invocations, 2);
      AME_STAT_ADD(g_ame_mm_single_tiles, 2);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        const float *ws_row0 = ws0 + (ptrdiff_t)g * (ptrdiff_t)ws_d +
                               (ptrdiff_t)scale_row0 + (ptrdiff_t)i0;
        const float *ws_row1 = ws1 + (ptrdiff_t)g * (ptrdiff_t)ws_d +
                               (ptrdiff_t)scale_row0 + (ptrdiff_t)i0;
        ame_mix_accum_f32_decode2_n64(xout0 + i0, xout1 + i0,
                                      c0, c1, ws_row0, ws_row1, xsg);
        AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
      }
    }
  }
}

static void ame_mix_matmul_noblk_pair_core(float *xout0,
                                           float *xout1,
                                           const int8_t *xq,
                                           const float *xs,
                                           const uint8_t *wq4_0,
                                           const float *ws0,
                                           const uint8_t *wq4_1,
                                           const float *ws1,
                                           int n,
                                           int d,
                                           int gs,
                                           int stride_w,
                                           int weight_row0,
                                           int scale_row0,
                                           int d_stride) {
  if (!xout0 || !xout1 || !xq || !xs || !wq4_0 || !ws0 || !wq4_1 || !ws1 ||
      n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS || (gs % 2) != 0) {
    ame_mix_fail("[ame_mix_offline_w4] pair: gs invalid, halting\r\n");
  }
  if (d % AME_MIX_OUTBLK) {
    ame_mix_fail("[ame_mix_offline] pair: d % OUTBLK != 0, halting\r\n");
  }
  if (stride_w % gs) {
    ame_mix_fail("[ame_mix_offline] pair: stride_w % gs != 0, halting\r\n");
  }
  if (weight_row0 != 0 && (weight_row0 % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] pair: weight_row0 % OUTBLK != 0, halting\r\n");
  }
  if (scale_row0 != 0 && (scale_row0 % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] pair: scale_row0 % OUTBLK != 0, halting\r\n");
  }

  const int ws_d = (d_stride > 0) ? d_stride : d;
  if (weight_row0 < 0 || scale_row0 < 0 || scale_row0 + d > ws_d) {
    ame_mix_fail("[ame_mix_offline] pair: row window out of range, halting\r\n");
  }

  const uint64_t total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_single_calls);
  AME_STAT_ADD(g_ame_mm_macs, 2 * (uint64_t)n * (uint64_t)d);

  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(1)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  ame_mix_matmul_active_k = 0;
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);

  ame_mix_matmul_noblk_pair_segment(xout0, xout1, xq, xs,
                                    wq4_0, ws0, wq4_1, ws1,
                                    n, d, gs, stride_w,
                                    weight_row0, scale_row0, ws_d);

  {
    const uint64_t total_cycles = ame_mix_rdcycle() - total_t0;
    AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
    AME_STAT_ADD(g_ame_mm_single_cycles, total_cycles);
  }
}

void ame_matmul_quant_noblk_i8_i8_f32_chunks(float *chunk_buf,
                                             int chunk_cap,
                                             const int8_t *xq,
                                             const float *xs,
                                             const int8_t *wq,
                                             const float *ws,
                                             int n,
                                             int d_full,
                                             int gs,
                                             int w_row_stride) {
  if (!chunk_buf || !xq || !xs || !wq || !ws || n <= 0 || d_full <= 0 || gs <= 0) return;
  if (chunk_cap <= 0 || (chunk_cap % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] chunk_cap % OUTBLK != 0, halting\r\n");
  }

  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  const uint64_t total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_single_calls);
  AME_STAT_ADD(g_ame_mm_macs, (uint64_t)n * (uint64_t)d_full);

  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(1)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  ame_mix_matmul_active_k = 0;
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);

  for (int row0 = 0; row0 < d_full; row0 += chunk_cap) {
    int chunk_m = d_full - row0;
    if (chunk_m > chunk_cap) chunk_m = chunk_cap;
    if ((chunk_m % AME_MIX_OUTBLK) != 0) {
      ame_mix_fail("[ame_mix_offline] tail chunk_m % OUTBLK != 0, halting\r\n");
    }
    ame_mix_matmul_noblk_segment(chunk_buf, xq, xs, (const uint8_t *)wq, ws,
                                 n, chunk_m, gs, stride_w, row0, row0, d_full);
  }

  {
    const uint64_t total_cycles = ame_mix_rdcycle() - total_t0;
    AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
    AME_STAT_ADD(g_ame_mm_single_cycles, total_cycles);
  }
}

static uint64_t ame_mix_scan_total_t0;
static int ame_mix_scan_active;

void ame_matmul_noblk_scan_begin(int n, int d_full, int gs) {
  if (gs > AME_MIX_MAX_GS || (gs % 2) != 0) {
    ame_mix_fail("[ame_mix_offline_w4] gs invalid, halting\r\n");
  }
  ame_mix_scan_total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_single_calls);
  AME_STAT_ADD(g_ame_mm_macs, (uint64_t)n * (uint64_t)d_full);
  asm volatile(
        "msettilem a6, %[v] # 0x04055877\n\t"
        :: [v] "r"((size_t)(1)) : "a6", "memory");
  asm volatile(
        "msettilen t2, %[v] # 0x040543f7\n\t"
        :: [v] "r"((size_t)(AME_MIX_HW_N)) : "t2", "memory");
  ame_mix_matmul_active_k = 0;
  AME_PROFILE_ADD(msettilem, 1);
  AME_PROFILE_ADD(msettilen, 1);
  ame_mix_scan_active = 1;
}

void ame_matmul_noblk_scan_chunk(float *chunk_buf,
                                 const int8_t *xq,
                                 const float *xs,
                                 const int8_t *wq,
                                 const float *ws,
                                 int n,
                                 int chunk_m,
                                 int gs,
                                 int w_row_stride,
                                 int row0,
                                 int d_full) {
  if (!ame_mix_scan_active) {
    ame_mix_fail("[ame_mix_offline] scan_chunk without begin, halting\r\n");
  }
  if (!chunk_buf || chunk_m <= 0 || (chunk_m % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] chunk_m invalid, halting\r\n");
  }
  if (row0 != 0 && (row0 % AME_MIX_OUTBLK) != 0) {
    ame_mix_fail("[ame_mix_offline] row0 % OUTBLK != 0, halting\r\n");
  }
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  ame_mix_matmul_noblk_segment(chunk_buf, xq, xs, (const uint8_t *)wq, ws,
                               n, chunk_m, gs, stride_w, row0, row0, d_full);
}

void ame_matmul_noblk_scan_end(void) {
  if (!ame_mix_scan_active) return;
  const uint64_t total_cycles = ame_mix_rdcycle() - ame_mix_scan_total_t0;
  AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
  AME_STAT_ADD(g_ame_mm_single_cycles, total_cycles);
  ame_mix_scan_active = 0;
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
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  ame_mix_matmul_noblk_core(xout, xq, xs, (const uint8_t *)wq, ws,
                            n, d, gs, stride_w, 0, 0, 0);
}

void ame_matmul_quant_noblk_i8_i8_f32_sramw_fullscale(float *xout,
                                                      const int8_t *xq,
                                                      const float *xs,
                                                      const int8_t *wq_sram_tile,
                                                      const float *ws_full,
                                                      int n,
                                                      int d_blk,
                                                      int d_full,
                                                      int d0,
                                                      int gs,
                                                      int w_row_stride) {
  if (!xout || !xq || !xs || !wq_sram_tile || !ws_full) return;
  if (n <= 0 || d_blk <= 0 || d_full <= 0 || gs <= 0) return;
  if (d0 < 0 || d0 + d_blk > d_full) {
    ame_mix_fail("[ame_mix_offline] sramw: d window out of range, halting\r\n");
  }
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  ame_mix_matmul_noblk_core(xout, xq, xs, (const uint8_t *)wq_sram_tile, ws_full,
                            n, d_blk, gs, stride_w, 0, d0, d_full);
}

void ame_matmul_quant_noblk_pair_i8_i8_f32(float *xout0,
                                           float *xout1,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq0,
                                           const float *ws0,
                                           const int8_t *wq1,
                                           const float *ws1,
                                           int n,
                                           int d,
                                           int gs,
                                           int w_row_stride) {
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  ame_mix_matmul_noblk_pair_core(xout0, xout1, xq, xs,
                                 (const uint8_t *)wq0, ws0,
                                 (const uint8_t *)wq1, ws1,
                                 n, d, gs, stride_w, 0, 0, 0);
}

void ame_matmul_quant_noblk_pair_i8_i8_f32_sramw_fullscale(float *xout0,
                                                           float *xout1,
                                                           const int8_t *xq,
                                                           const float *xs,
                                                           const int8_t *wq0_sram_tile,
                                                           const float *ws0_full,
                                                           const int8_t *wq1_sram_tile,
                                                           const float *ws1_full,
                                                           int n,
                                                           int d_blk,
                                                           int d_full,
                                                           int d0,
                                                           int gs,
                                                           int w_row_stride) {
  if (!xout0 || !xout1 || !xq || !xs || !wq0_sram_tile || !ws0_full ||
      !wq1_sram_tile || !ws1_full) return;
  if (n <= 0 || d_blk <= 0 || d_full <= 0 || gs <= 0) return;
  if (d0 < 0 || d0 + d_blk > d_full) {
    ame_mix_fail("[ame_mix_offline] pair sramw: d window out of range, halting\r\n");
  }
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  ame_mix_matmul_noblk_pair_core(xout0, xout1, xq, xs,
                                 (const uint8_t *)wq0_sram_tile, ws0_full,
                                 (const uint8_t *)wq1_sram_tile, ws1_full,
                                 n, d_blk, gs, stride_w, 0, d0, d_full);
}

void ame_matmul_quant_noblk_i8_i8_f32_at(float *xout,
                                         const int8_t *xq,
                                         const float *xs,
                                         const int8_t *wq,
                                         const float *ws,
                                         int n,
                                         int d,
                                         int gs,
                                         int w_row_stride,
                                         int d_row0,
                                         int d_full) {
  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  const int ws_d = (d_full > 0) ? d_full : d;
  ame_mix_matmul_noblk_core(xout, xq, xs, (const uint8_t *)wq, ws,
                            n, d, gs, stride_w, d_row0, d_row0, ws_d);
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
 * Batched quantized matmul (T tokens packed along M).
 *
 * Same math contract as the shared kernel:
 *   xout[t, j] += C[t, j] * ws[i0+j, g] * xs[t, g]
 *   C[t, j] = fp32(sum_k xq[t, base+k] * wq_tile[j, k])   (i8 -> i32 -> fp32)
 *
 * A is still packed per-token (activations are not known offline).
 * B is read directly from the tile-major weight blob.
 * --------------------------------------------------------------------*/

#define AME_MIX_BATCH_T_MAX  128

static float ame_mix_batch_c_pack[AME_MIX_BATCH_T_MAX * AME_MIX_OUTBLK] __attribute__((aligned(64)));
static float ame_mix_batch_zero_tile[AME_MIX_BREUSE_M * AME_MIX_OUTBLK] __attribute__((aligned(64)));

static inline void ame_mix_batch_accum_rows(float *xout,
                                            const float *ws,
                                            const float *xs,
                                            int d,
                                            int x_groups,
                                            int g,
                                            int i0,
                                            int row_base,
                                            int rows) {
  const float *ws_row = ws + (ptrdiff_t)g * (ptrdiff_t)d + (ptrdiff_t)i0;
  int t = 0;
  for (; t + 8 <= rows; t += 8) {
    const float *crow0 = ame_mix_batch_c_pack + (size_t)(t + 0) * AME_MIX_OUTBLK;
    const float *crow1 = ame_mix_batch_c_pack + (size_t)(t + 1) * AME_MIX_OUTBLK;
    const float *crow2 = ame_mix_batch_c_pack + (size_t)(t + 2) * AME_MIX_OUTBLK;
    const float *crow3 = ame_mix_batch_c_pack + (size_t)(t + 3) * AME_MIX_OUTBLK;
    const float *crow4 = ame_mix_batch_c_pack + (size_t)(t + 4) * AME_MIX_OUTBLK;
    const float *crow5 = ame_mix_batch_c_pack + (size_t)(t + 5) * AME_MIX_OUTBLK;
    const float *crow6 = ame_mix_batch_c_pack + (size_t)(t + 6) * AME_MIX_OUTBLK;
    const float *crow7 = ame_mix_batch_c_pack + (size_t)(t + 7) * AME_MIX_OUTBLK;
    const float xs0 = xs[(size_t)(row_base + t + 0) * (size_t)x_groups + (size_t)g];
    const float xs1 = xs[(size_t)(row_base + t + 1) * (size_t)x_groups + (size_t)g];
    const float xs2 = xs[(size_t)(row_base + t + 2) * (size_t)x_groups + (size_t)g];
    const float xs3 = xs[(size_t)(row_base + t + 3) * (size_t)x_groups + (size_t)g];
    const float xs4 = xs[(size_t)(row_base + t + 4) * (size_t)x_groups + (size_t)g];
    const float xs5 = xs[(size_t)(row_base + t + 5) * (size_t)x_groups + (size_t)g];
    const float xs6 = xs[(size_t)(row_base + t + 6) * (size_t)x_groups + (size_t)g];
    const float xs7 = xs[(size_t)(row_base + t + 7) * (size_t)x_groups + (size_t)g];
    float *xrow0 = xout + (size_t)(row_base + t + 0) * (size_t)d + (size_t)i0;
    float *xrow1 = xout + (size_t)(row_base + t + 1) * (size_t)d + (size_t)i0;
    float *xrow2 = xout + (size_t)(row_base + t + 2) * (size_t)d + (size_t)i0;
    float *xrow3 = xout + (size_t)(row_base + t + 3) * (size_t)d + (size_t)i0;
    float *xrow4 = xout + (size_t)(row_base + t + 4) * (size_t)d + (size_t)i0;
    float *xrow5 = xout + (size_t)(row_base + t + 5) * (size_t)d + (size_t)i0;
    float *xrow6 = xout + (size_t)(row_base + t + 6) * (size_t)d + (size_t)i0;
    float *xrow7 = xout + (size_t)(row_base + t + 7) * (size_t)d + (size_t)i0;
    ame_mix_accum_f32_prefill8(xrow0, xrow1, xrow2, xrow3,
                               xrow4, xrow5, xrow6, xrow7,
                               crow0, crow1, crow2, crow3,
                               crow4, crow5, crow6, crow7,
                               ws_row,
                               xs0, xs1, xs2, xs3, xs4, xs5, xs6, xs7,
                               AME_MIX_OUTBLK);
  }
  for (; t + 4 <= rows; t += 4) {
    const float *crow0 = ame_mix_batch_c_pack + (size_t)(t + 0) * AME_MIX_OUTBLK;
    const float *crow1 = ame_mix_batch_c_pack + (size_t)(t + 1) * AME_MIX_OUTBLK;
    const float *crow2 = ame_mix_batch_c_pack + (size_t)(t + 2) * AME_MIX_OUTBLK;
    const float *crow3 = ame_mix_batch_c_pack + (size_t)(t + 3) * AME_MIX_OUTBLK;
    const float xs0 = xs[(size_t)(row_base + t + 0) * (size_t)x_groups + (size_t)g];
    const float xs1 = xs[(size_t)(row_base + t + 1) * (size_t)x_groups + (size_t)g];
    const float xs2 = xs[(size_t)(row_base + t + 2) * (size_t)x_groups + (size_t)g];
    const float xs3 = xs[(size_t)(row_base + t + 3) * (size_t)x_groups + (size_t)g];
    float *xrow0 = xout + (size_t)(row_base + t + 0) * (size_t)d + (size_t)i0;
    float *xrow1 = xout + (size_t)(row_base + t + 1) * (size_t)d + (size_t)i0;
    float *xrow2 = xout + (size_t)(row_base + t + 2) * (size_t)d + (size_t)i0;
    float *xrow3 = xout + (size_t)(row_base + t + 3) * (size_t)d + (size_t)i0;
    ame_mix_accum_f32_prefill4(xrow0, xrow1, xrow2, xrow3,
                               crow0, crow1, crow2, crow3,
                               ws_row, xs0, xs1, xs2, xs3,
                               AME_MIX_OUTBLK);
  }
  if (t + 2 <= rows) {
    const float *crow0 = ame_mix_batch_c_pack + (size_t)(t + 0) * AME_MIX_OUTBLK;
    const float *crow1 = ame_mix_batch_c_pack + (size_t)(t + 1) * AME_MIX_OUTBLK;
    const float xs0 = xs[(size_t)(row_base + t + 0) * (size_t)x_groups + (size_t)g];
    const float xs1 = xs[(size_t)(row_base + t + 1) * (size_t)x_groups + (size_t)g];
    float *xrow0 = xout + (size_t)(row_base + t + 0) * (size_t)d + (size_t)i0;
    float *xrow1 = xout + (size_t)(row_base + t + 1) * (size_t)d + (size_t)i0;
    ame_mix_accum_f32_prefill2(xrow0, xrow1, crow0, crow1, ws_row,
                               xs0, xs1, AME_MIX_OUTBLK);
    t += 2;
  }
  for (; t < rows; t++) {
    const float *crow = ame_mix_batch_c_pack + (size_t)t * AME_MIX_OUTBLK;
    const float xst = xs[(size_t)(row_base + t) * (size_t)x_groups + (size_t)g];
    float *xrow = xout + (size_t)(row_base + t) * (size_t)d + (size_t)i0;
    ame_mix_accum_f32_prefill(xrow, crow, ws_row, xst, AME_MIX_OUTBLK);
  }
}

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
  const uint8_t *wq4 = (const uint8_t *)wq;

  if (!xout || !xq || !xs || !wq4 || !ws) return;
  if (T <= 0 || n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS || (gs % 2) != 0) {
    ame_mix_fail("[ame_mix_offline_w4] batch: gs invalid, halting\r\n");
  }

  if (T == 1) {
    ame_matmul_quant_noblk_i8_i8_f32(xout, xq, xs, wq, ws, n, d, gs,
                                     (w_row_stride > 0) ? w_row_stride : n);
    return;
  }

  if (T > AME_MIX_BATCH_T_MAX) {
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

  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;
  if (d % AME_MIX_OUTBLK) {
    ame_mix_fail("[ame_mix_offline] batch: d % OUTBLK != 0, halting\r\n");
  }
  if (stride_w % gs) {
    ame_mix_fail("[ame_mix_offline] batch: stride_w % gs != 0, halting\r\n");
  }

  const uint64_t total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_batch_calls);
  AME_STAT_ADD(g_ame_mm_macs, (uint64_t)T * (uint64_t)n * (uint64_t)d);

  const int groups        = (n + gs - 1) / gs;
  const int n_tile_groups = stride_w / gs;
  const int x_groups      = groups;

  {
    const uint64_t t0 = ame_mix_rdcycle();
    mix_zero_f32(xout, T * d);
    mix_zero_f32(ame_mix_batch_zero_tile, AME_MIX_BREUSE_M * AME_MIX_OUTBLK);
    AME_PROFILE_ADD(zero_f32_elements, T * d);
    AME_PROFILE_ADD(zero_f32_elements, AME_MIX_BREUSE_M * AME_MIX_OUTBLK);
    AME_STAT_ADD(g_ame_mm_zero_cycles, ame_mix_rdcycle() - t0);
  }

  for (int g = 0; g < groups; g++) {
    const int base  = g * gs;
    const int count = (base + gs <= n) ? gs : (n - base);
    if (count != gs) {
      ame_mix_fail("[ame_mix_offline] batch: partial tail group not supported\r\n");
    }

    /* A read directly from xq (no repack); xseg_g = this group's columns,
     * row stride n. */
    const int8_t *xseg_g = xq + (size_t)base;

    /* K128 A-resident schedule: keep four A tiles resident while walking
     * all output blocks. RVV accumulation does not touch the AME tile file. */
    if (count == 2 * AME_MIX_HW_K) {
      int mb = 0;
      for (; mb + 2 * AME_MIX_M_TILE <= T; mb += 2 * AME_MIX_M_TILE) {
        {
          const uint64_t t0 = ame_mix_rdcycle();
          amex_prefill_aresident_k128_load_a(
              xseg_g + (size_t)mb * (size_t)n, n);
          {
            const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
            AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
            AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
          }
        }

        for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
          const int8_t *btile = ame_mix_btile_i8((const int8_t *)wq4, i0, g,
                                                 gs, n_tile_groups);
          {
            const uint64_t t0 = ame_mix_rdcycle();
            amex_prefill_aresident_k128_n64_pair_i8_i8_f32(
                btile, ame_mix_batch_c_pack, ame_mix_batch_zero_tile);
            {
              const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
              AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
              AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
            }
          }
          AME_STAT_INC(g_ame_mm_tile_invocations);
          AME_STAT_INC(g_ame_mm_batch_tiles);

          {
            const uint64_t t0 = ame_mix_rdcycle();
            ame_mix_batch_accum_rows(xout, ws, xs, d, x_groups, g, i0,
                                     mb, 2 * AME_MIX_M_TILE);
            AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
          }
        }
      }

      /* Keep arbitrary T support: only the final rows miss the 32-row
       * A-resident shape and fall back to the generic kernel. */
      if (mb < T) {
        const int tail_rows = T - mb;
        for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
          const int8_t *btile = ame_mix_btile_i8((const int8_t *)wq4, i0, g,
                                                 gs, n_tile_groups);
          {
            const uint64_t t0 = ame_mix_rdcycle();
            amex_prefill_2a4b_n64_i8_i8_f32(
                xseg_g + (size_t)mb * (size_t)n,
                btile,
                ame_mix_batch_c_pack,
                ame_mix_batch_zero_tile,
                tail_rows,
                count,
                n);
            {
              const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
              AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
              AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
            }
          }
          AME_STAT_INC(g_ame_mm_tile_invocations);
          AME_STAT_INC(g_ame_mm_batch_tiles);

          {
            const uint64_t t0 = ame_mix_rdcycle();
            ame_mix_batch_accum_rows(xout, ws, xs, d, x_groups, g, i0,
                                     mb, tail_rows);
            AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
          }
        }
      }
      continue;
    }

    for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
      const int8_t *btile = ame_mix_btile_i8((const int8_t *)wq4, i0, g,
                                             gs, n_tile_groups);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        int fast_rows = 0;
        if (count == AME_MIX_HW_K) {
          fast_rows = amex_prefill_breuse_k64_n64_i8_i8_f32(xseg_g, btile,
                                                            ame_mix_batch_c_pack,
                                                            ame_mix_batch_zero_tile,
                                                            T, n);
        }
        if (fast_rows < T) {
          /* 8-accumulator 2A x 4B path (N = OUTBLK), A read from xq with
           * lda = n; the partial last M-tile covers T not a multiple of 16. */
          amex_prefill_2a4b_n64_i8_i8_f32(
              xseg_g + (size_t)fast_rows * (size_t)n,
              btile,
              ame_mix_batch_c_pack + (size_t)fast_rows * AME_MIX_OUTBLK,
              ame_mix_batch_zero_tile,
              T - fast_rows,
              count,
              n);
        }
        {
          const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
          AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
          AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
        }
      }
      AME_STAT_INC(g_ame_mm_tile_invocations);
      AME_STAT_INC(g_ame_mm_batch_tiles);

      {
        const uint64_t t0 = ame_mix_rdcycle();
        ame_mix_batch_accum_rows(xout, ws, xs, d, x_groups, g, i0, 0, T);
        AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
      }
    }
  }
  {
    const uint64_t total_cycles = ame_mix_rdcycle() - total_t0;
    AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
    AME_STAT_ADD(g_ame_mm_batch_cycles, total_cycles);
  }
}

void ame_matmul_quant_batch_pair_i8_i8_f32(float *xout0,
                                           float *xout1,
                                           const int8_t *xq,
                                           const float *xs,
                                           const int8_t *wq0,
                                           const float *ws0,
                                           const int8_t *wq1,
                                           const float *ws1,
                                           int T,
                                           int n,
                                           int d,
                                           int gs,
                                           int w_row_stride) {
  if (T == 1) {
    ame_matmul_quant_noblk_pair_i8_i8_f32(xout0, xout1, xq, xs,
                                          wq0, ws0, wq1, ws1,
                                          n, d, gs, w_row_stride);
    return;
  }

  if (!xout0 || !xout1 || !xq || !xs || !wq0 || !ws0 || !wq1 || !ws1) return;
  if (T <= 0 || n <= 0 || d <= 0 || gs <= 0) return;
  if (gs > AME_MIX_MAX_GS || (gs % 2) != 0) {
    ame_mix_fail("[ame_mix_offline_w4] batch_pair: gs invalid, halting\r\n");
  }

  const int stride_w = (w_row_stride > 0) ? w_row_stride : n;

  if (T > AME_MIX_BATCH_T_MAX || gs != 2 * AME_MIX_HW_K) {
    ame_matmul_quant_batch_i8_i8_f32(xout0, xq, xs, wq0, ws0,
                                     T, n, d, gs, stride_w);
    ame_matmul_quant_batch_i8_i8_f32(xout1, xq, xs, wq1, ws1,
                                     T, n, d, gs, stride_w);
    return;
  }

  if (d % AME_MIX_OUTBLK) {
    ame_mix_fail("[ame_mix_offline] batch_pair: d % OUTBLK != 0, halting\r\n");
  }
  if (stride_w % gs) {
    ame_mix_fail("[ame_mix_offline] batch_pair: stride_w % gs != 0, halting\r\n");
  }

  const uint8_t *wq4_0 = (const uint8_t *)wq0;
  const uint8_t *wq4_1 = (const uint8_t *)wq1;
  const uint64_t total_t0 = ame_mix_rdcycle();
  AME_STAT_INC(g_ame_mm_calls);
  AME_STAT_INC(g_ame_mm_batch_calls);
  AME_STAT_ADD(g_ame_mm_macs, 2 * (uint64_t)T * (uint64_t)n * (uint64_t)d);

  const int groups        = (n + gs - 1) / gs;
  const int n_tile_groups = stride_w / gs;
  const int x_groups      = groups;

  {
    const uint64_t t0 = ame_mix_rdcycle();
    mix_zero_f32(xout0, T * d);
    mix_zero_f32(xout1, T * d);
    mix_zero_f32(ame_mix_batch_zero_tile, AME_MIX_BREUSE_M * AME_MIX_OUTBLK);
    AME_PROFILE_ADD(zero_f32_elements, 2 * T * d);
    AME_PROFILE_ADD(zero_f32_elements, AME_MIX_BREUSE_M * AME_MIX_OUTBLK);
    AME_STAT_ADD(g_ame_mm_zero_cycles, ame_mix_rdcycle() - t0);
  }

  for (int g = 0; g < groups; g++) {
    const int base  = g * gs;
    const int count = (base + gs <= n) ? gs : (n - base);
    if (count != gs) {
      ame_mix_fail("[ame_mix_offline] batch_pair: partial tail group not supported\r\n");
    }

    const int8_t *xseg_g = xq + (size_t)base;
    int mb = 0;

    for (; mb + 2 * AME_MIX_M_TILE <= T; mb += 2 * AME_MIX_M_TILE) {
      {
        const uint64_t t0 = ame_mix_rdcycle();
        amex_prefill_aresident_k128_load_a(
            xseg_g + (size_t)mb * (size_t)n, n);
        {
          const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
          AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
          AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
        }
      }

      for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
        const int8_t *btile0 = ame_mix_btile_i8((const int8_t *)wq4_0, i0, g,
                                                gs, n_tile_groups);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          amex_prefill_aresident_k128_n64_pair_i8_i8_f32(
              btile0, ame_mix_batch_c_pack, ame_mix_batch_zero_tile);
          {
            const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
            AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
            AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
          }
        }
        AME_STAT_INC(g_ame_mm_tile_invocations);
        AME_STAT_INC(g_ame_mm_batch_tiles);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          ame_mix_batch_accum_rows(xout0, ws0, xs, d, x_groups, g, i0,
                                   mb, 2 * AME_MIX_M_TILE);
          AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
        }

        const int8_t *btile1 = ame_mix_btile_i8((const int8_t *)wq4_1, i0, g,
                                                gs, n_tile_groups);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          amex_prefill_aresident_k128_n64_pair_i8_i8_f32(
              btile1, ame_mix_batch_c_pack, ame_mix_batch_zero_tile);
          {
            const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
            AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
            AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
          }
        }
        AME_STAT_INC(g_ame_mm_tile_invocations);
        AME_STAT_INC(g_ame_mm_batch_tiles);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          ame_mix_batch_accum_rows(xout1, ws1, xs, d, x_groups, g, i0,
                                   mb, 2 * AME_MIX_M_TILE);
          AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
        }
      }
    }

    if (mb < T) {
      const int tail_rows = T - mb;
      for (int i0 = 0; i0 < d; i0 += AME_MIX_OUTBLK) {
        const int8_t *btile0 = ame_mix_btile_i8((const int8_t *)wq4_0, i0, g,
                                                gs, n_tile_groups);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          amex_prefill_2a4b_n64_i8_i8_f32(
              xseg_g + (size_t)mb * (size_t)n,
              btile0,
              ame_mix_batch_c_pack,
              ame_mix_batch_zero_tile,
              tail_rows,
              count,
              n);
          {
            const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
            AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
            AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
          }
        }
        AME_STAT_INC(g_ame_mm_tile_invocations);
        AME_STAT_INC(g_ame_mm_batch_tiles);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          ame_mix_batch_accum_rows(xout0, ws0, xs, d, x_groups, g, i0,
                                   mb, tail_rows);
          AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
        }

        const int8_t *btile1 = ame_mix_btile_i8((const int8_t *)wq4_1, i0, g,
                                                gs, n_tile_groups);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          amex_prefill_2a4b_n64_i8_i8_f32(
              xseg_g + (size_t)mb * (size_t)n,
              btile1,
              ame_mix_batch_c_pack,
              ame_mix_batch_zero_tile,
              tail_rows,
              count,
              n);
          {
            const uint64_t kernel_cycles = ame_mix_rdcycle() - t0;
            AME_STAT_ADD(g_ame_mm_mqma_cycles, kernel_cycles);
            AME_STAT_ADD(g_ame_mm_batch_kernel_cycles, kernel_cycles);
          }
        }
        AME_STAT_INC(g_ame_mm_tile_invocations);
        AME_STAT_INC(g_ame_mm_batch_tiles);
        {
          const uint64_t t0 = ame_mix_rdcycle();
          ame_mix_batch_accum_rows(xout1, ws1, xs, d, x_groups, g, i0,
                                   mb, tail_rows);
          AME_STAT_ADD(g_ame_mm_accum_cycles, ame_mix_rdcycle() - t0);
        }
      }
    }
  }

  {
    const uint64_t total_cycles = ame_mix_rdcycle() - total_t0;
    AME_STAT_ADD(g_ame_mm_total_cycles, total_cycles);
    AME_STAT_ADD(g_ame_mm_batch_cycles, total_cycles);
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
