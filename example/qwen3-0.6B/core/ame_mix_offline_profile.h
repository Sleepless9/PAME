#ifndef QWEN3_AME_MIX_OFFLINE_PROFILE_H
#define QWEN3_AME_MIX_OFFLINE_PROFILE_H

#include <stdint.h>

#define AME_PROFILE_SAMPLE_DECODE_KSTEP  1u
#define AME_PROFILE_SAMPLE_BREUSE_NBLK 2u
#define AME_PROFILE_SAMPLE_2A4B_KSTEP  3u
#define AME_PROFILE_SAMPLE_ARESIDENT   4u
#define AME_PROFILE_SAMPLE_2A4B_TAIL   5u

typedef struct {
  uint64_t calls;
  uint64_t single_calls;
  uint64_t batch_calls;
  uint64_t tiles;
  uint64_t single_tiles;
  uint64_t batch_tiles;

  uint64_t total_cycles;
  uint64_t single_cycles;
  uint64_t batch_cycles;
  uint64_t zero_cycles;
  uint64_t ame_kernel_cycles;
  uint64_t fp32_accum_cycles;
  uint64_t pack_cycles;

  uint64_t resync_calls;
  uint64_t msettilem;
  uint64_t msettilen;
  uint64_t msettilek;
  uint64_t msettype;
  uint64_t mlae8;
  uint64_t mlbe8;
  uint64_t mlbt8;
  uint64_t mlce32;
  uint64_t mqma_b;
  uint64_t mqma_b_aresident;
  uint64_t msce32;
  uint64_t fence_rw_rw;

  uint64_t rvv_accum_chunks;
  uint64_t zero_f32_elements;

  /* One representative inner-loop window (rdcycle), for ratio calibration. */
  uint64_t sample_tag;
  uint64_t sample_total;
  uint64_t sample_mset;
  uint64_t sample_mlae8;
  uint64_t sample_mlbe8;
  uint64_t sample_mqma;
  uint64_t sample_mlce32;
  uint64_t sample_msce32;
  uint64_t sample_fence;

  /* Measured ame_kernel_cycles attributed to each matmul kernel path. */
  uint64_t kernel_cycles_decode;
  uint64_t kernel_cycles_breuse;
  uint64_t kernel_cycles_2a4b;
  uint64_t kernel_cycles_aresident;
} AmeMixOfflineProfile;

void ame_matmul_profile_snapshot(AmeMixOfflineProfile *out);

#endif
