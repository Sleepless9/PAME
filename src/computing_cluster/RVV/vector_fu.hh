/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_FU_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_FU_HH__

#include <vector>

#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct VectorFuParams
{
    unsigned latency = 1;
    unsigned initiationInterval = 1;
    unsigned inputQueueDepth = 2;
    unsigned maxInflight = 8;
};

class VectorFuPipeline
{
  private:
    struct Inflight
    {
        VectorFuResp response;
        uint64_t readyCycle = 0;
    };

    const VecFuType type;
    const VectorFuParams params;
    RvvChannel<VectorFuReq> input;
    std::vector<Inflight> inflight;
    std::vector<Inflight> pendingInflight;
    uint64_t nextAcceptCycle = 0;

  public:
    VectorFuPipeline(VecFuType type_, const VectorFuParams &params_);

    bool canAcceptInputNext() const { return input.canPushNext(); }
    void acceptInputNext(VectorFuReq request);
    void evaluate(uint64_t cycle, LaneId lane_id,
                  RvvChannel<VectorFuResp> &result_queue);
    void advance();
    void cancel(RvvInstId id);
    bool active() const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_FU_HH__
