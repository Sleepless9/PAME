/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_PIPELINE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_PIPELINE_HH__

#include <vector>

#include "computing_cluster/RVV/completion_tracker.hh"
#include "computing_cluster/RVV/vector_masku.hh"
#include "computing_cluster/RVV/vector_sldu.hh"
#include "computing_cluster/RVV/global_sequencer.hh"
#include "computing_cluster/RVV/rvv_dispatcher.hh"
#include "computing_cluster/RVV/vector_lane.hh"
#include "computing_cluster/RVV/vector_memory.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct PipelineParams
{
    unsigned numLanes = 4;
    unsigned laneWidthBits = 64;
    unsigned vlenBits = 256;
    unsigned maxPackedElementsPerLane = 8;
    unsigned dispatchQueueDepth = 4;
    unsigned maxInflightInsts = 8;
    unsigned peRequestQueueDepth = 2;
    unsigned rvvInsnSlotCapacity = 8;
    unsigned completionQueueDepth = 4;
    unsigned completionWidth = 1;
    unsigned externalRequestQueueDepth = 16;
    unsigned externalResultQueueDepth = 16;
    VectorLaneParams lane;
    VectorMemoryParams memory;
    SlduParams sldu;
};

/** MinorRVV backend with instruction, lane-word and completion transactions. */
class Pipeline
{
  private:
    MinorInstQueue &instQueue;
    const PipelineParams params;

    InstContextTable contexts;
    GlobalVectorScoreboard globalScoreboard;
    RvvChannel<VectorInstToken> dispatchQueue;
    RvvChannel<PeRequest> peRequestQueue;
    RvvChannel<LaneComplete> laneCompletions;
    RvvChannel<RvvCompletion> completionQueue;
    RvvChannel<VectorFuReq> memoryRequests;
    RvvChannel<VectorFuResp> memoryResults;
    RvvChannel<VectorFuReq> slduOperandRequests;
    RvvChannel<VectorFuResp> slduResults;

    RvvDispatcher dispatcher;
    GlobalSequencer sequencer;
    VectorMemoryUnit memoryUnit;
    VectorSldu sldu;
    VectorMasku masku;
    std::vector<VectorLane> lanes;
    CompletionTracker completionTracker;
    uint64_t currentCycle = 0;
    uint64_t completedInstructions = 0;

    void evaluateMinorCompletion();
    void evaluateExternalResults();
    void evaluateTargetDone();
    void evaluatePeFanout();
    void advance();

  public:
    Pipeline(MinorInstQueue &inst_queue, const PipelineParams &params_,
             ConfigSnapshot config_snapshot = {});

    void evaluate();
    bool cancel(const InstQueueKey &key);
    uint64_t completionCount() const { return completedInstructions; }
    void completeTimingResponse(uint64_t transaction_id)
    { memoryUnit.completeTimingResponse(transaction_id); }
    bool active() const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_PIPELINE_HH__
