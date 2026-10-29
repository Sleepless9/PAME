/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_LANE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_LANE_HH__

#include <array>
#include <map>
#include <optional>
#include <string>

#include "computing_cluster/RVV/global_scoreboard.hh"
#include "computing_cluster/RVV/lane_scoreboard.hh"
#include "computing_cluster/RVV/vector_fu.hh"
#include "computing_cluster/RVV/vrf_bank.hh"
#include "computing_cluster/RVV/result_progress.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct VectorLaneParams
{
    unsigned numLanes = 4;
    unsigned laneWidthBits = 64;
    unsigned vlenBits = 256;
    unsigned banksPerLane = 8;
    unsigned laneScoreboardEntries = 4;
    unsigned laneInputDepth = 2;
    unsigned operandTaskDepth = 4;
    unsigned vrfRequestQueueDepth = 2;
    unsigned operandQueueDepth = 2;
    unsigned resultQueueDepth = 2;
    std::string vrfArbitrationPolicy = "two_level_priority";
    unsigned aluLatency = 1;
    unsigned aluII = 1;
    unsigned multiplyLatency = 3;
    unsigned multiplyII = 1;
    unsigned fpuLatency = 4;
    unsigned fpuII = 1;
    unsigned divideLatency = 8;
    unsigned divideII = 8;
    unsigned memoryLatency = 1;
    unsigned memoryII = 1;
    unsigned configLatency = 1;
    bool directFuBypass = false;
};

class VectorLane
{
  private:
    const LaneId laneId;
    const unsigned numLanes;
    const unsigned vlenBytes;
    const bool directFuBypass;
    InstContextTable &contexts;
    GlobalVectorScoreboard &globalScoreboard;
    RvvChannel<VectorFuReq> &memoryRequests;
    RvvChannel<VectorFuReq> &slduOperandRequests;
    RvvChannel<PeRequest> requestInput;
    std::optional<PeRequest> activeRequest;
    uint32_t nextElement = 0;
    uint32_t nextWord = 0;
    RvvChannel<LaneBeat> input;
    RvvChannel<LaneTag> operandTasks;
    LaneScoreboard scoreboard;
    VectorRegFile vrf;
    ResultProgressTable resultProgress;
    std::array<VectorFuPipeline,
               static_cast<size_t>(VecFuType::Sldu)> fus;
    RvvChannel<VectorFuResp> resultQueue;
    struct WordCompletion {
        unsigned completed = 0;
        unsigned expected = 0;
    };
    std::map<RvvInstId, WordCompletion> wordCompletions;
    void completeWord(const LaneTag &tag, bool last, TargetVfuMask target,
                      RvvChannel<LaneComplete> &completions);

    VectorFuPipeline &fu(VecFuType type)
    { return fus[static_cast<size_t>(type)]; }
    const VectorFuPipeline &fu(VecFuType type) const
    { return fus[static_cast<size_t>(type)]; }

    void evaluateWriteback(uint64_t cycle,
                           RvvChannel<LaneComplete> &completions);
    void evaluateCollector(uint64_t cycle);
    void evaluateOperandRequester(uint64_t cycle);
    void evaluateLaneSequencer(uint64_t cycle);
    static TargetVfuMask targetForFu(VecFuType type)
    {
        if (type == VecFuType::Memory)
            return TargetLane | TargetVlsu;
        if (type == VecFuType::Sldu)
            return TargetLane;
        return TargetLane;
    }

  public:
    VectorLane(LaneId lane_id, InstContextTable &contexts_,
               GlobalVectorScoreboard &global_scoreboard,
               RvvChannel<VectorFuReq> &memory_requests,
               RvvChannel<VectorFuReq> &sldu_operand_requests,
               const VectorLaneParams &params);

    bool canAcceptBeatNext() const
    { return input.canPushNext() && scoreboard.canAllocateNext(); }
    void acceptBeatNext(LaneBeat beat) { input.pushNext(std::move(beat)); }
    bool canAcceptRequestNext() const { return requestInput.canPushNext(); }
    void acceptRequestNext(PeRequest request)
    { requestInput.pushNext(std::move(request)); }
    bool canAcceptExternalResultNext() const
    { return resultQueue.canPushNext(); }
    void acceptExternalResultNext(VectorFuResp result)
    { resultQueue.pushNext(std::move(result)); }

    void evaluate(uint64_t cycle, RvvChannel<LaneComplete> &completions);
    void advance();
    void cancel(RvvInstId id);
    void release(RvvInsnSlotId slot_id) { resultProgress.release(slot_id); }
    bool active() const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_LANE_HH__
