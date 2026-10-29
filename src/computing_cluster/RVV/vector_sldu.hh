/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_SLDU_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_SLDU_HH__

#include <deque>
#include <map>
#include <utility>
#include <vector>

#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_context.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct SlduParams
{
    unsigned numLanes = 4;
    unsigned maxInflightWords = 4;
    unsigned slideLatency = 2;
    unsigned intraLaneReductionLatency = 2;
    unsigned reductionStageLatency = 1;
    unsigned resultWidth = 1;
};

enum class SlduPhase : uint8_t
{
    IntraLaneReduction,
    InterLaneExchange,
    FinalLaneReduction,
    Writeback
};

/**
 * Shared slide and reduction unit connected to every vector lane.
 *
 * The SLDU owns both instruction-level control and the per-lane operand/result
 * streams.  A separate cross-lane execution target would double-count the
 * same hardware.
 */
class VectorSldu
{
  private:
    struct WordKey
    {
        RvvInstId contextId = 0;
        uint32_t wordIndex = 0;

        bool operator<(const WordKey &other) const
        {
            return contextId < other.contextId ||
                (contextId == other.contextId &&
                 wordIndex < other.wordIndex);
        }
    };

    struct Operation
    {
        PeRequest request;
        bool reduction = false;
        uint32_t totalWords = 1;
        uint32_t completedWords = 0;
        bool reductionScheduled = false;
        uint64_t laneReductionBusyMask = 0;
        std::map<LaneId, std::deque<VectorFuReq>> reductionQueues;
        std::map<LaneId, VectorFuReq> lanePartials;
    };

    struct WordBatch
    {
        uint64_t expectedLaneMask = 0;
        std::map<LaneId, VectorFuReq> requests;
    };

    struct Inflight
    {
        RvvInstId contextId = 0;
        uint32_t wordIndex = 0;
        uint64_t participatingLaneMask = 0;
        SlduPhase phase = SlduPhase::InterLaneExchange;
        uint64_t readyCycle = 0;
        std::vector<VectorFuResp> responses;
        size_t nextResponse = 0;
        bool localReduction = false;
        bool finalReduction = false;
        VectorFuReq sourceRequest;
    };

    InstContextTable &contexts;
    RvvChannel<VectorFuReq> &laneRequests;
    RvvChannel<VectorFuResp> &laneResults;
    const SlduParams params;
    const unsigned instructionCapacity;
    RvvChannel<PeRequest> requests;
    RvvChannel<TargetDone> completions;
    std::map<RvvInstId, Operation> operations;
    std::map<WordKey, WordBatch> batches;
    std::vector<Inflight> inflight;

    static unsigned countBits(uint64_t value);
    uint32_t laneWordCount(const RvvInstDesc &desc, LaneId lane) const;
    uint64_t expectedLaneMask(const RvvInstDesc &desc,
                              uint32_t word_index) const;
    unsigned interLaneLatency(uint64_t lane_mask) const;
    void consumeInstructions();
    void consumeLaneRequests();
    void emitResults(uint64_t cycle);
    void advancePhases(uint64_t cycle);
    void scheduleReady(uint64_t cycle);
    void completeInstructions();

  public:
    VectorSldu(InstContextTable &contexts_,
               RvvChannel<VectorFuReq> &lane_requests,
               RvvChannel<VectorFuResp> &lane_results,
               const SlduParams &params_, unsigned instruction_capacity = 4);

    bool canAcceptNext() const { return requests.canPushNext(); }
    void acceptNext(PeRequest request) { requests.pushNext(std::move(request)); }
    const RvvChannel<TargetDone> &doneChannel() const { return completions; }
    RvvChannel<TargetDone> &doneChannel() { return completions; }
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(RvvInstId id);
    bool activeState() const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_SLDU_HH__
