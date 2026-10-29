/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/pipeline.hh"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

Pipeline::Pipeline(MinorInstQueue &inst_queue, const PipelineParams &params_,
                   ConfigSnapshot config_snapshot)
    : instQueue(inst_queue), params(params_),
      contexts(params.maxInflightInsts),
      dispatchQueue(params.dispatchQueueDepth),
      peRequestQueue(params.peRequestQueueDepth),
      laneCompletions(params.numLanes * params.lane.laneScoreboardEntries),
      completionQueue(params.completionQueueDepth),
      memoryRequests(params.externalRequestQueueDepth),
      memoryResults(params.externalResultQueueDepth),
      slduOperandRequests(params.externalRequestQueueDepth),
      slduResults(params.externalResultQueueDepth),
      dispatcher(inst_queue, contexts, globalScoreboard, dispatchQueue,
                 DispatcherParams{
                     params.numLanes, params.laneWidthBits,
                     params.maxPackedElementsPerLane},
                 std::move(config_snapshot)),
      sequencer(contexts, globalScoreboard, dispatchQueue,
                peRequestQueue, params.numLanes, params.rvvInsnSlotCapacity),
      memoryUnit(contexts, memoryRequests, memoryResults, params.memory),
      sldu(contexts, slduOperandRequests, slduResults, params.sldu,
           params.externalRequestQueueDepth),
      masku(params.externalRequestQueueDepth, params.lane.configLatency),
      completionTracker(contexts, globalScoreboard, laneCompletions,
                        completionQueue)
{
    fatal_if(params.numLanes == 0 || params.numLanes > 64 ||
             params.laneWidthBits == 0 || params.vlenBits == 0 ||
             params.completionWidth == 0 ||
             params.maxPackedElementsPerLane == 0,
             "Invalid RVV pipeline parameters");
    const uint64_t lane_array_bits =
        static_cast<uint64_t>(params.numLanes) * params.laneWidthBits;
    fatal_if(params.vlenBits % lane_array_bits != 0,
             "RVV VLEN (%u) must be a multiple of the lane array width "
             "(%u lanes * %u bits)",
             params.vlenBits, params.numLanes, params.laneWidthBits);
    fatal_if(params.lane.numLanes != params.numLanes ||
             params.lane.laneWidthBits != params.laneWidthBits ||
             params.lane.vlenBits != params.vlenBits ||
             params.sldu.numLanes != params.numLanes,
             "RVV lane topology must match pipeline topology");
    lanes.reserve(params.numLanes);
    for (unsigned lane = 0; lane < params.numLanes; ++lane)
        lanes.emplace_back(lane, contexts, globalScoreboard,
                           memoryRequests, slduOperandRequests, params.lane);
}

void
Pipeline::evaluateTargetDone()
{
    const auto consume = [this](RvvChannel<TargetDone> &channel) {
        const size_t available = channel.size();
        for (size_t index = 0; index < available; ++index) {
            const auto done = channel.at(index);
            completionTracker.targetDone(done.contextId, done.target);
            channel.popNext();
        }
    };
    consume(sldu.doneChannel());
    consume(masku.doneChannel());
}

void
Pipeline::evaluateMinorCompletion()
{
    unsigned completed = 0;
    while (!completionQueue.empty() && completed < params.completionWidth) {
        const RvvCompletion &rvv_completion = completionQueue.front();
        InstQueueCompletion completion;
        completion.key = rvv_completion.key;
        completion.fault = rvv_completion.fault;
        const bool accepted = instQueue.complete(std::move(completion));

        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=- lane=- Completion HANDSHAKE "
                "accepted=%d\n",
                currentCycle, rvv_completion.instId, accepted);

        panic_if(!accepted,
                 "RVV completion was rejected by MinorInstQueue "
                 "(inst=%llu tid=%d exec=%llu)",
                 rvv_completion.instId, rvv_completion.key.threadId,
                 rvv_completion.key.execSeqNum);

        const RvvInstId id = rvv_completion.instId;
        const RvvInsnSlotId slot_id = contexts.get(id).rvvSlotId;
        const bool slot_allocated = contexts.get(id).slotIdAllocated;
        for (auto &lane : lanes)
            lane.release(slot_id);
        completionQueue.popNext();
        dispatcher.release(id);
        contexts.release(id);
        if (slot_allocated)
            sequencer.releaseSlot(slot_id);
        ++completed;
        ++completedInstructions;
    }
}

void
Pipeline::evaluateExternalResults()
{
    const auto route = [this](RvvChannel<VectorFuResp> &results) {
        if (results.empty())
            return;
        const auto &response = results.front();
        const unsigned lane = response.tag.laneId;
        panic_if(lane >= lanes.size(), "RVV external result lane out of range");
        if (!lanes[lane].canAcceptExternalResultNext())
            return;
        lanes[lane].acceptExternalResultNext(response);
        results.popNext();
    };
    route(memoryResults);
    route(slduResults);
}

void
Pipeline::evaluatePeFanout()
{
    if (peRequestQueue.empty())
        return;
    const PeRequest &request = peRequestQueue.front();
    bool all_ready = true;
    for (unsigned lane = 0; lane < params.numLanes; ++lane) {
        if (!lanes[lane].canAcceptRequestNext()) {
            all_ready = false;
        }
    }
    if (!all_ready)
        return;

    TargetVfuMask accepted = TargetLane;
    if (request.targetVfus & TargetVlsu) {
        if (!memoryRequests.canPushNext())
            return;
        accepted |= TargetVlsu;
    }
    if (request.targetVfus & TargetSldu) {
        if (!sldu.canAcceptNext())
            return;
        accepted |= TargetSldu;
    }
    if ((request.targetVfus & TargetMasku) && !masku.canAcceptNext())
        return;
    if (request.targetVfus & TargetMasku)
        accepted |= TargetMasku;

    for (unsigned lane = 0; lane < params.numLanes; ++lane) {
        lanes[lane].acceptRequestNext(request);
    }
    if (request.targetVfus & TargetSldu)
        sldu.acceptNext(request);
    if (request.targetVfus & TargetMasku)
        masku.acceptNext(request);

    /* A single instruction-level request still needs one completion envelope
     * for the existing lane result streams.  This is not a global execution
     * beat: lane-local beat/word progress is generated by VectorLane. */
    VecBeat envelope;
    envelope.tag = {request.contextId, 0};
    envelope.activeLaneMask = params.numLanes == 64 ? ~0ULL :
        ((1ULL << params.numLanes) - 1);
    envelope.first = true;
    envelope.last = true;
    completionTracker.registerBeatNext(envelope);
    if (request.targetVfus & TargetVlsu)
        memoryUnit.registerBeatNext(envelope);
    sequencer.targetAccepted(request.contextId, accepted);
    peRequestQueue.popNext();
    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu slot=%u PE_FANOUT FIRE lanes=%u\n",
            currentCycle, request.contextId, request.rvvSlotId, params.numLanes);
}

void
Pipeline::evaluate()
{
    /* Back-to-front evaluation plus next-state channels makes call order
     * observationally irrelevant at module boundaries. */
    evaluateMinorCompletion();
    completionTracker.evaluate(currentCycle);
    evaluateExternalResults();
    evaluateTargetDone();
    sldu.evaluate(currentCycle);
    masku.evaluate(currentCycle);
    memoryUnit.evaluate(currentCycle);
    for (auto &lane : lanes)
        lane.evaluate(currentCycle, laneCompletions);
    evaluatePeFanout();
    sequencer.evaluate(currentCycle);
    dispatcher.evaluate(currentCycle);
    advance();
    ++currentCycle;
}

void
Pipeline::advance()
{
    globalScoreboard.advance();
    dispatchQueue.advance();
    peRequestQueue.advance();
    laneCompletions.advance();
    completionQueue.advance();
    memoryRequests.advance();
    memoryResults.advance();
    sldu.advance();
    masku.advance();
    completionTracker.advance();
    memoryUnit.advance();
    for (auto &lane : lanes)
        lane.advance();
}

bool
Pipeline::cancel(const InstQueueKey &key)
{
    const auto id = dispatcher.idForKey(key);
    if (!id)
        return false;
    sequencer.cancel(*id);
    for (auto &lane : lanes)
        lane.cancel(*id);
    completionTracker.cancel(*id);
    memoryUnit.cancel(*id);
    sldu.cancel(*id);
    masku.cancel(*id);
    peRequestQueue.eraseIf([id](const PeRequest &request) {
        return request.contextId == *id;
    });
    dispatcher.cancel(key);
    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu beat=- lane=- Pipeline CANCEL exec=%llu\n",
            currentCycle, *id, key.execSeqNum);
    return true;
}

bool
Pipeline::active() const
{
    if (!contexts.empty() || dispatcher.active() || sequencer.active() ||
        completionTracker.active() || dispatchQueue.active() ||
        peRequestQueue.active() || laneCompletions.active() ||
        completionQueue.active() || memoryUnit.active() ||
        sldu.activeState() || masku.activeState() ||
        instQueue.hasPending(InstQueueEngine::RVV)) {
        return true;
    }
    return std::any_of(lanes.begin(), lanes.end(),
        [](const VectorLane &lane) { return lane.active(); });
}

} // namespace rvv
} // namespace minor
} // namespace gem5
