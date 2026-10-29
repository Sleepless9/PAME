/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vector_lane.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

VectorFuParams
fuParams(unsigned latency, unsigned ii, unsigned input_depth,
         unsigned inflight)
{
    return VectorFuParams{latency, ii, input_depth, inflight};
}

} // anonymous namespace

VectorLane::VectorLane(
    LaneId lane_id, InstContextTable &contexts_,
    GlobalVectorScoreboard &global_scoreboard,
    RvvChannel<VectorFuReq> &memory_requests,
    RvvChannel<VectorFuReq> &sldu_operand_requests,
    const VectorLaneParams &params)
    : laneId(lane_id), numLanes(params.numLanes),
      vlenBytes(params.vlenBits / 8),
      directFuBypass(params.directFuBypass), contexts(contexts_),
      globalScoreboard(global_scoreboard), memoryRequests(memory_requests),
      slduOperandRequests(sldu_operand_requests),
      requestInput(params.laneInputDepth), input(params.laneInputDepth),
      operandTasks(params.operandTaskDepth),
      scoreboard(params.laneScoreboardEntries),
      vrf(lane_id, params.numLanes, params.laneWidthBits,
          params.banksPerLane, params.vlenBits,
          params.vrfRequestQueueDepth,
          params.laneScoreboardEntries * 4,
          params.vrfArbitrationPolicy),
      fus{{
          VectorFuPipeline(VecFuType::Alu,
              fuParams(params.aluLatency, params.aluII,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries)),
          VectorFuPipeline(VecFuType::Multiply,
              fuParams(params.multiplyLatency, params.multiplyII,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries)),
          VectorFuPipeline(VecFuType::Fpu,
              fuParams(params.fpuLatency, params.fpuII,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries)),
          VectorFuPipeline(VecFuType::Divide,
              fuParams(params.divideLatency, params.divideII,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries)),
          VectorFuPipeline(VecFuType::Memory,
              fuParams(params.memoryLatency, params.memoryII,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries)),
          VectorFuPipeline(VecFuType::Config,
              fuParams(params.configLatency, 1,
                       params.operandQueueDepth,
                       params.laneScoreboardEntries))}},
      resultQueue(params.resultQueueDepth)
{
}

void
VectorLane::completeWord(const LaneTag &tag, bool last, TargetVfuMask target,
                         RvvChannel<LaneComplete> &completions)
{
    // Memory transactions may return out of order. The final word identifies
    // the expected count; its return alone does not complete this lane.
    auto &progress = wordCompletions[tag.instId];
    ++progress.completed;
    if (last)
        progress.expected = tag.wordIndex + 1;
    panic_if(progress.expected && progress.completed > progress.expected,
             "RVV lane completed too many words inst=%llu", tag.instId);
    if (progress.expected && progress.completed == progress.expected) {
        completions.pushNext(LaneComplete{tag, true, target});
        wordCompletions.erase(tag.instId);
    }
}

void
VectorLane::evaluateWriteback(
    uint64_t cycle, RvvChannel<LaneComplete> &completions)
{
    auto &acks = vrf.writeAckChannel();
    for (size_t index = 0; index < acks.size(); ++index) {
        const auto &ack = acks.at(index);
        if (!completions.canPushNext())
            break;
        panic_if(!scoreboard.contains(ack.tag),
                 "RVV write ack has no lane scoreboard entry");
        const auto &entry = scoreboard.get(ack.tag);
        const auto target = targetForFu(entry.fuType);
        globalScoreboard.publishWriteReadyNext(ack.tag);
        globalScoreboard.publishWordWriteNext(ack.tag.rvvSlotId);
        resultProgress.publishWriteNext(ack.tag.rvvSlotId);
        vrf.updateEewTagNext(static_cast<RegIndex>(ack.reg.index()), ack.eew);
        completeWord(ack.tag, entry.last, target, completions);
        scoreboard.complete(ack.tag);
        acks.popNext();
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u WB COMPLETE bank=%u\n",
                cycle, ack.tag.instId, ack.tag.beatId,
                laneId, ack.bank);
    }

    if (resultQueue.empty())
        return;
    const auto &result = resultQueue.front();
    if (!scoreboard.contains(result.tag)) {
        resultQueue.popNext();
        return;
    }

    if (!result.destReg) {
        const bool last = scoreboard.get(result.tag).last;
        if (completions.canPushNext()) {
            completeWord(result.tag, last, targetForFu(result.fuType),
                         completions);
            scoreboard.complete(result.tag);
            resultQueue.popNext();
        }
        return;
    }

    LogicalVrfReq write;
    write.tag = result.tag;
    write.access = VrfAccess::Write;
    write.reg = *result.destReg;
    write.lmulLog2 = contexts.get(result.tag.instId).desc.config.lmulLog2;
    write.groupByteOffset = result.groupByteOffset;
    write.byteEnable = result.byteEnable;
    if (vrf.canAcceptWriteNext(write)) {
        vrf.acceptWriteNext(write);
        resultQueue.popNext();
    }
}

void
VectorLane::evaluateCollector(uint64_t cycle)
{
    auto &responses = vrf.readResponseChannel();
    for (size_t index = 0; index < responses.size(); ++index) {
        const auto &response = responses.at(index);
        if (scoreboard.contains(response.tag))
            scoreboard.markOperandReadyNext(response.tag);
        responses.popNext();
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u Collector READY "
                "operand=%u bank=%u\n",
                cycle, response.tag.instId, response.tag.beatId,
                laneId, response.operandIndex, response.bank);
    }

    const auto ready = scoreboard.firstReadyForFu();
    if (!ready)
        return;
    const auto &entry = scoreboard.get(*ready);
    RvvChannel<VectorFuReq> *external = nullptr;
    if (entry.fuType == VecFuType::Memory)
        external = &memoryRequests;
    else if (entry.fuType == VecFuType::Sldu)
        external = &slduOperandRequests;
    if (external && !external->canPushNext())
        return;
    if (!external && !fu(entry.fuType).canAcceptInputNext())
        return;

    VectorFuReq request;
    request.tag = entry.tag;
    request.fuType = entry.fuType;
    request.destReg = entry.destination;
    request.firstElem = entry.firstElem;
    request.elemCount = entry.elemCount;
    request.groupByteOffset = entry.groupByteOffset;
    request.byteEnable = entry.byteEnable;
    request.last = entry.last;
    if (entry.fuType == VecFuType::Memory) {
        const auto &packet = contexts.get(entry.tag.instId).desc.packet;
        request.memoryMode = classifyVlsuMode(packet);
        request.faultOnlyFirst = request.memoryMode ==
            VlsuAccessMode::FaultOnlyFirst;
        request.segmentFields = request.memoryMode == VlsuAccessMode::Segment ?
            2 : 1;
    }
    if (external)
        external->pushNext(std::move(request));
    else
        fu(entry.fuType).acceptInputNext(std::move(request));
    scoreboard.markFuIssuedNext(entry.tag);
}

void
VectorLane::evaluateOperandRequester(uint64_t cycle)
{
    if (operandTasks.empty())
        return;
    const LaneTag tag = operandTasks.front();
    if (!scoreboard.contains(tag)) {
        operandTasks.popNext();
        return;
    }

    const auto &entry = scoreboard.get(tag);
    if (entry.operandsRequested == entry.sources.size()) {
        operandTasks.popNext();
        return;
    }

    const RegId source = entry.sources[entry.operandsRequested];
    if (directFuBypass &&
        globalScoreboard.canBypass(tag.instId, source, tag)) {
        scoreboard.markOperandRequestedNext(tag);
        scoreboard.markOperandReadyNext(tag);
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u Operand BYPASS "
                "operand=%u\n",
                cycle, tag.instId, tag.beatId, laneId,
                entry.operandsRequested);
        return;
    }
    const auto producers = globalScoreboard.producersFor(tag.instId);
    if (!globalScoreboard.canReadWord(producers, tag.wordIndex)) {
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u Operand STALL "
                "hazard word=%u\n",
                cycle, tag.instId, tag.beatId, laneId, tag.wordIndex);
        return;
    }
    LogicalVrfReq request;
    request.tag = tag;
    request.operandIndex = entry.operandsRequested;
    request.access = VrfAccess::Read;
    request.reg = source;
    request.lmulLog2 = contexts.get(tag.instId).desc.config.lmulLog2;
    request.groupByteOffset = entry.groupByteOffset;
    request.byteEnable = entry.byteEnable;
    request.eew.value = static_cast<Eew>(
        contexts.get(tag.instId).desc.config.sewBits);
    request.highPriority = entry.fuType == VecFuType::Alu ||
        entry.fuType == VecFuType::Multiply || entry.fuType == VecFuType::Fpu;
    if (vrf.canAcceptReadNext(request)) {
        vrf.acceptReadNext(request);
        scoreboard.markOperandRequestedNext(tag);
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u Operand REQUEST "
                "operand=%u\n",
                cycle, tag.instId, tag.beatId, laneId,
                request.operandIndex);
    }
}

void
VectorLane::evaluateLaneSequencer(uint64_t cycle)
{
    if (!activeRequest && !requestInput.empty()) {
        activeRequest = requestInput.front();
        requestInput.popNext();
        nextElement = contexts.get(activeRequest->contextId).desc.config.vstart +
            laneId * std::max<unsigned>(1,
                contexts.get(activeRequest->contextId).desc.elementsPerLaneChunk);
        nextWord = 0;
        resultProgress.allocate(activeRequest->rvvSlotId);
    }
    if (activeRequest && input.canPushNext()) {
        const PeRequest &request = *activeRequest;
        const auto &context = contexts.get(request.contextId);
        const unsigned bytes_per_element = std::max(
            1U, static_cast<unsigned>(context.desc.config.sewBits) / 8U);
        /* The compatibility beat is a lane-local instruction envelope.  The
         * requester still performs all VRF accesses through the banked path. */
        const unsigned lane_span = std::max<unsigned>(1,
            context.desc.elementsPerLaneChunk);
        LaneBeat beat;
        beat.tag = {request.contextId, 0, laneId, request.rvvSlotId, nextWord};
        beat.firstElem = nextElement;
        if (beat.firstElem >= context.desc.config.vl)
            beat.elemCount = 0;
        else
            beat.elemCount = std::min<unsigned>(lane_span,
                context.desc.config.vl - beat.firstElem);
        const unsigned groupBytes = context.desc.config.lmulLog2 >= 0 ?
            (vlenBytes << context.desc.config.lmulLog2) :
            std::max(1U, vlenBytes >> -context.desc.config.lmulLog2);
        beat.groupByteOffset = (beat.firstElem * bytes_per_element) %
            groupBytes;
        const unsigned active_bytes = beat.elemCount * bytes_per_element;
        beat.byteEnable = active_bytes >= 64 ? ~0ULL :
            (active_bytes == 0 ? 0 : (1ULL << active_bytes) - 1);
        beat.elementEnable = beat.elemCount >= 64 ? ~0ULL :
            (beat.elemCount == 0 ? 0 : (1ULL << beat.elemCount) - 1);
        beat.first = nextWord == 0;
        beat.last = beat.elemCount == 0 ||
            beat.firstElem + beat.elemCount >= context.desc.config.vl ||
            context.desc.config.vl <= context.desc.config.vstart;
        const bool last = beat.last;
        input.pushNext(std::move(beat));
        nextElement += numLanes * lane_span;
        ++nextWord;
        if (last)
            activeRequest.reset();
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu slot=%u lane=%u LaneSeq REQUEST\n",
                cycle, request.contextId, request.rvvSlotId, laneId);
    }
    if (input.empty() || !scoreboard.canAllocateNext() ||
        !operandTasks.canPushNext()) {
        return;
    }
    const LaneBeat beat = input.front();
    const auto &context = contexts.get(beat.tag.instId);
    scoreboard.allocateNext(beat, context.desc);
    operandTasks.pushNext(beat.tag);
    input.popNext();
    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu beat=%u lane=%u LaneSeq ALLOC "
            "elements=%u\n",
            cycle, beat.tag.instId, beat.tag.beatId,
            laneId, beat.elemCount);
}

void
VectorLane::evaluate(
    uint64_t cycle, RvvChannel<LaneComplete> &completions)
{
    evaluateWriteback(cycle, completions);
    for (auto &pipeline : fus)
        pipeline.evaluate(cycle, laneId, resultQueue);
    evaluateCollector(cycle);
    evaluateOperandRequester(cycle);
    evaluateLaneSequencer(cycle);
    vrf.evaluate(cycle);
}

void
VectorLane::advance()
{
    requestInput.advance();
    input.advance();
    operandTasks.advance();
    scoreboard.advance();
    vrf.advance();
    for (auto &pipeline : fus)
        pipeline.advance();
    resultQueue.advance();
    resultProgress.advance();
}

void
VectorLane::cancel(RvvInstId id)
{
    wordCompletions.erase(id);
    input.eraseIf([id](const LaneBeat &beat) {
        return beat.tag.instId == id;
    });
    requestInput.eraseIf([id](const PeRequest &request) {
        return request.contextId == id;
    });
    if (activeRequest && activeRequest->contextId == id)
        activeRequest.reset();
    operandTasks.eraseIf([id](const LaneTag &tag) {
        return tag.instId == id;
    });
    scoreboard.cancel(id);
    if (contexts.contains(id))
        resultProgress.cancel(contexts.get(id).rvvSlotId);
    vrf.cancel(id);
    for (auto &pipeline : fus)
        pipeline.cancel(id);
    resultQueue.eraseIf([id](const VectorFuResp &response) {
        return response.tag.instId == id;
    });
}

bool
VectorLane::active() const
{
    if (requestInput.active() || input.active() || operandTasks.active() ||
        scoreboard.active() || resultProgress.active() ||
        vrf.active() || resultQueue.active()) {
        return true;
    }
    return std::any_of(fus.begin(), fus.end(),
        [](const VectorFuPipeline &pipeline) { return pipeline.active(); });
}

} // namespace rvv
} // namespace minor
} // namespace gem5
