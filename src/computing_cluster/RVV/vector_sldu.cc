/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vector_sldu.hh"

#include <algorithm>
#include <optional>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

unsigned
VectorSldu::countBits(uint64_t value)
{
    unsigned count = 0;
    while (value != 0) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

VectorSldu::VectorSldu(
    InstContextTable &contexts_, RvvChannel<VectorFuReq> &lane_requests,
    RvvChannel<VectorFuResp> &lane_results, const SlduParams &params_,
    unsigned instruction_capacity)
    : contexts(contexts_), laneRequests(lane_requests),
      laneResults(lane_results), params(params_),
      instructionCapacity(instruction_capacity), requests(instruction_capacity),
      completions(instruction_capacity)
{
    fatal_if(instructionCapacity == 0 || params.numLanes == 0 ||
             params.numLanes > 64 || params.maxInflightWords == 0 ||
             params.slideLatency == 0 ||
             params.intraLaneReductionLatency == 0 ||
             params.reductionStageLatency == 0 || params.resultWidth == 0,
             "Invalid RVV SLDU parameters");
}

uint32_t
VectorSldu::laneWordCount(const RvvInstDesc &desc, LaneId lane) const
{
    const uint64_t lane_span = std::max<uint64_t>(
        1, desc.elementsPerLaneChunk);
    uint64_t first = static_cast<uint64_t>(desc.config.vstart) +
        static_cast<uint64_t>(lane) * lane_span;
    uint32_t words = 1;

    while (first < desc.config.vl) {
        const uint64_t elements = std::min<uint64_t>(
            lane_span, static_cast<uint64_t>(desc.config.vl) - first);
        if (first + elements >= desc.config.vl)
            break;
        first += static_cast<uint64_t>(params.numLanes) * lane_span;
        ++words;
    }
    return words;
}

uint64_t
VectorSldu::expectedLaneMask(
    const RvvInstDesc &desc, uint32_t word_index) const
{
    uint64_t mask = 0;
    for (LaneId lane = 0; lane < params.numLanes; ++lane) {
        if (word_index < laneWordCount(desc, lane))
            mask |= 1ULL << lane;
    }
    return mask;
}

unsigned
VectorSldu::interLaneLatency(uint64_t lane_mask) const
{
    unsigned stages = 0;
    unsigned lanes = std::max(1U, countBits(lane_mask));
    while (lanes > 1) {
        lanes = (lanes + 1) / 2;
        ++stages;
    }
    return std::max(1U, stages) * params.reductionStageLatency;
}

void
VectorSldu::consumeInstructions()
{
    const size_t available = requests.size();
    for (size_t index = 0;
         index < available && operations.size() < instructionCapacity;
         ++index) {
        const PeRequest request = requests.at(index);
        panic_if(!contexts.contains(request.contextId),
                 "SLDU request has no instruction context");
        const auto &desc = contexts.get(request.contextId).desc;

        Operation operation;
        operation.request = request;
        operation.reduction = isReduction(desc.packet);
        operation.totalWords = 1;
        for (LaneId lane = 0; lane < params.numLanes; ++lane) {
            const uint32_t words = laneWordCount(desc, lane);
            operation.totalWords = std::max(operation.totalWords, words);
        }
        const auto [it, inserted] = operations.emplace(
            request.contextId, std::move(operation));
        panic_if(!inserted, "Duplicate RVV SLDU instruction request");
        requests.popNext();
    }
}

void
VectorSldu::consumeLaneRequests()
{
    const size_t available = laneRequests.size();
    for (size_t index = 0; index < available; ++index) {
        const VectorFuReq request = laneRequests.at(index);
        panic_if(request.fuType != VecFuType::Sldu,
                 "Non-SLDU request routed to the SLDU");
        const auto operation = operations.find(request.tag.instId);
        if (operation == operations.end())
            break;
        const auto &desc = contexts.get(request.tag.instId).desc;
        const uint64_t expected = expectedLaneMask(desc, request.tag.wordIndex);
        panic_if(request.tag.laneId >= 64 ||
                 (expected & (1ULL << request.tag.laneId)) == 0,
                 "Unexpected lane word routed to the SLDU");

        if (operation->second.reduction) {
            auto &queue = operation->second.reductionQueues[
                request.tag.laneId];
            panic_if(std::any_of(queue.begin(), queue.end(),
                         [&request](const VectorFuReq &queued) {
                             return queued.tag == request.tag;
                         }),
                     "Duplicate SLDU reduction lane word");
            queue.push_back(request);
        } else {
            const WordKey key{request.tag.instId, request.tag.wordIndex};
            auto [batch, created] = batches.emplace(
                key, WordBatch{expected, {}});
            panic_if(!created && batch->second.expectedLaneMask != expected,
                     "SLDU word participation mask changed");
            const auto [it, inserted] = batch->second.requests.emplace(
                request.tag.laneId, request);
            panic_if(!inserted, "Duplicate SLDU lane request for a word");
        }
        laneRequests.popNext();
    }
}

void
VectorSldu::emitResults(uint64_t cycle)
{
    unsigned returned = 0;
    while (returned < params.resultWidth && laneResults.canPushNext()) {
        auto ready = std::find_if(inflight.begin(), inflight.end(),
            [](const Inflight &entry) {
                return entry.phase == SlduPhase::Writeback;
            });
        if (ready == inflight.end())
            break;

        auto &response = ready->responses[ready->nextResponse++];
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu word=%u lane=%u SLDU RESULT\n",
                cycle, response.tag.instId, response.tag.wordIndex,
                response.tag.laneId);
        laneResults.pushNext(std::move(response));
        ++returned;

        if (ready->nextResponse != ready->responses.size())
            continue;
        auto operation = operations.find(ready->contextId);
        panic_if(operation == operations.end(),
                 "Completing an unknown SLDU operation");
        if (ready->finalReduction)
            operation->second.completedWords = operation->second.totalWords;
        else if (!ready->localReduction)
            ++operation->second.completedWords;
        inflight.erase(ready);
    }
}

void
VectorSldu::advancePhases(uint64_t cycle)
{
    for (auto entry = inflight.begin(); entry != inflight.end();) {
        if (entry->phase == SlduPhase::Writeback ||
            entry->readyCycle > cycle) {
            ++entry;
            continue;
        }
        if (entry->localReduction) {
            auto operation = operations.find(entry->contextId);
            panic_if(operation == operations.end(),
                     "Advancing an unknown SLDU reduction operation");
            const LaneId lane = entry->sourceRequest.tag.laneId;
            operation->second.laneReductionBusyMask &= ~(1ULL << lane);
            if (entry->sourceRequest.last) {
                const auto [it, inserted] =
                    operation->second.lanePartials.emplace(
                        lane, entry->sourceRequest);
                panic_if(!inserted,
                         "SLDU lane produced two final reduction partials");
                entry = inflight.erase(entry);
            } else {
                entry->phase = SlduPhase::Writeback;
                entry->readyCycle = cycle;
                ++entry;
            }
        } else if (entry->phase == SlduPhase::InterLaneExchange &&
                   entry->finalReduction) {
            entry->phase = SlduPhase::FinalLaneReduction;
            entry->readyCycle = cycle + params.reductionStageLatency;
            ++entry;
        } else {
            entry->phase = SlduPhase::Writeback;
            entry->readyCycle = cycle;
            ++entry;
        }
    }
}

void
VectorSldu::scheduleReady(uint64_t cycle)
{
    for (auto &[id, operation] : operations) {
        if (!operation.reduction)
            continue;
        for (LaneId lane = 0;
             lane < params.numLanes &&
             inflight.size() < params.maxInflightWords; ++lane) {
            if (operation.laneReductionBusyMask & (1ULL << lane))
                continue;
            auto queue = operation.reductionQueues.find(lane);
            if (queue == operation.reductionQueues.end() ||
                queue->second.empty()) {
                continue;
            }

            const VectorFuReq request = queue->second.front();
            queue->second.pop_front();
            Inflight scheduled;
            scheduled.contextId = id;
            scheduled.wordIndex = request.tag.wordIndex;
            scheduled.participatingLaneMask = 1ULL << lane;
            scheduled.phase = SlduPhase::IntraLaneReduction;
            scheduled.readyCycle =
                cycle + params.intraLaneReductionLatency;
            scheduled.localReduction = true;
            scheduled.sourceRequest = request;
            scheduled.responses.push_back(VectorFuResp{
                request.tag, request.fuType, std::nullopt,
                request.groupByteOffset, 0});
            operation.laneReductionBusyMask |= 1ULL << lane;
            DPRINTF(RVVEngine,
                    "cycle=%llu inst=%llu word=%u lane=%u "
                    "SLDU INTRA_REDUCTION_ISSUE\n",
                    cycle, id, request.tag.wordIndex, lane);
            inflight.push_back(std::move(scheduled));
        }
    }

    if (inflight.size() >= params.maxInflightWords)
        return;

    for (auto &[id, operation] : operations) {
        if (!operation.reduction || operation.reductionScheduled ||
            operation.lanePartials.size() != params.numLanes) {
            continue;
        }

        Inflight scheduled;
        scheduled.contextId = id;
        scheduled.finalReduction = true;
        scheduled.phase = SlduPhase::InterLaneExchange;
        for (const auto &[lane, request] : operation.lanePartials)
            scheduled.participatingLaneMask |= 1ULL << lane;
        scheduled.readyCycle = cycle +
            interLaneLatency(scheduled.participatingLaneMask);

        const auto architectural_result = operation.lanePartials.find(0);
        panic_if(architectural_result == operation.lanePartials.end(),
                 "SLDU reduction has no lane-zero result path");
        for (const auto &[lane, request] : operation.lanePartials) {
            if (lane == 0)
                continue;
            scheduled.responses.push_back(VectorFuResp{
                request.tag, request.fuType, std::nullopt,
                request.groupByteOffset, 0});
        }
        scheduled.responses.push_back(VectorFuResp{
            architectural_result->second.tag,
            architectural_result->second.fuType,
            architectural_result->second.destReg,
            architectural_result->second.groupByteOffset,
            architectural_result->second.byteEnable});
        operation.reductionScheduled = true;
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu word=- lane=- SLDU REDUCTION_ISSUE "
                "lanes=%#x words=%u\n",
                cycle, id, scheduled.participatingLaneMask,
                operation.totalWords);
        inflight.push_back(std::move(scheduled));
        return;
    }

    if (inflight.size() >= params.maxInflightWords)
        return;

    for (auto batch = batches.begin(); batch != batches.end(); ++batch) {
        if (batch->second.requests.size() !=
            countBits(batch->second.expectedLaneMask)) {
            continue;
        }

        Inflight scheduled;
        scheduled.contextId = batch->first.contextId;
        scheduled.wordIndex = batch->first.wordIndex;
        scheduled.participatingLaneMask = batch->second.expectedLaneMask;
        scheduled.phase = SlduPhase::InterLaneExchange;
        scheduled.readyCycle = cycle + params.slideLatency;
        for (const auto &[lane, request] : batch->second.requests) {
            scheduled.responses.push_back(VectorFuResp{
                request.tag, request.fuType, request.destReg,
                request.groupByteOffset, request.byteEnable});
        }
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu word=%u lane=- SLDU SLIDE_ISSUE "
                "lanes=%#x latency=%u\n",
                cycle, batch->first.contextId, batch->first.wordIndex,
                batch->second.expectedLaneMask, params.slideLatency);
        inflight.push_back(std::move(scheduled));
        batches.erase(batch);
        return;
    }
}

void
VectorSldu::completeInstructions()
{
    for (auto operation = operations.begin();
         operation != operations.end(); ++operation) {
        if (operation->second.completedWords != operation->second.totalWords ||
            !completions.canPushNext()) {
            continue;
        }
        completions.pushNext(TargetDone{
            operation->second.request.contextId,
            operation->second.request.rvvSlotId, TargetSldu});
        operations.erase(operation);
        return;
    }
}

void
VectorSldu::evaluate(uint64_t cycle)
{
    emitResults(cycle);
    completeInstructions();
    advancePhases(cycle);
    consumeInstructions();
    consumeLaneRequests();
    scheduleReady(cycle);
}

void
VectorSldu::advance()
{
    laneRequests.advance();
    laneResults.advance();
    requests.advance();
    completions.advance();
}

void
VectorSldu::cancel(RvvInstId id)
{
    requests.eraseIf([id](const PeRequest &request) {
        return request.contextId == id;
    });
    laneRequests.eraseIf([id](const VectorFuReq &request) {
        return request.tag.instId == id;
    });
    laneResults.eraseIf([id](const VectorFuResp &response) {
        return response.tag.instId == id;
    });
    completions.eraseIf([id](const TargetDone &done) {
        return done.contextId == id;
    });
    operations.erase(id);
    for (auto batch = batches.begin(); batch != batches.end();) {
        if (batch->first.contextId == id)
            batch = batches.erase(batch);
        else
            ++batch;
    }
    inflight.erase(
        std::remove_if(inflight.begin(), inflight.end(),
            [id](const Inflight &entry) {
                return entry.contextId == id;
            }),
        inflight.end());
}

bool
VectorSldu::activeState() const
{
    return requests.active() || laneRequests.active() ||
        laneResults.active() || completions.active() ||
        !operations.empty() || !batches.empty() || !inflight.empty();
}

} // namespace rvv
} // namespace minor
} // namespace gem5
