/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/rvv_dispatcher.hh"

#include <algorithm>
#include <utility>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

RvvDispatcher::RvvDispatcher(
    MinorInstQueue &inst_queue, InstContextTable &contexts_,
    GlobalVectorScoreboard &scoreboard_,
    RvvChannel<VectorInstToken> &dispatch_queue,
    const DispatcherParams &params_, ConfigSnapshot config_snapshot)
    : instQueue(inst_queue), contexts(contexts_), scoreboard(scoreboard_),
      dispatchQueue(dispatch_queue), params(params_),
      configSnapshot(std::move(config_snapshot))
{
    fatal_if(params.numLanes == 0 || params.laneWidthBits == 0 ||
             params.laneWidthBits % 8 != 0,
             "Invalid RVV dispatcher topology");
}

void
RvvDispatcher::evaluate(uint64_t cycle)
{
    if (!instQueue.hasPending(InstQueueEngine::RVV) ||
        !contexts.canAllocate() || !dispatchQueue.canPushNext()) {
        return;
    }

    InstQueuePacket packet = instQueue.pop(InstQueueEngine::RVV);
    RvvInstContext context;
    context.desc.id = nextId++;
    context.desc.key = packet.key;
    context.desc.packet = std::move(packet);
    context.desc.config = decodeConfig(context.desc.packet);
    if (configSnapshot)
        configSnapshot(context.desc.config, context.desc.packet);
    context.desc.fuType = classifyFu(context.desc.packet);
    context.desc.serialized = context.desc.fuType == VecFuType::Config;
    context.desc.targetVfus = TargetLane;
    if (context.desc.fuType == VecFuType::Memory)
        context.desc.targetVfus |= TargetVlsu;
    if (context.desc.fuType == VecFuType::Sldu)
        context.desc.targetVfus |= TargetSldu;
    if (!context.desc.config.vm)
        context.desc.targetVfus |= TargetMasku;
    context.targetVfus = context.desc.targetVfus;

    const unsigned elements_per_chunk = std::max(
        1U, params.laneWidthBits /
                static_cast<unsigned>(context.desc.config.sewBits));
    const unsigned packed = std::max(1U, std::min(
        params.maxPackedElementsPerLane,
        elements_per_chunk));
    context.desc.elementsPerLaneChunk = elements_per_chunk;
    context.desc.elementsPerLaneBeat = packed;
    /* Global scheduling is instruction-level.  Any word/element iteration is
     * owned by the lane sequencers and must not gate global completion. */
    context.desc.totalBeats = 1;
    context.state.acceptCycle = cycle;

    const RvvInstId id = context.desc.id;
    const InstQueueKey key = context.desc.key;
    contexts.allocate(std::move(context));
    scoreboard.allocate(contexts.get(id));
    const auto [it, inserted] = keyToId.emplace(key, id);
    panic_if(!inserted, "Duplicate RVV queue key at dispatch");
    dispatchQueue.pushNext(VectorInstToken{id});

    const auto &allocated = contexts.get(id);
    if (allocated.desc.config.fractionalLmul) {
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=- lane=- Dispatcher "
                "FRACTIONAL_LMUL mapped_slice lmul=%d\n",
                cycle, id, allocated.desc.config.lmulLog2);
    }
    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu beat=- lane=- Dispatcher ACCEPT "
            "exec=%llu vl=%u sew=%u lmul=%d beats=%u\n",
            cycle, id, key.execSeqNum, allocated.desc.config.vl,
            allocated.desc.config.sewBits,
            allocated.desc.config.lmulLog2, allocated.desc.totalBeats);
}

std::optional<RvvInstId>
RvvDispatcher::idForKey(const InstQueueKey &key) const
{
    const auto it = keyToId.find(key);
    if (it == keyToId.end())
        return std::nullopt;
    return it->second;
}

void
RvvDispatcher::cancel(const InstQueueKey &key)
{
    const auto it = keyToId.find(key);
    if (it == keyToId.end())
        return;
    const RvvInstId id = it->second;
    dispatchQueue.eraseIf(
        [id](const VectorInstToken &token) { return token.instId == id; });
    scoreboard.cancel(id);
    contexts.cancel(id);
    keyToId.erase(it);
}

void
RvvDispatcher::release(RvvInstId id)
{
    const auto it = std::find_if(keyToId.begin(), keyToId.end(),
        [id](const auto &entry) { return entry.second == id; });
    panic_if(it == keyToId.end(), "Missing RVV dispatcher context id=%llu", id);
    keyToId.erase(it);
}

} // namespace rvv
} // namespace minor
} // namespace gem5
