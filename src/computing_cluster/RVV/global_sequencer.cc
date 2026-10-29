/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/global_sequencer.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

GlobalSequencer::GlobalSequencer(
    InstContextTable &contexts_, GlobalVectorScoreboard &scoreboard_,
    RvvChannel<VectorInstToken> &input_, RvvChannel<PeRequest> &output_,
    unsigned num_lanes, unsigned slot_id_capacity)
    : contexts(contexts_), scoreboard(scoreboard_), input(input_),
      output(output_), rvvSlotIds(slot_id_capacity), numLanes(num_lanes)
{
    fatal_if(numLanes == 0 || numLanes > 64,
             "RVV sequencer supports 1 to 64 lanes");
}

PeRequest
GlobalSequencer::makeRequest(const RvvInstContext &context) const
{
    PeRequest request;
    request.contextId = context.desc.id;
    request.rvvSlotId = context.rvvSlotId;
    request.opcodeClass = context.desc.fuType;
    request.targetVfus = context.targetVfus;
    request.vl = context.desc.config.vl;
    request.vstart = context.desc.config.vstart;
    request.config = context.desc.config;
    request.producerHazards = context.producerHazards;
    request.vm = context.desc.config.vm;
    request.eew.value = static_cast<Eew>(context.desc.config.sewBits);
    request.eew.widening = context.desc.packet.mnemonic.find("w") !=
        std::string::npos;
    request.eew.narrowing = context.desc.packet.mnemonic.find("n") !=
        std::string::npos;
    const auto destination = vectorDestination(context.desc.packet);
    if (destination)
        request.vd = *destination;
    const auto sources = vectorSources(context.desc.packet,
                                       context.desc.config);
    if (!sources.empty())
        request.vs1 = sources.front();
    if (sources.size() > 1)
        request.vs2 = sources[1];
    request.useVdAsOperand = std::find(sources.begin(), sources.end(),
                                       request.vd) != sources.end();
    if (!context.desc.packet.scalarSrcs.empty()) {
        request.scalarOperand = context.desc.packet.scalarSrcs.front().value;
        request.useScalar = true;
    }
    return request;
}

void
GlobalSequencer::evaluate(uint64_t cycle)
{
    if (!pending && !input.empty() && rvvSlotIds.canAllocate()) {
        const RvvInstId id = input.front().instId;
        const auto &candidate = contexts.get(id);
        const bool serialized_ready = candidate.desc.serialized ?
            !contexts.hasOlder(id) : !contexts.hasOlderSerialized(id);
        if (serialized_ready && scoreboard.canSequence(id)) {
            auto &context = contexts.get(id);
            context.rvvSlotId = rvvSlotIds.allocate();
            context.slotIdAllocated = true;
            context.desc.rvvSlotId = context.rvvSlotId;
            scoreboard.bindSlotId(id, context.rvvSlotId);
            context.producerHazards.clear();
            for (const auto producer : scoreboard.producersFor(id)) {
                if (contexts.contains(producer))
                    context.producerHazards.push_back(
                        contexts.get(producer).rvvSlotId);
            }
            pending = PendingBroadcast{makeRequest(context),
                                       context.targetVfus, 0, false};
            input.popNext();
            context.state.status = ContextStatus::Sequencing;
            context.state.firstBeatCycle = cycle;
            DPRINTF(RVVEngine,
                    "cycle=%llu inst=%llu slot=%u GlobalSeq ALLOC\n",
                    cycle, id, context.rvvSlotId);
        }
    }

    if (!pending)
        return;

    auto &context = contexts.get(pending->request.contextId);
    if (!pending->sent && output.canPushNext()) {
        output.pushNext(pending->request);
        pending->sent = true;
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu slot=%u PE_BROADCAST_START target=%#x\n",
                cycle, context.desc.id, context.rvvSlotId, pending->required);
    }

    if (pending->accepted == pending->required) {
        context.peRequestAccepted = true;
        context.state.status = ContextStatus::Executing;
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu slot=%u PE_BROADCAST_COMPLETE\n",
                cycle, context.desc.id, context.rvvSlotId);
        pending.reset();
    }
}

void
GlobalSequencer::targetAccepted(RvvInstId id, TargetVfuMask target)
{
    if (!pending || pending->request.contextId != id)
        return;
    panic_if((target & pending->required) == 0 ||
             (target & pending->accepted) != 0,
             "RVV target acceptance does not match pending request");
    pending->accepted |= target;
    auto &context = contexts.get(id);
    context.targetAccepted |= target;
}

void
GlobalSequencer::cancel(RvvInstId id)
{
    input.eraseIf([id](const VectorInstToken &token) {
        return token.instId == id;
    });
    output.eraseIf([id](const PeRequest &request) {
        return request.contextId == id;
    });
    if (pending && pending->request.contextId == id) {
        const RvvInsnSlotId slot_id = pending->request.rvvSlotId;
        pending.reset();
        if (contexts.contains(id))
            contexts.get(id).slotIdAllocated = false;
        if (rvvSlotIds.running(slot_id))
            rvvSlotIds.release(slot_id);
        return;
    }
    if (contexts.contains(id)) {
        const RvvInsnSlotId slot_id = contexts.get(id).rvvSlotId;
        if (contexts.get(id).slotIdAllocated && rvvSlotIds.running(slot_id)) {
            rvvSlotIds.release(slot_id);
            contexts.get(id).slotIdAllocated = false;
        }
    }
}

} // namespace rvv
} // namespace minor
} // namespace gem5
