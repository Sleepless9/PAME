/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/global_sequencer.hh"

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

AmeGlobalSequencer::AmeGlobalSequencer(
    AmeContextTable &contexts_, AmeScoreboard &scoreboard_,
    AmeChannel<AmeInstToken> &input_, AmeChannel<AmePeRequest> &output_,
    unsigned id_capacity)
    : contexts(contexts_), scoreboard(scoreboard_), input(input_),
      output(output_), ids(id_capacity)
{
}

void
AmeGlobalSequencer::evaluate(uint64_t cycle)
{
    if (!pending && !input.empty() && ids.canAllocate()) {
        /* The dispatch FIFO is the architectural command window, not a
         * single-head issue pipe.  Pick the oldest command whose MRF/ACC
         * hazards are clear.  This lets an independent bank/accumulator
         * command pass an older drain waiting for the active SA result while
         * AmeScoreboard still enforces RAW/WAR/WAW and serialized barriers. */
        std::optional<size_t> ready_index;
        for (size_t index = 0; index < input.size(); ++index) {
            if (scoreboard.canSequence(input.at(index).contextId)) {
                ready_index = index;
                break;
            }
        }
        if (ready_index) {
            const AmeContextId id = input.at(*ready_index).contextId;
            auto &context = contexts.get(id);
            context.desc.ameId = ids.allocate();
            context.slotAllocated = true;
            scoreboard.bindSlot(id, context.desc.ameId);
            pending = Broadcast{AmePeRequest{id, context.desc.ameId,
                context.desc.opcode, context.desc.targets,
                context.desc.config}, context.desc.targets, TargetNone, false};
            input.eraseAt(*ready_index);
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu slot=%u AME_ID_ALLOC queue_index=%u\n",
                    cycle, id, context.desc.ameId,
                    static_cast<unsigned>(*ready_index));
        }
    }
    if (!pending)
        return;
    if (!pending->sent && output.canPushNext()) {
        output.pushNext(pending->request);
        pending->sent = true;
        DPRINTF(AMEFlow,
                "cycle=%llu inst=%llu slot=%u AME_GLOBAL_BROADCAST targets=%#x\n",
                cycle, pending->request.contextId, pending->request.ameId,
                pending->required);
    }
    if (pending->accepted == pending->required)
        pending.reset();
}

void
AmeGlobalSequencer::targetAccepted(
    AmeContextId id, AmeTargetMask targets)
{
    if (!pending || pending->request.contextId != id)
        return;
    panic_if((targets & ~pending->required) != 0 ||
             (targets & pending->accepted) != 0,
             "Invalid AME target acceptance");
    pending->accepted |= targets;
    contexts.get(id).acceptedTargets |= targets;
}

void
AmeGlobalSequencer::cancel(AmeContextId id)
{
    input.eraseIf([id](const AmeInstToken &token) {
        return token.contextId == id;
    });
    output.eraseIf([id](const AmePeRequest &request) {
        return request.contextId == id;
    });
    if (pending && pending->request.contextId == id)
        pending.reset();
    if (contexts.contains(id) && contexts.get(id).slotAllocated) {
        ids.release(contexts.get(id).desc.ameId);
        contexts.get(id).slotAllocated = false;
    }
}

} // namespace ame
} // namespace minor
} // namespace gem5
