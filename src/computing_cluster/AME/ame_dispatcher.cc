/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/ame_dispatcher.hh"

#include <algorithm>

#include "arch/riscv/regs/matrix.hh"
#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

AmeDispatcher::AmeDispatcher(
    MinorInstQueue &inst_queue, AmeContextTable &contexts_,
    AmeScoreboard &scoreboard_, AmeChannel<AmeInstToken> &output_,
    ConfigSnapshot config_snapshot, DmaEventReady dma_event_ready,
    DmaEventConsume dma_event_consume)
    : instQueue(inst_queue), contexts(contexts_), scoreboard(scoreboard_),
      output(output_), configSnapshot(std::move(config_snapshot)),
      dmaEventReady(std::move(dma_event_ready)),
      dmaEventConsume(std::move(dma_event_consume))
{
}

void
AmeDispatcher::evaluate(uint64_t cycle)
{
    for (auto it = retiringSerialBarriers.begin();
         it != retiringSerialBarriers.end();) {
        if (!instQueue.hasPacket(*it))
            it = retiringSerialBarriers.erase(it);
        else
            ++it;
    }

    const bool active_serial = std::any_of(
        keyToId.begin(), keyToId.end(), [this](const auto &entry) {
            return contexts.contains(entry.second) &&
                contexts.get(entry.second).desc.serialized;
        });
    if (active_serial || !retiringSerialBarriers.empty()) {
        DPRINTF(AMEFlow,
                "cycle=%llu AME_DISPATCH_GATE active_serial=%u "
                "retiring_barriers=%zu\n",
                cycle, active_serial, retiringSerialBarriers.size());
        return;
    }

    if (!instQueue.hasPending(InstQueueEngine::AME) ||
        !contexts.canAllocate() || !output.canPushNext()) {
        return;
    }
    const auto &front = instQueue.front(InstQueueEngine::AME);
    uint64_t wait_events = 0;
    if (front.mnemonic.find("event_wait") != std::string::npos) {
        /* RD is immediate storage for the four-entry DMA event table.  The
         * instruction has no architectural integer destination. */
        const unsigned event_id = (front.encodedInst >> 7) & 0x1f;
        wait_events = 1ULL << event_id;
    } else if (front.mnemonic.find("acquire") != std::string::npos &&
               !front.scalarSrcs.empty()) {
        wait_events = front.scalarSrcs.back().value >> 8;
    }
    if (wait_events && dmaEventReady && !dmaEventReady(wait_events)) {
        DPRINTF(AMEFlow,
                "cycle=%llu AME_DMA_EVENT_WAIT exec=%llu descriptor=%#llx "
                "events=%#llx\n",
                cycle, front.key.execSeqNum,
                front.encodedInst, wait_events);
        return;
    }
    InstQueuePacket packet = instQueue.pop(InstQueueEngine::AME);
    if (wait_events && dmaEventConsume)
        dmaEventConsume(wait_events);
    AmeInstContext context;
    context.desc.id = nextId++;
    context.desc.key = packet.key;
    context.desc.packet = std::move(packet);
    context.desc.opcode = classifyOpcode(context.desc.packet);
    context.desc.targets = targetMask(context.desc.opcode);
    /* Datatype metadata is versioned in command packets.  Configuration
     * therefore follows queue order without globally draining independent
     * engines.  Release remains the ownership boundary. */
    const bool scalar_result = std::any_of(
        context.desc.packet.destRegs.begin(),
        context.desc.packet.destRegs.end(),
        [](const RegId &reg) {
            return reg.is(IntRegClass) || reg.is(FloatRegClass) ||
                reg.is(CCRegClass) || reg.is(MiscRegClass);
        });
    context.desc.serialized =
        context.desc.opcode == AmeOpcode::Release || scalar_result;
    if (configSnapshot)
        configSnapshot(context.desc.config, context.desc.packet);
    const auto &mnemonic = context.desc.packet.mnemonic;

    /* O3 deliberately does not rename AME datatype pseudo-registers.  Keep
     * their speculative view here, where packets arrive in architectural
     * command order.  This lets msettyp/asettyp and their consumers enqueue
     * back-to-back while preserving the exact type snapshot carried by each
     * AME context. */
    auto &type_shadow = typeShadows[context.desc.key.threadId];
    if ((mnemonic.find("msettyp") != std::string::npos ||
         mnemonic.find("asettyp") != std::string::npos) &&
        !context.desc.packet.scalarSrcs.empty()) {
        const uint16_t bits =
            context.desc.packet.scalarSrcs.back().value & 0xff;
        if (bits != 0) {
            for (const auto &reg : context.desc.packet.destRegs) {
                if (!reg.is(MatRegClass))
                    continue;
                if (reg.index() >= RiscvISA::ZttMatRegBase &&
                    reg.index() < RiscvISA::ZttMatRegBase +
                                      RiscvISA::NumZttMatRegs) {
                    type_shadow.m[reg.index() - RiscvISA::ZttMatRegBase] =
                        bits;
                } else if (reg.index() >= RiscvISA::ZttAccRegBase &&
                           reg.index() < RiscvISA::ZttAccRegBase +
                                             RiscvISA::NumZttAccRegs) {
                    type_shadow.acc[reg.index() - RiscvISA::ZttAccRegBase] =
                        bits;
                }
            }
        }
    }
    for (unsigned index = 0; index < type_shadow.m.size(); ++index) {
        if (type_shadow.m[index] != 0)
            context.desc.config.mRegBits[index] = type_shadow.m[index];
    }
    for (unsigned index = 0; index < type_shadow.acc.size(); ++index) {
        if (type_shadow.acc[index] != 0)
            context.desc.config.accRegBits[index] = type_shadow.acc[index];
    }
    context.desc.config.transposeA =
        mnemonic.find("mmulat") != std::string::npos;
    context.desc.config.transposeB =
        mnemonic.find("mmulbt") != std::string::npos;
    unsigned max_squares = 1;
    for (const auto &reg : context.desc.packet.srcRegs) {
        if (reg.is(MatRegClass) &&
            reg.index() >= RiscvISA::ZttMatRegBase &&
            reg.index() < RiscvISA::ZttMatRegBase +
                              RiscvISA::NumZttMatRegs) {
            const unsigned index = reg.index() - RiscvISA::ZttMatRegBase;
            const unsigned bits = mElementBits(context.desc.config, index);
            max_squares = std::max(max_squares,
                static_cast<unsigned>(context.desc.config.unitBits) /
                    std::max(1U, bits));
        }
    }
    for (const auto &reg : context.desc.packet.destRegs) {
        if (reg.is(MatRegClass) &&
            reg.index() >= RiscvISA::ZttMatRegBase &&
            reg.index() < RiscvISA::ZttMatRegBase +
                              RiscvISA::NumZttMatRegs) {
            const unsigned index = reg.index() - RiscvISA::ZttMatRegBase;
            const unsigned bits = mElementBits(context.desc.config, index);
            max_squares = std::max(max_squares,
                static_cast<unsigned>(context.desc.config.unitBits) /
                    std::max(1U, bits));
        }
    }
    context.desc.config.nSquares = max_squares;
    const bool k128_group =
        mnemonic.find("mmulacc_k128") != std::string::npos;
    if (k128_group)
        context.desc.config.nSquares *= 8;
    context.desc.config.kLogical = context.desc.config.nSquares *
        context.desc.config.nAme;
    populateRegisterRegions(context.desc);
    if (k128_group) {
        panic_if(context.desc.mSources.size() != 16,
                 "K128 AME command requires exactly 8 A and 8 B M registers");
        context.desc.mSourceGroups.clear();
        context.desc.mSourceGroups.emplace_back(
            context.desc.mSources.begin(), context.desc.mSources.begin() + 8);
        context.desc.mSourceGroups.emplace_back(
            context.desc.mSources.begin() + 8, context.desc.mSources.end());
    }
    context.acceptCycle = cycle;
    const AmeContextId id = context.desc.id;
    const InstQueueKey key = context.desc.key;
    contexts.allocate(std::move(context));
    scoreboard.allocate(contexts.get(id));
    const auto [it, inserted] = keyToId.emplace(key, id);
    panic_if(!inserted, "Duplicate AME queue key at dispatch");
    output.pushNext(AmeInstToken{id});
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu AME_DISPATCH_ACCEPT exec=%llu opcode=%s "
            "n=%u md=%u ad=%u squares=%u k=%u targets=%#x\n",
            cycle, id, key.execSeqNum,
            opcodeName(contexts.get(id).desc.opcode),
            contexts.get(id).desc.config.nAme,
            contexts.get(id).desc.config.mdBits,
            contexts.get(id).desc.config.adBits,
            contexts.get(id).desc.config.nSquares,
            contexts.get(id).desc.config.kLogical,
            contexts.get(id).desc.targets);
}

std::optional<AmeContextId>
AmeDispatcher::idForKey(const InstQueueKey &key) const
{
    const auto it = keyToId.find(key);
    return it == keyToId.end() ? std::nullopt :
                                std::optional<AmeContextId>(it->second);
}

void
AmeDispatcher::cancel(const InstQueueKey &key)
{
    retiringSerialBarriers.erase(key);
    const auto it = keyToId.find(key);
    if (it == keyToId.end())
        return;
    const AmeContextId id = it->second;
    output.eraseIf([id](const AmeInstToken &token) {
        return token.contextId == id;
    });
    scoreboard.cancel(id);
    contexts.release(id);
    keyToId.erase(it);
}

void
AmeDispatcher::release(AmeContextId id)
{
    const auto it = std::find_if(keyToId.begin(), keyToId.end(),
        [id](const auto &entry) { return entry.second == id; });
    panic_if(it == keyToId.end(), "Missing AME dispatcher context id=%llu", id);
    if (contexts.get(id).desc.serialized)
        retiringSerialBarriers.insert(it->first);
    keyToId.erase(it);
}

} // namespace ame
} // namespace minor
} // namespace gem5
