/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/global_scoreboard.hh"

#include <algorithm>

#include "arch/riscv/regs/vector.hh"
#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

bool
GlobalVectorScoreboard::overlaps(const std::vector<RegId> &left,
                                 const std::vector<RegId> &right)
{
    return std::any_of(left.begin(), left.end(), [&right](const RegId &reg) {
        return std::find(right.begin(), right.end(), reg) != right.end();
    });
}

std::vector<RegId>
GlobalVectorScoreboard::expand(const std::vector<RegId> &regs,
                               const RvvConfig &config)
{
    std::vector<RegId> expanded;
    const unsigned group_regs = config.lmulLog2 > 0 ?
        1U << config.lmulLog2 : 1U;
    for (const auto &reg : regs) {
        if (!isVectorRegister(reg))
            continue;
        for (unsigned offset = 0; offset < group_regs; ++offset) {
            const RegId member(reg.regClass(), reg.index() + offset);
            if (std::find(expanded.begin(), expanded.end(), member) ==
                expanded.end()) {
                expanded.push_back(member);
            }
        }
    }
    return expanded;
}

std::vector<GlobalVectorScoreboard::RegisterGroup>
GlobalVectorScoreboard::sourceGroups(
    const InstQueuePacket &packet, const RvvConfig &config)
{
    std::vector<RegisterGroup> groups;
    const auto sources = vectorSources(packet, config);
    const RegId mask(RiscvISA::vecRegClass, 0);
    const auto packet_mask_count = std::count(
        packet.srcRegs.begin(), packet.srcRegs.end(), mask);
    for (const auto &source : sources) {
        RvvConfig group_config = config;
        /* A lone v0 on a masked instruction is the one-register predicate.
         * Repeated v0 denotes a data/old-vd group in addition to the mask. */
        if (!config.vm && source == mask && packet_mask_count <= 1)
            group_config.lmulLog2 = 0;
        groups.push_back(RegisterGroup{
            source, expand({source}, group_config)});
    }
    return groups;
}

void
GlobalVectorScoreboard::allocate(const RvvInstContext &context)
{
    Reservation reservation;
    reservation.id = context.desc.id;
    reservation.rvvSlotId = context.rvvSlotId;
    reservation.threadId = context.desc.key.threadId;
    reservation.sourceGroups = sourceGroups(
        context.desc.packet, context.desc.config);
    reservation.destinationGroups = context.desc.packet.destRegs;
    for (const auto &group : reservation.sourceGroups) {
        for (const auto &member : group.members) {
            if (std::find(reservation.sources.begin(),
                          reservation.sources.end(), member) ==
                reservation.sources.end()) {
                reservation.sources.push_back(member);
            }
        }
    }
    reservation.destinations = expand(reservation.destinationGroups,
                                      context.desc.config);
    reservation.config = context.desc.config;
    reservation.elementsPerLaneChunk = context.desc.elementsPerLaneChunk;
    reservation.elementsPerLaneBeat = context.desc.elementsPerLaneBeat;
    const auto [it, inserted] =
        reservations.emplace(reservation.id, std::move(reservation));
    panic_if(!inserted, "Duplicate RVV scoreboard reservation id=%llu",
             context.desc.id);
}

void
GlobalVectorScoreboard::bindSlotId(RvvInstId id, RvvInsnSlotId slot_id)
{
    const auto it = reservations.find(id);
    panic_if(it == reservations.end(),
             "Binding RVV slot ID to unknown RVV instruction id=%llu", id);
    it->second.rvvSlotId = slot_id;
    it->second.slotIdBound = true;
}

bool
GlobalVectorScoreboard::compatibleMapping(
    const Reservation &producer, const Reservation &consumer)
{
    return producer.config.vl == consumer.config.vl &&
        producer.config.vstart == consumer.config.vstart &&
        producer.config.sewBits == consumer.config.sewBits &&
        producer.config.lmulLog2 == consumer.config.lmulLog2 &&
        producer.elementsPerLaneChunk == consumer.elementsPerLaneChunk &&
        producer.elementsPerLaneBeat == consumer.elementsPerLaneBeat;
}

bool
GlobalVectorScoreboard::exactRawMapping(
    const Reservation &producer, const Reservation &consumer)
{
    for (const auto &source : consumer.sourceGroups) {
        if (!overlaps(source.members, producer.destinations))
            continue;
        if (std::find(producer.destinationGroups.begin(),
                      producer.destinationGroups.end(), source.base) ==
            producer.destinationGroups.end()) {
            return false;
        }
    }
    return true;
}

bool
GlobalVectorScoreboard::canSequence(RvvInstId id) const
{
    const auto current = reservations.find(id);
    panic_if(current == reservations.end(),
             "Missing RVV scoreboard reservation id=%llu", id);

    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.threadId != current->second.threadId)
            continue;
        const bool raw = overlaps(current->second.sources,
                                  older->second.destinations);
        const bool waw = overlaps(current->second.destinations,
                                  older->second.destinations);
        const bool war = overlaps(current->second.destinations,
                                  older->second.sources);
        if (waw || war || (raw &&
                (!compatibleMapping(older->second, current->second) ||
                 !exactRawMapping(older->second, current->second)))) {
            return false;
        }
    }
    return true;
}

bool
GlobalVectorScoreboard::readyForRaw(
    const Reservation &producer, BeatId beat, uint64_t active_lanes) const
{
    for (LaneId lane = 0; lane < 64; ++lane) {
        if ((active_lanes & (1ULL << lane)) == 0)
            continue;
        const ReadyKey key{producer.id, beat, lane};
        if (resultReady.find(key) == resultReady.end() &&
            writeReady.find(key) == writeReady.end()) {
            return false;
        }
    }
    return true;
}

bool
GlobalVectorScoreboard::canSequenceBeat(
    RvvInstId id, BeatId beat, uint64_t active_lanes) const
{
    const auto current = reservations.find(id);
    panic_if(current == reservations.end(),
             "Missing RVV scoreboard reservation id=%llu", id);
    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.threadId != current->second.threadId ||
            !overlaps(current->second.sources,
                      older->second.destinations)) {
            continue;
        }
        if (!compatibleMapping(older->second, current->second) ||
            !exactRawMapping(older->second, current->second) ||
            !readyForRaw(older->second, beat, active_lanes)) {
            return false;
        }
    }
    return true;
}

bool
GlobalVectorScoreboard::canBypass(
    RvvInstId consumer, const RegId &source, const LaneTag &tag) const
{
    const auto current = reservations.find(consumer);
    panic_if(current == reservations.end(),
             "Missing RVV scoreboard consumer id=%llu", consumer);
    for (auto older = std::make_reverse_iterator(current);
         older != reservations.rend(); ++older) {
        if (older->second.threadId != current->second.threadId ||
            std::find(older->second.destinationGroups.begin(),
                      older->second.destinationGroups.end(), source) ==
                older->second.destinationGroups.end()) {
            continue;
        }
        if (!compatibleMapping(older->second, current->second))
            return false;
        return resultReady.find(
            ReadyKey{older->second.id, tag.beatId, tag.laneId}) !=
            resultReady.end();
    }
    return false;
}

void
GlobalVectorScoreboard::publishResultReadyNext(const LaneTag &tag)
{
    if (reservations.find(tag.instId) != reservations.end())
        pendingResultReady.insert({tag.instId, tag.beatId, tag.laneId});
}

void
GlobalVectorScoreboard::publishWriteReadyNext(const LaneTag &tag)
{
    if (reservations.find(tag.instId) != reservations.end())
        pendingWriteReady.insert({tag.instId, tag.beatId, tag.laneId});
}

std::vector<RvvInstId>
GlobalVectorScoreboard::producersFor(RvvInstId consumer) const
{
    const auto current = reservations.find(consumer);
    panic_if(current == reservations.end(),
             "Missing RVV scoreboard consumer id=%llu", consumer);
    std::vector<RvvInstId> producers;
    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.threadId == current->second.threadId &&
            overlaps(current->second.sources, older->second.destinations)) {
            producers.push_back(older->first);
        }
    }
    return producers;
}

void
GlobalVectorScoreboard::publishWordWriteNext(RvvInsnSlotId producer, uint64_t words)
{
    panic_if(words == 0, "RVV word progress increment must be non-zero");
    pendingWordProgress[producer] += words;
}

uint64_t
GlobalVectorScoreboard::wordsWritten(RvvInsnSlotId producer) const
{
    const auto it = wordProgress.find(producer);
    return it == wordProgress.end() ? 0 : it->second;
}

bool
GlobalVectorScoreboard::canReadWord(
    const std::vector<RvvInstId> &producers, uint32_t word_index) const
{
    for (const auto producer_id : producers) {
        const auto producer = reservations.find(producer_id);
        if (producer == reservations.end() ||
            wordsWritten(producer->second.rvvSlotId) <= word_index)
            return false;
    }
    return true;
}

bool
GlobalVectorScoreboard::canReadWord(
    const std::vector<RvvInsnSlotId> &producers, uint32_t word_index) const
{
    for (const auto producer : producers) {
        if (wordsWritten(producer) <= word_index)
            return false;
    }
    return true;
}

void
GlobalVectorScoreboard::release(RvvInstId id)
{
    const auto reservation = reservations.find(id);
    panic_if(reservation == reservations.end(),
             "Releasing unknown RVV scoreboard reservation id=%llu", id);
    const RvvInsnSlotId slot_id = reservation->second.rvvSlotId;
    const bool slot_bound = reservation->second.slotIdBound;
    panic_if(reservations.erase(id) != 1,
             "Releasing unknown RVV scoreboard reservation id=%llu", id);
    const auto erase_id = [id](auto &set) {
        for (auto it = set.begin(); it != set.end(); ) {
            if (it->producer == id)
                it = set.erase(it);
            else
                ++it;
        }
    };
    erase_id(resultReady);
    erase_id(writeReady);
    erase_id(pendingResultReady);
    erase_id(pendingWriteReady);
    if (slot_bound) {
        wordProgress.erase(slot_id);
        pendingWordProgress.erase(slot_id);
    }
}

void
GlobalVectorScoreboard::releaseNext(RvvInstId id)
{
    panic_if(reservations.find(id) == reservations.end(),
             "Scheduling unknown RVV scoreboard release id=%llu", id);
    pendingReleases.insert(id);
}

void
GlobalVectorScoreboard::advance()
{
    resultReady.insert(pendingResultReady.begin(), pendingResultReady.end());
    writeReady.insert(pendingWriteReady.begin(), pendingWriteReady.end());
    pendingResultReady.clear();
    pendingWriteReady.clear();
    for (const auto &[producer, words] : pendingWordProgress)
        wordProgress[producer] += words;
    pendingWordProgress.clear();
    for (const auto id : pendingReleases)
        release(id);
    pendingReleases.clear();
}

void
GlobalVectorScoreboard::cancel(RvvInstId id)
{
    pendingReleases.erase(id);
    if (reservations.find(id) != reservations.end())
        release(id);
}

} // namespace rvv
} // namespace minor
} // namespace gem5
