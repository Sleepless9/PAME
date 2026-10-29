/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/ame_scoreboard.hh"

#include <algorithm>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

bool
AmeScoreboard::overlaps(const MRegion &left, const MRegion &right)
{
    return left.tid == right.tid && left.reg == right.reg &&
        left.byteBegin < right.byteEnd && right.byteBegin < left.byteEnd;
}

bool
AmeScoreboard::overlaps(const AccRegion &left, const AccRegion &right)
{
    return left.tid == right.tid && left.reg == right.reg;
}

bool
AmeScoreboard::overlaps(const std::vector<MRegion> &left,
                        const std::vector<MRegion> &right)
{
    for (const auto &a : left)
        for (const auto &b : right)
            if (overlaps(a, b))
                return true;
    return false;
}

bool
AmeScoreboard::overlaps(const std::vector<AccRegion> &left,
                        const std::vector<AccRegion> &right)
{
    for (const auto &a : left)
        for (const auto &b : right)
            if (overlaps(a, b))
                return true;
    return false;
}

void
AmeScoreboard::allocate(const AmeInstContext &context)
{
    Reservation reservation;
    reservation.id = context.desc.id;
    reservation.ameId = context.desc.ameId;
    reservation.tid = context.desc.key.threadId;
    reservation.mSources = context.desc.mSources;
    reservation.mDestinations = context.desc.mDestinations;
    reservation.accSources = context.desc.accSources;
    reservation.accDestinations = context.desc.accDestinations;
    reservation.serialized = context.desc.serialized;
    const auto [it, inserted] =
        reservations.emplace(reservation.id, std::move(reservation));
    panic_if(!inserted, "Duplicate AME scoreboard context id=%llu", it->first);
}

void
AmeScoreboard::bindSlot(AmeContextId id, AmeInsnId ame_id)
{
    const auto it = reservations.find(id);
    panic_if(it == reservations.end(), "Binding unknown AME context id=%llu", id);
    it->second.ameId = ame_id;
}

bool
AmeScoreboard::canSequence(AmeContextId id) const
{
    const auto current = reservations.find(id);
    panic_if(current == reservations.end(), "Sequencing unknown AME context id=%llu", id);
    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.tid != current->second.tid)
            continue;
        if (older->second.serialized || current->second.serialized)
            return false;
        const bool m_waw = overlaps(current->second.mDestinations,
                                    older->second.mDestinations);
        const bool m_war = overlaps(current->second.mDestinations,
                                    older->second.mSources);
        const bool acc_raw = overlaps(current->second.accSources,
                                      older->second.accDestinations);
        const bool acc_waw = overlaps(current->second.accDestinations,
                                      older->second.accDestinations);
        const bool acc_war = overlaps(current->second.accDestinations,
                                      older->second.accSources);
        if (m_waw || m_war || acc_raw || acc_waw || acc_war)
            return false;
        /* M RAW is deliberately allowed here. The operand requester blocks
         * on exact byte coverage, enabling fragment-level chaining. */
    }
    return true;
}

std::vector<AmeInsnId>
AmeScoreboard::mProducersFor(AmeContextId id) const
{
    const auto current = reservations.find(id);
    panic_if(current == reservations.end(), "Unknown AME consumer id=%llu", id);
    std::vector<AmeInsnId> result;
    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.tid == current->second.tid &&
            overlaps(current->second.mSources,
                     older->second.mDestinations)) {
            result.push_back(older->second.ameId);
        }
    }
    return result;
}

bool
AmeScoreboard::mRegionReady(AmeInsnId producer, const MRegion &region) const
{
    const auto it = committedCoverage.find(producer);
    if (it == committedCoverage.end())
        return false;
    uint32_t cursor = region.byteBegin;
    std::vector<Coverage> matching;
    for (const auto &coverage : it->second) {
        if (coverage.reg == region.reg && coverage.end > region.byteBegin &&
            coverage.begin < region.byteEnd) {
            matching.push_back(coverage);
        }
    }
    std::sort(matching.begin(), matching.end(),
              [](const Coverage &a, const Coverage &b) {
                  return a.begin < b.begin;
              });
    for (const auto &coverage : matching) {
        if (coverage.begin > cursor)
            return false;
        cursor = std::max(cursor, coverage.end);
        if (cursor >= region.byteEnd)
            return true;
    }
    return cursor >= region.byteEnd;
}

bool
AmeScoreboard::mRegionReadyForConsumer(
    AmeContextId consumer, const MRegion &region) const
{
    const auto current = reservations.find(consumer);
    panic_if(current == reservations.end(), "Unknown AME consumer id=%llu",
             consumer);
    for (auto older = reservations.begin(); older != current; ++older) {
        if (older->second.tid != current->second.tid)
            continue;
        const bool produces_region = std::any_of(
            older->second.mDestinations.begin(),
            older->second.mDestinations.end(),
            [&region](const MRegion &destination) {
                return overlaps(region, destination);
            });
        if (produces_region && !mRegionReady(older->second.ameId, region))
            return false;
    }
    return true;
}

bool
AmeScoreboard::accReady(AmeInsnId producer) const
{
    return accCommitted.find(producer) != accCommitted.end();
}

void
AmeScoreboard::publishMWriteNext(AmeInsnId producer, const MRegion &region)
{
    panic_if(region.empty(), "Publishing empty AME MRF coverage");
    pendingCoverage[producer].push_back(
        Coverage{region.reg, region.byteBegin, region.byteEnd});
}

void
AmeScoreboard::publishAccWriteNext(AmeInsnId producer)
{
    pendingAccCommitted.insert(producer);
}

void
AmeScoreboard::advance()
{
    for (auto &[producer, coverage] : pendingCoverage) {
        auto &committed = committedCoverage[producer];
        committed.insert(committed.end(), coverage.begin(), coverage.end());
    }
    pendingCoverage.clear();
    accCommitted.insert(pendingAccCommitted.begin(), pendingAccCommitted.end());
    pendingAccCommitted.clear();
}

void
AmeScoreboard::release(AmeContextId id)
{
    const auto it = reservations.find(id);
    panic_if(it == reservations.end(), "Releasing unknown AME context id=%llu", id);
    const AmeInsnId slot = it->second.ameId;
    reservations.erase(it);
    committedCoverage.erase(slot);
    pendingCoverage.erase(slot);
    accCommitted.erase(slot);
    pendingAccCommitted.erase(slot);
}

void
AmeScoreboard::cancel(AmeContextId id)
{
    if (reservations.find(id) != reservations.end())
        release(id);
}

} // namespace ame
} // namespace minor
} // namespace gem5
