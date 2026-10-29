/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/lane_scoreboard.hh"

#include <algorithm>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

LaneScoreboard::LaneScoreboard(size_t capacity_) : capacity(capacity_)
{
    fatal_if(capacity == 0, "RVV lane scoreboard capacity must be non-zero");
}

bool
LaneScoreboard::canAllocateNext() const
{
    return entries.size() + pendingAllocations.size() < capacity;
}

void
LaneScoreboard::allocateNext(const LaneBeat &beat, const RvvInstDesc &desc)
{
    panic_if(!canAllocateNext(), "RVV lane scoreboard overflow");
    LaneScoreEntry entry;
    entry.tag = beat.tag;
    entry.fuType = desc.fuType;
    entry.sources = vectorSources(desc.packet, desc.config);
    entry.destination = vectorDestination(desc.packet);
    entry.firstElem = beat.firstElem;
    entry.elemCount = beat.elemCount;
    entry.groupByteOffset = beat.groupByteOffset;
    entry.byteEnable = beat.byteEnable;
    entry.last = beat.last;
    pendingAllocations.push_back(std::move(entry));
}

bool
LaneScoreboard::contains(const LaneTag &tag) const
{
    return entries.find(tag) != entries.end();
}

const LaneScoreEntry &
LaneScoreboard::get(const LaneTag &tag) const
{
    const auto it = entries.find(tag);
    panic_if(it == entries.end(), "Unknown RVV lane scoreboard tag");
    return it->second;
}

LaneScoreEntry &
LaneScoreboard::get(const LaneTag &tag)
{
    const auto it = entries.find(tag);
    panic_if(it == entries.end(), "Unknown RVV lane scoreboard tag");
    return it->second;
}

void
LaneScoreboard::markOperandRequestedNext(const LaneTag &tag)
{
    ++pendingRequested[tag];
}

void
LaneScoreboard::markOperandReadyNext(const LaneTag &tag)
{
    ++pendingReady[tag];
}

void
LaneScoreboard::markFuIssuedNext(const LaneTag &tag)
{
    pendingFuIssued.push_back(tag);
}

std::optional<LaneTag>
LaneScoreboard::firstReadyForFu() const
{
    for (const auto &[tag, entry] : entries) {
        if (!entry.fuIssued && entry.operandsReady == entry.sources.size())
            return tag;
    }
    return std::nullopt;
}

void
LaneScoreboard::complete(const LaneTag &tag)
{
    panic_if(entries.erase(tag) != 1,
             "Completing an unknown RVV lane scoreboard entry");
}

void
LaneScoreboard::advance()
{
    for (auto &entry : pendingAllocations) {
        const auto [it, inserted] = entries.emplace(entry.tag, std::move(entry));
        panic_if(!inserted, "Duplicate RVV lane scoreboard tag");
    }
    pendingAllocations.clear();

    for (const auto &[tag, count] : pendingRequested) {
        auto &entry = get(tag);
        entry.operandsRequested += count;
        panic_if(entry.operandsRequested > entry.sources.size(),
                 "RVV operand requested twice");
    }
    pendingRequested.clear();

    for (const auto &[tag, count] : pendingReady) {
        auto &entry = get(tag);
        entry.operandsReady += count;
        panic_if(entry.operandsReady > entry.sources.size(),
                 "RVV operand response received twice");
    }
    pendingReady.clear();

    for (const auto &tag : pendingFuIssued) {
        auto &entry = get(tag);
        panic_if(entry.fuIssued, "RVV lane beat issued to FU twice");
        entry.fuIssued = true;
    }
    pendingFuIssued.clear();
}

void
LaneScoreboard::cancel(RvvInstId id)
{
    for (auto it = entries.begin(); it != entries.end(); ) {
        if (it->first.instId == id)
            it = entries.erase(it);
        else
            ++it;
    }
    pendingAllocations.erase(
        std::remove_if(pendingAllocations.begin(), pendingAllocations.end(),
            [id](const LaneScoreEntry &entry) {
                return entry.tag.instId == id;
            }), pendingAllocations.end());
    for (auto it = pendingRequested.begin(); it != pendingRequested.end(); ) {
        if (it->first.instId == id)
            it = pendingRequested.erase(it);
        else
            ++it;
    }
    for (auto it = pendingReady.begin(); it != pendingReady.end(); ) {
        if (it->first.instId == id)
            it = pendingReady.erase(it);
        else
            ++it;
    }
    pendingFuIssued.erase(
        std::remove_if(pendingFuIssued.begin(), pendingFuIssued.end(),
            [id](const LaneTag &tag) { return tag.instId == id; }),
        pendingFuIssued.end());
}

} // namespace rvv
} // namespace minor
} // namespace gem5
