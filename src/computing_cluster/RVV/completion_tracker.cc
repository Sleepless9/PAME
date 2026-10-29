/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/completion_tracker.hh"

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

unsigned
countBits(uint64_t value)
{
    unsigned count = 0;
    while (value != 0) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

} // anonymous namespace

CompletionTracker::CompletionTracker(
    InstContextTable &contexts_, GlobalVectorScoreboard &scoreboard_,
    RvvChannel<LaneComplete> &lane_completions,
    RvvChannel<RvvCompletion> &completions_)
    : contexts(contexts_), scoreboard(scoreboard_),
      laneCompletions(lane_completions), completions(completions_)
{
}

void
CompletionTracker::registerBeatNext(const VecBeat &beat)
{
    const unsigned lanes = countBits(beat.activeLaneMask);
    panic_if(lanes == 0, "RVV beat has no participating lane");
    pendingBeats.push_back({beat.tag, BeatState{lanes, 0}});
}

void
CompletionTracker::targetDone(RvvInstId id, TargetVfuMask target)
{
    if (!contexts.contains(id))
        return;
    auto &context = contexts.get(id);
    context.targetDone |= target;
    if (context.state.completedBeats == context.desc.totalBeats &&
        context.targetDone == context.targetVfus) {
        context.state.status = ContextStatus::Completing;
        readyInsts.insert(id);
    }
}

void
CompletionTracker::evaluate(uint64_t cycle)
{
    const size_t available = laneCompletions.size();
    for (size_t index = 0; index < available; ++index) {
        const LaneComplete &completion = laneCompletions.at(index);
        const BeatTag beat_tag{
            completion.tag.instId, completion.tag.beatId};
        auto beat = beats.find(beat_tag);
        panic_if(beat == beats.end(),
                 "RVV LaneComplete has no Beat tracker");
        ++beat->second.completedLanes;
        panic_if(beat->second.completedLanes > beat->second.expectedLanes,
                 "RVV beat produced duplicate LaneComplete");
        laneCompletions.popNext();

        if (beat->second.completedLanes != beat->second.expectedLanes)
            continue;

        auto &context = contexts.get(beat_tag.instId);
        targetDone(beat_tag.instId, completion.target);
        ++context.state.completedBeats;
        panic_if(context.state.completedBeats > context.desc.totalBeats,
                 "RVV completedBeats invariant id=%llu", beat_tag.instId);
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=- Completion "
                "BEAT_DONE (%u/%u)\n",
                cycle, beat_tag.instId, beat_tag.beatId,
                context.state.completedBeats, context.desc.totalBeats);
        beats.erase(beat);
        if (context.state.completedBeats == context.desc.totalBeats &&
            context.targetDone == context.targetVfus) {
            context.state.status = ContextStatus::Completing;
            readyInsts.insert(context.desc.id);
        }
    }

    if (readyInsts.empty() || !completions.canPushNext())
        return;
    const RvvInstId id = *readyInsts.begin();
    auto &context = contexts.get(id);
    context.state.status = ContextStatus::Done;
    context.state.completionCycle = cycle;
    completions.pushNext(RvvCompletion{
        id, context.desc.key, context.state.fault});
    scoreboard.releaseNext(id);
    readyInsts.erase(readyInsts.begin());
    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu beat=- lane=- Completion INST_DONE "
            "latency=%llu\n",
            cycle, id, cycle - context.state.acceptCycle);
}

void
CompletionTracker::advance()
{
    for (const auto &[tag, state] : pendingBeats) {
        const auto [it, inserted] = beats.emplace(tag, state);
        panic_if(!inserted, "Duplicate RVV beat tracker");
    }
    pendingBeats.clear();
}

void
CompletionTracker::cancel(RvvInstId id)
{
    for (auto it = beats.begin(); it != beats.end(); ) {
        if (it->first.instId == id)
            it = beats.erase(it);
        else
            ++it;
    }
    pendingBeats.erase(
        std::remove_if(pendingBeats.begin(), pendingBeats.end(),
            [id](const auto &entry) { return entry.first.instId == id; }),
        pendingBeats.end());
    readyInsts.erase(id);
    laneCompletions.eraseIf([id](const LaneComplete &completion) {
        return completion.tag.instId == id;
    });
    completions.eraseIf([id](const RvvCompletion &completion) {
        return completion.instId == id;
    });
}

bool
CompletionTracker::active() const
{
    return !beats.empty() || !pendingBeats.empty() || !readyInsts.empty() ||
        laneCompletions.active() || completions.active();
}

} // namespace rvv
} // namespace minor
} // namespace gem5
