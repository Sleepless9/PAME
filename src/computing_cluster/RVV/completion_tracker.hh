/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_COMPLETION_TRACKER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_COMPLETION_TRACKER_HH__

#include <map>
#include <set>

#include "computing_cluster/RVV/global_scoreboard.hh"
#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_context.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

class CompletionTracker
{
  private:
    struct BeatState
    {
        unsigned expectedLanes = 0;
        unsigned completedLanes = 0;
    };

    InstContextTable &contexts;
    GlobalVectorScoreboard &scoreboard;
    RvvChannel<LaneComplete> &laneCompletions;
    RvvChannel<RvvCompletion> &completions;
    std::map<BeatTag, BeatState> beats;
    std::vector<std::pair<BeatTag, BeatState>> pendingBeats;
    std::set<RvvInstId> readyInsts;

  public:
    CompletionTracker(InstContextTable &contexts_,
                      GlobalVectorScoreboard &scoreboard_,
                      RvvChannel<LaneComplete> &lane_completions,
                      RvvChannel<RvvCompletion> &completions_);

    void registerBeatNext(const VecBeat &beat);
    void targetDone(RvvInstId id, TargetVfuMask target);
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(RvvInstId id);
    bool active() const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_COMPLETION_TRACKER_HH__
