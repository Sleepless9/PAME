/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_LANE_SCOREBOARD_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_LANE_SCOREBOARD_HH__

#include <map>

#include "computing_cluster/RVV/rvv_context.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct LaneScoreEntry
{
    LaneTag tag;
    VecFuType fuType = VecFuType::Alu;
    std::vector<RegId> sources;
    std::optional<RegId> destination;
    uint32_t firstElem = 0;
    uint16_t elemCount = 0;
    uint32_t groupByteOffset = 0;
    uint64_t byteEnable = 0;
    bool last = true;
    unsigned operandsRequested = 0;
    unsigned operandsReady = 0;
    bool fuIssued = false;
};

class LaneScoreboard
{
  private:
    const size_t capacity;
    std::map<LaneTag, LaneScoreEntry> entries;
    std::vector<LaneScoreEntry> pendingAllocations;
    std::map<LaneTag, unsigned> pendingRequested;
    std::map<LaneTag, unsigned> pendingReady;
    std::vector<LaneTag> pendingFuIssued;

  public:
    explicit LaneScoreboard(size_t capacity_);

    bool canAllocateNext() const;
    void allocateNext(const LaneBeat &beat, const RvvInstDesc &desc);
    bool contains(const LaneTag &tag) const;
    const LaneScoreEntry &get(const LaneTag &tag) const;
    LaneScoreEntry &get(const LaneTag &tag);

    void markOperandRequestedNext(const LaneTag &tag);
    void markOperandReadyNext(const LaneTag &tag);
    void markFuIssuedNext(const LaneTag &tag);
    std::optional<LaneTag> firstReadyForFu() const;
    void complete(const LaneTag &tag);

    void advance();
    void cancel(RvvInstId id);
    bool active() const { return !entries.empty() || !pendingAllocations.empty(); }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_LANE_SCOREBOARD_HH__
