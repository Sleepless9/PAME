/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SEQUENCER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SEQUENCER_HH__

#include <optional>

#include "computing_cluster/RVV/global_scoreboard.hh"
#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_context.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

/** RVV-style instruction sequencer.  The output is one PE request per
 * instruction; lane-local word work is generated after target acceptance. */
class GlobalSequencer
{
  private:
    struct PendingBroadcast
    {
        PeRequest request;
        TargetVfuMask required = 0;
        TargetVfuMask accepted = 0;
        bool sent = false;
    };

    InstContextTable &contexts;
    GlobalVectorScoreboard &scoreboard;
    RvvChannel<VectorInstToken> &input;
    RvvChannel<PeRequest> &output;
    RvvInsnSlotAllocator rvvSlotIds;
    const unsigned numLanes;
    std::optional<PendingBroadcast> pending;

    PeRequest makeRequest(const RvvInstContext &context) const;

  public:
    GlobalSequencer(InstContextTable &contexts_,
                    GlobalVectorScoreboard &scoreboard_,
                    RvvChannel<VectorInstToken> &input_,
                    RvvChannel<PeRequest> &output_, unsigned num_lanes,
                    unsigned slot_id_capacity = 8);

    void evaluate(uint64_t cycle);
    void cancel(RvvInstId id);
    void targetAccepted(RvvInstId id, TargetVfuMask target);
    void releaseSlot(RvvInsnSlotId id) { rvvSlotIds.release(id); }
    bool active() const { return pending.has_value() || input.active(); }
    unsigned slotIdsUsed() const { return rvvSlotIds.used(); }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SEQUENCER_HH__
