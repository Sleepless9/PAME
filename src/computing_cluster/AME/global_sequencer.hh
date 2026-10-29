/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_GLOBAL_SEQUENCER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_GLOBAL_SEQUENCER_HH__

#include <optional>

#include "computing_cluster/AME/ame_channel.hh"
#include "computing_cluster/AME/ame_context.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

class AmeGlobalSequencer
{
  private:
    struct Broadcast
    {
        AmePeRequest request;
        AmeTargetMask required = TargetNone;
        AmeTargetMask accepted = TargetNone;
        bool sent = false;
    };

    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    AmeChannel<AmeInstToken> &input;
    AmeChannel<AmePeRequest> &output;
    AmeInsnIdAllocator ids;
    std::optional<Broadcast> pending;

  public:
    AmeGlobalSequencer(AmeContextTable &contexts_,
                       AmeScoreboard &scoreboard_,
                       AmeChannel<AmeInstToken> &input_,
                       AmeChannel<AmePeRequest> &output_,
                       unsigned id_capacity);
    void evaluate(uint64_t cycle);
    void targetAccepted(AmeContextId id, AmeTargetMask targets);
    void releaseSlot(AmeInsnId id) { ids.release(id); }
    void cancel(AmeContextId id);
    bool active() const { return pending.has_value() || input.active(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_GLOBAL_SEQUENCER_HH__
