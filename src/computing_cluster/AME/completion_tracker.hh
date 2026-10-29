/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_COMPLETION_TRACKER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_COMPLETION_TRACKER_HH__

#include <set>

#include "computing_cluster/AME/ame_channel.hh"
#include "computing_cluster/AME/ame_context.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

class AmeCompletionTracker
{
  private:
    AmeContextTable &contexts;
    AmeChannel<AmeCompletion> &completions;
    std::set<AmeContextId> ready;

  public:
    AmeCompletionTracker(AmeContextTable &contexts_,
                         AmeChannel<AmeCompletion> &completions_)
        : contexts(contexts_), completions(completions_)
    {}
    void targetDone(const TargetDone &done);
    void evaluate(uint64_t cycle);
    void cancel(AmeContextId id);
    bool active() const { return !ready.empty() || completions.active(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_COMPLETION_TRACKER_HH__
