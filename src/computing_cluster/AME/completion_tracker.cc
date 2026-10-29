/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/completion_tracker.hh"

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

void
AmeCompletionTracker::targetDone(const TargetDone &done)
{
    if (!contexts.contains(done.contextId))
        return;
    auto &context = contexts.get(done.contextId);
    panic_if((context.desc.targets & done.target) == 0 ||
             (context.completedTargets & done.target) != 0,
             "Invalid or duplicate AME target completion");
    context.completedTargets |= done.target;
    if (context.completedTargets == context.desc.targets)
        ready.insert(done.contextId);
}

void
AmeCompletionTracker::evaluate(uint64_t cycle)
{
    if (ready.empty() || !completions.canPushNext())
        return;
    const AmeContextId id = *ready.begin();
    const auto &context = contexts.get(id);
    completions.pushNext(AmeCompletion{id, context.desc.key, context.fault});
    ready.erase(ready.begin());
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u AME_COMPLETE latency=%llu\n",
            cycle, id, context.desc.ameId, cycle - context.acceptCycle);
}

void
AmeCompletionTracker::cancel(AmeContextId id)
{
    ready.erase(id);
    completions.eraseIf([id](const AmeCompletion &completion) {
        return completion.contextId == id;
    });
}

} // namespace ame
} // namespace minor
} // namespace gem5
