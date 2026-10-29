/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/result_progress.hh"

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

void
ResultProgressTable::publishWriteNext(RvvInsnSlotId id, uint64_t words, bool final)
{
    panic_if(words == 0, "RVV result progress requires a non-zero word count");
    auto &entry = pending[id];
    entry.wordsWritten += words;
    entry.pulseThisCycle = true;
    entry.instructionWriteDone |= final;
}

const ResultProgressTable::Entry *
ResultProgressTable::lookup(RvvInsnSlotId id) const
{
    const auto it = current.find(id);
    return it == current.end() ? nullptr : &it->second;
}

uint64_t
ResultProgressTable::wordsWritten(RvvInsnSlotId id) const
{
    const auto *entry = lookup(id);
    return entry ? entry->wordsWritten : 0;
}

void
ResultProgressTable::advance()
{
    for (auto &[id, update] : pending) {
        auto &entry = current[id];
        entry.wordsWritten += update.wordsWritten;
        entry.pulseThisCycle = update.pulseThisCycle;
        entry.instructionWriteDone |= update.instructionWriteDone;
    }
    pending.clear();
    for (auto &[id, entry] : current)
        entry.pulseThisCycle = false;
}

void
ResultProgressTable::cancel(RvvInsnSlotId id)
{
    current.erase(id);
    pending.erase(id);
}

void
ResultProgressTable::release(RvvInsnSlotId id)
{
    cancel(id);
}

} // namespace rvv
} // namespace minor
} // namespace gem5
