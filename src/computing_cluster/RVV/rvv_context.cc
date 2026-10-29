/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/rvv_context.hh"

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

InstContextTable::InstContextTable(size_t capacity_) : capacity(capacity_)
{
    fatal_if(capacity == 0, "RVV context table capacity must be non-zero");
}

void
InstContextTable::allocate(RvvInstContext context)
{
    panic_if(!canAllocate(), "RVV context table overflow");
    const RvvInstId id = context.desc.id;
    const auto [it, inserted] = contexts.emplace(id, std::move(context));
    panic_if(!inserted, "Duplicate RVV context id=%llu", id);
}

bool
InstContextTable::contains(RvvInstId id) const
{
    return contexts.find(id) != contexts.end();
}

bool
InstContextTable::hasOlder(RvvInstId id) const
{
    return !contexts.empty() && contexts.begin()->first < id;
}

bool
InstContextTable::hasOlderSerialized(RvvInstId id) const
{
    for (auto it = contexts.begin(); it != contexts.lower_bound(id); ++it) {
        if (it->second.desc.serialized)
            return true;
    }
    return false;
}

RvvInstContext &
InstContextTable::get(RvvInstId id)
{
    auto it = contexts.find(id);
    panic_if(it == contexts.end(), "Unknown RVV context id=%llu", id);
    return it->second;
}

const RvvInstContext &
InstContextTable::get(RvvInstId id) const
{
    auto it = contexts.find(id);
    panic_if(it == contexts.end(), "Unknown RVV context id=%llu", id);
    return it->second;
}

void
InstContextTable::release(RvvInstId id)
{
    panic_if(contexts.erase(id) != 1, "Releasing unknown RVV context id=%llu", id);
}

void
InstContextTable::cancel(RvvInstId id)
{
    auto it = contexts.find(id);
    if (it == contexts.end())
        return;
    it->second.state.killed = true;
    it->second.state.status = ContextStatus::Squashed;
    contexts.erase(it);
}

RvvInsnSlotAllocator::RvvInsnSlotAllocator(unsigned capacity_) : capacity(capacity_)
{
    fatal_if(capacity == 0 || capacity > 255,
             "RVV instruction ID capacity must be in [1, 255]");
    for (unsigned id = 0; id < capacity; ++id)
        freeIds.insert(static_cast<RvvInsnSlotId>(id));
}

RvvInsnSlotId
RvvInsnSlotAllocator::allocate()
{
    panic_if(freeIds.empty(), "RVV instruction ID allocator exhausted");
    const auto it = freeIds.begin();
    const RvvInsnSlotId id = *it;
    freeIds.erase(it);
    return id;
}

void
RvvInsnSlotAllocator::release(RvvInsnSlotId id)
{
    panic_if(id >= capacity, "Invalid RVV instruction ID %u", id);
    const auto [it, inserted] = freeIds.insert(id);
    panic_if(!inserted, "RVV instruction ID %u released twice", id);
}

} // namespace rvv
} // namespace minor
} // namespace gem5
