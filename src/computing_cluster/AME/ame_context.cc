/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/ame_context.hh"

#include <algorithm>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

AmeContextTable::AmeContextTable(size_t capacity_) : capacity(capacity_)
{
    fatal_if(capacity == 0, "AME context capacity must be non-zero");
}

void
AmeContextTable::allocate(AmeInstContext context)
{
    panic_if(!canAllocate(), "AME context table overflow");
    const auto [it, inserted] =
        entries.emplace(context.desc.id, std::move(context));
    panic_if(!inserted, "Duplicate AME context id=%llu", it->first);
}

bool
AmeContextTable::contains(AmeContextId id) const
{
    return entries.find(id) != entries.end();
}

AmeInstContext &
AmeContextTable::get(AmeContextId id)
{
    const auto it = entries.find(id);
    panic_if(it == entries.end(), "Unknown AME context id=%llu", id);
    return it->second;
}

const AmeInstContext &
AmeContextTable::get(AmeContextId id) const
{
    const auto it = entries.find(id);
    panic_if(it == entries.end(), "Unknown AME context id=%llu", id);
    return it->second;
}

void
AmeContextTable::cancel(AmeContextId id)
{
    if (contains(id))
        entries.at(id).killed = true;
}

void
AmeContextTable::release(AmeContextId id)
{
    panic_if(entries.erase(id) != 1, "Releasing unknown AME context id=%llu", id);
}

AmeInsnIdAllocator::AmeInsnIdAllocator(unsigned capacity) : used(capacity, false)
{
    fatal_if(capacity == 0 || capacity > 256,
             "AME instruction ID capacity must be in [1, 256]");
}

bool
AmeInsnIdAllocator::canAllocate() const
{
    return std::find(used.begin(), used.end(), false) != used.end();
}

AmeInsnId
AmeInsnIdAllocator::allocate()
{
    const auto it = std::find(used.begin(), used.end(), false);
    panic_if(it == used.end(), "AME instruction IDs exhausted");
    *it = true;
    return static_cast<AmeInsnId>(std::distance(used.begin(), it));
}

void
AmeInsnIdAllocator::release(AmeInsnId id)
{
    panic_if(id >= used.size() || !used[id], "Releasing free AME ID %u", id);
    used[id] = false;
}

bool
AmeInsnIdAllocator::empty() const
{
    return std::none_of(used.begin(), used.end(), [](bool entry) {
        return entry;
    });
}

} // namespace ame
} // namespace minor
} // namespace gem5
