/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CONTEXT_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CONTEXT_HH__

#include <map>
#include <set>

#include "base/logging.hh"
#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

class InstContextTable
{
  private:
    const size_t capacity;
    std::map<RvvInstId, RvvInstContext> contexts;

  public:
    explicit InstContextTable(size_t capacity_);

    bool canAllocate() const { return contexts.size() < capacity; }
    void allocate(RvvInstContext context);
    bool contains(RvvInstId id) const;
    bool hasOlder(RvvInstId id) const;
    bool hasOlderSerialized(RvvInstId id) const;
    RvvInstContext &get(RvvInstId id);
    const RvvInstContext &get(RvvInstId id) const;
    void release(RvvInstId id);
    void cancel(RvvInstId id);

    bool empty() const { return contexts.empty(); }
    size_t size() const { return contexts.size(); }
};

/** Finite RVV instruction-slot allocator owned by the global sequencer. */
class RvvInsnSlotAllocator
{
  private:
    const unsigned capacity;
    std::set<RvvInsnSlotId> freeIds;

  public:
    explicit RvvInsnSlotAllocator(unsigned capacity_);

    bool canAllocate() const { return !freeIds.empty(); }
    RvvInsnSlotId allocate();
    void release(RvvInsnSlotId id);
    bool running(RvvInsnSlotId id) const
    { return freeIds.find(id) == freeIds.end(); }
    unsigned capacitySize() const { return capacity; }
    unsigned used() const { return capacity - freeIds.size(); }
};

struct RvvRunningBitmap
{
    uint64_t bits = 0;
    void set(RvvInsnSlotId id)
    {
        panic_if(id >= 64, "RVV running bitmap supports at most 64 IDs");
        bits |= 1ULL << id;
    }
    void clear(RvvInsnSlotId id)
    {
        panic_if(id >= 64, "RVV running bitmap supports at most 64 IDs");
        bits &= ~(1ULL << id);
    }
    bool test(RvvInsnSlotId id) const
    { return id < 64 && (bits & (1ULL << id)) != 0; }
    bool empty() const { return bits == 0; }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CONTEXT_HH__
