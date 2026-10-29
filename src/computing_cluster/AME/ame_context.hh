/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CONTEXT_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CONTEXT_HH__

#include <map>

#include "computing_cluster/AME/ame_types.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

class AmeContextTable
{
  private:
    const size_t capacity;
    std::map<AmeContextId, AmeInstContext> entries;

  public:
    explicit AmeContextTable(size_t capacity_);
    bool canAllocate() const { return entries.size() < capacity; }
    void allocate(AmeInstContext context);
    bool contains(AmeContextId id) const;
    AmeInstContext &get(AmeContextId id);
    const AmeInstContext &get(AmeContextId id) const;
    void cancel(AmeContextId id);
    void release(AmeContextId id);
    bool empty() const { return entries.empty(); }
};

class AmeInsnIdAllocator
{
  private:
    std::vector<bool> used;

  public:
    explicit AmeInsnIdAllocator(unsigned capacity);
    bool canAllocate() const;
    AmeInsnId allocate();
    void release(AmeInsnId id);
    bool empty() const;
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CONTEXT_HH__
