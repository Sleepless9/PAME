/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_RESULT_PROGRESS_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_RESULT_PROGRESS_HH__

#include <map>

#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

/** Per-lane architectural write progress used by RVV-style chaining. */
class ResultProgressTable
{
  public:
    struct Entry
    {
        uint64_t wordsWritten = 0;
        bool pulseThisCycle = false;
        bool instructionWriteDone = false;
    };

  private:
    std::map<RvvInsnSlotId, Entry> current;
    std::map<RvvInsnSlotId, Entry> pending;

  public:
    void allocate(RvvInsnSlotId id) { current.emplace(id, Entry{}); }
    void publishWriteNext(RvvInsnSlotId id, uint64_t words = 1,
                          bool final = false);
    const Entry *lookup(RvvInsnSlotId id) const;
    uint64_t wordsWritten(RvvInsnSlotId id) const;
    bool canRead(RvvInsnSlotId id, uint32_t word) const
    { return wordsWritten(id) > word; }
    void advance();
    void cancel(RvvInsnSlotId id);
    void release(RvvInsnSlotId id);
    bool active() const { return !current.empty() || !pending.empty(); }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_RESULT_PROGRESS_HH__
