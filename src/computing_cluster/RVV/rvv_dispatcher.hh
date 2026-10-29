/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_DISPATCHER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_DISPATCHER_HH__

#include <functional>
#include <map>
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

struct DispatcherParams
{
    unsigned numLanes = 1;
    unsigned laneWidthBits = 64;
    unsigned maxPackedElementsPerLane = 8;
};

using ConfigSnapshot =
    std::function<void(RvvConfig &, const InstQueuePacket &)>;

class RvvDispatcher
{
  private:
    MinorInstQueue &instQueue;
    InstContextTable &contexts;
    GlobalVectorScoreboard &scoreboard;
    RvvChannel<VectorInstToken> &dispatchQueue;
    const DispatcherParams params;
    const ConfigSnapshot configSnapshot;
    RvvInstId nextId = 1;
    std::map<InstQueueKey, RvvInstId> keyToId;

  public:
    RvvDispatcher(MinorInstQueue &inst_queue, InstContextTable &contexts_,
                  GlobalVectorScoreboard &scoreboard_,
                  RvvChannel<VectorInstToken> &dispatch_queue,
                  const DispatcherParams &params_,
                  ConfigSnapshot config_snapshot = {});

    void evaluate(uint64_t cycle);
    std::optional<RvvInstId> idForKey(const InstQueueKey &key) const;
    void cancel(const InstQueueKey &key);
    void release(RvvInstId id);
    bool active() const { return !keyToId.empty(); }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_DISPATCHER_HH__
