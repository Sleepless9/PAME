/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_DISPATCHER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_DISPATCHER_HH__

#include <array>
#include <functional>
#include <map>
#include <optional>
#include <set>

#include "computing_cluster/AME/ame_channel.hh"
#include "computing_cluster/AME/ame_context.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

using ConfigSnapshot =
    std::function<void(AmeConfig &, const InstQueuePacket &)>;
using DmaEventReady = std::function<bool(uint64_t)>;
using DmaEventConsume = std::function<void(uint64_t)>;

class AmeDispatcher
{
  private:
    MinorInstQueue &instQueue;
    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    AmeChannel<AmeInstToken> &output;
    const ConfigSnapshot configSnapshot;
    const DmaEventReady dmaEventReady;
    const DmaEventConsume dmaEventConsume;
    AmeContextId nextId = 1;
    std::map<InstQueueKey, AmeContextId> keyToId;
    std::set<InstQueueKey> retiringSerialBarriers;

    struct TypeShadow
    {
        std::array<uint16_t, 32> m{};
        std::array<uint16_t, 4> acc{};
    };
    std::map<ThreadID, TypeShadow> typeShadows;

  public:
    AmeDispatcher(MinorInstQueue &inst_queue, AmeContextTable &contexts_,
                  AmeScoreboard &scoreboard_,
                  AmeChannel<AmeInstToken> &output_,
                  ConfigSnapshot config_snapshot = {},
                  DmaEventReady dma_event_ready = {},
                  DmaEventConsume dma_event_consume = {});
    void evaluate(uint64_t cycle);
    std::optional<AmeContextId> idForKey(const InstQueueKey &key) const;
    void cancel(const InstQueueKey &key);
    void release(AmeContextId id);
    bool active() const
    { return !keyToId.empty() || !retiringSerialBarriers.empty(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_DISPATCHER_HH__
