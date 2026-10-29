/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SCOREBOARD_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SCOREBOARD_HH__

#include <map>
#include <set>
#include <vector>

#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

/** Instruction hazards plus P1 beat/lane RAW readiness. */
class GlobalVectorScoreboard
{
  private:
    struct RegisterGroup
    {
        RegId base;
        std::vector<RegId> members;
    };

    struct Reservation
    {
        RvvInstId id = 0;
        RvvInsnSlotId rvvSlotId = 0;
        bool slotIdBound = false;
        ThreadID threadId = InvalidThreadID;
        std::vector<RegisterGroup> sourceGroups;
        std::vector<RegId> destinationGroups;
        std::vector<RegId> sources;
        std::vector<RegId> destinations;
        RvvConfig config;
        uint16_t elementsPerLaneChunk = 1;
        uint16_t elementsPerLaneBeat = 1;
    };

    struct ReadyKey
    {
        RvvInstId producer = 0;
        BeatId beat = 0;
        LaneId lane = 0;

        bool operator<(const ReadyKey &other) const
        {
            if (producer != other.producer)
                return producer < other.producer;
            if (beat != other.beat)
                return beat < other.beat;
            return lane < other.lane;
        }
    };

    std::map<RvvInstId, Reservation> reservations;
    std::set<RvvInstId> pendingReleases;
    std::set<ReadyKey> resultReady;
    std::set<ReadyKey> writeReady;
    std::set<ReadyKey> pendingResultReady;
    std::set<ReadyKey> pendingWriteReady;
    std::map<RvvInsnSlotId, uint64_t> wordProgress;
    std::map<RvvInsnSlotId, uint64_t> pendingWordProgress;

    static bool overlaps(const std::vector<RegId> &left,
                         const std::vector<RegId> &right);
    static std::vector<RegId> expand(const std::vector<RegId> &regs,
                                     const RvvConfig &config);
    static std::vector<RegisterGroup> sourceGroups(
        const InstQueuePacket &packet, const RvvConfig &config);
    static bool compatibleMapping(const Reservation &producer,
                                  const Reservation &consumer);
    static bool exactRawMapping(const Reservation &producer,
                                const Reservation &consumer);
    bool readyForRaw(const Reservation &producer, BeatId beat,
                     uint64_t active_lanes) const;

  public:
    void allocate(const RvvInstContext &context);
    void bindSlotId(RvvInstId id, RvvInsnSlotId slot_id);
    bool canSequence(RvvInstId id) const;
    bool canSequenceBeat(RvvInstId id, BeatId beat,
                         uint64_t active_lanes) const;
    bool canBypass(RvvInstId consumer, const RegId &source,
                   const LaneTag &tag) const;
    std::vector<RvvInstId> producersFor(RvvInstId consumer) const;
    void publishWordWriteNext(RvvInsnSlotId producer, uint64_t words = 1);
    uint64_t wordsWritten(RvvInsnSlotId producer) const;
    bool canReadWord(const std::vector<RvvInstId> &producers,
                     uint32_t word_index) const;
    bool canReadWord(const std::vector<RvvInsnSlotId> &producers,
                        uint32_t word_index) const;
    void publishResultReadyNext(const LaneTag &tag);
    void publishWriteReadyNext(const LaneTag &tag);
    void release(RvvInstId id);
    void releaseNext(RvvInstId id);
    void advance();
    void cancel(RvvInstId id);
    bool empty() const { return reservations.empty(); }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_GLOBAL_SCOREBOARD_HH__
