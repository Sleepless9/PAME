/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_SCOREBOARD_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_SCOREBOARD_HH__

#include <map>
#include <set>
#include <vector>

#include "computing_cluster/AME/ame_types.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

class AmeScoreboard
{
  private:
    struct Reservation
    {
        AmeContextId id = 0;
        AmeInsnId ameId = 0;
        ThreadID tid = InvalidThreadID;
        std::vector<MRegion> mSources;
        std::vector<MRegion> mDestinations;
        std::vector<AccRegion> accSources;
        std::vector<AccRegion> accDestinations;
        bool serialized = false;
    };

    struct Coverage
    {
        uint16_t reg = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
    };

    std::map<AmeContextId, Reservation> reservations;
    std::map<AmeInsnId, std::vector<Coverage>> committedCoverage;
    std::map<AmeInsnId, std::vector<Coverage>> pendingCoverage;
    std::set<AmeInsnId> accCommitted;
    std::set<AmeInsnId> pendingAccCommitted;

    static bool overlaps(const MRegion &left, const MRegion &right);
    static bool overlaps(const AccRegion &left, const AccRegion &right);
    static bool overlaps(const std::vector<MRegion> &left,
                         const std::vector<MRegion> &right);
    static bool overlaps(const std::vector<AccRegion> &left,
                         const std::vector<AccRegion> &right);

  public:
    void allocate(const AmeInstContext &context);
    void bindSlot(AmeContextId id, AmeInsnId ame_id);
    bool canSequence(AmeContextId id) const;
    std::vector<AmeInsnId> mProducersFor(AmeContextId id) const;
    bool mRegionReady(AmeInsnId producer, const MRegion &region) const;
    bool mRegionReadyForConsumer(AmeContextId consumer,
                                 const MRegion &region) const;
    bool accReady(AmeInsnId producer) const;
    void publishMWriteNext(AmeInsnId producer, const MRegion &region);
    void publishAccWriteNext(AmeInsnId producer);
    void advance();
    void release(AmeContextId id);
    void cancel(AmeContextId id);
    bool empty() const { return reservations.empty(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_SCOREBOARD_HH__
