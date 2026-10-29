/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_ACCUMULATOR_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_ACCUMULATOR_HH__

#include <map>

#include "computing_cluster/AME/ame_context.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"
#include "computing_cluster/AME/register_file.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

class AccumulatorSubsystem
{
  private:
    struct Progress
    {
        AmeInsnId owner = 0;
        uint8_t reg = 0;
        AccLocation location = AccLocation::AccRF;
        unsigned totalBytes = 0;
        unsigned totalFragments = 0;
        unsigned readsIssued = 0;
        unsigned readsCompleted = 0;
        unsigned writesIssued = 0;
        unsigned writesCompleted = 0;
        bool committed = false;
    };

    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    BankedRegisterFile &accRf;
    std::map<AmeContextId, Progress> progress;

  public:
    AccumulatorSubsystem(AmeContextTable &contexts_,
                         AmeScoreboard &scoreboard_,
                         BankedRegisterFile &acc_rf);

    void start(AmeContextId id);
    void evaluateSeed(AmeContextId id);
    bool seedReady(AmeContextId id) const;
    void markResident(AmeContextId id);
    void beginDrain(AmeContextId id);
    void evaluateDrain(AmeContextId id);
    void acceptResponse(const BankResponse &response);
    bool committed(AmeContextId id) const;
    AccLocation location(AmeContextId id) const;
    void release(AmeContextId id);
    void cancel(AmeContextId id);
    bool active() const { return !progress.empty(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_ACCUMULATOR_HH__
