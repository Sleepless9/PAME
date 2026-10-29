/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AUXILIARY_UNITS_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AUXILIARY_UNITS_HH__

#include <deque>
#include <optional>
#include <vector>

#include "computing_cluster/AME/ame_context.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"
#include "computing_cluster/AME/register_file.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

struct AuxiliaryUnitParams
{
    unsigned queueDepth = 2;
    unsigned localLatency = 2;
    unsigned crossTileLatency = 1;
    unsigned tileRows = 1;
    unsigned tileCols = 1;
    unsigned resultWidth = 1;
};

class MatrixAuxiliaryUnit
{
  private:
    enum class Phase : uint8_t
    {
        Read,
        LocalCompute,
        CrossTile,
        PrefixPropagate,
        Format,
        Write,
        Done
    };

    struct ComputeToken
    {
        uint64_t readyCycle = 0;
        unsigned line = 0;
        unsigned ordinal = 0;
    };

    struct Work
    {
        AmePeRequest request;
        Phase phase = Phase::Read;
        unsigned readsExpected = 0;
        unsigned readsIssued = 0;
        unsigned readsDone = 0;
        uint32_t mReadBytesIssued = 0;
        unsigned writesExpected = 0;
        unsigned writesIssued = 0;
        unsigned writesDone = 0;
        uint32_t mWriteBytesIssued = 0;
        unsigned localItemsExpected = 0;
        unsigned localItemsIssued = 0;
        unsigned localItemsDone = 0;
        std::vector<unsigned> localIssuedByLine;
        std::vector<unsigned> localDoneByLine;
        std::deque<ComputeToken> localPipeline;
        unsigned axisTiles = 1;
        unsigned crossPartials = 1;
        unsigned crossMergesExpected = 0;
        unsigned crossMergesIssued = 0;
        unsigned crossMergesDone = 0;
        std::deque<ComputeToken> crossPipeline;
        std::vector<unsigned> carryIssuedByLine;
        std::vector<unsigned> carryDoneByLine;
        size_t roundRobinLine = 0;
        uint64_t readyCycle = 0;
    };

    const AmeTarget target;
    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    BankedRegisterFile &mrf;
    BankedRegisterFile &accRf;
    const AuxiliaryUnitParams params;
    AmeChannel<AmePeRequest> requests;
    AmeChannel<TargetDone> done;
    std::optional<Work> work;
    uint64_t localCycleCount = 0;
    uint64_t crossTileCycleCount = 0;
    uint64_t prefixCycleCount = 0;
    uint64_t computeItemCount = 0;

    void start(const AmePeRequest &request);
    bool issueRead();
    bool issueWrite();
    unsigned axisTileCount(const AmePeRequest &request) const;
    void evaluateLocal(uint64_t cycle);
    void startCrossStage(Work &state);
    void evaluateCrossTile(uint64_t cycle);
    void evaluatePrefix(uint64_t cycle);
    MRegion mapMByte(const std::vector<std::vector<MRegion>> &groups,
                     uint32_t byte_offset, uint32_t max_bytes) const;

  public:
    MatrixAuxiliaryUnit(AmeTarget target_, AmeContextTable &contexts_,
                        AmeScoreboard &scoreboard_,
                        BankedRegisterFile &mrf_,
                        BankedRegisterFile &acc_rf,
                        const AuxiliaryUnitParams &params_);
    bool canAcceptNext() const { return requests.canPushNext(); }
    void acceptNext(AmePeRequest request)
    { requests.pushNext(std::move(request)); }
    AmeChannel<TargetDone> &doneChannel() { return done; }
    void acceptResponse(const BankResponse &response);
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(AmeContextId id);
    bool active() const
    { return work.has_value() || requests.active() || done.active(); }
    uint64_t localCycles() const { return localCycleCount; }
    uint64_t crossTileCycles() const { return crossTileCycleCount; }
    uint64_t prefixCycles() const { return prefixCycleCount; }
    uint64_t computeItems() const { return computeItemCount; }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AUXILIARY_UNITS_HH__
