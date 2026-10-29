/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_SYSTOLIC_ARRAY_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_SYSTOLIC_ARRAY_HH__

#include <deque>
#include <optional>
#include <vector>

#include "computing_cluster/AME/accumulator.hh"
#include "computing_cluster/AME/operand_subsystem.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

struct SystolicArrayParams
{
    unsigned peRows = 4;
    unsigned peCols = 4;
    unsigned tileRows = 1;
    unsigned tileCols = 1;
    unsigned kBlock = 4;
    unsigned commandQueueDepth = 2;
    unsigned tileIngressDepth = 2;
    unsigned edgeFifoDepth = 2;
    unsigned drainQueueDepth = 2;
    unsigned peHopLatency = 1;
    unsigned tileBoundaryLatency = 1;
    unsigned macLatency = 3;
    unsigned macII = 1;
    bool pssContinuous = false;
    unsigned operationDepth = 2;
};

struct WaveToken
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    uint32_t outputBlock = 0;
    uint32_t kBlock = 0;
    uint16_t kStep = 0;
    uint16_t tile = 0;
    uint16_t validRows = 0;
    uint16_t validCols = 0;
    MatrixDataflow dataflow = MatrixDataflow::PartialSumStationary;
    bool forwardEast = false;
    bool forwardSouth = false;
};

/**
 * One physical tile.  West and north operands are independently buffered;
 * a tile operation fires only after matching tokens are visible on both
 * inputs.  Every valid PE owns its own MAC pipeline and resident state.
 */
class MatrixTile
{
  private:
    struct MacToken
    {
        uint64_t readyCycle = 0;
        uint64_t operation = 0;
        WaveToken wave;
        bool issueCounted = false;
    };

    struct PeState
    {
        bool weightValid = false;
        bool psumValid = false;
        bool weightPending = false;
        bool psumPending = false;
        AmeContextId weightContext = 0;
        AmeContextId psumContext = 0;
        uint32_t weightKBlock = 0;
        uint64_t weightReadyCycle = 0;
        uint64_t psumReadyCycle = 0;
        AmeContextId pendingWeightContext = 0;
        AmeContextId pendingPsumContext = 0;
        uint32_t pendingWeightKBlock = 0;
        std::deque<MacToken> macPipe;
    };

    struct TileOperation
    {
        uint64_t id = 0;
        WaveToken wave;
        unsigned expectedPes = 0;
        unsigned completedPes = 0;
        uint64_t eastReadyCycle = 0;
        uint64_t southReadyCycle = 0;
        bool eastSent = false;
        bool southSent = false;
    };

    const unsigned id;
    const unsigned row;
    const unsigned col;
    const unsigned peRows;
    const unsigned peCols;
    const unsigned peHopLatency;
    const unsigned macLatency;
    const unsigned macII;
    const unsigned operationDepth;
    const bool hasEast;
    const bool hasSouth;
    uint64_t nextIssueCycle = 0;
    uint64_t nextOperationId = 1;
    AmeChannel<WaveToken> westIngress;
    AmeChannel<WaveToken> northIngress;
    AmeChannel<WaveToken> eastOutput;
    AmeChannel<WaveToken> southOutput;
    AmeChannel<WaveToken> completions;
    std::vector<PeState> pes;
    std::deque<TileOperation> operations;
    uint64_t peIssueCount = 0;
    uint64_t peResultCount = 0;

    PeState &pe(unsigned pe_row, unsigned pe_col)
    { return pes.at(pe_row * peCols + pe_col); }
    const PeState &pe(unsigned pe_row, unsigned pe_col) const
    { return pes.at(pe_row * peCols + pe_col); }
    void updateResidentState(uint64_t cycle);
    void updateMacPipelines(uint64_t cycle);
    void startOperation(const WaveToken &wave, uint64_t cycle);

  public:
    MatrixTile(unsigned id_, unsigned row_, unsigned col_,
               unsigned ingress_depth, unsigned edge_depth,
               unsigned pe_rows, unsigned pe_cols,
               unsigned pe_hop_latency, unsigned mac_latency,
               unsigned mac_ii, bool has_east, bool has_south,
               unsigned operation_depth = 0);

    bool canAcceptWestNext() const { return westIngress.canPushNext(); }
    bool canAcceptNorthNext() const { return northIngress.canPushNext(); }
    void acceptWestNext(WaveToken token)
    { westIngress.pushNext(std::move(token)); }
    void acceptNorthNext(WaveToken token)
    { northIngress.pushNext(std::move(token)); }

    bool eastReady() const { return !eastOutput.empty(); }
    bool southReady() const { return !southOutput.empty(); }
    bool completionReady() const { return !completions.empty(); }
    const WaveToken &east() const { return eastOutput.front(); }
    const WaveToken &south() const { return southOutput.front(); }
    const WaveToken &completion() const { return completions.front(); }
    void popEastNext() { eastOutput.popNext(); }
    void popSouthNext() { southOutput.popNext(); }
    void popCompletionNext() { completions.popNext(); }

    void beginWeightPreload(AmeContextId context_id, uint32_t k_block,
                            unsigned valid_rows, unsigned valid_cols,
                            uint64_t cycle);
    bool weightsReady(AmeContextId context_id, uint32_t k_block,
                      unsigned valid_rows, unsigned valid_cols) const;
    void beginPsumSeed(AmeContextId context_id, unsigned valid_rows,
                       unsigned valid_cols, uint64_t cycle);
    bool psumsReady(AmeContextId context_id, unsigned valid_rows,
                    unsigned valid_cols) const;
    void clearResident(AmeContextId context_id);

    void evaluate(uint64_t cycle);
    void advance();
    void cancel(AmeContextId id);
    bool active() const;
    bool transientFor(AmeContextId id) const;
    uint64_t peIssues() const { return peIssueCount; }
    uint64_t peResults() const { return peResultCount; }
};

class SystolicBackend
{
  private:
    struct BoundaryToken
    {
        uint64_t readyCycle = 0;
        WaveToken token;
    };

    struct Work
    {
        AmePeRequest request;
        MatrixPhase phase = MatrixPhase::Allocate;
        uint32_t outputBlocksM = 1;
        uint32_t outputBlocksN = 1;
        uint32_t outputBlocks = 1;
        uint32_t kBlocks = 1;
        uint32_t outputBlock = 0;
        uint32_t kBlock = 0;
        uint32_t kSteps = 1;
        uint32_t injected = 0;
        uint32_t completed = 0;
        uint32_t expected = 0;
        unsigned activeTileRows = 1;
        unsigned activeTileCols = 1;
        unsigned drainsExpected = 0;
        unsigned drainsIssued = 0;
        unsigned drainsCompleted = 0;
        bool weightsStarted = false;
        bool seedStarted = false;
        bool outputSeeded = false;
        bool accumulatorResident = false;
        bool accumulatorDrainStarted = false;
        uint64_t phaseStartCycle = 0;
        bool supplyStalled = false;
        uint64_t supplyStallStart = 0;
    };

    AmeContextTable &contexts;
    MatrixOperandSubsystem &operands;
    AccumulatorSubsystem &accumulator;
    const SystolicArrayParams params;
    AmeChannel<AmePeRequest> commands;
    AmeChannel<TargetDone> done;
    std::vector<MatrixTile> tiles;
    std::vector<std::deque<BoundaryToken>> horizontalBoundaries;
    std::vector<std::deque<BoundaryToken>> verticalBoundaries;
    std::deque<BoundaryToken> drainQueue;
    std::optional<Work> work;
    uint64_t macIssueCount = 0;
    uint64_t tileCompletionCount = 0;
    uint64_t edgeTransferCount = 0;
    uint64_t edgeStallCount = 0;
    uint64_t injectionStallCount = 0;
    uint64_t drainStallCount = 0;

    unsigned tileId(unsigned row, unsigned col) const
    { return row * params.tileCols + col; }
    unsigned horizontalId(unsigned row, unsigned col) const
    { return row * (params.tileCols - 1) + col; }
    unsigned verticalId(unsigned row, unsigned col) const
    { return row * params.tileCols + col; }
    unsigned validRows(const Work &state, unsigned tile_row) const;
    unsigned validCols(const Work &state, unsigned tile_col) const;
    WaveToken tokenForTile(const WaveToken &token, unsigned tile) const;
    void configureOutputBlock(Work &state);
    void evaluateGrid(uint64_t cycle);
    bool injectWavefront(Work &state, uint64_t cycle);
    bool gridIdleFor(AmeContextId id) const;
    void startWork(const AmePeRequest &request, uint64_t cycle);
    void beginTileDrain(Work &state);
    bool evaluateTileDrain(Work &state, uint64_t cycle);
    void finishOutputBlock(Work &state, uint64_t cycle);

  public:
    SystolicBackend(AmeContextTable &contexts_,
                    MatrixOperandSubsystem &operands_,
                    AccumulatorSubsystem &accumulator_,
                    const SystolicArrayParams &params_);

    bool canAcceptNext() const { return commands.canPushNext(); }
    void acceptNext(AmePeRequest request)
    { commands.pushNext(std::move(request)); }
    AmeChannel<TargetDone> &doneChannel() { return done; }
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(AmeContextId id);
    bool active() const;
    uint64_t macIssues() const { return macIssueCount; }
    uint64_t peIssues() const {
        uint64_t count = 0;
        for (const auto &tile : tiles) count += tile.peIssues();
        return count;
    }
    uint64_t peResults() const {
        uint64_t count = 0;
        for (const auto &tile : tiles) count += tile.peResults();
        return count;
    }
    uint64_t tileCompletions() const { return tileCompletionCount; }
    uint64_t edgeTransfers() const { return edgeTransferCount; }
    uint64_t edgeStalls() const { return edgeStallCount; }
    uint64_t injectionStalls() const { return injectionStallCount; }
    uint64_t drainStalls() const { return drainStallCount; }
    bool workActive() const { return work.has_value(); }
    bool commandQueued() const { return !commands.empty(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_SYSTOLIC_ARRAY_HH__
