/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_P1_PIPELINE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_P1_PIPELINE_HH__

#include <deque>

#include "computing_cluster/AME/ame_dispatcher.hh"
#include "computing_cluster/AME/auxiliary_units.hh"
#include "computing_cluster/AME/completion_tracker.hh"
#include "computing_cluster/AME/global_sequencer.hh"
#include "computing_cluster/AME/matrix_lsu.hh"
#include "computing_cluster/AME/systolic_array.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

struct AmePipelineParams
{
    unsigned dispatchQueueDepth = 16;
    unsigned maxInflightInsts = 32;
    unsigned instructionSlots = 16;
    unsigned peRequestQueueDepth = 8;
    unsigned completionQueueDepth = 16;
    unsigned completionWidth = 4;
    unsigned controlLatency = 1;
    unsigned mrfBanks = 8;
    unsigned mrfWordBits = 64;
    unsigned mrfRegisterBytes = 256;
    unsigned mrfBankSkewGroupRegs = 0;
    unsigned mrfBankSkewWords = 0;
    unsigned mrfRequestQueueDepth = 4;
    unsigned mrfResponseQueueDepth = 16;
    unsigned mrfReadLatency = 1;
    unsigned mrfWriteVisibility = 1;
    unsigned accBanks = 4;
    unsigned accWordBits = 128;
    unsigned accRegisterBytes = 256;
    unsigned accRequestQueueDepth = 4;
    unsigned accResponseQueueDepth = 16;
    unsigned accReadLatency = 1;
    unsigned accWriteVisibility = 1;
    OperandSubsystemParams operands;
    SystolicArrayParams systolic;
    MatrixLsuParams mlsu;
    AuxiliaryUnitParams mrsu;
    AuxiliaryUnitParams mptu;
};

class AmePipeline
{
  private:
    struct ControlToken
    {
        uint64_t readyCycle = 0;
        AmeContextId contextId = 0;
    };

    MinorInstQueue &instQueue;
    const AmePipelineParams params;
    AmeContextTable contexts;
    AmeScoreboard scoreboard;
    AmeChannel<AmeInstToken> dispatchQueue;
    AmeChannel<AmePeRequest> peRequests;
    AmeChannel<AmeCompletion> completions;
    BankedRegisterFile mrf;
    BankedRegisterFile accRf;
    MatrixOperandSubsystem operands;
    AccumulatorSubsystem accumulator;
    SystolicBackend systolic;
    MatrixLsu mlsu;
    MatrixAuxiliaryUnit mrsu;
    MatrixAuxiliaryUnit mptu;
    AmeDispatcher dispatcher;
    AmeGlobalSequencer sequencer;
    AmeCompletionTracker completionTracker;
    std::deque<ControlToken> controls;
    uint64_t currentCycle = 0;
    uint64_t completedInstructions = 0;

    void evaluateMinorCompletion();
    void evaluateRegisterResponses();
    void evaluateTargetDone();
    void evaluateFanout();
    void evaluateControl();
    void advance();

  public:
    AmePipeline(MinorInstQueue &inst_queue,
                const AmePipelineParams &params_,
                ConfigSnapshot config_snapshot = {},
                DmaEventReady dma_event_ready = {},
                DmaEventConsume dma_event_consume = {});
    void evaluate();
    void completeTimingResponse(uint64_t transaction_id)
    { mlsu.completeTimingResponse(transaction_id); }
    bool cancel(const InstQueueKey &key);
    bool active() const;
    uint64_t cycles() const { return currentCycle; }
    uint64_t completionCount() const { return completedInstructions; }
    uint64_t mrfGrants() const { return mrf.grantCount(); }
    uint64_t mrfConflicts() const { return mrf.conflictCount(); }
    uint64_t accGrants() const { return accRf.grantCount(); }
    uint64_t accConflicts() const { return accRf.conflictCount(); }
    size_t mlsuOutstanding() const { return mlsu.outstandingSize(); }
    uint64_t saMacIssues() const { return systolic.macIssues(); }
    uint64_t saPeIssues() const { return systolic.peIssues(); }
    uint64_t saPeResults() const { return systolic.peResults(); }
    bool saActive() const { return systolic.active(); }
    bool saWorkActive() const { return systolic.workActive(); }
    bool saCommandQueued() const { return systolic.commandQueued(); }
    bool mlsuActive() const { return mlsu.active(); }
    uint64_t saTileCompletions() const
    { return systolic.tileCompletions(); }
    uint64_t saEdgeTransfers() const { return systolic.edgeTransfers(); }
    uint64_t saEdgeStalls() const { return systolic.edgeStalls(); }
    uint64_t saInjectionStalls() const { return systolic.injectionStalls(); }
    uint64_t saDrainStalls() const { return systolic.drainStalls(); }
    uint64_t mrsuLocalCycles() const { return mrsu.localCycles(); }
    uint64_t mrsuCrossTileCycles() const { return mrsu.crossTileCycles(); }
    uint64_t mrsuPrefixCycles() const { return mrsu.prefixCycles(); }
    uint64_t mrsuComputeItems() const { return mrsu.computeItems(); }
    uint64_t mptuComputeItems() const { return mptu.computeItems(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_P1_PIPELINE_HH__
