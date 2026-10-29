/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_MEMORY_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_MEMORY_HH__

#include <deque>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_context.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

struct VectorMemoryParams
{
    struct TimingRequest
    {
        uint64_t transactionId = 0;
        Addr address = 0;
        unsigned bytes = 0;
        bool isStore = false;
    };

    using TimingRequestSender =
        std::function<bool(const TimingRequest &)>;

    unsigned numLanes = 4;
    unsigned maxOutstanding = 16;
    unsigned lineBytes = 64;
    unsigned responseLatency = 8;
    unsigned reorderWindow = 4;
    unsigned responseWidth = 1;
    TimingRequestSender sendTimingRequest;
};

/** P1 timing VLSU.  Architectural memory effects remain in Minor's LSQ. */
class VectorMemoryUnit
{
  private:
    struct WordTag
    {
        RvvInstId instId = 0;
        uint32_t wordIndex = 0;

        bool operator<(const WordTag &other) const
        {
            return instId < other.instId ||
                (instId == other.instId && wordIndex < other.wordIndex);
        }
    };

    struct Batch
    {
        uint64_t expectedMask = 0;
        std::map<LaneId, VectorFuReq> requests;
    };

    struct Operation
    {
        VectorFuResp response;
        unsigned pendingTransactions = 0;
    };

    struct Registration
    {
        uint64_t activeLaneMask = 0;
        uint64_t pendingFinalLaneMask = 0;
    };

    struct Transaction
    {
        uint64_t id = 0;
        Addr line = 0;
        unsigned bytes = 0;
        bool isStore = false;
        uint64_t readyCycle = 0;
        std::vector<LaneTag> consumers;
    };

    InstContextTable &contexts;
    RvvChannel<VectorFuReq> &requests;
    RvvChannel<VectorFuResp> &results;
    const VectorMemoryParams params;
    std::map<WordTag, Batch> batches;
    std::map<RvvInstId, Registration> registeredInsts;
    std::vector<std::pair<RvvInstId, uint64_t>> pendingInsts;
    std::map<LaneTag, Operation> operations;
    std::deque<Transaction> pendingTransactions;
    std::vector<Transaction> outstanding;
    uint64_t nextTxnId = 1;

    Addr elementAddress(const VectorFuReq &request,
                        const RvvInstDesc &desc, unsigned element) const;
    uint32_t laneWordCount(const RvvInstDesc &desc, LaneId lane) const;
    uint64_t expectedLaneMask(const RvvInstDesc &desc,
                              uint32_t word_index) const;
    bool hasPendingWork(RvvInstId id) const;
    void completeTransaction(uint64_t transaction_id);

  public:
    VectorMemoryUnit(InstContextTable &contexts_,
                     RvvChannel<VectorFuReq> &requests_,
                     RvvChannel<VectorFuResp> &results_,
                     const VectorMemoryParams &params_);

    void registerBeatNext(const VecBeat &beat);
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(RvvInstId id);
    void completeTimingResponse(uint64_t transaction_id);
    bool active() const;
    size_t outstandingSize() const { return outstanding.size(); }
    size_t pendingTransactionSize() const
    { return pendingTransactions.size(); }
    uint64_t issuedTransactions() const { return nextTxnId - 1; }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_MEMORY_HH__
