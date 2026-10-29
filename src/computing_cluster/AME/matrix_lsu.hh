/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_MATRIX_LSU_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_MATRIX_LSU_HH__

#include <deque>
#include <functional>
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

struct MatrixLsuParams
{
    struct TimingRequest
    {
        AmeContextId contextId = 0;
        uint64_t transactionId = 0;
        Addr address = 0;
        uint16_t bytes = 0;
        bool isStore = false;
    };

    using TimingRequestSender = std::function<bool(const TimingRequest &)>;

    unsigned descriptorQueueDepth = 4;
    unsigned packetBytes = 64;
    unsigned maxOutstanding = 16;
    unsigned responseLatency = 8;
    unsigned reorderWindow = 4;
    unsigned responseWidth = 1;
    TimingRequestSender sendTimingRequest;
};

struct MatrixBurstDesc
{
    struct Transfer
    {
        struct Mapping
        {
            uint16_t packetOffset = 0;
            uint32_t mOffset = 0;
            uint16_t bytes = 0;
        };

        uint32_t memoryOffset = 0;
        uint16_t bytes = 0;
        std::vector<Mapping> mappings;
    };

    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    uint64_t descId = 0;
    Addr base = 0;
    uint32_t bytes = 0;
    uint16_t elementBytes = 1;
    bool isLoad = false;
    bool isStore = false;
    uint16_t mBase = 0;
    uint64_t stride = 0;
    std::vector<Transfer> transfers;
};

/** Timing M-LSU. Architectural memory effects occur at Minor commit. */
class MatrixLsu
{
  private:
    struct Transaction
    {
        AmeContextId contextId = 0;
        uint64_t id = 0;
        uint32_t offset = 0;
        uint16_t bytes = 0;
        std::vector<MatrixBurstDesc::Transfer::Mapping> mappings;
        unsigned mappingIndex = 0;
        uint16_t mappingBytesIssued = 0;
        uint64_t readyCycle = 0;
    };

    struct Work
    {
        MatrixBurstDesc desc;
        unsigned packetCount = 0;
        unsigned storeReadsIssued = 0;
        unsigned storeReadsDone = 0;
        unsigned packetsIssued = 0;
        unsigned packetsReturned = 0;
        unsigned nextTransfer = 0;
        uint32_t storeBytesIssued = 0;
        uint32_t loadBytesIssued = 0;
        uint32_t loadBytesDone = 0;
        std::vector<bool> storeBytesReady;
        std::deque<Transaction> loadWriteQueue;
        bool killed = false;
    };

    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    BankedRegisterFile &mrf;
    const MatrixLsuParams params;
    AmeChannel<AmePeRequest> requests;
    AmeChannel<TargetDone> done;
    std::map<AmeContextId, Work> works;
    std::deque<Transaction> outstanding;
    std::deque<Transaction> returnedTransactions;
    uint64_t nextDescId = 1;
    uint64_t nextTxnId = 1;

    void acceptCommands();
    void issueStoreRead(Work &work);
    void issueLoadWrite(Work &work);
    void issueTransaction(Work &work, uint64_t cycle);
    void returnTransaction(const Transaction &transaction, uint64_t cycle);
    void buildTransfers(Work &work);
    bool storeTransferReady(const Work &work,
                            const MatrixBurstDesc::Transfer &transfer) const;
    MRegion mapMByte(const Work &work, uint32_t byte_offset,
                     uint32_t max_bytes) const;

  public:
    MatrixLsu(AmeContextTable &contexts_, AmeScoreboard &scoreboard_,
              BankedRegisterFile &mrf_, const MatrixLsuParams &params_);
    bool canAcceptNext() const { return requests.canPushNext(); }
    void acceptNext(AmePeRequest request)
    { requests.pushNext(std::move(request)); }
    AmeChannel<TargetDone> &doneChannel() { return done; }
    void acceptResponse(const BankResponse &response);
    void completeTimingResponse(uint64_t transaction_id);
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(AmeContextId id);
    bool active() const;
    size_t outstandingSize() const
    { return outstanding.size() + returnedTransactions.size(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_MATRIX_LSU_HH__
