/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_REGISTER_FILE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_REGISTER_FILE_HH__

#include <cstdint>
#include <deque>
#include <map>
#include <vector>

#include "computing_cluster/AME/ame_channel.hh"
#include "computing_cluster/AME/ame_types.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

enum class RegisterFileKind : uint8_t
{
    Mrf,
    Acc
};

enum class BankAccess : uint8_t
{
    Read,
    Write
};

enum class BankClient : uint8_t
{
    SaA,
    SaB,
    SaAcc,
    SaWriteback,
    Mlsu,
    Mrsu,
    Mptu
};

struct LogicalBankRequest
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    RegisterFileKind file = RegisterFileKind::Mrf;
    BankAccess access = BankAccess::Read;
    BankClient client = BankClient::SaA;
    uint16_t reg = 0;
    uint32_t byteOffset = 0;
    uint16_t byteCount = 0;
    bool highPriority = false;
    bool final = false;
    MFragmentTag tag;
};

struct BankRequest : public LogicalBankRequest
{
    uint16_t bank = 0;
    uint32_t row = 0;
    uint64_t sequence = 0;
};

struct BankResponse
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    RegisterFileKind file = RegisterFileKind::Mrf;
    BankAccess access = BankAccess::Read;
    BankClient client = BankClient::SaA;
    uint16_t reg = 0;
    uint32_t byteOffset = 0;
    uint16_t byteCount = 0;
    uint16_t bank = 0;
    uint32_t row = 0;
    bool final = false;
    MFragmentTag tag;
};

class BankedRegisterFile
{
  private:
    struct DelayedResponse
    {
        uint64_t readyCycle = 0;
        BankResponse response;
    };

    class Bank
    {
      private:
        const unsigned id;
        AmeChannel<BankRequest> highReads;
        AmeChannel<BankRequest> highWrites;
        AmeChannel<BankRequest> lowReads;
        AmeChannel<BankRequest> lowWrites;
        unsigned highRoundRobin = 0;
        unsigned lowRoundRobin = 0;

      public:
        Bank(unsigned id_, unsigned queue_depth);
        bool canAcceptNext(const BankRequest &request) const;
        void acceptNext(BankRequest request);
        bool grant(BankRequest &request);
        void advance();
        void cancel(AmeContextId id);
        bool active() const;
        bool contended() const;
    };

    const RegisterFileKind kind;
    const unsigned numBanks;
    const unsigned wordBytes;
    const unsigned registerBytes;
    const unsigned bankSkewGroupRegs;
    const unsigned bankSkewWords;
    const unsigned readLatency;
    const unsigned writeVisibility;
    std::vector<Bank> banks;
    AmeChannel<BankResponse> responses;
    std::deque<DelayedResponse> delayed;
    uint64_t nextSequence = 1;
    uint64_t grants = 0;
    uint64_t conflicts = 0;

    BankRequest map(const LogicalBankRequest &request) const;

  public:
    BankedRegisterFile(RegisterFileKind kind_, unsigned num_banks,
                       unsigned word_bits, unsigned register_bytes,
                       unsigned request_queue_depth,
                       unsigned response_queue_depth,
                       unsigned read_latency, unsigned write_visibility,
                       unsigned bank_skew_group_regs = 0,
                       unsigned bank_skew_words = 0);

    bool canAcceptNext(const LogicalBankRequest &request) const;
    void acceptNext(LogicalBankRequest request);
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(AmeContextId id);
    bool active() const;
    unsigned fragmentBytes() const { return wordBytes; }
    unsigned bytesPerRegister() const { return registerBytes; }
    AmeChannel<BankResponse> &responseChannel() { return responses; }
    const AmeChannel<BankResponse> &responseChannel() const { return responses; }
    uint64_t grantCount() const { return grants; }
    uint64_t conflictCount() const { return conflicts; }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_REGISTER_FILE_HH__
