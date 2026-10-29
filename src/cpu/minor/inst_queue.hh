/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __CPU_MINOR_INST_QUEUE_HH__
#define __CPU_MINOR_INST_QUEUE_HH__

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "base/named.hh"
#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "cpu/op_class.hh"
#include "cpu/reg_class.hh"
#include "cpu/static_inst.hh"
#include "sim/faults.hh"

namespace gem5
{

class ThreadContext;

namespace minor
{

/** Engine selected by Minor's instruction offload boundary. */
enum class InstQueueEngine : uint8_t
{
    RVV,
    AME
};

/** Stable identity used from dispatch until completion. */
struct InstQueueKey
{
    ThreadID threadId = InvalidThreadID;
    InstSeqNum execSeqNum = 0;

    bool
    operator<(const InstQueueKey &other) const
    {
        return threadId < other.threadId ||
            (threadId == other.threadId && execSeqNum < other.execSeqNum);
    }

    bool
    operator==(const InstQueueKey &other) const
    {
        return threadId == other.threadId &&
            execSeqNum == other.execSeqNum;
    }
};

/** Scalar input copied across the MinorCPU/compute-engine boundary. */
struct InstQueueScalarOperand
{
    unsigned operandIndex = 0;
    RegId reg;
    RegVal value = 0;
};

/**
 * A compact snapshot of an implicit matrix datatype register.
 *
 * AME matrix data remains in the accelerator-owned register file.  Datatype
 * pseudo-registers are different: O3 renames them to order msettyp/asettyp
 * against later AME commands, so the command packet must carry the value read
 * from that physical version instead of consulting architectural state later.
 */
struct InstQueueMatrixTypeOperand
{
    unsigned operandIndex = 0;
    RegId reg;
    uint32_t value = 0;
};

/**
 * One architectural compute instruction submitted by Minor Execute.
 *
 * Register identifiers describe the architectural contract. Vector and
 * matrix values deliberately do not cross this boundary: they belong to the
 * RVV/AME component. Scalar source values are snapshotted at issue.
 */
struct InstQueuePacket
{
    InstQueueKey key;
    InstQueueEngine engine = InstQueueEngine::RVV;
    Addr pc = 0;
    uint64_t encodedInst = 0;
    std::string mnemonic;
    OpClass opClass = No_OpClass;
    bool isMemory = false;
    bool isMicroop = false;
    bool isFirstMicroop = false;
    bool isLastMicroop = false;
    bool vectorLengthValid = false;
    uint32_t vectorLength = 0;
    std::vector<RegId> srcRegs;
    std::vector<RegId> destRegs;
    std::vector<InstQueueScalarOperand> scalarSrcs;
    std::vector<InstQueueMatrixTypeOperand> matrixTypeSrcs;
};

/** Scalar architectural result returned by an RVV/AME component. */
struct InstQueueScalarResult
{
    /** Index in InstQueuePacket::destRegs, not a physical register number. */
    unsigned destOperandIndex = 0;
    RegVal value = 0;
};

/** Completion returned by the component which consumed a packet. */
struct InstQueueCompletion
{
    InstQueueKey key;
    Fault fault = NoFault;
    std::vector<InstQueueScalarResult> scalarResults;
};

/**
 * MinorCPU boundary queue for decoupled RVV and AME execution.
 *
 * A consumer observes dispatch availability through dispatchNotify, removes
 * packets with pop(), and eventually calls complete(). Completion may be out
 * of order, but Minor Execute retires it in architectural program order.
 */
class MinorInstQueue : public Named
{
  public:
    using Notify = std::function<void()>;
    using CancelNotify = std::function<void(const InstQueueKey &)>;

  private:
    const unsigned capacity;
    const bool offloadRVV;
    const bool offloadAME;

    std::deque<InstQueuePacket> dispatchQueue;
    std::map<InstQueueKey, InstQueuePacket> inflight;
    std::map<InstQueueKey, InstQueueCompletion> completions;
    std::set<InstQueueKey> unissuedMemoryPackets;

    std::vector<Notify> dispatchNotifies;
    std::vector<CancelNotify> cancelNotifies;
    Notify completionNotify;

  public:
    MinorInstQueue(const std::string &name, unsigned capacity,
                   bool offload_rvv, bool offload_ame,
                   Notify completion_notify);

    static bool isRVV(const StaticInstPtr &inst);
    static bool isAME(const StaticInstPtr &inst);
    static bool isOffloadCandidate(const StaticInstPtr &inst,
                                   bool offload_rvv, bool offload_ame);

    bool enabled() const { return offloadRVV || offloadAME; }
    bool isOffloadable(const StaticInstPtr &inst) const;
    bool canDispatch() const { return inflight.size() < capacity; }

    InstQueuePacket makePacket(const StaticInstPtr &inst,
                               ThreadContext *tc, ThreadID tid,
                               InstSeqNum exec_seq_num, Addr pc) const;
    InstQueuePacket makePacket(
        const StaticInstPtr &inst, ThreadID tid, InstSeqNum exec_seq_num,
        Addr pc, const std::function<RegVal(unsigned)> &scalar_reader,
        const std::function<uint32_t(unsigned)> &matrix_type_reader = {}) const;
    bool dispatch(InstQueuePacket packet);

    bool hasPending() const { return !dispatchQueue.empty(); }
    bool hasPending(InstQueueEngine engine) const;
    const InstQueuePacket &front() const;
    const InstQueuePacket &front(InstQueueEngine engine) const;
    InstQueuePacket pop();
    InstQueuePacket pop(InstQueueEngine engine);

    bool complete(InstQueueCompletion completion);
    bool hasPacket(const InstQueueKey &key) const;
    bool hasCompletion(const InstQueueKey &key) const;
    const InstQueueCompletion &getCompletion(const InstQueueKey &key) const;

    /** Preserve LSQ program order across the external execution boundary. */
    bool hasOlderUnissuedMemory(ThreadID tid, InstSeqNum exec_seq_num) const;
    void markMemoryIssued(const InstQueueKey &key);

    void retire(const InstQueueKey &key);
    void cancel(const InstQueueKey &key);
    void cancelAll();

    bool empty() const { return inflight.empty(); }
    size_t pendingSize() const { return dispatchQueue.size(); }
    size_t pendingSize(InstQueueEngine engine) const;
    size_t inflightSize() const { return inflight.size(); }

    void setDispatchNotify(Notify notify)
    {
        dispatchNotifies.clear();
        dispatchNotifies.push_back(std::move(notify));
    }

    void addDispatchNotify(Notify notify)
    {
        dispatchNotifies.push_back(std::move(notify));
    }

    /** Register an engine callback invoked synchronously when Minor flushes. */
    void addCancelNotify(CancelNotify notify)
    {
        cancelNotifies.push_back(std::move(notify));
    }
};

} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_INST_QUEUE_HH__
