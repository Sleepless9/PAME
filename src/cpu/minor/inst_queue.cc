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

#include "cpu/minor/inst_queue.hh"

#include <algorithm>
#include <utility>

#include "base/logging.hh"
#include "arch/riscv/regs/matrix.hh"
#include "cpu/thread_context.hh"
#include "debug/MinorInstQueue.hh"

namespace gem5
{

namespace minor
{

MinorInstQueue::MinorInstQueue(const std::string &name, unsigned capacity_,
    bool offload_rvv, bool offload_ame, Notify completion_notify)
    : Named(name), capacity(capacity_), offloadRVV(offload_rvv),
      offloadAME(offload_ame),
      completionNotify(std::move(completion_notify))
{
    if (capacity == 0)
        fatal("%s: instruction queue capacity must be non-zero", name);
}

bool
MinorInstQueue::isRVV(const StaticInstPtr &inst)
{
    /* Minor Decode owns macro-op decomposition.  Sending a RISC-V vector
     * macro-op across the queue would make host commit call the deliberate
     * RiscvMacroInst::execute() panic; queue only the executable micro-op. */
    return inst && inst->isVector() && !inst->isMacroop();
}

bool
MinorInstQueue::isAME(const StaticInstPtr &inst)
{
    if (!inst)
        return false;

    const std::string name = inst->getName();
    return inst->isMatrix() || name.rfind("ztt_", 0) == 0;
}

bool
MinorInstQueue::isOffloadCandidate(const StaticInstPtr &inst,
    bool offload_rvv, bool offload_ame)
{
    return (offload_rvv && isRVV(inst)) ||
        (offload_ame && isAME(inst));
}

bool
MinorInstQueue::isOffloadable(const StaticInstPtr &inst) const
{
    return isOffloadCandidate(inst, offloadRVV, offloadAME);
}

InstQueuePacket
MinorInstQueue::makePacket(const StaticInstPtr &inst, ThreadContext *tc,
    ThreadID tid, InstSeqNum exec_seq_num, Addr pc) const
{
    assert(tc);
    return makePacket(inst, tid, exec_seq_num, pc,
        [inst, tc](unsigned operand) {
            return tc->getReg(inst->srcRegIdx(operand));
        },
        [inst, tc](unsigned operand) {
            RiscvISA::MatRegContainer value;
            value.zero();
            tc->getReg(inst->srcRegIdx(operand), &value);
            const auto rows = value.as<uint8_t>();
            uint32_t raw = 0;
            for (unsigned byte = 0; byte < sizeof(raw); ++byte)
                raw |= static_cast<uint32_t>(rows[0][byte]) << (byte * 8);
            return raw;
        });
}

InstQueuePacket
MinorInstQueue::makePacket(
    const StaticInstPtr &inst, ThreadID tid, InstSeqNum exec_seq_num,
    Addr pc, const std::function<RegVal(unsigned)> &scalar_reader,
    const std::function<uint32_t(unsigned)> &matrix_type_reader) const
{
    assert(inst);
    assert(scalar_reader);
    assert(isOffloadable(inst));

    InstQueuePacket packet;
    packet.key = {tid, exec_seq_num};
    packet.engine = isAME(inst) ? InstQueueEngine::AME :
                                  InstQueueEngine::RVV;
    packet.pc = pc;
    packet.encodedInst = inst->getEMI();
    packet.mnemonic = inst->getName();
    packet.opClass = inst->opClass();
    packet.isMemory = inst->isMemRef();
    packet.isMicroop = inst->isMicroop();
    packet.isFirstMicroop = inst->isFirstMicroop();
    packet.isLastMicroop = inst->isLastMicroop();
    packet.vectorLengthValid = inst->hasVectorLength();
    if (packet.vectorLengthValid)
        packet.vectorLength = inst->getVectorLength();

    for (unsigned i = 0; i < inst->numSrcRegs(); ++i) {
        const RegId reg = inst->srcRegIdx(i);
        packet.srcRegs.push_back(reg);
        if (reg.is(IntRegClass))
            packet.scalarSrcs.push_back({i, reg, scalar_reader(i)});
        if (matrix_type_reader && reg.is(MatRegClass) &&
            ((reg.index() >= RiscvISA::ZttMatDataRegBase &&
              reg.index() < RiscvISA::ZttMatDataRegBase +
                                RiscvISA::NumZttMatDataRegs) ||
             (reg.index() >= RiscvISA::ZttAccDataRegBase &&
              reg.index() < RiscvISA::ZttAccDataRegBase +
                                RiscvISA::NumZttAccDataRegs))) {
            packet.matrixTypeSrcs.push_back(
                {i, reg, matrix_type_reader(i)});
        }
    }

    for (unsigned i = 0; i < inst->numDestRegs(); ++i)
        packet.destRegs.push_back(inst->destRegIdx(i));

    return packet;
}

bool
MinorInstQueue::dispatch(InstQueuePacket packet)
{
    if (!canDispatch())
        return false;

    const InstQueueKey key = packet.key;
    auto [it, inserted] = inflight.emplace(key, packet);
    if (!inserted) {
        panic("%s: duplicate dispatch for tid=%d exec=%llu", name(),
              key.threadId, key.execSeqNum);
    }
    if (it->second.isMemory)
        unissuedMemoryPackets.insert(key);

    dispatchQueue.push_back(std::move(packet));
    DPRINTF(MinorInstQueue,
            "Dispatch tid=%d exec=%llu engine=%s pc=%#x inst=%s\n",
            key.threadId, key.execSeqNum,
            it->second.engine == InstQueueEngine::RVV ? "RVV" : "AME",
            it->second.pc, it->second.mnemonic);

    for (const auto &notify : dispatchNotifies) {
        if (notify)
            notify();
    }

    return true;
}

const InstQueuePacket &
MinorInstQueue::front() const
{
    assert(!dispatchQueue.empty());
    return dispatchQueue.front();
}

const InstQueuePacket &
MinorInstQueue::front(InstQueueEngine engine) const
{
    const auto it = std::find_if(dispatchQueue.begin(), dispatchQueue.end(),
        [engine](const InstQueuePacket &packet) {
            return packet.engine == engine;
        });
    assert(it != dispatchQueue.end());
    return *it;
}

InstQueuePacket
MinorInstQueue::pop()
{
    assert(!dispatchQueue.empty());
    InstQueuePacket packet = std::move(dispatchQueue.front());
    dispatchQueue.pop_front();
    return packet;
}

bool
MinorInstQueue::hasPending(InstQueueEngine engine) const
{
    return std::any_of(dispatchQueue.begin(), dispatchQueue.end(),
        [engine](const InstQueuePacket &packet) {
            return packet.engine == engine;
        });
}

InstQueuePacket
MinorInstQueue::pop(InstQueueEngine engine)
{
    const auto it = std::find_if(dispatchQueue.begin(), dispatchQueue.end(),
        [engine](const InstQueuePacket &packet) {
            return packet.engine == engine;
        });
    assert(it != dispatchQueue.end());
    InstQueuePacket packet = std::move(*it);
    dispatchQueue.erase(it);
    return packet;
}

size_t
MinorInstQueue::pendingSize(InstQueueEngine engine) const
{
    return std::count_if(dispatchQueue.begin(), dispatchQueue.end(),
        [engine](const InstQueuePacket &packet) {
            return packet.engine == engine;
        });
}

bool
MinorInstQueue::complete(InstQueueCompletion completion)
{
    const InstQueueKey key = completion.key;
    if (inflight.find(key) == inflight.end()) {
        DPRINTF(MinorInstQueue,
                "Ignoring stale completion tid=%d exec=%llu\n",
                key.threadId, key.execSeqNum);
        return false;
    }

    auto [it, inserted] = completions.emplace(key, std::move(completion));
    if (!inserted) {
        warn("%s: duplicate completion for tid=%d exec=%llu ignored",
             name(), key.threadId, key.execSeqNum);
        return false;
    }

    DPRINTF(MinorInstQueue,
            "Complete tid=%d exec=%llu scalar_results=%u fault=%s\n",
            key.threadId, key.execSeqNum, it->second.scalarResults.size(),
            it->second.fault == NoFault ? "none" : it->second.fault->name());

    if (completionNotify)
        completionNotify();

    return true;
}

bool
MinorInstQueue::hasPacket(const InstQueueKey &key) const
{
    return inflight.find(key) != inflight.end();
}

bool
MinorInstQueue::hasCompletion(const InstQueueKey &key) const
{
    return completions.find(key) != completions.end();
}

const InstQueueCompletion &
MinorInstQueue::getCompletion(const InstQueueKey &key) const
{
    const auto it = completions.find(key);
    if (it == completions.end()) {
        panic("%s: no completion for tid=%d exec=%llu", name(),
              key.threadId, key.execSeqNum);
    }
    return it->second;
}

bool
MinorInstQueue::hasOlderUnissuedMemory(
    ThreadID tid, InstSeqNum exec_seq_num) const
{
    const auto it = unissuedMemoryPackets.lower_bound({tid, 0});
    return it != unissuedMemoryPackets.end() && it->threadId == tid &&
        it->execSeqNum < exec_seq_num;
}

void
MinorInstQueue::markMemoryIssued(const InstQueueKey &key)
{
    const auto packet = inflight.find(key);
    assert(packet != inflight.end() && packet->second.isMemory);
    const size_t erased = unissuedMemoryPackets.erase(key);
    assert(erased == 1);
    DPRINTF(MinorInstQueue,
            "Memory issued to LSQ tid=%d exec=%llu\n",
            key.threadId, key.execSeqNum);
}

void
MinorInstQueue::retire(const InstQueueKey &key)
{
    assert(hasCompletion(key));
    completions.erase(key);
    unissuedMemoryPackets.erase(key);
    inflight.erase(key);
    DPRINTF(MinorInstQueue, "Retire tid=%d exec=%llu\n",
            key.threadId, key.execSeqNum);
}

void
MinorInstQueue::cancel(const InstQueueKey &key)
{
    if (!hasPacket(key)) {
        DPRINTF(MinorInstQueue,
                "Ignoring stale cancel tid=%d exec=%llu\n",
                key.threadId, key.execSeqNum);
        return;
    }

    dispatchQueue.erase(
        std::remove_if(dispatchQueue.begin(), dispatchQueue.end(),
            [&key](const InstQueuePacket &packet) {
                return packet.key.threadId == key.threadId &&
                    packet.key.execSeqNum == key.execSeqNum;
            }),
        dispatchQueue.end());
    completions.erase(key);
    unissuedMemoryPackets.erase(key);
    inflight.erase(key);
    for (const auto &notify : cancelNotifies) {
        if (notify)
            notify(key);
    }
    DPRINTF(MinorInstQueue, "Cancel tid=%d exec=%llu\n",
            key.threadId, key.execSeqNum);
}

void
MinorInstQueue::cancelAll()
{
    std::vector<InstQueueKey> keys;
    keys.reserve(inflight.size());
    for (const auto &[key, packet] : inflight)
        keys.push_back(key);

    for (const auto &key : keys)
        cancel(key);
}

} // namespace minor
} // namespace gem5
