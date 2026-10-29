/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/register_file.hh"

#include <algorithm>
#include <array>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

BankedRegisterFile::Bank::Bank(unsigned id_, unsigned queue_depth)
    : id(id_), highReads(queue_depth), highWrites(queue_depth),
      lowReads(queue_depth), lowWrites(queue_depth)
{
}

bool
BankedRegisterFile::Bank::canAcceptNext(const BankRequest &request) const
{
    if (request.highPriority)
        return request.access == BankAccess::Read ?
            highReads.canPushNext() : highWrites.canPushNext();
    return request.access == BankAccess::Read ?
        lowReads.canPushNext() : lowWrites.canPushNext();
}

void
BankedRegisterFile::Bank::acceptNext(BankRequest request)
{
    panic_if(request.bank != id, "AME request delivered to wrong bank");
    if (request.highPriority) {
        if (request.access == BankAccess::Read)
            highReads.pushNext(std::move(request));
        else
            highWrites.pushNext(std::move(request));
    } else if (request.access == BankAccess::Read) {
        lowReads.pushNext(std::move(request));
    } else {
        lowWrites.pushNext(std::move(request));
    }
}

bool
BankedRegisterFile::Bank::grant(BankRequest &request)
{
    std::array<AmeChannel<BankRequest> *, 4> queues = {
        &highReads, &highWrites, &lowReads, &lowWrites};
    for (unsigned priority = 0; priority < 2; ++priority) {
        unsigned &round_robin = priority == 0 ? highRoundRobin : lowRoundRobin;
        const unsigned base = priority * 2;
        for (unsigned offset = 0; offset < 2; ++offset) {
            const unsigned local = (round_robin + offset) % 2;
            auto *queue = queues[base + local];
            if (queue->empty())
                continue;
            request = queue->front();
            queue->popNext();
            round_robin = (local + 1) % 2;
            return true;
        }
    }
    return false;
}

void
BankedRegisterFile::Bank::advance()
{
    highReads.advance();
    highWrites.advance();
    lowReads.advance();
    lowWrites.advance();
}

void
BankedRegisterFile::Bank::cancel(AmeContextId context_id)
{
    const auto killed = [context_id](const BankRequest &request) {
        return request.contextId == context_id;
    };
    highReads.eraseIf(killed);
    highWrites.eraseIf(killed);
    lowReads.eraseIf(killed);
    lowWrites.eraseIf(killed);
}

bool
BankedRegisterFile::Bank::active() const
{
    return highReads.active() || highWrites.active() || lowReads.active() ||
        lowWrites.active();
}

bool
BankedRegisterFile::Bank::contended() const
{
    return highReads.size() + highWrites.size() + lowReads.size() +
        lowWrites.size() > 1;
}

BankedRegisterFile::BankedRegisterFile(
    RegisterFileKind kind_, unsigned num_banks, unsigned word_bits,
    unsigned register_bytes, unsigned request_queue_depth,
    unsigned response_queue_depth, unsigned read_latency,
    unsigned write_visibility, unsigned bank_skew_group_regs,
    unsigned bank_skew_words)
    : kind(kind_), numBanks(num_banks), wordBytes(word_bits / 8),
      registerBytes(register_bytes),
      bankSkewGroupRegs(bank_skew_group_regs),
      bankSkewWords(bank_skew_words), readLatency(read_latency),
      writeVisibility(write_visibility), responses(response_queue_depth)
{
    fatal_if(numBanks == 0 || word_bits == 0 || word_bits % 8 != 0 ||
             registerBytes == 0 || readLatency == 0 || writeVisibility == 0,
             "Invalid AME banked register file parameters");
    banks.reserve(numBanks);
    for (unsigned bank = 0; bank < numBanks; ++bank)
        banks.emplace_back(bank, request_queue_depth);
}

BankRequest
BankedRegisterFile::map(const LogicalBankRequest &request) const
{
    panic_if(request.file != kind, "AME register file kind mismatch");
    const uint64_t absolute_byte =
        static_cast<uint64_t>(request.reg) * registerBytes +
        request.byteOffset;
    panic_if(request.byteCount == 0 || request.byteCount > wordBytes ||
             request.byteOffset + request.byteCount > registerBytes,
             "Invalid AME register fragment reg=%u offset=%u bytes=%u",
             request.reg, request.byteOffset, request.byteCount);
    panic_if(absolute_byte % wordBytes + request.byteCount > wordBytes,
             "AME register fragment crosses a bank word reg=%u "
             "offset=%u bytes=%u", request.reg, request.byteOffset,
             request.byteCount);
    const uint64_t word_address = absolute_byte / wordBytes;
    const uint64_t bank_skew = bankSkewGroupRegs == 0 ? 0 :
        (request.reg / bankSkewGroupRegs) * bankSkewWords;
    BankRequest mapped;
    static_cast<LogicalBankRequest &>(mapped) = request;
    /* K128 consumes A and B as consecutive eight-register groups.  A small
     * group-dependent cyclic skew prevents equal offsets in the two operands
     * from requesting the same single-port bank in the same cycle.  Row stays
     * based on the unskewed linear word address, so the mapping is bijective
     * within every register row. */
    mapped.bank = (word_address + bank_skew) % numBanks;
    mapped.row = word_address / numBanks;
    return mapped;
}

bool
BankedRegisterFile::canAcceptNext(const LogicalBankRequest &request) const
{
    const auto mapped = map(request);
    return banks[mapped.bank].canAcceptNext(mapped);
}

void
BankedRegisterFile::acceptNext(LogicalBankRequest request)
{
    auto mapped = map(request);
    mapped.sequence = nextSequence++;
    banks[mapped.bank].acceptNext(std::move(mapped));
}

void
BankedRegisterFile::evaluate(uint64_t cycle)
{
    for (auto it = delayed.begin(); it != delayed.end();) {
        if (it->readyCycle > cycle || !responses.canPushNext()) {
            ++it;
            continue;
        }
        responses.pushNext(it->response);
        it = delayed.erase(it);
    }

    for (auto &bank : banks) {
        BankRequest request;
        if (!bank.grant(request))
            continue;
        ++grants;
        BankResponse response;
        response.contextId = request.contextId;
        response.ameId = request.ameId;
        response.file = request.file;
        response.access = request.access;
        response.client = request.client;
        response.reg = request.reg;
        response.byteOffset = request.byteOffset;
        response.byteCount = request.byteCount;
        response.bank = request.bank;
        response.row = request.row;
        response.final = request.final;
        response.tag = request.tag;
        const unsigned latency = request.access == BankAccess::Read ?
            readLatency : writeVisibility;
        delayed.push_back(DelayedResponse{cycle + latency, response});
        DPRINTF(AMEFlow,
                "cycle=%llu inst=%llu slot=%u %s_GRANT bank=%u row=%u "
                "offset=%u bytes=%u client=%u\n",
                cycle, request.contextId, request.ameId,
                request.access == BankAccess::Read ? "READ" : "WRITE",
                request.bank, request.row, request.byteOffset,
                request.byteCount, static_cast<unsigned>(request.client));
    }

    for (const auto &bank : banks) {
        if (bank.contended())
            ++conflicts;
    }
}

void
BankedRegisterFile::advance()
{
    for (auto &bank : banks)
        bank.advance();
    responses.advance();
}

void
BankedRegisterFile::cancel(AmeContextId context_id)
{
    for (auto &bank : banks)
        bank.cancel(context_id);
    responses.eraseIf([context_id](const BankResponse &response) {
        return response.contextId == context_id;
    });
    delayed.erase(std::remove_if(delayed.begin(), delayed.end(),
        [context_id](const DelayedResponse &response) {
            return response.response.contextId == context_id;
        }), delayed.end());
}

bool
BankedRegisterFile::active() const
{
    return !delayed.empty() || responses.active() ||
        std::any_of(banks.begin(), banks.end(),
            [](const Bank &bank) { return bank.active(); });
}

} // namespace ame
} // namespace minor
} // namespace gem5
