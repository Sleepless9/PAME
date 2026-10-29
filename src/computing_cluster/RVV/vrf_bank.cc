/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vrf_bank.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

VrfArbitrationPolicy
parsePolicy(const std::string &policy)
{
    if (policy == "write_priority")
        return VrfArbitrationPolicy::WritePriority;
    if (policy == "round_robin")
        return VrfArbitrationPolicy::RoundRobin;
    if (policy == "oldest")
        return VrfArbitrationPolicy::Oldest;
    if (policy == "two_level_priority")
        return VrfArbitrationPolicy::TwoLevelPriority;
    fatal("Unknown RVV VRF arbitration policy '%s'", policy);
}

} // anonymous namespace

VrfBank::VrfBank(unsigned bank_id, unsigned request_queue_depth,
                 VrfArbitrationPolicy policy_)
    : bankId(bank_id), policy(policy_), reads(request_queue_depth),
      writes(request_queue_depth)
{
}

void
VrfBank::acceptReadNext(VrfBankReq request)
{
    panic_if(request.bank != bankId || request.access != VrfAccess::Read,
             "Invalid RVV VRF bank read request");
    reads.pushNext(std::move(request));
}

void
VrfBank::acceptWriteNext(VrfBankReq request)
{
    panic_if(request.bank != bankId || request.access != VrfAccess::Write,
             "Invalid RVV VRF bank write request");
    writes.pushNext(std::move(request));
}

void
VrfBank::evaluate(RvvChannel<VrfReadResp> &read_responses,
                  RvvChannel<VrfWriteAck> &write_acks,
                  uint64_t cycle, LaneId lane_id)
{
    const bool can_write = !writes.empty() && write_acks.canPushNext();
    const bool can_read = !reads.empty() && read_responses.canPushNext();
    bool choose_write = can_write;
    if (can_write && can_read) {
        if (policy == VrfArbitrationPolicy::TwoLevelPriority &&
            writes.front().highPriority != reads.front().highPriority) {
            choose_write = writes.front().highPriority;
        } else if (policy == VrfArbitrationPolicy::RoundRobin)
            choose_write = preferWrite;
        else if (policy == VrfArbitrationPolicy::Oldest)
            choose_write = writes.front().sequence <= reads.front().sequence;
    }

    /* Every bank is single port; policy only selects the one firing request. */
    if (choose_write) {
        const auto &request = writes.front();
        write_acks.pushNext(VrfWriteAck{
            request.tag, request.bank, request.reg, request.eew});
        writes.popNext();
        if (policy == VrfArbitrationPolicy::RoundRobin)
            preferWrite = false;
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u VRF WRITE_FIRE bank=%u\n",
                cycle, request.tag.instId, request.tag.beatId,
                lane_id, bankId);
    } else if (can_read) {
        const auto &request = reads.front();
        read_responses.pushNext(VrfReadResp{
            request.tag, request.operandIndex, request.bank});
        reads.popNext();
        if (policy == VrfArbitrationPolicy::RoundRobin)
            preferWrite = true;
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u VRF READ_FIRE "
                "bank=%u operand=%u\n",
                cycle, request.tag.instId, request.tag.beatId,
                lane_id, bankId, request.operandIndex);
    }
}

void
VrfBank::advance()
{
    reads.advance();
    writes.advance();
}

void
VrfBank::cancel(RvvInstId id)
{
    reads.eraseIf([id](const VrfBankReq &request) {
        return request.tag.instId == id;
    });
    writes.eraseIf([id](const VrfBankReq &request) {
        return request.tag.instId == id;
    });
}

VectorRegFile::VectorRegFile(
    LaneId lane_id, unsigned num_lanes, unsigned lane_width_bits,
    unsigned banks_per_lane, unsigned vlen_bits,
    unsigned request_queue_depth, unsigned response_queue_depth,
    const std::string &arbitration_policy)
    : laneId(lane_id),
      mapper(num_lanes, lane_width_bits, banks_per_lane, vlen_bits),
      readResponses(response_queue_depth), writeAcks(response_queue_depth)
{
    banks.reserve(banks_per_lane);
    for (unsigned bank = 0; bank < banks_per_lane; ++bank)
        banks.emplace_back(
            bank, request_queue_depth, parsePolicy(arbitration_policy));
}

bool
VectorRegFile::canAcceptReadNext(const LogicalVrfReq &request) const
{
    const auto mapped = mapper.map(request);
    return banks[mapped.bank].canAcceptReadNext();
}

bool
VectorRegFile::canAcceptWriteNext(const LogicalVrfReq &request) const
{
    const auto mapped = mapper.map(request);
    return banks[mapped.bank].canAcceptWriteNext();
}

void
VectorRegFile::acceptReadNext(const LogicalVrfReq &request)
{
    auto mapped = mapper.map(request);
    mapped.sequence = nextRequestSequence++;
    banks[mapped.bank].acceptReadNext(std::move(mapped));
}

void
VectorRegFile::acceptWriteNext(const LogicalVrfReq &request)
{
    auto mapped = mapper.map(request);
    mapped.sequence = nextRequestSequence++;
    banks[mapped.bank].acceptWriteNext(std::move(mapped));
}

void
VectorRegFile::evaluate(uint64_t cycle)
{
    for (auto &bank : banks)
        bank.evaluate(readResponses, writeAcks, cycle, laneId);
}

void
VectorRegFile::advance()
{
    for (auto &bank : banks)
        bank.advance();
    readResponses.advance();
    writeAcks.advance();
    for (const auto &[reg, tag] : pendingEewTags)
        eewTags.at(static_cast<size_t>(reg)) = tag;
    pendingEewTags.clear();
}

void
VectorRegFile::cancel(RvvInstId id)
{
    for (auto &bank : banks)
        bank.cancel(id);
    readResponses.eraseIf([id](const VrfReadResp &response) {
        return response.tag.instId == id;
    });
    writeAcks.eraseIf([id](const VrfWriteAck &ack) {
        return ack.tag.instId == id;
    });
}

bool
VectorRegFile::active() const
{
    return readResponses.active() || writeAcks.active() ||
        std::any_of(banks.begin(), banks.end(),
            [](const VrfBank &bank) { return bank.active(); });
}

} // namespace rvv
} // namespace minor
} // namespace gem5
