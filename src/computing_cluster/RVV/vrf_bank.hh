/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_BANK_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_BANK_HH__

#include <array>
#include <map>
#include <string>
#include <vector>

#include "arch/riscv/regs/vector.hh"
#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/vrf_mapper.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

enum class VrfArbitrationPolicy
{
    WritePriority,
    RoundRobin,
    Oldest,
    TwoLevelPriority
};

class VrfBank
{
  private:
    const unsigned bankId;
    const VrfArbitrationPolicy policy;
    RvvChannel<VrfBankReq> reads;
    RvvChannel<VrfBankReq> writes;
    bool preferWrite = true;

  public:
    VrfBank(unsigned bank_id, unsigned request_queue_depth,
            VrfArbitrationPolicy policy_ =
                VrfArbitrationPolicy::WritePriority);

    bool canAcceptReadNext() const { return reads.canPushNext(); }
    bool canAcceptWriteNext() const { return writes.canPushNext(); }
    void acceptReadNext(VrfBankReq request);
    void acceptWriteNext(VrfBankReq request);
    void evaluate(RvvChannel<VrfReadResp> &read_responses,
                  RvvChannel<VrfWriteAck> &write_acks,
                  uint64_t cycle, LaneId lane_id);
    void advance();
    void cancel(RvvInstId id);
    bool active() const { return reads.active() || writes.active(); }
};

class VectorRegFile
{
  private:
    const LaneId laneId;
    VrfAddressMapper mapper;
    std::vector<VrfBank> banks;
    RvvChannel<VrfReadResp> readResponses;
    RvvChannel<VrfWriteAck> writeAcks;
    uint64_t nextRequestSequence = 1;
    // Stable gem5 expands vector instructions into micro-ops which also
    // name vtmp0..vtmp7. They need the same EEW tracking as v0..v31.
    std::array<EewInfo, RiscvISA::NumVecRegs> eewTags{};

  public:
    VectorRegFile(LaneId lane_id, unsigned num_lanes,
                  unsigned lane_width_bits, unsigned banks_per_lane,
                  unsigned vlen_bits, unsigned request_queue_depth,
                  unsigned response_queue_depth,
                  const std::string &arbitration_policy = "oldest");

    bool canAcceptReadNext(const LogicalVrfReq &request) const;
    bool canAcceptWriteNext(const LogicalVrfReq &request) const;
    void acceptReadNext(const LogicalVrfReq &request);
    void acceptWriteNext(const LogicalVrfReq &request);

    const RvvChannel<VrfReadResp> &readResponseChannel() const
    { return readResponses; }
    RvvChannel<VrfReadResp> &readResponseChannel() { return readResponses; }
    const RvvChannel<VrfWriteAck> &writeAckChannel() const
    { return writeAcks; }
    RvvChannel<VrfWriteAck> &writeAckChannel() { return writeAcks; }

    void evaluate(uint64_t cycle);
    void advance();
    void cancel(RvvInstId id);
    EewInfo eewTag(RegIndex reg) const
    { return eewTags.at(static_cast<size_t>(reg)); }
    void updateEewTagNext(RegIndex reg, EewInfo tag)
    { pendingEewTags[static_cast<size_t>(reg)] = tag; }
    bool active() const;

  private:
    std::map<RegIndex, EewInfo> pendingEewTags;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_BANK_HH__
