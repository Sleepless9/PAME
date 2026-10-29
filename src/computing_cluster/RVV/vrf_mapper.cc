/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vrf_mapper.hh"

#include <algorithm>
#include <cstdint>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

unsigned
calculateChunksPerVregPerLane(unsigned num_lanes, unsigned lane_width_bits,
                             unsigned banks_per_lane, unsigned vlen_bits)
{
    fatal_if(num_lanes == 0 || lane_width_bits == 0 ||
             lane_width_bits % 8 != 0 || banks_per_lane == 0 ||
             vlen_bits == 0 || vlen_bits % 8 != 0,
             "Invalid RVV VRF mapper topology");
    const uint64_t lane_array_bits =
        static_cast<uint64_t>(num_lanes) * lane_width_bits;
    fatal_if(vlen_bits % lane_array_bits != 0,
             "RVV VLEN (%u) must be a multiple of the lane array width "
             "(%u lanes * %u bits)",
             vlen_bits, num_lanes, lane_width_bits);
    return vlen_bits / lane_array_bits;
}

} // anonymous namespace

VrfAddressMapper::VrfAddressMapper(
    unsigned num_lanes, unsigned lane_width_bits,
    unsigned banks_per_lane, unsigned vlen_bits)
    : numLanes(num_lanes), laneBytes(lane_width_bits / 8),
      banksPerLane(banks_per_lane), vlenBytes(vlen_bits / 8),
      chunksPerVregPerLane(calculateChunksPerVregPerLane(
          num_lanes, lane_width_bits, banks_per_lane, vlen_bits))
{
}

VrfBankReq
VrfAddressMapper::map(const LogicalVrfReq &request) const
{
    panic_if(!isVectorRegister(request.reg),
             "RVV VRF request names a non-vector register");
    const unsigned group_bytes = request.lmulLog2 >= 0 ?
        vlenBytes << request.lmulLog2 :
        std::max(1U, vlenBytes >> -request.lmulLog2);
    panic_if(request.groupByteOffset >= group_bytes,
             "RVV VRF group offset %u exceeds LMUL slice size %u",
             request.groupByteOffset, group_bytes);
    const unsigned reg_offset = request.groupByteOffset / vlenBytes;
    const RegId physical_reg(
        request.reg.regClass(), request.reg.index() + reg_offset);
    const unsigned byte_in_vreg = request.groupByteOffset % vlenBytes;
    const unsigned global_chunk = byte_in_vreg / laneBytes;
    const unsigned mapped_lane = global_chunk % numLanes;
    panic_if(mapped_lane != request.tag.laneId,
             "RVV VRF mapping lane mismatch: mapped=%u token=%u",
             mapped_lane, request.tag.laneId);
    const unsigned lane_chunk = global_chunk / numLanes;
    const unsigned linear_word =
        physical_reg.index() * chunksPerVregPerLane + lane_chunk;

    VrfBankReq mapped;
    mapped.tag = request.tag;
    mapped.operandIndex = request.operandIndex;
    mapped.access = request.access;
    mapped.reg = physical_reg;
    mapped.bank = linear_word % banksPerLane;
    mapped.row = linear_word / banksPerLane;
    mapped.byteEnable = request.byteEnable;
    mapped.eew = request.eew;
    mapped.highPriority = request.highPriority;
    return mapped;
}

LaneId
VrfAddressMapper::laneForElement(uint32_t elem_index, Eew eew) const
{
    panic_if(static_cast<unsigned>(eew) == 0,
             "RVV element width must be non-zero");
    /* Elements are striped across lanes; packed narrow elements share a
     * 64-bit lane word. */
    return static_cast<LaneId>(elem_index % numLanes);
}

uint32_t
VrfAddressMapper::laneWordForElement(RegIndex vreg, uint32_t elem_index,
                                     Eew eew) const
{
    const unsigned element_bits = static_cast<unsigned>(eew);
    const unsigned packed = std::max(1U, 64U / element_bits);
    const uint32_t lane_element = elem_index / numLanes;
    return static_cast<uint32_t>(vreg) * chunksPerVregPerLane +
        lane_element / packed;
}

uint8_t
VrfAddressMapper::byteOffsetInWord(uint32_t elem_index, Eew eew) const
{
    const unsigned element_bits = static_cast<unsigned>(eew);
    const unsigned element_bytes = std::max(1U, element_bits / 8U);
    const unsigned packed = std::max(1U, 8U / element_bytes);
    return static_cast<uint8_t>((elem_index / numLanes % packed) *
                                element_bytes);
}

uint16_t
VrfAddressMapper::bankForLaneWord(uint32_t lane_word) const
{
    return static_cast<uint16_t>(lane_word % banksPerLane);
}

uint32_t
VrfAddressMapper::rowForLaneWord(uint32_t lane_word) const
{
    return lane_word / banksPerLane;
}

} // namespace rvv
} // namespace minor
} // namespace gem5
