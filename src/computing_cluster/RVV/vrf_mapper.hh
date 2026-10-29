/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_MAPPER_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_MAPPER_HH__

#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

class VrfAddressMapper
{
  private:
    const unsigned numLanes;
    const unsigned laneBytes;
    const unsigned banksPerLane;
    const unsigned vlenBytes;
    const unsigned chunksPerVregPerLane;

  public:
    VrfAddressMapper(unsigned num_lanes, unsigned lane_width_bits,
                     unsigned banks_per_lane, unsigned vlen_bits);

    VrfBankReq map(const LogicalVrfReq &request) const;

    /** RVV lane organization helpers.  These are pure layout operations and
     * do not consume a bank port. */
    LaneId laneForElement(uint32_t elem_index, Eew eew) const;
    uint32_t laneWordForElement(RegIndex vreg, uint32_t elem_index,
                                Eew eew) const;
    uint8_t byteOffsetInWord(uint32_t elem_index, Eew eew) const;
    uint16_t bankForLaneWord(uint32_t lane_word) const;
    uint32_t rowForLaneWord(uint32_t lane_word) const;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_VRF_MAPPER_HH__
