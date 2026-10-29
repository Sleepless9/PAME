/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_TYPES_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_TYPES_HH__

#include <cstdint>
#include <optional>
#include <vector>

#include "cpu/minor/inst_queue.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

using RvvInstId = uint64_t;
/** Finite RVV-style instruction slot allocated by the global sequencer. */
using RvvInsnSlotId = uint8_t;
using BeatId = uint32_t;
using LaneId = uint16_t;
using TargetVfuMask = uint8_t;

enum class VecFuType : uint8_t
{
    Alu,
    Multiply,
    Fpu,
    Divide,
    Memory,
    Config,
    Sldu,
    NumTypes
};

enum TargetVfu : uint8_t
{
    TargetLane = 1 << 0,
    TargetVlsu = 1 << 1,
    TargetSldu = 1 << 2,
    TargetMasku = 1 << 3
};

enum class Eew : uint16_t
{
    E8 = 8,
    E16 = 16,
    E32 = 32,
    E64 = 64,
    E128 = 128
};

struct EewInfo
{
    Eew value = Eew::E8;
    bool widening = false;
    bool narrowing = false;
};

enum class ContextStatus : uint8_t
{
    Dispatched,
    Waiting,
    Sequencing,
    Executing,
    Completing,
    Done,
    Squashed
};

struct RvvConfig
{
    uint32_t vl = 0;
    uint32_t vstart = 0;
    uint16_t sewBits = 8;
    int8_t lmulLog2 = 0;
    bool fractionalLmul = false;
    bool vill = false;
    bool vta = false;
    bool vma = false;
    bool vm = true;
    uint8_t vxrm = 0;
    uint8_t frm = 0;
};

struct RvvInstDesc
{
    RvvInstId id = 0;
    InstQueueKey key;
    InstQueuePacket packet;
    RvvConfig config;
    VecFuType fuType = VecFuType::Alu;
    uint32_t totalBeats = 1;
    uint16_t elementsPerLaneChunk = 1;
    uint16_t elementsPerLaneBeat = 1;
    bool serialized = false;
    RvvInsnSlotId rvvSlotId = 0;
    TargetVfuMask targetVfus = TargetLane;
};

struct RvvInstState
{
    ContextStatus status = ContextStatus::Dispatched;
    uint32_t issuedBeats = 0;
    uint32_t completedBeats = 0;
    uint64_t acceptCycle = 0;
    uint64_t firstBeatCycle = 0;
    uint64_t completionCycle = 0;
    Fault fault = NoFault;
    bool killed = false;
};

struct RvvInstContext
{
    RvvInstDesc desc;
    RvvInstState state;
    RvvInsnSlotId rvvSlotId = 0;
    bool slotIdAllocated = false;
    TargetVfuMask targetVfus = TargetLane;
    TargetVfuMask targetAccepted = 0;
    TargetVfuMask targetDone = 0;
    bool peRequestAccepted = false;
    std::vector<RvvInsnSlotId> producerHazards;
};

struct VectorInstToken
{
    RvvInstId instId = 0;
};

/** Instruction-level PE request.  It is deliberately not a vector beat. */
struct PeRequest
{
    RvvInstId contextId = 0;
    RvvInsnSlotId rvvSlotId = 0;
    VecFuType opcodeClass = VecFuType::Alu;
    TargetVfuMask targetVfus = TargetLane;
    RegId vd;
    RegId vs1;
    RegId vs2;
    bool useVdAsOperand = false;
    bool vm = true;
    uint32_t vl = 0;
    uint32_t vstart = 0;
    RvvConfig config;
    EewInfo eew;
    std::vector<RvvInsnSlotId> producerHazards;
    RegVal scalarOperand = 0;
    bool useScalar = false;
};

enum class OperandSlot : uint8_t
{
    AluA,
    AluB,
    MulFpuA,
    MulFpuB,
    MulFpuC,
    MaskB,
    MaskM,
    StoreA,
    SlideAddrGenA,
    OldVd
};

enum class ResizeMode : uint8_t
{
    None,
    Widen,
    Narrow
};

enum class RequesterState : uint8_t
{
    Idle,
    Requesting,
    WaitingResponse,
    Complete,
    Killed
};

enum class VlsuAccessMode : uint8_t
{
    UnitStride,
    Strided,
    Indexed,
    Segment,
    FaultOnlyFirst
};

struct LaneWordTag
{
    RvvInstId contextId = 0;
    RvvInsnSlotId rvvSlotId = 0;
    LaneId lane = 0;
    OperandSlot slot = OperandSlot::AluA;
    uint32_t wordIndex = 0;
};

struct OperandRequestCmd
{
    RvvInsnSlotId rvvSlotId = 0;
    RegId vs;
    OperandSlot slot = OperandSlot::AluA;
    uint32_t laneVl = 0;
    uint32_t laneVstart = 0;
    EewInfo eew;
    ResizeMode resize = ResizeMode::None;
    VecFuType targetFu = VecFuType::Alu;
    std::vector<RvvInsnSlotId> producerHazards;
    bool scaleVl = false;
    bool isReduction = false;
    bool isSlide = false;
};

struct VfuOperation
{
    RvvInsnSlotId rvvSlotId = 0;
    VecFuType fuType = VecFuType::Alu;
    RegId vd;
    uint32_t vl = 0;
    uint32_t vstart = 0;
    RvvConfig config;
    RegVal scalarOperand = 0;
    bool useScalar = false;
    bool useOldVd = false;
    bool vm = true;
};

struct VectorBurstDesc
{
    uint64_t txnId = 0;
    RvvInsnSlotId rvvSlotId = 0;
    Addr addr = 0;
    uint32_t bytes = 0;
    uint16_t beats = 0;
    bool incrementing = true;
    bool load = false;
    bool store = false;
    uint32_t firstElement = 0;
};

struct BeatTag
{
    RvvInstId instId = 0;
    BeatId beatId = 0;

    bool operator==(const BeatTag &other) const
    {
        return instId == other.instId && beatId == other.beatId;
    }

    bool operator<(const BeatTag &other) const
    {
        return instId < other.instId ||
            (instId == other.instId && beatId < other.beatId);
    }
};

struct VecBeat
{
    BeatTag tag;
    uint32_t elemBase = 0;
    uint16_t elemCount = 0;
    uint16_t elementsPerLane = 1;
    uint32_t laneChunkRound = 0;
    uint16_t laneSlotOffset = 0;
    uint16_t laneSlotCapacity = 1;
    uint64_t activeLaneMask = 0;
    bool first = false;
    bool last = false;
};

struct LaneTag
{
    RvvInstId instId = 0;
    BeatId beatId = 0;
    LaneId laneId = 0;
    RvvInsnSlotId rvvSlotId = 0;
    uint32_t wordIndex = 0;

    bool operator==(const LaneTag &other) const
    {
        return instId == other.instId && beatId == other.beatId &&
            laneId == other.laneId && rvvSlotId == other.rvvSlotId &&
            wordIndex == other.wordIndex;
    }

    bool operator<(const LaneTag &other) const
    {
        if (instId != other.instId)
            return instId < other.instId;
        if (beatId != other.beatId)
            return beatId < other.beatId;
        if (laneId != other.laneId)
            return laneId < other.laneId;
        if (rvvSlotId != other.rvvSlotId)
            return rvvSlotId < other.rvvSlotId;
        return wordIndex < other.wordIndex;
    }
};

struct LaneBeat
{
    LaneTag tag;
    uint32_t firstElem = 0;
    uint8_t elemCount = 0;
    uint32_t groupByteOffset = 0;
    uint64_t elementEnable = 0;
    uint64_t byteEnable = 0;
    bool first = false;
    bool last = false;
};

enum class VrfAccess : uint8_t
{
    Read,
    Write
};

struct LogicalVrfReq
{
    LaneTag tag;
    uint8_t operandIndex = 0;
    VrfAccess access = VrfAccess::Read;
    RegId reg;
    int8_t lmulLog2 = 0;
    uint32_t groupByteOffset = 0;
    uint16_t sizeBytes = 0;
    uint64_t byteEnable = 0;
    EewInfo eew;
    bool highPriority = false;
};

struct VrfBankReq
{
    LaneTag tag;
    uint8_t operandIndex = 0;
    VrfAccess access = VrfAccess::Read;
    RegId reg;
    uint16_t bank = 0;
    uint32_t row = 0;
    uint64_t byteEnable = 0;
    uint64_t sequence = 0;
    EewInfo eew;
    bool highPriority = false;
};

struct VrfReadResp
{
    LaneTag tag;
    uint8_t operandIndex = 0;
    uint16_t bank = 0;
};

struct VrfWriteAck
{
    LaneTag tag;
    uint16_t bank = 0;
    RegId reg;
    EewInfo eew;
};

struct VectorFuReq
{
    LaneTag tag;
    VecFuType fuType = VecFuType::Alu;
    std::optional<RegId> destReg;
    uint32_t firstElem = 0;
    uint16_t elemCount = 0;
    uint32_t groupByteOffset = 0;
    uint64_t byteEnable = 0;
    bool last = true;
    VlsuAccessMode memoryMode = VlsuAccessMode::UnitStride;
    uint16_t segmentFields = 1;
    bool faultOnlyFirst = false;
    VectorBurstDesc burst;
};

struct VectorFuResp
{
    LaneTag tag;
    VecFuType fuType = VecFuType::Alu;
    std::optional<RegId> destReg;
    uint32_t groupByteOffset = 0;
    uint64_t byteEnable = 0;
};

struct LaneComplete
{
    LaneTag tag;
    bool instructionDone = false;
    TargetVfuMask target = TargetLane;
};

struct TargetDone
{
    RvvInstId contextId = 0;
    RvvInsnSlotId rvvSlotId = 0;
    TargetVfu target = TargetSldu;
};

struct RvvCompletion
{
    RvvInstId instId = 0;
    InstQueueKey key;
    Fault fault = NoFault;
};

RvvConfig decodeConfig(const InstQueuePacket &packet);
VecFuType classifyFu(const InstQueuePacket &packet);
bool isVectorRegister(const RegId &reg);
bool isVectorLoad(const InstQueuePacket &packet);
bool isVectorStore(const InstQueuePacket &packet);
VlsuAccessMode classifyVlsuMode(const InstQueuePacket &packet);
bool isSlide(const InstQueuePacket &packet);
bool isReduction(const InstQueuePacket &packet);
std::vector<RegId> vectorSources(const InstQueuePacket &packet);
std::vector<RegId> vectorSources(const InstQueuePacket &packet,
                                 const RvvConfig &config);
std::optional<RegId> vectorDestination(const InstQueuePacket &packet);

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_TYPES_HH__
