/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_TYPES_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_TYPES_HH__

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu/minor/inst_queue.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

using AmeContextId = uint64_t;
using AmeInsnId = uint8_t;
using AmeTargetMask = uint8_t;

enum AmeTarget : uint8_t
{
    TargetNone = 0,
    TargetSa = 1 << 0,
    TargetMlsu = 1 << 1,
    TargetMrsu = 1 << 2,
    TargetMptu = 1 << 3,
    TargetConfig = 1 << 4
};

enum class AmeOpcode : uint8_t
{
    MatrixMultiply,
    Elementwise,
    Reduction,
    Prefix,
    Permutation,
    Load,
    Store,
    Move,
    Zero,
    Config,
    Acquire,
    EventWait,
    Release,
    Unknown
};

enum class MatrixDataflow : uint8_t
{
    WeightStationary,
    PartialSumStationary
};

enum class MatrixPhase : uint8_t
{
    Allocate,
    FormOperands,
    Prefetch,
    LoadWeight,
    LoadAcc,
    Wavefront,
    NextKBlock,
    Drain,
    AccCommit,
    Done,
    Killed
};

enum class OperandSlot : uint8_t
{
    A,
    B,
    OldDest,
    AccSeed,
    StoreSource,
    ReduceSource,
    PermuteSource
};

enum class AccLocation : uint8_t
{
    AccRF,
    LoadingToArray,
    ResidentInArray,
    Draining,
    CommitPending
};

struct AmeConfig
{
    uint16_t nAme = 4;
    uint16_t unitBits = 32;
    uint16_t mdBits = 32;
    uint16_t adBits = 32;
    uint16_t nSquares = 1;
    uint16_t kLogical = 4;
    bool transposeA = false;
    bool transposeB = false;
    MatrixDataflow dataflow = MatrixDataflow::PartialSumStationary;
    uint64_t ameType = 0;
    uint64_t exceptionFlags = 0;
    std::array<uint16_t, 32> mRegBits{};
    std::array<uint16_t, 4> accRegBits{};
};

struct MRegion
{
    ThreadID tid = InvalidThreadID;
    uint16_t reg = 0;
    uint32_t byteBegin = 0;
    uint32_t byteEnd = 0;

    bool empty() const { return byteBegin >= byteEnd; }
};

struct AccRegion
{
    ThreadID tid = InvalidThreadID;
    uint8_t reg = 0;
};

struct AmeInstDesc
{
    AmeContextId id = 0;
    AmeInsnId ameId = 0;
    InstQueueKey key;
    InstQueuePacket packet;
    AmeOpcode opcode = AmeOpcode::Unknown;
    AmeTargetMask targets = TargetNone;
    AmeConfig config;
    std::vector<std::vector<MRegion>> mSourceGroups;
    std::vector<std::vector<MRegion>> mDestinationGroups;
    std::vector<MRegion> mSources;
    std::vector<MRegion> mDestinations;
    std::vector<AccRegion> accSources;
    std::vector<AccRegion> accDestinations;
    uint32_t mBytesRead = 0;
    uint32_t mBytesWritten = 0;
    uint32_t accElementsRead = 0;
    uint32_t accElementsWritten = 0;
    bool serialized = false;
};

struct AmeInstContext
{
    AmeInstDesc desc;
    AmeTargetMask acceptedTargets = TargetNone;
    AmeTargetMask completedTargets = TargetNone;
    bool slotAllocated = false;
    bool killed = false;
    Fault fault = NoFault;
    uint64_t acceptCycle = 0;
};

struct AmeInstToken
{
    AmeContextId contextId = 0;
};

struct AmePeRequest
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    AmeOpcode opcode = AmeOpcode::Unknown;
    AmeTargetMask targets = TargetNone;
    AmeConfig config;
};

struct TargetDone
{
    AmeContextId contextId = 0;
    AmeTarget target = TargetNone;
};

struct AmeCompletion
{
    AmeContextId contextId = 0;
    InstQueueKey key;
    Fault fault = NoFault;
};

struct SaBlockTag
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    uint16_t outputBlockRow = 0;
    uint16_t outputBlockCol = 0;
    uint16_t kBlock = 0;
};

struct TileTag
{
    SaBlockTag block;
    uint16_t tile = 0;
    uint16_t kStep = 0;
    MatrixPhase phase = MatrixPhase::Allocate;
};

struct MFragmentTag
{
    AmeContextId contextId = 0;
    AmeInsnId ameId = 0;
    OperandSlot slot = OperandSlot::A;
    uint16_t squareIndex = 0;
    uint16_t logicalRow = 0;
    uint16_t fragmentIndex = 0;
};

AmeOpcode classifyOpcode(const InstQueuePacket &packet);
AmeTargetMask targetMask(AmeOpcode opcode);
bool isMatrixMultiply(AmeOpcode opcode);
bool isMemory(AmeOpcode opcode);
std::string opcodeName(AmeOpcode opcode);
uint32_t physicalMRegisterBytes(const AmeConfig &config);
uint32_t activeAccRegisterBytes(const AmeConfig &config);
uint32_t activeAccRegisterBytes(const AmeConfig &config, unsigned reg);
uint16_t mElementBits(const AmeConfig &config, unsigned reg);
uint16_t accElementBits(const AmeConfig &config, unsigned reg);
void populateRegisterRegions(AmeInstDesc &desc);

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_TYPES_HH__
