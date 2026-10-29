/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/rvv_types.hh"

#include <algorithm>

#include "base/bitfield.hh"
#include "arch/riscv/regs/vector.hh"
#include "enums/OpClass.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

RvvConfig
decodeConfig(const InstQueuePacket &packet)
{
    RvvConfig config;
    const uint64_t emi = packet.encodedInst;
    config.vl = packet.vectorLengthValid ? packet.vectorLength :
        bits(emi, 57, 41);
    config.sewBits = 8U << std::min<uint64_t>(bits(emi, 37, 35), 3);
    const uint8_t raw_lmul = bits(emi, 34, 32);
    config.lmulLog2 = raw_lmul & 0x4 ?
        static_cast<int8_t>(raw_lmul | 0xf8) : raw_lmul;
    config.fractionalLmul = config.lmulLog2 < 0;
    config.vill = bits(emi, 40);
    config.vta = bits(emi, 38);
    config.vma = bits(emi, 39);
    config.vm = bits(emi, 25);

    if (packet.opClass == enums::SimdConfig)
        config.vl = 1;
    return config;
}

VecFuType
classifyFu(const InstQueuePacket &packet)
{
    if (isSlide(packet))
        return VecFuType::Sldu;

    switch (packet.opClass) {
      case enums::SimdMult:
      case enums::SimdMultAcc:
      case enums::SimdDotProd:
        return VecFuType::Multiply;

      case enums::SimdFloatAdd:
      case enums::SimdFloatAlu:
      case enums::SimdFloatCmp:
      case enums::SimdFloatCvt:
      case enums::SimdFloatMisc:
      case enums::SimdFloatMult:
      case enums::SimdFloatMultAcc:
      case enums::SimdBf16Add:
      case enums::SimdBf16Cmp:
      case enums::SimdBf16Cvt:
      case enums::SimdBf16DotProd:
      case enums::SimdBf16Mult:
      case enums::SimdBf16MultAcc:
        return VecFuType::Fpu;

      case enums::SimdDiv:
      case enums::SimdSqrt:
      case enums::SimdFloatDiv:
      case enums::SimdFloatSqrt:
        return VecFuType::Divide;

      case enums::SimdReduceAdd:
      case enums::SimdReduceAlu:
      case enums::SimdReduceCmp:
      case enums::SimdFloatReduceAdd:
      case enums::SimdFloatReduceCmp:
        return VecFuType::Sldu;

      case enums::SimdUnitStrideLoad:
      case enums::SimdUnitStrideStore:
      case enums::SimdUnitStrideMaskLoad:
      case enums::SimdUnitStrideMaskStore:
      case enums::SimdStridedLoad:
      case enums::SimdStridedStore:
      case enums::SimdIndexedLoad:
      case enums::SimdIndexedStore:
      case enums::SimdWholeRegisterLoad:
      case enums::SimdWholeRegisterStore:
      case enums::SimdUnitStrideFaultOnlyFirstLoad:
      case enums::SimdUnitStrideSegmentedLoad:
      case enums::SimdUnitStrideSegmentedStore:
      case enums::SimdUnitStrideSegmentedFaultOnlyFirstLoad:
      case enums::SimdStrideSegmentedLoad:
      case enums::SimdStrideSegmentedStore:
        // Vector memory macro-ops also contain register-only micro-ops
        // (e.g. copying index operands to vtmp). Their inherited OpClass
        // must not turn them into a second, address-less memory request.
        return packet.isMemory ? VecFuType::Memory : VecFuType::Alu;

      case enums::SimdConfig:
        return VecFuType::Config;

      default:
        return VecFuType::Alu;
    }
}

bool
isVectorLoad(const InstQueuePacket &packet)
{
    if (!packet.isMemory)
        return false;
    switch (packet.opClass) {
      case enums::SimdUnitStrideLoad:
      case enums::SimdUnitStrideMaskLoad:
      case enums::SimdStridedLoad:
      case enums::SimdIndexedLoad:
      case enums::SimdWholeRegisterLoad:
      case enums::SimdUnitStrideFaultOnlyFirstLoad:
      case enums::SimdUnitStrideSegmentedLoad:
      case enums::SimdUnitStrideSegmentedFaultOnlyFirstLoad:
      case enums::SimdStrideSegmentedLoad:
        return true;
      default:
        return false;
    }
}

bool
isVectorStore(const InstQueuePacket &packet)
{
    return packet.isMemory && !isVectorLoad(packet);
}

VlsuAccessMode
classifyVlsuMode(const InstQueuePacket &packet)
{
    if (packet.opClass == enums::SimdIndexedLoad ||
        packet.opClass == enums::SimdIndexedStore)
        return VlsuAccessMode::Indexed;
    if (packet.opClass == enums::SimdStridedLoad ||
        packet.opClass == enums::SimdStridedStore ||
        packet.opClass == enums::SimdStrideSegmentedLoad ||
        packet.opClass == enums::SimdStrideSegmentedStore)
        return VlsuAccessMode::Strided;
    if (packet.opClass == enums::SimdUnitStrideSegmentedLoad ||
        packet.opClass == enums::SimdUnitStrideSegmentedStore ||
        packet.opClass == enums::SimdUnitStrideSegmentedFaultOnlyFirstLoad)
        return VlsuAccessMode::Segment;
    if (packet.opClass == enums::SimdUnitStrideFaultOnlyFirstLoad ||
        packet.opClass == enums::SimdUnitStrideSegmentedFaultOnlyFirstLoad)
        return VlsuAccessMode::FaultOnlyFirst;
    return VlsuAccessMode::UnitStride;
}

bool
isSlide(const InstQueuePacket &packet)
{
    return packet.mnemonic.find("slide") != std::string::npos;
}

bool
isReduction(const InstQueuePacket &packet)
{
    switch (packet.opClass) {
      case enums::SimdReduceAdd:
      case enums::SimdReduceAlu:
      case enums::SimdReduceCmp:
      case enums::SimdFloatReduceAdd:
      case enums::SimdFloatReduceCmp:
        return true;
      default:
        return packet.mnemonic.find("red") != std::string::npos;
    }
}

bool
isVectorRegister(const RegId &reg)
{
    return reg.is(VecRegClass) || reg.is(VecElemClass) ||
        reg.is(VecPredRegClass);
}

std::vector<RegId>
vectorSources(const InstQueuePacket &packet)
{
    std::vector<RegId> sources;
    for (const auto &reg : packet.srcRegs) {
        if (isVectorRegister(reg) &&
            std::find(sources.begin(), sources.end(), reg) == sources.end()) {
            sources.push_back(reg);
        }
    }
    return sources;
}

std::vector<RegId>
vectorSources(const InstQueuePacket &packet, const RvvConfig &config)
{
    auto sources = vectorSources(packet);
    const auto add = [&sources](const RegId &reg) {
        if (std::find(sources.begin(), sources.end(), reg) == sources.end())
            sources.push_back(reg);
    };

    /* Mask-undisturbed instructions read v0.  Tail/mask-undisturbed writes
     * also consume the old destination value. */
    if (!config.vm)
        add(RegId(RiscvISA::vecRegClass, 0));
    if (!config.vta || (!config.vm && !config.vma)) {
        const auto destination = vectorDestination(packet);
        if (destination)
            add(*destination);
    }
    return sources;
}

std::optional<RegId>
vectorDestination(const InstQueuePacket &packet)
{
    for (const auto &reg : packet.destRegs) {
        if (isVectorRegister(reg))
            return reg;
    }
    return std::nullopt;
}

} // namespace rvv
} // namespace minor
} // namespace gem5
