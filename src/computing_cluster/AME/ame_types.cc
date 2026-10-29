/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/ame_types.hh"

#include <algorithm>

#include "arch/riscv/regs/matrix.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

namespace
{

bool
contains(const std::string &name, const std::string &fragment)
{
    return name.find(fragment) != std::string::npos;
}

bool
isAcc(const RegId &reg)
{
    return reg.is(MatRegClass) &&
        reg.index() >= RiscvISA::ZttAccRegBase &&
        reg.index() < RiscvISA::ZttAccRegBase + RiscvISA::NumZttAccRegs;
}

bool
isM(const RegId &reg)
{
    return reg.is(MatRegClass) &&
        reg.index() >= RiscvISA::ZttMatRegBase &&
        reg.index() < RiscvISA::ZttMatRegBase + RiscvISA::NumZttMatRegs;
}

} // anonymous namespace

AmeOpcode
classifyOpcode(const InstQueuePacket &packet)
{
    const std::string &name = packet.mnemonic;
    if ((contains(name, "mmul") && contains(name, "_2d")) ||
        contains(name, "matmul"))
        return AmeOpcode::MatrixMultiply;
    if (contains(name, "mprefix"))
        return AmeOpcode::Prefix;
    if (contains(name, "mreduce"))
        return AmeOpcode::Reduction;
    if (contains(name, "unzip") || contains(name, "zip") ||
        contains(name, "gather") || contains(name, "scatter") ||
        contains(name, "transpose")) {
        return AmeOpcode::Permutation;
    }
    if (contains(name, "mls") || contains(name, "mld") ||
        (packet.isMemory && contains(name, "load"))) {
        return AmeOpcode::Load;
    }
    if (contains(name, "mss") || contains(name, "mst") ||
        (packet.isMemory && contains(name, "store"))) {
        return AmeOpcode::Store;
    }
    if (contains(name, "mzero"))
        return AmeOpcode::Zero;
    if (contains(name, "mmov") || contains(name, "mmv"))
        return AmeOpcode::Move;
    if (contains(name, "settyp") || contains(name, "gettyp"))
        return AmeOpcode::Config;
    if (contains(name, "acquire"))
        return AmeOpcode::Acquire;
    if (contains(name, "event_wait"))
        return AmeOpcode::EventWait;
    if (contains(name, "release"))
        return AmeOpcode::Release;
    return AmeOpcode::Elementwise;
}

AmeTargetMask
targetMask(AmeOpcode opcode)
{
    switch (opcode) {
      case AmeOpcode::MatrixMultiply:
        return TargetSa;
      case AmeOpcode::Load:
      case AmeOpcode::Store:
        return TargetMlsu;
      case AmeOpcode::Reduction:
      case AmeOpcode::Prefix:
        return TargetMrsu;
      case AmeOpcode::Permutation:
      case AmeOpcode::Move:
      case AmeOpcode::Zero:
      case AmeOpcode::Elementwise:
        return TargetMptu;
      case AmeOpcode::Config:
      case AmeOpcode::Acquire:
      case AmeOpcode::EventWait:
      case AmeOpcode::Release:
      case AmeOpcode::Unknown:
        return TargetConfig;
    }
    return TargetConfig;
}

bool
isMatrixMultiply(AmeOpcode opcode)
{
    return opcode == AmeOpcode::MatrixMultiply;
}

bool
isMemory(AmeOpcode opcode)
{
    return opcode == AmeOpcode::Load || opcode == AmeOpcode::Store;
}

std::string
opcodeName(AmeOpcode opcode)
{
    switch (opcode) {
      case AmeOpcode::MatrixMultiply: return "matrix_multiply";
      case AmeOpcode::Elementwise: return "elementwise";
      case AmeOpcode::Reduction: return "reduction";
      case AmeOpcode::Prefix: return "prefix";
      case AmeOpcode::Permutation: return "permutation";
      case AmeOpcode::Load: return "load";
      case AmeOpcode::Store: return "store";
      case AmeOpcode::Move: return "move";
      case AmeOpcode::Zero: return "zero";
      case AmeOpcode::Config: return "config";
      case AmeOpcode::Acquire: return "acquire";
      case AmeOpcode::EventWait: return "event_wait";
      case AmeOpcode::Release: return "release";
      case AmeOpcode::Unknown: return "unknown";
    }
    return "unknown";
}

uint32_t
physicalMRegisterBytes(const AmeConfig &config)
{
    const uint64_t bits = static_cast<uint64_t>(config.nAme) *
        config.nAme * config.unitBits;
    return std::max<uint64_t>(1, (bits + 7) / 8);
}

uint32_t
activeAccRegisterBytes(const AmeConfig &config)
{
    const uint64_t bits = static_cast<uint64_t>(config.nAme) *
        config.nAme * config.adBits;
    return std::max<uint64_t>(1, (bits + 7) / 8);
}

uint32_t
activeAccRegisterBytes(const AmeConfig &config, unsigned reg)
{
    const uint64_t bits = static_cast<uint64_t>(config.nAme) *
        config.nAme * accElementBits(config, reg);
    return std::max<uint64_t>(1, (bits + 7) / 8);
}

uint16_t
mElementBits(const AmeConfig &config, unsigned reg)
{
    return reg < config.mRegBits.size() && config.mRegBits[reg] != 0 ?
        config.mRegBits[reg] : config.mdBits;
}

uint16_t
accElementBits(const AmeConfig &config, unsigned reg)
{
    return reg < config.accRegBits.size() && config.accRegBits[reg] != 0 ?
        config.accRegBits[reg] : config.adBits;
}

void
populateRegisterRegions(AmeInstDesc &desc)
{
    const uint32_t m_bytes = physicalMRegisterBytes(desc.config);
    const uint32_t acc_elements = static_cast<uint32_t>(desc.config.nAme) *
        desc.config.nAme;

    const auto add = [&desc, m_bytes](const RegId &reg, bool source) {
        if (isM(reg)) {
            const unsigned base = reg.index() - RiscvISA::ZttMatRegBase;
            const unsigned element_bits = mElementBits(desc.config, base);
            const unsigned span = std::max(1U,
                (element_bits +
                 std::max(1U, static_cast<unsigned>(desc.config.unitBits)) - 1) /
                std::max(1U, static_cast<unsigned>(desc.config.unitBits)));
            auto &regions = source ? desc.mSources : desc.mDestinations;
            auto &groups = source ? desc.mSourceGroups :
                                    desc.mDestinationGroups;
            if (std::any_of(groups.begin(), groups.end(),
                    [base](const auto &group) {
                        return std::any_of(group.begin(), group.end(),
                            [base](const MRegion &region) {
                                return region.reg == base;
                            });
                    })) {
                return;
            }
            std::vector<MRegion> group;
            for (unsigned index = 0;
                 index < span && base + index < RiscvISA::NumZttMatRegs;
                 ++index) {
                MRegion region{desc.key.threadId,
                    static_cast<uint16_t>(base + index), 0, m_bytes};
                group.push_back(region);
                regions.push_back(region);
            }
            groups.push_back(std::move(group));
        } else if (isAcc(reg)) {
            AccRegion region{desc.key.threadId,
                static_cast<uint8_t>(reg.index() - RiscvISA::ZttAccRegBase)};
            auto &regions = source ? desc.accSources : desc.accDestinations;
            if (std::find_if(regions.begin(), regions.end(),
                    [&region](const AccRegion &other) {
                        return region.reg == other.reg &&
                            region.tid == other.tid;
                    }) == regions.end()) {
                regions.push_back(region);
            }
        }
    };

    for (const auto &reg : desc.packet.srcRegs)
        add(reg, true);
    for (const auto &reg : desc.packet.destRegs)
        add(reg, false);

    const auto bytes = [](const std::vector<MRegion> &regions) {
        uint64_t result = 0;
        for (const auto &region : regions)
            result += region.byteEnd - region.byteBegin;
        return static_cast<uint32_t>(result);
    };
    desc.mBytesRead = bytes(desc.mSources);
    desc.mBytesWritten = bytes(desc.mDestinations);
    desc.accElementsRead = desc.accSources.size() * acc_elements;
    desc.accElementsWritten = desc.accDestinations.size() * acc_elements;
}

} // namespace ame
} // namespace minor
} // namespace gem5
