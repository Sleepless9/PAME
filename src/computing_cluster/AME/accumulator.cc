/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/accumulator.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

AccumulatorSubsystem::AccumulatorSubsystem(
    AmeContextTable &contexts_, AmeScoreboard &scoreboard_,
    BankedRegisterFile &acc_rf)
    : contexts(contexts_), scoreboard(scoreboard_), accRf(acc_rf)
{
}

void
AccumulatorSubsystem::start(AmeContextId id)
{
    panic_if(progress.find(id) != progress.end(),
             "Starting duplicate AME accumulator work id=%llu", id);
    const auto &desc = contexts.get(id).desc;
    Progress state;
    state.owner = desc.ameId;
    if (!desc.accDestinations.empty())
        state.reg = desc.accDestinations.front().reg;
    else if (!desc.accSources.empty())
        state.reg = desc.accSources.front().reg;
    const unsigned bytes = activeAccRegisterBytes(desc.config, state.reg);
    panic_if(bytes > accRf.bytesPerRegister(),
             "AME Acc configuration needs %u bytes but Acc RF entry has %u",
             bytes, accRf.bytesPerRegister());
    state.totalBytes = bytes;
    state.totalFragments = std::max(1U,
        (bytes + accRf.fragmentBytes() - 1) / accRf.fragmentBytes());
    state.location = AccLocation::LoadingToArray;
    progress.emplace(id, state);
}

void
AccumulatorSubsystem::evaluateSeed(AmeContextId id)
{
    auto &state = progress.at(id);
    if (state.readsIssued >= state.totalFragments)
        return;
    const unsigned offset = state.readsIssued * accRf.fragmentBytes();
    const unsigned bytes = std::min<unsigned>(accRf.fragmentBytes(),
        state.totalBytes - offset);
    LogicalBankRequest request;
    request.contextId = id;
    request.ameId = state.owner;
    request.file = RegisterFileKind::Acc;
    request.access = BankAccess::Read;
    request.client = BankClient::SaAcc;
    request.reg = state.reg;
    request.byteOffset = offset;
    request.byteCount = bytes;
    request.highPriority = true;
    request.final = state.readsIssued + 1 == state.totalFragments;
    request.tag = {id, state.owner, OperandSlot::AccSeed, 0, 0,
        static_cast<uint16_t>(state.readsIssued)};
    if (!accRf.canAcceptNext(request))
        return;
    accRf.acceptNext(std::move(request));
    ++state.readsIssued;
}

bool
AccumulatorSubsystem::seedReady(AmeContextId id) const
{
    const auto it = progress.find(id);
    return it != progress.end() &&
        it->second.readsCompleted == it->second.totalFragments;
}

void
AccumulatorSubsystem::markResident(AmeContextId id)
{
    auto &state = progress.at(id);
    panic_if(!seedReady(id), "AME Acc became resident before seed was ready");
    state.location = AccLocation::ResidentInArray;
}

void
AccumulatorSubsystem::beginDrain(AmeContextId id)
{
    auto &state = progress.at(id);
    panic_if(state.location != AccLocation::ResidentInArray,
             "AME Acc drain began from invalid location");
    state.location = AccLocation::Draining;
}

void
AccumulatorSubsystem::evaluateDrain(AmeContextId id)
{
    auto &state = progress.at(id);
    if (state.location != AccLocation::Draining ||
        state.writesIssued >= state.totalFragments) {
        return;
    }
    const unsigned offset = state.writesIssued * accRf.fragmentBytes();
    const unsigned bytes = std::min<unsigned>(accRf.fragmentBytes(),
        state.totalBytes - offset);
    LogicalBankRequest request;
    request.contextId = id;
    request.ameId = state.owner;
    request.file = RegisterFileKind::Acc;
    request.access = BankAccess::Write;
    request.client = BankClient::SaWriteback;
    request.reg = state.reg;
    request.byteOffset = offset;
    request.byteCount = bytes;
    request.highPriority = true;
    request.final = state.writesIssued + 1 == state.totalFragments;
    request.tag = {id, state.owner, OperandSlot::AccSeed, 0, 0,
        static_cast<uint16_t>(state.writesIssued)};
    if (!accRf.canAcceptNext(request))
        return;
    accRf.acceptNext(std::move(request));
    ++state.writesIssued;
}

void
AccumulatorSubsystem::acceptResponse(const BankResponse &response)
{
    const auto it = progress.find(response.contextId);
    if (it == progress.end())
        return;
    auto &state = it->second;
    if (response.access == BankAccess::Read) {
        ++state.readsCompleted;
        panic_if(state.readsCompleted > state.totalFragments,
                 "Duplicate AME Acc seed response");
        return;
    }
    ++state.writesCompleted;
    panic_if(state.writesCompleted > state.totalFragments,
             "Duplicate AME Acc write acknowledgement");
    if (state.writesCompleted == state.totalFragments) {
        state.location = AccLocation::CommitPending;
        state.committed = true;
        scoreboard.publishAccWriteNext(state.owner);
    }
}

bool
AccumulatorSubsystem::committed(AmeContextId id) const
{
    const auto it = progress.find(id);
    return it != progress.end() && it->second.committed;
}

AccLocation
AccumulatorSubsystem::location(AmeContextId id) const
{
    return progress.at(id).location;
}

void
AccumulatorSubsystem::release(AmeContextId id)
{
    panic_if(progress.erase(id) != 1,
             "Releasing unknown AME accumulator context id=%llu", id);
}

void
AccumulatorSubsystem::cancel(AmeContextId id)
{
    progress.erase(id);
}

} // namespace ame
} // namespace minor
} // namespace gem5
