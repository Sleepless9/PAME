/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/auxiliary_units.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

MatrixAuxiliaryUnit::MatrixAuxiliaryUnit(
    AmeTarget target_, AmeContextTable &contexts_,
    AmeScoreboard &scoreboard_, BankedRegisterFile &mrf_,
    BankedRegisterFile &acc_rf, const AuxiliaryUnitParams &params_)
    : target(target_), contexts(contexts_), scoreboard(scoreboard_), mrf(mrf_),
      accRf(acc_rf), params(params_), requests(params.queueDepth),
      done(params.queueDepth)
{
    fatal_if((target != TargetMrsu && target != TargetMptu) ||
             params.localLatency == 0 || params.crossTileLatency == 0 ||
             params.tileRows == 0 || params.tileCols == 0 ||
             params.resultWidth == 0,
             "Invalid AME auxiliary unit parameters");
}

void
MatrixAuxiliaryUnit::start(const AmePeRequest &request)
{
    Work next;
    next.request = request;
    const auto &desc = contexts.get(request.contextId).desc;
    const unsigned m_fragment = mrf.fragmentBytes();
    const unsigned acc_fragment = accRf.fragmentBytes();
    const auto acc_fragments = [&desc, acc_fragment](const auto &regions) {
        unsigned total = 0;
        for (const auto &region : regions) {
            const unsigned bytes = activeAccRegisterBytes(
                desc.config, region.reg);
            total += (bytes + acc_fragment - 1) / acc_fragment;
        }
        return total;
    };
    next.readsExpected = (desc.mBytesRead + m_fragment - 1) / m_fragment +
        acc_fragments(desc.accSources);
    next.writesExpected =
        (desc.mBytesWritten + m_fragment - 1) / m_fragment +
        acc_fragments(desc.accDestinations);
    if (target == TargetMrsu) {
        const unsigned lines = std::max(1U,
            static_cast<unsigned>(desc.config.nAme));
        next.localItemsExpected = lines * lines;
        next.localIssuedByLine.assign(lines, 0);
        next.localDoneByLine.assign(lines, 0);
        next.axisTiles = axisTileCount(request);
        next.crossPartials = next.axisTiles;
        next.carryIssuedByLine.assign(lines, 0);
        next.carryDoneByLine.assign(lines, 0);
    } else {
        next.localItemsExpected = std::max(
            1U, std::max(next.readsExpected, next.writesExpected));
    }
    next.phase = next.readsExpected == 0 ? Phase::LocalCompute : Phase::Read;
    work = next;
}

bool
MatrixAuxiliaryUnit::issueRead()
{
    auto &state = *work;
    if (state.readsIssued >= state.readsExpected)
        return false;
    const auto &desc = contexts.get(state.request.contextId).desc;
    const unsigned m_fragments =
        (desc.mBytesRead + mrf.fragmentBytes() - 1) / mrf.fragmentBytes();
    LogicalBankRequest request;
    request.contextId = state.request.contextId;
    request.ameId = state.request.ameId;
    request.access = BankAccess::Read;
    request.client = target == TargetMrsu ? BankClient::Mrsu : BankClient::Mptu;
    request.highPriority = target == TargetMrsu;
    if (state.readsIssued < m_fragments) {
        const MRegion fragment = mapMByte(
            desc.mSourceGroups, state.mReadBytesIssued, mrf.fragmentBytes());
        request.file = RegisterFileKind::Mrf;
        request.reg = fragment.reg;
        request.byteOffset = fragment.byteBegin;
        request.byteCount = fragment.byteEnd - fragment.byteBegin;
        if (!scoreboard.mRegionReadyForConsumer(desc.id, fragment) ||
            !mrf.canAcceptNext(request)) {
            return false;
        }
        mrf.acceptNext(std::move(request));
        state.mReadBytesIssued += fragment.byteEnd - fragment.byteBegin;
    } else {
        unsigned ordinal = state.readsIssued - m_fragments;
        const AccRegion *selected = nullptr;
        unsigned selected_fragment = 0;
        unsigned bytes_per_acc = 0;
        for (const auto &region : desc.accSources) {
            const unsigned bytes = activeAccRegisterBytes(
                desc.config, region.reg);
            const unsigned fragments =
                (bytes + accRf.fragmentBytes() - 1) /
                    accRf.fragmentBytes();
            if (ordinal < fragments) {
                selected = &region;
                selected_fragment = ordinal;
                bytes_per_acc = bytes;
                break;
            }
            ordinal -= fragments;
        }
        panic_if(!selected,
                 "AME auxiliary Acc read ordinal outside source list");
        request.file = RegisterFileKind::Acc;
        request.reg = selected->reg;
        request.byteOffset = selected_fragment * accRf.fragmentBytes();
        request.byteCount = std::min<unsigned>(accRf.fragmentBytes(),
            bytes_per_acc - request.byteOffset);
        if (!accRf.canAcceptNext(request))
            return false;
        accRf.acceptNext(std::move(request));
    }
    ++state.readsIssued;
    return true;
}

bool
MatrixAuxiliaryUnit::issueWrite()
{
    auto &state = *work;
    if (state.writesIssued >= state.writesExpected)
        return false;
    const auto &desc = contexts.get(state.request.contextId).desc;
    const unsigned m_fragments =
        (desc.mBytesWritten + mrf.fragmentBytes() - 1) / mrf.fragmentBytes();
    LogicalBankRequest request;
    request.contextId = state.request.contextId;
    request.ameId = state.request.ameId;
    request.access = BankAccess::Write;
    request.client = target == TargetMrsu ? BankClient::Mrsu : BankClient::Mptu;
    request.highPriority = true;
    request.final = state.writesIssued + 1 == state.writesExpected;
    if (state.writesIssued < m_fragments) {
        const MRegion fragment = mapMByte(
            desc.mDestinationGroups, state.mWriteBytesIssued,
            mrf.fragmentBytes());
        request.file = RegisterFileKind::Mrf;
        request.reg = fragment.reg;
        request.byteOffset = fragment.byteBegin;
        request.byteCount = fragment.byteEnd - fragment.byteBegin;
        if (!mrf.canAcceptNext(request))
            return false;
        mrf.acceptNext(std::move(request));
        state.mWriteBytesIssued += fragment.byteEnd - fragment.byteBegin;
    } else {
        unsigned ordinal = state.writesIssued - m_fragments;
        const AccRegion *selected = nullptr;
        unsigned selected_fragment = 0;
        unsigned bytes_per_acc = 0;
        for (const auto &region : desc.accDestinations) {
            const unsigned bytes = activeAccRegisterBytes(
                desc.config, region.reg);
            const unsigned fragments =
                (bytes + accRf.fragmentBytes() - 1) /
                    accRf.fragmentBytes();
            if (ordinal < fragments) {
                selected = &region;
                selected_fragment = ordinal;
                bytes_per_acc = bytes;
                break;
            }
            ordinal -= fragments;
        }
        panic_if(!selected,
                 "AME auxiliary Acc write ordinal outside destination list");
        request.file = RegisterFileKind::Acc;
        request.reg = selected->reg;
        request.byteOffset = selected_fragment * accRf.fragmentBytes();
        request.byteCount = std::min<unsigned>(accRf.fragmentBytes(),
            bytes_per_acc - request.byteOffset);
        if (!accRf.canAcceptNext(request))
            return false;
        accRf.acceptNext(std::move(request));
    }
    ++state.writesIssued;
    return true;
}

unsigned
MatrixAuxiliaryUnit::axisTileCount(const AmePeRequest &request) const
{
    const auto &name = contexts.get(request.contextId).desc.packet.mnemonic;
    if (name.find("_row") != std::string::npos)
        return params.tileCols;
    if (name.find("_col") != std::string::npos)
        return params.tileRows;
    return std::max(params.tileRows, params.tileCols);
}

void
MatrixAuxiliaryUnit::evaluateLocal(uint64_t cycle)
{
    auto &state = *work;
    unsigned retired = 0;
    while (!state.localPipeline.empty() &&
           state.localPipeline.front().readyCycle <= cycle &&
           retired < params.resultWidth) {
        const auto token = state.localPipeline.front();
        state.localPipeline.pop_front();
        ++state.localItemsDone;
        ++computeItemCount;
        if (state.request.opcode == AmeOpcode::Prefix) {
            panic_if(token.line >= state.localDoneByLine.size() ||
                     state.localDoneByLine[token.line] != token.ordinal,
                     "AME prefix local scan retired out of order");
            ++state.localDoneByLine[token.line];
        }
        ++retired;
    }

    unsigned issued = 0;
    if (state.request.opcode == AmeOpcode::Prefix) {
        const unsigned lines = state.localIssuedByLine.size();
        for (unsigned checked = 0;
             checked < lines && issued < params.resultWidth; ++checked) {
            const unsigned line = (state.roundRobinLine + checked) % lines;
            auto &line_issued = state.localIssuedByLine[line];
            if (line_issued >= lines ||
                line_issued != state.localDoneByLine[line]) {
                continue;
            }
            state.localPipeline.push_back(ComputeToken{
                cycle + params.localLatency, line, line_issued});
            ++line_issued;
            ++state.localItemsIssued;
            ++issued;
        }
        if (lines != 0)
            state.roundRobinLine = (state.roundRobinLine + 1) % lines;
    } else {
        while (state.localItemsIssued < state.localItemsExpected &&
               issued < params.resultWidth) {
            state.localPipeline.push_back(ComputeToken{
                cycle + params.localLatency, 0, state.localItemsIssued});
            ++state.localItemsIssued;
            ++issued;
        }
    }

    if (state.localItemsDone != state.localItemsExpected ||
        !state.localPipeline.empty()) {
        return;
    }

    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u %s LOCAL_COMPLETE items=%u\n",
            cycle, state.request.contextId, state.request.ameId,
            target == TargetMrsu ? "MRSU" : "MPTU",
            state.localItemsExpected);
    if (target == TargetMptu || state.axisTiles <= 1) {
        state.phase = Phase::Format;
        state.readyCycle = cycle + 1;
    } else if (state.request.opcode == AmeOpcode::Prefix) {
        state.phase = Phase::PrefixPropagate;
    } else {
        state.phase = Phase::CrossTile;
        startCrossStage(state);
    }
}

void
MatrixAuxiliaryUnit::startCrossStage(Work &state)
{
    panic_if(state.crossPartials <= 1,
             "AME reduction started an empty merge stage");
    const unsigned lines = std::max(1U,
        static_cast<unsigned>(state.localIssuedByLine.size()));
    state.crossMergesExpected = lines * (state.crossPartials / 2);
    state.crossMergesIssued = 0;
    state.crossMergesDone = 0;
    state.crossPipeline.clear();
}

void
MatrixAuxiliaryUnit::evaluateCrossTile(uint64_t cycle)
{
    auto &state = *work;
    unsigned retired = 0;
    while (!state.crossPipeline.empty() &&
           state.crossPipeline.front().readyCycle <= cycle &&
           retired < params.resultWidth) {
        state.crossPipeline.pop_front();
        ++state.crossMergesDone;
        ++retired;
    }
    unsigned issued = 0;
    while (state.crossMergesIssued < state.crossMergesExpected &&
           issued < params.resultWidth) {
        state.crossPipeline.push_back(ComputeToken{
            cycle + params.crossTileLatency, 0,
            state.crossMergesIssued});
        ++state.crossMergesIssued;
        ++issued;
    }
    if (state.crossMergesDone != state.crossMergesExpected ||
        !state.crossPipeline.empty()) {
        return;
    }

    state.crossPartials = (state.crossPartials + 1) / 2;
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u MRSU_MERGE_STAGE partials=%u\n",
            cycle, state.request.contextId, state.request.ameId,
            state.crossPartials);
    if (state.crossPartials > 1) {
        startCrossStage(state);
    } else {
        state.phase = Phase::Format;
        state.readyCycle = cycle + 1;
    }
}

void
MatrixAuxiliaryUnit::evaluatePrefix(uint64_t cycle)
{
    auto &state = *work;
    unsigned retired = 0;
    while (!state.crossPipeline.empty() &&
           state.crossPipeline.front().readyCycle <= cycle &&
           retired < params.resultWidth) {
        const auto token = state.crossPipeline.front();
        state.crossPipeline.pop_front();
        panic_if(token.line >= state.carryDoneByLine.size() ||
                 state.carryDoneByLine[token.line] != token.ordinal,
                 "AME prefix carry retired out of order");
        ++state.carryDoneByLine[token.line];
        ++retired;
    }

    unsigned issued = 0;
    const unsigned lines = state.carryIssuedByLine.size();
    for (unsigned checked = 0;
         checked < lines && issued < params.resultWidth; ++checked) {
        const unsigned line = (state.roundRobinLine + checked) % lines;
        auto &line_issued = state.carryIssuedByLine[line];
        if (line_issued >= state.axisTiles - 1 ||
            line_issued != state.carryDoneByLine[line]) {
            continue;
        }
        state.crossPipeline.push_back(ComputeToken{
            cycle + params.crossTileLatency, line, line_issued});
        ++line_issued;
        ++issued;
    }
    if (lines != 0)
        state.roundRobinLine = (state.roundRobinLine + 1) % lines;

    const bool complete = std::all_of(
        state.carryDoneByLine.begin(), state.carryDoneByLine.end(),
        [&state](unsigned count) { return count == state.axisTiles - 1; });
    if (complete && state.crossPipeline.empty()) {
        DPRINTF(AMEFlow,
                "cycle=%llu inst=%llu slot=%u MRSU_PREFIX_PROPAGATE "
                "tiles=%u\n", cycle, state.request.contextId,
                state.request.ameId, state.axisTiles);
        state.phase = Phase::Format;
        state.readyCycle = cycle + 1;
    }
}

MRegion
MatrixAuxiliaryUnit::mapMByte(
    const std::vector<std::vector<MRegion>> &groups,
    uint32_t byte_offset, uint32_t max_bytes) const
{
    uint32_t cursor = byte_offset;
    for (const auto &group : groups) {
        for (const auto &region : group) {
            const uint32_t length = region.byteEnd - region.byteBegin;
            if (cursor >= length) {
                cursor -= length;
                continue;
            }
            const uint32_t begin = region.byteBegin + cursor;
            const uint32_t word_remaining = mrf.fragmentBytes() -
                (begin % mrf.fragmentBytes());
            const uint32_t bytes = std::min<uint32_t>(
                {max_bytes, length - cursor, word_remaining});
            return MRegion{
                region.tid, region.reg, begin, begin + bytes};
        }
    }
    panic("AME auxiliary byte offset %u outside M register groups",
          byte_offset);
}

void
MatrixAuxiliaryUnit::acceptResponse(const BankResponse &response)
{
    if (!work || response.contextId != work->request.contextId)
        return;
    if (response.access == BankAccess::Read) {
        ++work->readsDone;
    } else {
        ++work->writesDone;
        const auto &desc = contexts.get(response.contextId).desc;
        if (response.file == RegisterFileKind::Mrf) {
            scoreboard.publishMWriteNext(response.ameId,
                MRegion{desc.key.threadId, response.reg, response.byteOffset,
                    response.byteOffset + response.byteCount});
        } else if (response.final) {
            scoreboard.publishAccWriteNext(response.ameId);
        }
    }
}

void
MatrixAuxiliaryUnit::evaluate(uint64_t cycle)
{
    if (!work && !requests.empty()) {
        start(requests.front());
        requests.popNext();
    }
    if (!work)
        return;
    auto &state = *work;
    if (state.phase == Phase::LocalCompute)
        ++localCycleCount;
    else if (state.phase == Phase::CrossTile)
        ++crossTileCycleCount;
    else if (state.phase == Phase::PrefixPropagate)
        ++prefixCycleCount;
    switch (state.phase) {
      case Phase::Read:
        for (unsigned issued = 0; issued < params.resultWidth; ++issued) {
            if (!issueRead())
                break;
        }
        if (state.readsDone == state.readsExpected) {
            state.phase = Phase::LocalCompute;
        }
        break;
      case Phase::LocalCompute:
        evaluateLocal(cycle);
        break;
      case Phase::CrossTile:
        evaluateCrossTile(cycle);
        break;
      case Phase::PrefixPropagate:
        evaluatePrefix(cycle);
        break;
      case Phase::Format:
        if (cycle >= state.readyCycle)
            state.phase = Phase::Write;
        break;
      case Phase::Write:
        for (unsigned issued = 0; issued < params.resultWidth; ++issued) {
            if (!issueWrite())
                break;
        }
        if (state.writesDone == state.writesExpected)
            state.phase = Phase::Done;
        break;
      case Phase::Done:
        if (done.canPushNext()) {
            done.pushNext(TargetDone{state.request.contextId, target});
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu slot=%u %s TARGET_DONE\n",
                    cycle, state.request.contextId, state.request.ameId,
                    target == TargetMrsu ? "MRSU" : "MPTU");
            work.reset();
        }
        break;
    }
}

void
MatrixAuxiliaryUnit::advance()
{
    requests.advance();
    done.advance();
}

void
MatrixAuxiliaryUnit::cancel(AmeContextId id)
{
    requests.eraseIf([id](const AmePeRequest &request) {
        return request.contextId == id;
    });
    done.eraseIf([id](const TargetDone &completion) {
        return completion.contextId == id;
    });
    if (work && work->request.contextId == id)
        work.reset();
}

} // namespace ame
} // namespace minor
} // namespace gem5
