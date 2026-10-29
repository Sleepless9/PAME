/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/matrix_lsu.hh"

#include <algorithm>
#include <limits>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

MatrixLsu::MatrixLsu(
    AmeContextTable &contexts_, AmeScoreboard &scoreboard_,
    BankedRegisterFile &mrf_, const MatrixLsuParams &params_)
    : contexts(contexts_), scoreboard(scoreboard_), mrf(mrf_), params(params_),
      requests(params.descriptorQueueDepth), done(params.descriptorQueueDepth)
{
    fatal_if(params.packetBytes == 0 ||
             params.packetBytes > std::numeric_limits<uint16_t>::max() ||
             params.maxOutstanding == 0 ||
             (!params.sendTimingRequest && params.responseLatency == 0) ||
             params.reorderWindow == 0 ||
             params.responseWidth == 0,
             "Invalid AME M-LSU parameters");
}

void
MatrixLsu::acceptCommands()
{
    if (requests.empty() || works.size() >= params.descriptorQueueDepth)
        return;
    const auto &request = requests.front();
    const auto &context = contexts.get(request.contextId);
    Work work;
    work.desc.contextId = request.contextId;
    work.desc.ameId = request.ameId;
    work.desc.descId = nextDescId++;
    work.desc.base = context.desc.packet.scalarSrcs.empty() ? 0 :
        context.desc.packet.scalarSrcs.front().value;
    work.desc.stride = context.desc.packet.scalarSrcs.size() < 2 ? 0 :
        context.desc.packet.scalarSrcs[1].value;
    work.desc.isLoad = context.desc.opcode == AmeOpcode::Load;
    work.desc.isStore = context.desc.opcode == AmeOpcode::Store;
    work.desc.bytes = std::max(1U, work.desc.isLoad ?
        context.desc.mBytesWritten : context.desc.mBytesRead);
    work.desc.elementBytes = std::max(1U,
        static_cast<unsigned>(context.desc.config.mdBits) / 8U);
    if (work.desc.isLoad && !context.desc.mDestinations.empty())
        work.desc.mBase = context.desc.mDestinations.front().reg;
    else if (work.desc.isStore && !context.desc.mSources.empty())
        work.desc.mBase = context.desc.mSources.front().reg;
    buildTransfers(work);
    work.packetCount = work.desc.transfers.size();
    work.storeBytesReady.assign(work.desc.bytes, false);
    works.emplace(request.contextId, work);
    requests.popNext();
}

void
MatrixLsu::buildTransfers(Work &work)
{
    const auto &desc = contexts.get(work.desc.contextId).desc;
    const auto &name = desc.packet.mnemonic;
    const unsigned n = std::max(1U,
        static_cast<unsigned>(desc.config.nAme));
    const unsigned squares = std::max(1U,
        static_cast<unsigned>(desc.config.nSquares));
    const auto &groups = work.desc.isLoad ? desc.mDestinationGroups :
                                           desc.mSourceGroups;
    panic_if(groups.empty() || groups.front().empty(),
             "AME M-LSU descriptor has no M register group");
    const unsigned elem_bytes = std::max(1U,
        (static_cast<unsigned>(mElementBits(
             desc.config, groups.front().front().reg)) + 7) / 8);

    const auto append = [this, &work](uint64_t memory_offset,
                                      uint64_t m_offset,
                                      unsigned bytes) {
        panic_if(memory_offset > std::numeric_limits<uint32_t>::max() ||
                 m_offset > std::numeric_limits<uint32_t>::max(),
                 "AME M-LSU transfer offset exceeds P1 descriptor width");
        while (bytes != 0) {
            const unsigned line_remaining = params.packetBytes -
                ((work.desc.base + memory_offset) % params.packetBytes);
            const unsigned chunk = std::min(bytes, line_remaining);
            MatrixBurstDesc::Transfer *transfer = nullptr;
            if (!work.desc.transfers.empty()) {
                auto &last = work.desc.transfers.back();
                const bool adjacent =
                    last.memoryOffset + last.bytes == memory_offset;
                const bool same_packet =
                    (work.desc.base + last.memoryOffset) /
                        params.packetBytes ==
                    (work.desc.base + memory_offset + chunk - 1) /
                        params.packetBytes;
                if (adjacent && same_packet &&
                    last.bytes + chunk <= params.packetBytes) {
                    transfer = &last;
                    last.bytes += chunk;
                }
            }
            if (!transfer) {
                work.desc.transfers.push_back(MatrixBurstDesc::Transfer{
                    static_cast<uint32_t>(memory_offset),
                    static_cast<uint16_t>(chunk), {}});
                transfer = &work.desc.transfers.back();
            }
            const unsigned packet_offset =
                memory_offset - transfer->memoryOffset;
            if (!transfer->mappings.empty()) {
                auto &mapping = transfer->mappings.back();
                if (mapping.packetOffset + mapping.bytes == packet_offset &&
                    mapping.mOffset + mapping.bytes == m_offset) {
                    mapping.bytes += chunk;
                } else {
                    transfer->mappings.push_back(
                        MatrixBurstDesc::Transfer::Mapping{
                            static_cast<uint16_t>(packet_offset),
                            static_cast<uint32_t>(m_offset),
                            static_cast<uint16_t>(chunk)});
                }
            } else {
                transfer->mappings.push_back(
                    MatrixBurstDesc::Transfer::Mapping{
                        static_cast<uint16_t>(packet_offset),
                        static_cast<uint32_t>(m_offset),
                        static_cast<uint16_t>(chunk)});
            }
            memory_offset += chunk;
            m_offset += chunk;
            bytes -= chunk;
        }
    };

    const bool whole_register = name.find("_1r") != std::string::npos;
    const bool strided = name.find("_st") != std::string::npos ||
        name.find("_tst") != std::string::npos;
    const bool transposed_stride = name.find("_tst") != std::string::npos;
    const bool column_major = name.find("_cm") != std::string::npos;
    const uint64_t represented_bytes = static_cast<uint64_t>(squares) * n *
        n * elem_bytes;

    if (whole_register || represented_bytes != work.desc.bytes) {
        append(0, 0, work.desc.bytes);
    } else if (strided && transposed_stride) {
        for (unsigned square = 0; square < squares; ++square) {
            for (unsigned col = 0; col < n; ++col) {
                const uint64_t segment = square * n + col;
                for (unsigned row = 0; row < n; ++row) {
                    const uint64_t memory_offset =
                        segment * work.desc.stride + row * elem_bytes;
                    const uint64_t m_offset =
                        ((square * n + row) * n + col) * elem_bytes;
                    append(memory_offset, m_offset, elem_bytes);
                }
            }
        }
    } else if (strided) {
        for (unsigned row = 0; row < n; ++row) {
            for (unsigned square = 0; square < squares; ++square) {
                for (unsigned col = 0; col < n; ++col) {
                    const uint64_t memory_offset = row * work.desc.stride +
                        (square * n + col) * elem_bytes;
                    const uint64_t m_offset =
                        ((square * n + row) * n + col) * elem_bytes;
                    append(memory_offset, m_offset, elem_bytes);
                }
            }
        }
    } else if (column_major) {
        for (unsigned square = 0; square < squares; ++square) {
            for (unsigned col = 0; col < n; ++col) {
                for (unsigned row = 0; row < n; ++row) {
                    const uint64_t memory_offset =
                        ((square * n + col) * n + row) * elem_bytes;
                    const uint64_t m_offset =
                        ((square * n + row) * n + col) * elem_bytes;
                    append(memory_offset, m_offset, elem_bytes);
                }
            }
        }
    } else {
        append(0, 0, work.desc.bytes);
    }
    panic_if(work.desc.transfers.empty(),
             "AME M-LSU built an empty transfer descriptor");
}

void
MatrixLsu::issueStoreRead(Work &work)
{
    if (!work.desc.isStore || work.storeBytesIssued >= work.desc.bytes)
        return;
    const MRegion region = mapMByte(
        work, work.storeBytesIssued,
        std::min(params.packetBytes, mrf.fragmentBytes()));
    if (!scoreboard.mRegionReadyForConsumer(work.desc.contextId, region))
        return;
    LogicalBankRequest request;
    request.contextId = work.desc.contextId;
    request.ameId = work.desc.ameId;
    request.file = RegisterFileKind::Mrf;
    request.access = BankAccess::Read;
    request.client = BankClient::Mlsu;
    request.reg = region.reg;
    request.byteOffset = region.byteBegin;
    request.byteCount = region.byteEnd - region.byteBegin;
    request.final = work.storeBytesIssued + request.byteCount == work.desc.bytes;
    request.tag = {work.desc.contextId, work.desc.ameId,
        OperandSlot::StoreSource,
        static_cast<uint16_t>(work.storeBytesIssued), 0,
        static_cast<uint16_t>(work.storeReadsIssued)};
    if (!mrf.canAcceptNext(request))
        return;
    mrf.acceptNext(std::move(request));
    ++work.storeReadsIssued;
    work.storeBytesIssued += region.byteEnd - region.byteBegin;
}

bool
MatrixLsu::storeTransferReady(
    const Work &work, const MatrixBurstDesc::Transfer &transfer) const
{
    if (!work.desc.isStore)
        return true;
    for (const auto &mapping : transfer.mappings) {
        panic_if(mapping.mOffset + mapping.bytes >
                     work.storeBytesReady.size(),
                 "AME store transfer exceeds source coverage");
        if (!std::all_of(
                work.storeBytesReady.begin() + mapping.mOffset,
                work.storeBytesReady.begin() + mapping.mOffset +
                    mapping.bytes,
                [](bool ready) { return ready; })) {
            return false;
        }
    }
    return true;
}

void
MatrixLsu::issueTransaction(Work &work, uint64_t cycle)
{
    if (work.nextTransfer >= work.desc.transfers.size() ||
        outstanding.size() >= params.maxOutstanding) {
        return;
    }
    const auto &transfer = work.desc.transfers[work.nextTransfer];
    if (!storeTransferReady(work, transfer))
        return;
    const uint64_t id = nextTxnId;
    const unsigned reorder_delay =
        (params.reorderWindow - (id % params.reorderWindow)) %
        params.reorderWindow;
    Transaction transaction{work.desc.contextId, id, transfer.memoryOffset,
        transfer.bytes, transfer.mappings, 0, 0,
        cycle + params.responseLatency + reorder_delay};
    if (params.sendTimingRequest &&
        !params.sendTimingRequest(MatrixLsuParams::TimingRequest{
            work.desc.contextId, id, work.desc.base + transfer.memoryOffset,
            transfer.bytes, work.desc.isStore})) {
        return;
    }
    ++nextTxnId;
    outstanding.push_back(transaction);
    ++work.packetsIssued;
    ++work.nextTransfer;
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u MLSU_REQ desc=%llu txn=%llu "
            "addr=%#x bytes=%u\n",
            cycle, work.desc.contextId, work.desc.ameId, work.desc.descId, id,
            work.desc.base + transfer.memoryOffset, transfer.bytes);
}

void
MatrixLsu::returnTransaction(
    const Transaction &transaction, uint64_t cycle)
{
    auto work = works.find(transaction.contextId);
    if (work != works.end() && !work->second.killed) {
        ++work->second.packetsReturned;
        if (work->second.desc.isLoad)
            work->second.loadWriteQueue.push_back(transaction);
    }
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu MLSU_RESP txn=%llu bytes=%u\n",
            cycle, transaction.contextId, transaction.id, transaction.bytes);
}

void
MatrixLsu::completeTimingResponse(uint64_t transaction_id)
{
    const auto it = std::find_if(outstanding.begin(), outstanding.end(),
        [transaction_id](const Transaction &transaction) {
            return transaction.id == transaction_id;
        });
    panic_if(it == outstanding.end(),
             "AME M-LSU response for unknown transaction %llu",
             transaction_id);
    returnedTransactions.push_back(*it);
    outstanding.erase(it);
}

void
MatrixLsu::issueLoadWrite(Work &work)
{
    if (!work.desc.isLoad || work.loadWriteQueue.empty())
        return;
    auto &transaction = work.loadWriteQueue.front();
    panic_if(transaction.mappingIndex >= transaction.mappings.size(),
             "AME load packet has no remaining MRF mapping");
    const auto &mapping =
        transaction.mappings[transaction.mappingIndex];
    const MRegion region = mapMByte(
        work, mapping.mOffset + transaction.mappingBytesIssued,
        std::min<unsigned>(
            mapping.bytes - transaction.mappingBytesIssued,
            mrf.fragmentBytes()));
    LogicalBankRequest request;
    request.contextId = work.desc.contextId;
    request.ameId = work.desc.ameId;
    request.file = RegisterFileKind::Mrf;
    request.access = BankAccess::Write;
    request.client = BankClient::Mlsu;
    request.reg = region.reg;
    request.byteOffset = region.byteBegin;
    request.byteCount = region.byteEnd - region.byteBegin;
    request.final = work.loadBytesIssued + request.byteCount == work.desc.bytes;
    request.tag = {work.desc.contextId, work.desc.ameId, OperandSlot::OldDest,
        static_cast<uint16_t>(mapping.mOffset +
                              transaction.mappingBytesIssued),
        0, static_cast<uint16_t>(work.loadBytesIssued /
                                 mrf.fragmentBytes())};
    if (!mrf.canAcceptNext(request))
        return;
    mrf.acceptNext(std::move(request));
    const unsigned bytes = region.byteEnd - region.byteBegin;
    work.loadBytesIssued += bytes;
    transaction.mappingBytesIssued += bytes;
    if (transaction.mappingBytesIssued == mapping.bytes) {
        ++transaction.mappingIndex;
        transaction.mappingBytesIssued = 0;
    }
    if (transaction.mappingIndex == transaction.mappings.size())
        work.loadWriteQueue.pop_front();
}

MRegion
MatrixLsu::mapMByte(
    const Work &work, uint32_t byte_offset, uint32_t max_bytes) const
{
    const auto &desc = contexts.get(work.desc.contextId).desc;
    const auto &groups = work.desc.isLoad ? desc.mDestinationGroups :
                                           desc.mSourceGroups;
    panic_if(groups.empty() || groups.front().empty(),
             "AME M-LSU has no matrix register group");
    uint32_t cursor = byte_offset;
    for (const auto &region : groups.front()) {
        const uint32_t length = region.byteEnd - region.byteBegin;
        if (cursor >= length) {
            cursor -= length;
            continue;
        }
        const uint32_t begin = region.byteBegin + cursor;
        const uint32_t word_remaining = mrf.fragmentBytes() -
            (begin % mrf.fragmentBytes());
        const uint32_t bytes = std::min<uint32_t>(
            {max_bytes, length - cursor, word_remaining,
             work.desc.bytes - byte_offset});
        return MRegion{region.tid, region.reg, begin, begin + bytes};
    }
    panic("AME M-LSU byte offset %u outside matrix register group",
          byte_offset);
}

void
MatrixLsu::acceptResponse(const BankResponse &response)
{
    const auto it = works.find(response.contextId);
    if (it == works.end())
        return;
    auto &work = it->second;
    if (response.access == BankAccess::Read) {
        ++work.storeReadsDone;
        const unsigned offset = response.tag.squareIndex;
        panic_if(offset + response.byteCount > work.storeBytesReady.size(),
                 "AME store read response exceeds source coverage");
        std::fill(work.storeBytesReady.begin() + offset,
                  work.storeBytesReady.begin() + offset + response.byteCount,
                  true);
    } else {
        work.loadBytesDone += response.byteCount;
        const auto &desc = contexts.get(response.contextId).desc;
        const ThreadID tid = desc.key.threadId;
        scoreboard.publishMWriteNext(response.ameId,
            MRegion{tid, response.reg, response.byteOffset,
                    response.byteOffset + response.byteCount});
    }
}

void
MatrixLsu::evaluate(uint64_t cycle)
{
    acceptCommands();
    unsigned returned = 0;
    if (params.sendTimingRequest) {
        while (!returnedTransactions.empty() &&
               returned < params.responseWidth) {
            returnTransaction(returnedTransactions.front(), cycle);
            returnedTransactions.pop_front();
            ++returned;
        }
    } else {
        for (auto it = outstanding.begin();
             it != outstanding.end() && returned < params.responseWidth;) {
            if (it->readyCycle > cycle) {
                ++it;
                continue;
            }
            returnTransaction(*it, cycle);
            it = outstanding.erase(it);
            ++returned;
        }
    }

    for (auto it = works.begin(); it != works.end();) {
        auto &work = it->second;
        if (work.killed) {
            if (std::none_of(outstanding.begin(), outstanding.end(),
                    [&work](const Transaction &transaction) {
                        return transaction.contextId == work.desc.contextId;
                    }) &&
                std::none_of(returnedTransactions.begin(),
                    returnedTransactions.end(),
                    [&work](const Transaction &transaction) {
                        return transaction.contextId == work.desc.contextId;
                    })) {
                it = works.erase(it);
            } else {
                ++it;
            }
            continue;
        }
        issueStoreRead(work);
        issueTransaction(work, cycle);
        issueLoadWrite(work);
        const bool complete = work.desc.isLoad ?
            work.loadBytesDone == work.desc.bytes :
            work.packetsReturned == work.packetCount;
        if (complete && done.canPushNext()) {
            done.pushNext(TargetDone{work.desc.contextId, TargetMlsu});
            it = works.erase(it);
        } else {
            ++it;
        }
    }
}

void
MatrixLsu::advance()
{
    requests.advance();
    done.advance();
}

void
MatrixLsu::cancel(AmeContextId id)
{
    requests.eraseIf([id](const AmePeRequest &request) {
        return request.contextId == id;
    });
    done.eraseIf([id](const TargetDone &completion) {
        return completion.contextId == id;
    });
    const auto it = works.find(id);
    if (it != works.end())
        it->second.killed = true;
}

bool
MatrixLsu::active() const
{
    return requests.active() || done.active() || !works.empty() ||
        !outstanding.empty() || !returnedTransactions.empty();
}

} // namespace ame
} // namespace minor
} // namespace gem5
