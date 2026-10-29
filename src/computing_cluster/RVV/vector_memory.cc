/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vector_memory.hh"

#include <algorithm>
#include <set>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"
#include "enums/OpClass.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

unsigned
countBits(uint64_t value)
{
    unsigned count = 0;
    while (value != 0) {
        count += value & 1;
        value >>= 1;
    }
    return count;
}

} // anonymous namespace

VectorMemoryUnit::VectorMemoryUnit(
    InstContextTable &contexts_, RvvChannel<VectorFuReq> &requests_,
    RvvChannel<VectorFuResp> &results_, const VectorMemoryParams &params_)
    : contexts(contexts_), requests(requests_), results(results_),
      params(params_)
{
    fatal_if(params.numLanes == 0 || params.numLanes > 64 ||
             params.maxOutstanding == 0 || params.lineBytes == 0 ||
             params.responseLatency == 0 || params.reorderWindow == 0 ||
             params.responseWidth == 0,
             "Invalid RVV P1 VLSU parameters");
}

void
VectorMemoryUnit::registerBeatNext(const VecBeat &beat)
{
    pendingInsts.emplace_back(beat.tag.instId, beat.activeLaneMask);
}

uint32_t
VectorMemoryUnit::laneWordCount(const RvvInstDesc &desc, LaneId lane) const
{
    const uint64_t lane_span = std::max<uint64_t>(
        1, desc.elementsPerLaneChunk);
    uint64_t first = static_cast<uint64_t>(desc.config.vstart) +
        static_cast<uint64_t>(lane) * lane_span;
    uint32_t words = 1;

    while (first < desc.config.vl) {
        const uint64_t elements = std::min<uint64_t>(
            lane_span, static_cast<uint64_t>(desc.config.vl) - first);
        if (first + elements >= desc.config.vl)
            break;
        first += static_cast<uint64_t>(params.numLanes) * lane_span;
        ++words;
    }
    return words;
}

uint64_t
VectorMemoryUnit::expectedLaneMask(
    const RvvInstDesc &desc, uint32_t word_index) const
{
    uint64_t mask = 0;
    for (LaneId lane = 0; lane < params.numLanes; ++lane) {
        if (word_index < laneWordCount(desc, lane))
            mask |= 1ULL << lane;
    }
    return mask;
}

bool
VectorMemoryUnit::hasPendingWork(RvvInstId id) const
{
    for (const auto &[tag, batch] : batches) {
        if (tag.instId == id)
            return true;
    }
    for (const auto &[tag, operation] : operations) {
        if (tag.instId == id)
            return true;
    }
    const auto transaction_has_id = [id](const auto &transactions) {
        for (const auto &transaction : transactions) {
            for (const auto &consumer : transaction.consumers) {
                if (consumer.instId == id)
                    return true;
            }
        }
        return false;
    };
    return transaction_has_id(pendingTransactions) ||
        transaction_has_id(outstanding);
}

Addr
VectorMemoryUnit::elementAddress(
    const VectorFuReq &request, const RvvInstDesc &desc,
    unsigned element) const
{
    const Addr base = desc.packet.scalarSrcs.empty() ? 0 :
        desc.packet.scalarSrcs.front().value;
    const unsigned element_bytes = std::max(
        1U, static_cast<unsigned>(desc.config.sewBits) / 8U);
    int64_t stride = element_bytes;
    if ((desc.packet.opClass == enums::SimdStridedLoad ||
         desc.packet.opClass == enums::SimdStridedStore) &&
        desc.packet.scalarSrcs.size() > 1) {
        stride = static_cast<int64_t>(desc.packet.scalarSrcs[1].value);
    }
    return base + static_cast<int64_t>(request.firstElem + element) * stride;
}

void
VectorMemoryUnit::evaluate(uint64_t cycle)
{
    if (!params.sendTimingRequest) {
        for (auto it = outstanding.begin(); it != outstanding.end(); ) {
            if (it->readyCycle > cycle) {
                ++it;
                continue;
            }
            DPRINTF(RVVEngine,
                    "cycle=%llu inst=- beat=- lane=- VLSU RESP txn=%llu "
                    "line=%#x consumers=%u\n",
                    cycle, it->id, it->line, it->consumers.size());
            const uint64_t id = it->id;
            completeTransaction(id);
            it = outstanding.begin();
        }
    }

    unsigned returned = 0;
    for (auto it = operations.begin();
         it != operations.end() && returned < params.responseWidth; ) {
        if (it->second.pendingTransactions != 0 || !results.canPushNext()) {
            ++it;
            continue;
        }
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u VLSU RESULT\n",
                cycle, it->first.instId, it->first.beatId,
                it->first.laneId);
        results.pushNext(std::move(it->second.response));
        it = operations.erase(it);
        ++returned;
    }

    const size_t request_count = requests.size();
    for (size_t index = 0; index < request_count; ++index) {
        const auto &request = requests.at(index);
        auto registration = registeredInsts.find(request.tag.instId);
        panic_if(registration == registeredInsts.end(),
                 "VLSU request has no registered instruction");
        const WordTag tag{request.tag.instId, request.tag.wordIndex};
        const auto &desc = contexts.get(request.tag.instId).desc;
        auto batch_result = batches.try_emplace(tag, Batch{
            expectedLaneMask(desc, request.tag.wordIndex) &
                registration->second.activeLaneMask,
            {}});
        auto batch = batch_result.first;
        panic_if(request.tag.laneId >= 64 ||
                 (batch->second.expectedMask &
                  (1ULL << request.tag.laneId)) == 0,
                 "VLSU request lane is not active for its beat");
        const auto [it, inserted] = batch->second.requests.emplace(
            request.tag.laneId, request);
        panic_if(!inserted, "Duplicate VLSU lane request");
        if (request.last)
            registration->second.pendingFinalLaneMask &=
                ~(1ULL << request.tag.laneId);
        requests.popNext();
    }

    if (pendingTransactions.empty()) {
        for (auto batch = batches.begin(); batch != batches.end(); ++batch) {
            if (batch->second.requests.size() !=
                countBits(batch->second.expectedMask)) {
                continue;
            }

            std::map<Addr, std::set<LaneTag>> line_consumers;
            std::map<Addr, bool> line_stores;
            std::map<LaneTag, VectorFuResp> responses;
            for (auto &[lane, request] : batch->second.requests) {
                const auto &desc = contexts.get(request.tag.instId).desc;
                responses.emplace(request.tag, VectorFuResp{
                    request.tag, request.fuType, request.destReg,
                    request.groupByteOffset, request.byteEnable});
                for (unsigned elem = 0; elem < request.elemCount; ++elem) {
                    const Addr address = elementAddress(request, desc, elem);
                    const Addr line = address - address % params.lineBytes;
                    line_consumers[line].insert(request.tag);
                    const bool store = isVectorStore(desc.packet);
                    const auto [store_it, inserted] =
                        line_stores.emplace(line, store);
                    panic_if(!inserted && store_it->second != store,
                             "RVV VLSU coalesced load and store on one line");
                }
                const Addr base = desc.packet.scalarSrcs.empty() ? 0 :
                    desc.packet.scalarSrcs.front().value;
                const unsigned element_bytes = std::max(
                    1U, static_cast<unsigned>(desc.config.sewBits) / 8U);
                request.burst = VectorBurstDesc{
                    0, request.tag.rvvSlotId, base + request.firstElem *
                    element_bytes,
                    request.elemCount * element_bytes,
                    static_cast<uint16_t>(request.elemCount), true,
                    isVectorLoad(desc.packet), isVectorStore(desc.packet),
                    request.firstElem};
            }

            for (auto &[tag, response] : responses) {
                unsigned transactions = 0;
                for (const auto &[line, consumers] : line_consumers)
                    transactions += consumers.find(tag) != consumers.end();
                const auto [it, inserted] = operations.emplace(
                    tag, Operation{std::move(response), transactions});
                panic_if(!inserted, "Duplicate RVV VLSU lane operation");
            }
            for (const auto &[line, consumers] : line_consumers) {
                pendingTransactions.push_back(Transaction{
                    0, line, params.lineBytes, line_stores.at(line), 0,
                    std::vector<LaneTag>(consumers.begin(),
                                         consumers.end())});
            }
            batches.erase(batch);
            break;
        }
    }

    while (!pendingTransactions.empty() &&
           outstanding.size() < params.maxOutstanding) {
        Transaction transaction = std::move(pendingTransactions.front());
        pendingTransactions.pop_front();
        const uint64_t id = nextTxnId++;
        const unsigned reorder_delay =
            (params.reorderWindow - (id % params.reorderWindow)) %
            params.reorderWindow;
        transaction.id = id;
        if (params.sendTimingRequest) {
            const bool accepted = params.sendTimingRequest(
                VectorMemoryParams::TimingRequest{
                    id, transaction.line, transaction.bytes,
                    transaction.isStore});
            if (!accepted) {
                --nextTxnId;
                pendingTransactions.push_front(std::move(transaction));
                break;
            }
            transaction.readyCycle = 0;
        } else {
            transaction.readyCycle =
                cycle + params.responseLatency + reorder_delay;
        }
        DPRINTF(RVVEngine,
                "cycle=%llu inst=- beat=- lane=- VLSU ISSUE txn=%llu "
                "line=%#x fragments=%u ready=%llu\n",
                cycle, id, transaction.line, transaction.consumers.size(),
                transaction.readyCycle);
        outstanding.push_back(std::move(transaction));
    }

    for (auto it = registeredInsts.begin(); it != registeredInsts.end(); ) {
        if (it->second.pendingFinalLaneMask == 0 &&
            !hasPendingWork(it->first))
            it = registeredInsts.erase(it);
        else
            ++it;
    }
}

void
VectorMemoryUnit::completeTransaction(uint64_t transaction_id)
{
    auto it = std::find_if(outstanding.begin(), outstanding.end(),
        [transaction_id](const Transaction &transaction) {
            return transaction.id == transaction_id;
        });
    if (it == outstanding.end())
        return;

    DPRINTF(RVVEngine,
            "VLSU timing response txn=%llu line=%#x consumers=%u\n",
            it->id, it->line, it->consumers.size());
    for (const auto &tag : it->consumers) {
        auto operation = operations.find(tag);
        if (operation == operations.end())
            continue;
        panic_if(operation->second.pendingTransactions == 0,
                 "RVV VLSU transaction completed twice");
        --operation->second.pendingTransactions;
    }
    outstanding.erase(it);
}

void
VectorMemoryUnit::completeTimingResponse(uint64_t transaction_id)
{
    if (!params.sendTimingRequest)
        return;
    completeTransaction(transaction_id);
}

void
VectorMemoryUnit::advance()
{
    for (const auto &[id, mask] : pendingInsts) {
        const auto [it, inserted] = registeredInsts.emplace(
            id, Registration{mask, mask});
        panic_if(!inserted, "Duplicate VLSU instruction registration");
    }
    pendingInsts.clear();
}

void
VectorMemoryUnit::cancel(RvvInstId id)
{
    requests.eraseIf([id](const VectorFuReq &request) {
        return request.tag.instId == id;
    });
    results.eraseIf([id](const VectorFuResp &response) {
        return response.tag.instId == id;
    });
    for (auto it = batches.begin(); it != batches.end(); ) {
        if (it->first.instId == id)
            it = batches.erase(it);
        else
            ++it;
    }
    registeredInsts.erase(id);
    pendingInsts.erase(
        std::remove_if(pendingInsts.begin(), pendingInsts.end(),
            [id](const auto &entry) { return entry.first == id; }),
        pendingInsts.end());
    for (auto it = operations.begin(); it != operations.end(); ) {
        if (it->first.instId == id)
            it = operations.erase(it);
        else
            ++it;
    }
    const auto erase_consumers = [id](auto &transactions,
                                      bool keep_empty) {
        for (auto it = transactions.begin(); it != transactions.end(); ) {
            it->consumers.erase(
                std::remove_if(it->consumers.begin(), it->consumers.end(),
                    [id](const LaneTag &tag) { return tag.instId == id; }),
                it->consumers.end());
            if (it->consumers.empty() && !keep_empty)
                it = transactions.erase(it);
            else
                ++it;
        }
    };
    erase_consumers(pendingTransactions, false);
    /* A real timing response must still be absorbed after its instruction is
     * squashed. Keeping an empty transaction also keeps drain honest. */
    erase_consumers(outstanding, bool(params.sendTimingRequest));
}

bool
VectorMemoryUnit::active() const
{
    return requests.active() || results.active() || !batches.empty() ||
        !registeredInsts.empty() || !pendingInsts.empty() ||
        !operations.empty() ||
        !pendingTransactions.empty() || !outstanding.empty();
}

} // namespace rvv
} // namespace minor
} // namespace gem5
