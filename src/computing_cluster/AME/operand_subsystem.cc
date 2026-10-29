/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/operand_subsystem.hh"

#include <algorithm>
#include <tuple>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

MatrixOperandSubsystem::MatrixOperandSubsystem(
    AmeContextTable &contexts_, AmeScoreboard &scoreboard_,
    BankedRegisterFile &mrf_, const OperandSubsystemParams &params_)
    : contexts(contexts_), scoreboard(scoreboard_), mrf(mrf_), params(params_)
{
    fatal_if(params.kBlock == 0 || params.requestWidth == 0,
             "Invalid AME operand subsystem parameters");
}

void
MatrixOperandSubsystem::start(AmeContextId id, uint32_t total_blocks)
{
    panic_if(works.find(id) != works.end(),
             "Starting duplicate AME operand work id=%llu", id);
    const auto &desc = contexts.get(id).desc;
    Work work;
    work.contextId = id;
    work.ameId = desc.ameId;
    work.config = desc.config;
    work.totalBlocks = std::max(1U, total_blocks);
    if (!desc.mSourceGroups.empty()) {
        const unsigned bits = mElementBits(
            desc.config, desc.mSourceGroups[0].front().reg);
        work.aRegions = formFragments(
            desc.mSourceGroups[0], desc.config, desc.config.transposeA, bits);
        work.hasA = true;
    }
    if (desc.mSourceGroups.size() > 1) {
        const unsigned bits = mElementBits(
            desc.config, desc.mSourceGroups[1].front().reg);
        work.bRegions = formFragments(
            desc.mSourceGroups[1], desc.config, desc.config.transposeB, bits);
        work.hasB = true;
    } else if (work.hasA) {
        work.bRegions = work.aRegions;
        work.hasB = true;
    }
    const auto [it, inserted] = works.emplace(id, std::move(work));
    panic_if(!inserted, "Duplicate AME MOFU context id=%llu", id);
    fillFreeBuffers(it->second);
}

void
MatrixOperandSubsystem::fillFreeBuffers(Work &work)
{
    for (auto &buffer : work.buffers) {
        if (buffer.valid || work.nextBlock >= work.totalBlocks)
            continue;
        buffer = Buffer{};
        buffer.valid = true;
        buffer.block = work.nextBlock++;
        const unsigned k_blocks = std::max(1U,
            static_cast<unsigned>((work.config.kLogical + params.kBlock - 1) /
                                  params.kBlock));
        const unsigned k_block = buffer.block % k_blocks;
        const auto count = [k_blocks, k_block](size_t fragments) {
            const unsigned base = fragments / k_blocks;
            const unsigned remainder = fragments % k_blocks;
            return base + (k_block < remainder ? 1 : 0);
        };
        buffer.aExpected = work.hasA ? count(work.aRegions.size()) : 0;
        buffer.bExpected = work.hasB ? count(work.bRegions.size()) : 0;
    }
}

std::vector<MRegion>
MatrixOperandSubsystem::formFragments(
    const std::vector<MRegion> &regions, const AmeConfig &config,
    bool transpose, unsigned element_bits) const
{
    std::vector<MRegion> result;
    const unsigned elem_bytes = std::max(1U,
        (element_bits + 7) / 8);
    for (const auto &region : regions) {
        for (uint32_t offset = region.byteBegin; offset < region.byteEnd;) {
            const unsigned word_remaining = mrf.fragmentBytes() -
                (offset % mrf.fragmentBytes());
            const uint16_t bytes = std::min<uint32_t>(
                transpose ? std::min(word_remaining, elem_bytes) :
                            word_remaining,
                region.byteEnd - offset);
            result.push_back(MRegion{
                region.tid, region.reg, offset, offset + bytes});
            offset += bytes;
        }
    }

    if (!transpose || result.size() < 2)
        return result;

    const unsigned n = std::max(1U, static_cast<unsigned>(config.nAme));
    std::stable_sort(result.begin(), result.end(),
        [n, elem_bytes](const MRegion &left, const MRegion &right) {
            const auto rank = [n, elem_bytes](const MRegion &region) {
                const unsigned fragment = region.byteBegin /
                    std::max(1U, elem_bytes);
                const unsigned square = fragment / (n * n);
                const unsigned element = fragment % (n * n);
                const unsigned row = element / n;
                const unsigned col = element % n;
                return square * n * n + col * n + row;
            };
            const auto left_key = std::make_tuple(
                left.reg, rank(left), left.byteBegin);
            const auto right_key = std::make_tuple(
                right.reg, rank(right), right.byteBegin);
            return left_key < right_key;
        });
    return result;
}

MRegion
MatrixOperandSubsystem::fragment(
    const Work &work, const Buffer &buffer, bool operand_a,
    unsigned index) const
{
    const auto &fragments = operand_a ? work.aRegions : work.bRegions;
    const unsigned k_blocks = std::max(1U,
        static_cast<unsigned>((work.config.kLogical + params.kBlock - 1) /
                              params.kBlock));
    const unsigned k_block = buffer.block % k_blocks;
    const unsigned base = fragments.size() / k_blocks;
    const unsigned remainder = fragments.size() % k_blocks;
    const unsigned begin = k_block * base + std::min(k_block, remainder);
    panic_if(begin + index >= fragments.size(),
             "AME MOFU fragment index outside operand coverage");
    return fragments[begin + index];
}

bool
MatrixOperandSubsystem::issueOne(
    Work &work, Buffer &buffer, bool operand_a)
{
    unsigned &issued = operand_a ? buffer.aIssued : buffer.bIssued;
    const unsigned expected = operand_a ? buffer.aExpected : buffer.bExpected;
    if (issued >= expected)
        return false;
    const MRegion source = fragment(work, buffer, operand_a, issued);
    if (!scoreboard.mRegionReadyForConsumer(work.contextId, source))
        return false;

    LogicalBankRequest request;
    request.contextId = work.contextId;
    request.ameId = work.ameId;
    request.file = RegisterFileKind::Mrf;
    request.access = BankAccess::Read;
    request.client = operand_a ? BankClient::SaA : BankClient::SaB;
    request.reg = source.reg;
    request.byteOffset = source.byteBegin;
    request.byteCount = source.byteEnd - source.byteBegin;
    request.highPriority = buffer.block == 0;
    request.final = issued + 1 == expected;
    request.tag = {work.contextId, work.ameId,
        operand_a ? OperandSlot::A : OperandSlot::B,
        static_cast<uint16_t>(buffer.block), 0,
        static_cast<uint16_t>(issued)};
    if (!mrf.canAcceptNext(request))
        return false;
    mrf.acceptNext(std::move(request));
    ++issued;
    return true;
}

void
MatrixOperandSubsystem::evaluate(uint64_t cycle)
{
    if (works.empty())
        return;
    std::vector<AmeContextId> ids;
    ids.reserve(works.size());
    for (const auto &[id, work] : works)
        ids.push_back(id);

    unsigned issued = 0;
    for (size_t offset = 0; offset < ids.size() && issued < params.requestWidth;
         ++offset) {
        const size_t index = (roundRobin + offset) % ids.size();
        auto &work = works.at(ids[index]);
        for (auto &buffer : work.buffers) {
            if (!buffer.valid)
                continue;
            if (issued < params.requestWidth && issueOne(work, buffer, true))
                ++issued;
            if (issued < params.requestWidth && issueOne(work, buffer, false))
                ++issued;
        }
    }
    roundRobin = (roundRobin + 1) % ids.size();
    if (issued != 0) {
        DPRINTF(AMEFlow, "cycle=%llu MOFU issued=%u\n", cycle, issued);
    }
}

void
MatrixOperandSubsystem::acceptResponse(const BankResponse &response)
{
    const auto it = works.find(response.contextId);
    if (it == works.end())
        return;
    Work &work = it->second;
    const uint32_t block = response.tag.squareIndex;
    const auto buffer = std::find_if(work.buffers.begin(), work.buffers.end(),
        [block](const Buffer &entry) {
            return entry.valid && entry.block == block;
        });
    panic_if(buffer == work.buffers.end(),
             "AME MOFU response has no matching double buffer");
    if (response.client == BankClient::SaA) {
        ++buffer->aReceived;
        panic_if(buffer->aReceived > buffer->aExpected,
                 "Duplicate AME A fragment response");
    } else {
        ++buffer->bReceived;
        panic_if(buffer->bReceived > buffer->bExpected,
                 "Duplicate AME B fragment response");
    }
}

bool
MatrixOperandSubsystem::blockReady(AmeContextId id, uint32_t block) const
{
    const auto work = works.find(id);
    if (work == works.end())
        return false;
    const auto buffer = std::find_if(
        work->second.buffers.begin(), work->second.buffers.end(),
        [block](const Buffer &entry) {
            return entry.valid && entry.block == block;
        });
    return buffer != work->second.buffers.end() &&
        buffer->aReceived == buffer->aExpected &&
        buffer->bReceived == buffer->bExpected;
}

void
MatrixOperandSubsystem::consumeBlock(AmeContextId id, uint32_t block)
{
    auto work = works.find(id);
    panic_if(work == works.end(), "Consuming unknown AME MOFU work");
    const auto buffer = std::find_if(
        work->second.buffers.begin(), work->second.buffers.end(),
        [block](const Buffer &entry) {
            return entry.valid && entry.block == block;
        });
    panic_if(buffer == work->second.buffers.end() ||
             buffer->aReceived != buffer->aExpected ||
             buffer->bReceived != buffer->bExpected,
             "Consuming incomplete AME operand buffer");
    buffer->valid = false;
    fillFreeBuffers(work->second);
    if (work->second.nextBlock == work->second.totalBlocks &&
        std::none_of(work->second.buffers.begin(), work->second.buffers.end(),
            [](const Buffer &entry) { return entry.valid; })) {
        works.erase(work);
    }
}

void
MatrixOperandSubsystem::cancel(AmeContextId id)
{
    works.erase(id);
}

} // namespace ame
} // namespace minor
} // namespace gem5
