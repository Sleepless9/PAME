/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <gtest/gtest.h>

#include "arch/riscv/regs/vector.hh"
#include "computing_cluster/RVV/vector_masku.hh"
#include "computing_cluster/RVV/vector_sldu.hh"
#include "computing_cluster/RVV/global_scoreboard.hh"
#include "computing_cluster/RVV/pipeline.hh"
#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/vector_fu.hh"
#include "computing_cluster/RVV/vector_memory.hh"
#include "computing_cluster/RVV/vrf_bank.hh"
#include "computing_cluster/RVV/result_progress.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

RegId
vreg(unsigned index)
{
    return RegId(RiscvISA::vecRegClass, index);
}

RvvInstContext
context(RvvInstId id, ThreadID tid, std::vector<RegId> src,
        std::vector<RegId> dest, int8_t lmul = 0)
{
    RvvInstContext result;
    result.desc.id = id;
    result.desc.key = {tid, id};
    result.desc.packet.key = result.desc.key;
    result.desc.packet.srcRegs = std::move(src);
    result.desc.packet.destRegs = std::move(dest);
    result.desc.config.lmulLog2 = lmul;
    return result;
}

VectorFuReq
fuRequest(RvvInstId id, BeatId beat = 0)
{
    VectorFuReq request;
    request.tag = {id, beat, 0};
    request.fuType = VecFuType::Alu;
    request.destReg = vreg(1);
    return request;
}

InstQueuePacket
rvvPacket(InstSeqNum seq, unsigned vl)
{
    InstQueuePacket packet;
    packet.key = {0, seq};
    packet.engine = InstQueueEngine::RVV;
    packet.mnemonic = "vadd.vv";
    packet.opClass = SimdAluOp;
    packet.encodedInst = (2ULL << 35) | (1ULL << 38) |
        (1ULL << 25); // SEW=32, tail agnostic, unmasked.
    packet.vectorLengthValid = true;
    packet.vectorLength = vl;
    packet.srcRegs = {vreg(1), vreg(2)};
    packet.destRegs = {vreg(3)};
    return packet;
}

PipelineParams
pipelineParams()
{
    PipelineParams params;
    params.numLanes = 2;
    params.laneWidthBits = 64;
    params.maxPackedElementsPerLane = 1;
    params.maxInflightInsts = 4;
    params.lane.numLanes = params.numLanes;
    params.lane.laneWidthBits = params.laneWidthBits;
    params.lane.vlenBits = params.vlenBits;
    params.sldu.numLanes = params.numLanes;
    return params;
}

} // anonymous namespace

TEST(RvvDispatcherTest, SnapshotsDynamicConfigAtAccept)
{
    MinorInstQueue queue("queue", 2, true, false, {});
    InstContextTable contexts(2);
    GlobalVectorScoreboard scoreboard;
    RvvChannel<VectorInstToken> dispatch(2);
    RvvDispatcher dispatcher(
        queue, contexts, scoreboard, dispatch,
        DispatcherParams{2, 64, 1},
        [](RvvConfig &config, const InstQueuePacket &) {
            config.vstart = 2;
            config.vxrm = 3;
            config.frm = 4;
        });
    ASSERT_TRUE(queue.dispatch(rvvPacket(1, 4)));

    dispatcher.evaluate(7);

    const auto &config = contexts.get(1).desc.config;
    EXPECT_EQ(config.vl, 4);
    EXPECT_EQ(config.vstart, 2);
    EXPECT_EQ(config.vxrm, 3);
    EXPECT_EQ(config.frm, 4);
}

TEST(RvvInsnSlotAllocatorTest, ReusesFiniteInstructionSlots)
{
    RvvInsnSlotAllocator allocator(2);
    const auto first = allocator.allocate();
    const auto second = allocator.allocate();
    EXPECT_NE(first, second);
    EXPECT_FALSE(allocator.canAllocate());
    allocator.release(first);
    EXPECT_TRUE(allocator.canAllocate());
    EXPECT_EQ(allocator.allocate(), first);
}

TEST(ResultProgressTest, VisibilityStartsAfterWriteCommit)
{
    ResultProgressTable progress;
    progress.allocate(3);
    EXPECT_EQ(progress.wordsWritten(3), 0);
    progress.publishWriteNext(3);
    EXPECT_EQ(progress.wordsWritten(3), 0);
    progress.advance();
    EXPECT_EQ(progress.wordsWritten(3), 1);
    EXPECT_TRUE(progress.canRead(3, 0));
    EXPECT_FALSE(progress.canRead(3, 1));
}

TEST(VectorSlduTest, ReductionUsesExplicitPhases)
{
    InstContextTable contexts(2);
    auto reduction = context(7, 0, {vreg(1)}, {vreg(2)});
    reduction.desc.packet.opClass = SimdReduceAddOp;
    reduction.desc.packet.mnemonic = "vredsum.vs";
    reduction.desc.config.vl = 1;
    reduction.desc.elementsPerLaneChunk = 1;
    contexts.allocate(std::move(reduction));
    RvvChannel<VectorFuReq> lane_requests(4);
    RvvChannel<VectorFuResp> lane_results(4);
    SlduParams params;
    params.numLanes = 2;
    params.intraLaneReductionLatency = 2;
    params.reductionStageLatency = 2;
    params.resultWidth = 2;
    VectorSldu sldu(
        contexts, lane_requests, lane_results, params, 2);
    PeRequest request;
    request.contextId = 7;
    request.rvvSlotId = 1;
    request.opcodeClass = VecFuType::Sldu;
    sldu.acceptNext(request);
    sldu.advance();
    sldu.evaluate(0);
    sldu.advance();

    auto lane0 = fuRequest(7);
    lane0.fuType = VecFuType::Sldu;
    auto lane1 = lane0;
    lane1.tag.laneId = 1;
    lane_requests.pushNext(lane0);
    lane_requests.pushNext(lane1);
    lane_requests.advance();
    for (uint64_t cycle = 1; cycle < 8; ++cycle) {
        sldu.evaluate(cycle);
        sldu.advance();
        EXPECT_TRUE(lane_results.empty());
    }
    sldu.evaluate(8);
    sldu.advance();
    EXPECT_EQ(lane_results.size(), 2);
    ASSERT_FALSE(sldu.doneChannel().empty());
    EXPECT_EQ(sldu.doneChannel().front().contextId, 7);
}

TEST(RvvChannelTest, UpdatesOnlyAtCycleBoundary)
{
    RvvChannel<unsigned> channel(2);
    channel.pushNext(7);
    EXPECT_TRUE(channel.empty());
    channel.advance();
    ASSERT_FALSE(channel.empty());
    EXPECT_EQ(channel.front(), 7);

    channel.popNext();
    EXPECT_FALSE(channel.empty());
    channel.advance();
    EXPECT_TRUE(channel.empty());
}

TEST(GlobalVectorScoreboardTest, TracksRawWawWarAndThreads)
{
    GlobalVectorScoreboard scoreboard;
    const auto producer = context(1, 0, {}, {vreg(4)});
    const auto raw = context(2, 0, {vreg(4)}, {vreg(5)});
    const auto other_thread = context(3, 1, {vreg(4)}, {vreg(5)});
    scoreboard.allocate(producer);
    scoreboard.allocate(raw);
    scoreboard.allocate(other_thread);

    EXPECT_TRUE(scoreboard.canSequence(1));
    EXPECT_TRUE(scoreboard.canSequence(2));
    EXPECT_FALSE(scoreboard.canSequenceBeat(2, 0, 1));
    EXPECT_TRUE(scoreboard.canSequence(3));

    scoreboard.publishResultReadyNext({1, 0, 0});
    scoreboard.advance();
    EXPECT_TRUE(scoreboard.canSequenceBeat(2, 0, 1));
    EXPECT_TRUE(scoreboard.canBypass(2, vreg(4), {2, 0, 0}));

    scoreboard.release(1);
    EXPECT_TRUE(scoreboard.canSequence(2));
}

TEST(GlobalVectorScoreboardTest, BlocksPartiallyOverlappingLmulGroups)
{
    GlobalVectorScoreboard scoreboard;
    scoreboard.allocate(context(1, 0, {}, {vreg(4)}, 2));
    scoreboard.allocate(context(2, 0, {vreg(7)}, {}, 2));
    EXPECT_FALSE(scoreboard.canSequence(2));
}

TEST(GlobalVectorScoreboardTest, KeepsWawAndWarAtInstructionGranularity)
{
    GlobalVectorScoreboard waw;
    waw.allocate(context(1, 0, {}, {vreg(4)}));
    waw.allocate(context(2, 0, {}, {vreg(4)}));
    EXPECT_FALSE(waw.canSequence(2));

    GlobalVectorScoreboard war;
    war.allocate(context(1, 0, {vreg(4)}, {}));
    war.allocate(context(2, 0, {}, {vreg(4)}));
    EXPECT_FALSE(war.canSequence(2));
}

TEST(GlobalVectorScoreboardTest, DoesNotExpandMaskRegisterWithLmul)
{
    GlobalVectorScoreboard scoreboard;
    scoreboard.allocate(context(1, 0, {}, {vreg(1)}, 1));
    auto masked = context(2, 0, {vreg(0)}, {}, 1);
    masked.desc.config.vm = false;
    scoreboard.allocate(masked);

    EXPECT_TRUE(scoreboard.canSequence(2));
}

TEST(VrfMapperTest, UsesFixedLaneBankRowMapping)
{
    VrfAddressMapper mapper(4, 64, 8, 256);
    LogicalVrfReq request;
    request.tag = {1, 0, 0};
    request.reg = vreg(3);
    request.groupByteOffset = 0;
    auto mapped = mapper.map(request);
    EXPECT_EQ(mapped.bank, 3);
    EXPECT_EQ(mapped.row, 0);
    EXPECT_EQ(mapped.reg.index(), 3);

    request.tag.laneId = 1;
    request.groupByteOffset = 8;
    mapped = mapper.map(request);
    EXPECT_EQ(mapped.bank, 3);
    EXPECT_EQ(mapped.row, 0);
    EXPECT_EQ(mapped.reg.index(), 3);

    request.tag.laneId = 0;
    request.lmulLog2 = 2;
    request.groupByteOffset = 32;
    mapped = mapper.map(request);
    EXPECT_EQ(mapped.bank, 4);
    EXPECT_EQ(mapped.reg.index(), 4);
}

TEST(VrfMapperTest, MapsFractionalLmulIntoLowRegisterSlice)
{
    VrfAddressMapper mapper(2, 64, 4, 128);
    LogicalVrfReq request;
    request.tag = {1, 0, 0};
    request.reg = vreg(6);
    request.lmulLog2 = -1;
    request.groupByteOffset = 0;
    const auto mapped = mapper.map(request);
    EXPECT_EQ(mapped.reg.index(), 6);
    EXPECT_EQ(mapped.bank, 2);
}

TEST(VrfBankTest, SinglePortPrioritizesWriteback)
{
    VrfBank bank(0, 2);
    RvvChannel<VrfReadResp> reads(2);
    RvvChannel<VrfWriteAck> writes(2);
    VrfBankReq read;
    read.tag = {1, 0, 0};
    read.bank = 0;
    read.access = VrfAccess::Read;
    VrfBankReq write = read;
    write.tag.instId = 2;
    write.access = VrfAccess::Write;

    bank.acceptReadNext(read);
    bank.acceptWriteNext(write);
    bank.advance();
    bank.evaluate(reads, writes, 0, 0);
    bank.advance();
    reads.advance();
    writes.advance();
    ASSERT_FALSE(writes.empty());
    EXPECT_EQ(writes.front().tag.instId, 2);
    EXPECT_TRUE(reads.empty());

    writes.popNext();
    bank.evaluate(reads, writes, 1, 0);
    bank.advance();
    reads.advance();
    writes.advance();
    ASSERT_FALSE(reads.empty());
    EXPECT_EQ(reads.front().tag.instId, 1);
}

TEST(VrfBankTest, RoundRobinPreventsReadWriteStarvation)
{
    VrfBank bank(0, 4, VrfArbitrationPolicy::RoundRobin);
    RvvChannel<VrfReadResp> reads(4);
    RvvChannel<VrfWriteAck> writes(4);
    VrfBankReq read;
    read.tag = {1, 0, 0};
    read.bank = 0;
    read.access = VrfAccess::Read;
    VrfBankReq write = read;
    write.tag.instId = 2;
    write.access = VrfAccess::Write;
    bank.acceptReadNext(read);
    bank.acceptWriteNext(write);
    bank.advance();

    bank.evaluate(reads, writes, 0, 0);
    bank.advance();
    reads.advance();
    writes.advance();
    ASSERT_EQ(writes.size(), 1);

    bank.evaluate(reads, writes, 1, 0);
    bank.advance();
    reads.advance();
    writes.advance();
    ASSERT_EQ(reads.size(), 1);
}

TEST(VectorMemoryTest, CoalescesLaneFragmentsByCacheLine)
{
    InstContextTable contexts(2);
    auto memory = context(1, 0, {}, {vreg(3)});
    memory.desc.packet.opClass = enums::SimdUnitStrideLoad;
    memory.desc.packet.isMemory = true;
    memory.desc.packet.scalarSrcs.push_back({0, RegId(), 0x1000});
    memory.desc.config.sewBits = 32;
    contexts.allocate(std::move(memory));
    RvvChannel<VectorFuReq> requests(4);
    RvvChannel<VectorFuResp> results(4);
    VectorMemoryParams params;
    params.responseLatency = 2;
    params.reorderWindow = 1;
    params.responseWidth = 2;
    VectorMemoryUnit unit(contexts, requests, results, params);
    VecBeat beat;
    beat.tag = {1, 0};
    beat.activeLaneMask = 3;
    unit.registerBeatNext(beat);
    unit.advance();

    VectorFuReq lane0;
    lane0.tag = {1, 0, 0};
    lane0.fuType = VecFuType::Memory;
    lane0.destReg = vreg(3);
    lane0.firstElem = 0;
    lane0.elemCount = 2;
    VectorFuReq lane1 = lane0;
    lane1.tag.laneId = 1;
    lane1.firstElem = 2;
    requests.pushNext(lane0);
    requests.pushNext(lane1);
    requests.advance();

    unit.evaluate(0);
    requests.advance();
    EXPECT_EQ(unit.issuedTransactions(), 1);
    EXPECT_EQ(unit.outstandingSize(), 1);
    unit.evaluate(1);
    results.advance();
    unit.evaluate(2);
    results.advance();
    EXPECT_EQ(results.size(), 2);
}

TEST(VectorMemoryTest, HandlesMultipleWordsWithUnevenLaneTails)
{
    InstContextTable contexts(2);
    auto memory = context(1, 0, {vreg(2)}, {});
    memory.desc.packet.opClass = enums::SimdUnitStrideStore;
    memory.desc.packet.isMemory = true;
    memory.desc.packet.scalarSrcs.push_back({0, RegId(), 0x1000});
    memory.desc.config.sewBits = 8;
    memory.desc.config.vl = 4;
    memory.desc.elementsPerLaneChunk = 1;
    contexts.allocate(std::move(memory));
    RvvChannel<VectorFuReq> requests(8);
    RvvChannel<VectorFuResp> results(8);
    VectorMemoryParams params;
    params.numLanes = 2;
    params.responseLatency = 1;
    params.reorderWindow = 1;
    params.responseWidth = 8;
    VectorMemoryUnit unit(contexts, requests, results, params);
    VecBeat envelope;
    envelope.tag = {1, 0};
    envelope.activeLaneMask = 3;
    unit.registerBeatNext(envelope);
    unit.advance();

    const auto push_word = [&requests](LaneId lane, uint32_t word,
                                       uint32_t first, bool last) {
        VectorFuReq request;
        request.tag = {1, 0, lane, 0, word};
        request.fuType = VecFuType::Memory;
        request.firstElem = first;
        request.elemCount = first < 4 ? 1 : 0;
        request.last = last;
        requests.pushNext(request);
    };
    push_word(0, 0, 0, false);
    push_word(1, 0, 1, false);
    push_word(0, 1, 2, false);
    push_word(1, 1, 3, true);
    push_word(0, 2, 4, true);
    requests.advance();

    for (unsigned cycle = 0; cycle < 10 && results.size() != 5; ++cycle) {
        unit.evaluate(cycle);
        requests.advance();
        results.advance();
        unit.advance();
    }
    EXPECT_EQ(unit.issuedTransactions(), 2);
    EXPECT_EQ(results.size(), 5);
}

TEST(VectorRegFileTest, TracksStableGem5InternalVectorRegisters)
{
    VectorRegFile vrf(0, 4, 64, 8, 512, 2, 8);
    EewInfo tag;
    tag.value = Eew::E64;
    vrf.updateEewTagNext(RiscvISA::VecMemInternalReg0, tag);
    vrf.updateEewTagNext(RiscvISA::NumVecRegs - 1, tag);
    EXPECT_NO_THROW(vrf.advance());
    EXPECT_EQ(vrf.eewTag(RiscvISA::VecMemInternalReg0).value, Eew::E64);
    EXPECT_EQ(vrf.eewTag(RiscvISA::NumVecRegs - 1).value, Eew::E64);
}

TEST(VectorLaneTest, LastMemoryWordReturningFirstDoesNotCompleteLane)
{
    InstContextTable contexts(2);
    auto inst = context(1, 0, {}, {vreg(3)});
    inst.desc.packet.opClass = enums::SimdUnitStrideLoad;
    inst.desc.packet.isMemory = true;
    inst.desc.fuType = VecFuType::Memory;
    inst.desc.config.sewBits = 32;
    inst.desc.config.vl = 2;
    GlobalVectorScoreboard scoreboard;
    contexts.allocate(std::move(inst));
    scoreboard.allocate(contexts.get(1));
    RvvChannel<VectorFuReq> memory(8), slide(8);
    RvvChannel<LaneComplete> completions(8);
    VectorLaneParams params;
    params.numLanes = 1;
    VectorLane lane(0, contexts, scoreboard, memory, slide, params);
    uint64_t cycle = 0;
    const auto step = [&] {
        lane.evaluate(cycle++, completions);
        lane.advance();
        memory.advance(); slide.advance(); completions.advance();
        scoreboard.advance();
    };
    for (unsigned word = 0; word < 2; ++word) {
        LaneBeat beat;
        beat.tag = {1, 0, 0, 0, word};
        beat.firstElem = word;
        beat.elemCount = 1;
        beat.groupByteOffset = word * 4;
        beat.byteEnable = 15;
        beat.last = word == 1;
        lane.acceptBeatNext(beat);
        step();
    }
    for (unsigned i = 0; i < 20; ++i) step();
    ASSERT_EQ(memory.size(), 2);
    const auto respond = [&](unsigned word) {
        VectorFuResp result;
        result.tag = {1, 0, 0, 0, word};
        result.fuType = VecFuType::Memory;
        result.destReg = vreg(3);
        result.groupByteOffset = word * 4;
        result.byteEnable = 15;
        lane.acceptExternalResultNext(result);
        for (unsigned i = 0; i < 20; ++i) step();
    };
    respond(1);
    EXPECT_TRUE(completions.empty());
    respond(0);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_EQ(completions.front().tag.instId, 1);
}

TEST(VectorMemoryTest, StridedTransactionsCanRespondOutOfOrder)
{
    InstContextTable contexts(2);
    auto memory = context(1, 0, {}, {vreg(3)});
    memory.desc.packet.opClass = enums::SimdStridedLoad;
    memory.desc.packet.isMemory = true;
    memory.desc.packet.scalarSrcs.push_back({0, RegId(), 0x1000});
    memory.desc.packet.scalarSrcs.push_back({1, RegId(), 64});
    memory.desc.config.sewBits = 32;
    contexts.allocate(std::move(memory));
    RvvChannel<VectorFuReq> requests(2);
    RvvChannel<VectorFuResp> results(2);
    VectorMemoryParams params;
    params.responseLatency = 2;
    params.reorderWindow = 2;
    VectorMemoryUnit unit(contexts, requests, results, params);
    VecBeat beat;
    beat.tag = {1, 0};
    beat.activeLaneMask = 1;
    unit.registerBeatNext(beat);
    unit.advance();
    VectorFuReq request;
    request.tag = {1, 0, 0};
    request.fuType = VecFuType::Memory;
    request.destReg = vreg(3);
    request.elemCount = 2;
    requests.pushNext(request);
    requests.advance();

    unit.evaluate(0);
    requests.advance();
    ASSERT_EQ(unit.issuedTransactions(), 2);
    unit.evaluate(2);
    EXPECT_EQ(unit.outstandingSize(), 1);
    unit.evaluate(3);
    results.advance();
    EXPECT_FALSE(results.empty());
}

TEST(VectorMemoryTest, MakesProgressWithOneOutstandingTransaction)
{
    InstContextTable contexts(2);
    auto memory = context(1, 0, {}, {vreg(3)});
    memory.desc.packet.opClass = enums::SimdStridedLoad;
    memory.desc.packet.isMemory = true;
    memory.desc.packet.scalarSrcs.push_back({0, RegId(), 0x1000});
    memory.desc.packet.scalarSrcs.push_back({1, RegId(), 64});
    memory.desc.config.sewBits = 32;
    contexts.allocate(std::move(memory));
    RvvChannel<VectorFuReq> requests(2);
    RvvChannel<VectorFuResp> results(2);
    VectorMemoryParams params;
    params.maxOutstanding = 1;
    params.responseLatency = 1;
    params.reorderWindow = 1;
    VectorMemoryUnit unit(contexts, requests, results, params);
    VecBeat beat;
    beat.tag = {1, 0};
    beat.activeLaneMask = 1;
    unit.registerBeatNext(beat);
    unit.advance();

    VectorFuReq request;
    request.tag = {1, 0, 0};
    request.fuType = VecFuType::Memory;
    request.destReg = vreg(3);
    request.elemCount = 3;
    requests.pushNext(request);
    requests.advance();

    for (unsigned cycle = 0; cycle < 8 && results.empty(); ++cycle) {
        unit.evaluate(cycle);
        EXPECT_LE(unit.outstandingSize(), 1);
        requests.advance();
        results.advance();
        unit.advance();
    }
    EXPECT_EQ(unit.issuedTransactions(), 3);
    EXPECT_FALSE(results.empty());
}

TEST(VectorMemoryTest, CancelDropsPendingAndOutstandingTransactions)
{
    InstContextTable contexts(2);
    auto memory = context(1, 0, {}, {vreg(3)});
    memory.desc.packet.opClass = enums::SimdStridedLoad;
    memory.desc.packet.isMemory = true;
    memory.desc.packet.scalarSrcs.push_back({0, RegId(), 0x1000});
    memory.desc.packet.scalarSrcs.push_back({1, RegId(), 64});
    memory.desc.config.sewBits = 32;
    contexts.allocate(std::move(memory));
    RvvChannel<VectorFuReq> requests(2);
    RvvChannel<VectorFuResp> results(2);
    VectorMemoryParams params;
    params.maxOutstanding = 1;
    params.responseLatency = 10;
    params.reorderWindow = 1;
    VectorMemoryUnit unit(contexts, requests, results, params);
    VecBeat beat;
    beat.tag = {1, 0};
    beat.activeLaneMask = 1;
    unit.registerBeatNext(beat);
    unit.advance();

    VectorFuReq request;
    request.tag = {1, 0, 0};
    request.fuType = VecFuType::Memory;
    request.destReg = vreg(3);
    request.elemCount = 3;
    requests.pushNext(request);
    requests.advance();
    unit.evaluate(0);
    requests.advance();
    ASSERT_EQ(unit.outstandingSize(), 1);
    ASSERT_EQ(unit.pendingTransactionSize(), 2);

    unit.cancel(1);
    EXPECT_FALSE(unit.active());
}

TEST(VectorSlduTest, HandlesMultipleLaneWords)
{
    InstContextTable contexts(2);
    auto slide = context(1, 0, {vreg(1)}, {vreg(2)});
    slide.desc.packet.opClass = SimdAluOp;
    slide.desc.packet.mnemonic = "vslideup.vx";
    slide.desc.config.vl = 4;
    slide.desc.elementsPerLaneChunk = 1;
    contexts.allocate(std::move(slide));
    RvvChannel<VectorFuReq> requests(8);
    RvvChannel<VectorFuResp> results(8);
    SlduParams params;
    params.numLanes = 2;
    params.maxInflightWords = 2;
    params.slideLatency = 1;
    params.resultWidth = 2;
    VectorSldu unit(contexts, requests, results, params, 2);
    PeRequest instruction;
    instruction.contextId = 1;
    instruction.opcodeClass = VecFuType::Sldu;
    unit.acceptNext(instruction);
    unit.advance();
    unit.evaluate(0);
    unit.advance();

    const auto push_word = [&requests](LaneId lane, uint32_t word) {
        auto request = fuRequest(1);
        request.fuType = VecFuType::Sldu;
        request.tag.laneId = lane;
        request.tag.wordIndex = word;
        requests.pushNext(request);
    };
    push_word(0, 0);
    push_word(1, 0);
    push_word(0, 1);
    push_word(1, 1);
    push_word(0, 2);
    requests.advance();
    for (uint64_t cycle = 1;
         cycle < 20 && unit.doneChannel().empty(); ++cycle) {
        unit.evaluate(cycle);
        unit.advance();
    }
    EXPECT_EQ(results.size(), 5);
    ASSERT_FALSE(unit.doneChannel().empty());
    EXPECT_EQ(unit.doneChannel().front().contextId, 1);
}

TEST(VectorSlduTest, CancelDropsOperandsAndInflightResults)
{
    InstContextTable contexts(2);
    auto slide = context(1, 0, {vreg(1)}, {vreg(2)});
    slide.desc.packet.opClass = SimdAluOp;
    slide.desc.packet.mnemonic = "vslideup.vx";
    slide.desc.config.vl = 1;
    slide.desc.elementsPerLaneChunk = 1;
    contexts.allocate(std::move(slide));
    RvvChannel<VectorFuReq> requests(2);
    RvvChannel<VectorFuResp> results(2);
    SlduParams params;
    params.numLanes = 1;
    params.slideLatency = 10;
    VectorSldu unit(contexts, requests, results, params, 2);
    PeRequest instruction;
    instruction.contextId = 1;
    instruction.opcodeClass = VecFuType::Sldu;
    unit.acceptNext(instruction);
    unit.advance();
    unit.evaluate(0);
    unit.advance();

    auto request = fuRequest(1);
    request.fuType = VecFuType::Sldu;
    requests.pushNext(request);
    requests.advance();
    unit.evaluate(1);
    requests.advance();
    ASSERT_TRUE(unit.activeState());

    unit.cancel(1);
    EXPECT_FALSE(unit.activeState());
}

TEST(VectorSlduTest, ClassifiesSlideForSldu)
{
    InstQueuePacket packet;
    packet.opClass = SimdAluOp;
    packet.mnemonic = "vslideup.vx";
    EXPECT_TRUE(isSlide(packet));
    EXPECT_EQ(classifyFu(packet), VecFuType::Sldu);
}

TEST(VectorOperandTrafficTest, AddsMaskAndOldDestinationReads)
{
    InstQueuePacket packet;
    packet.srcRegs = {vreg(1)};
    packet.destRegs = {vreg(3)};
    RvvConfig config;
    config.vm = false;
    config.vma = false;
    config.vta = false;
    const auto sources = vectorSources(packet, config);
    EXPECT_NE(std::find(sources.begin(), sources.end(), vreg(0)),
              sources.end());
    EXPECT_NE(std::find(sources.begin(), sources.end(), vreg(3)),
              sources.end());
}

TEST(VectorFuTest, SeparatesLatencyFromInitiationInterval)
{
    VectorFuPipeline fu(VecFuType::Alu, VectorFuParams{3, 2, 2, 4});
    RvvChannel<VectorFuResp> results(4);
    fu.acceptInputNext(fuRequest(1));
    fu.acceptInputNext(fuRequest(2));
    fu.advance();

    fu.evaluate(0, 0, results);
    fu.advance();
    results.advance();
    EXPECT_TRUE(results.empty());

    fu.evaluate(1, 0, results);
    fu.advance();
    results.advance();
    EXPECT_TRUE(results.empty());

    fu.evaluate(2, 0, results);
    fu.advance();
    EXPECT_TRUE(results.empty());
    results.advance();
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().tag.instId, 1);
}

TEST(VectorFuTest, CancelDropsQueuedAndInflightTokens)
{
    VectorFuPipeline fu(VecFuType::Alu, VectorFuParams{4, 1, 2, 4});
    RvvChannel<VectorFuResp> results(2);
    fu.acceptInputNext(fuRequest(9));
    fu.advance();
    fu.evaluate(0, 0, results);
    fu.advance();
    fu.cancel(9);
    EXPECT_FALSE(fu.active());
}

TEST(RvvPipelineTest, MemoryPreparationMicroopDoesNotAccessAddressZero)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    auto params = pipelineParams();
    unsigned memory_requests = 0;
    params.memory.sendTimingRequest = [&](const VectorMemoryParams::TimingRequest &) {
        ++memory_requests;
        return true;
    };
    Pipeline pipeline(queue, params);
    auto packet = rvvPacket(1, 4);
    packet.opClass = enums::SimdIndexedLoad;
    packet.isMicroop = true;
    packet.isMemory = false;
    packet.srcRegs = {vreg(2)};
    packet.destRegs = {vreg(RiscvISA::VecMemInternalReg0)};
    ASSERT_TRUE(queue.dispatch(packet));
    for (unsigned cycle = 0;
         cycle < 100 && !queue.hasCompletion({0, 1}); ++cycle)
        pipeline.evaluate();
    EXPECT_TRUE(queue.hasCompletion({0, 1}));
    EXPECT_EQ(memory_requests, 0);
}

TEST(RvvPipelineTest, CompletesMultiBeatInstruction)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    Pipeline pipeline(queue, pipelineParams());
    ASSERT_TRUE(queue.dispatch(rvvPacket(1, 4)));

    for (unsigned cycle = 0;
         cycle < 100 && !queue.hasCompletion({0, 1}); ++cycle) {
        pipeline.evaluate();
    }

    EXPECT_TRUE(queue.hasCompletion({0, 1}));
    EXPECT_FALSE(pipeline.active());
}

TEST(RvvPipelineTest, HostCancelDropsInflightInstruction)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    auto params = pipelineParams();
    params.lane.aluLatency = 10;
    Pipeline pipeline(queue, params);
    queue.addCancelNotify(
        [&pipeline](const InstQueueKey &key) { pipeline.cancel(key); });
    ASSERT_TRUE(queue.dispatch(rvvPacket(2, 16)));

    for (unsigned cycle = 0; cycle < 8; ++cycle)
        pipeline.evaluate();
    ASSERT_TRUE(pipeline.active());

    queue.cancel({0, 2});
    EXPECT_FALSE(queue.hasPacket({0, 2}));
    EXPECT_FALSE(pipeline.active());
    for (unsigned cycle = 0; cycle < 20; ++cycle)
        pipeline.evaluate();
    EXPECT_FALSE(queue.hasCompletion({0, 2}));
    EXPECT_FALSE(pipeline.active());
}

TEST(RvvPipelineTest, MaskedInstructionWaitsForMaskTarget)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    auto params = pipelineParams();
    Pipeline pipeline(queue, params);
    auto packet = rvvPacket(5, 4);
    packet.encodedInst &= ~(1ULL << 25); // vm=0, MASKU target required
    ASSERT_TRUE(queue.dispatch(packet));
    for (unsigned cycle = 0;
         cycle < 100 && !queue.hasCompletion({0, 5}); ++cycle)
        pipeline.evaluate();
    EXPECT_TRUE(queue.hasCompletion({0, 5}));
    EXPECT_FALSE(pipeline.active());
}

TEST(RvvPipelineTest, LongSlideCompletesThroughSldu)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    Pipeline pipeline(queue, pipelineParams());
    auto packet = rvvPacket(6, 16);
    packet.mnemonic = "vslideup.vx";
    ASSERT_TRUE(queue.dispatch(std::move(packet)));

    for (unsigned cycle = 0;
         cycle < 200 && !queue.hasCompletion({0, 6}); ++cycle) {
        pipeline.evaluate();
    }
    EXPECT_TRUE(queue.hasCompletion({0, 6}));
    EXPECT_FALSE(pipeline.active());
}

TEST(RvvPipelineTest, ReductionCompletesThroughSlduPhases)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    Pipeline pipeline(queue, pipelineParams());
    auto packet = rvvPacket(7, 16);
    packet.mnemonic = "vredsum.vs";
    packet.opClass = SimdReduceAddOp;
    ASSERT_TRUE(queue.dispatch(std::move(packet)));

    for (unsigned cycle = 0;
         cycle < 300 && !queue.hasCompletion({0, 7}); ++cycle) {
        pipeline.evaluate();
    }
    EXPECT_TRUE(queue.hasCompletion({0, 7}));
    EXPECT_FALSE(pipeline.active());
}

TEST(RvvPipelineTest, RawConsumerCompletesWithBeatLevelChaining)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    auto params = pipelineParams();
    params.lane.aluLatency = 4;
    Pipeline pipeline(queue, params);
    auto producer = rvvPacket(3, 4);
    auto consumer = rvvPacket(4, 4);
    consumer.srcRegs = {vreg(3)};
    consumer.destRegs = {vreg(4)};
    ASSERT_TRUE(queue.dispatch(std::move(producer)));
    ASSERT_TRUE(queue.dispatch(std::move(consumer)));

    for (unsigned cycle = 0;
         cycle < 200 && !queue.hasCompletion({0, 4}); ++cycle) {
        pipeline.evaluate();
    }
    EXPECT_TRUE(queue.hasCompletion({0, 3}));
    EXPECT_TRUE(queue.hasCompletion({0, 4}));
    EXPECT_FALSE(pipeline.active());
}

} // namespace rvv
} // namespace minor
} // namespace gem5
