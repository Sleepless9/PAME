/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include <gtest/gtest.h>

#include "arch/riscv/regs/matrix.hh"
#include "computing_cluster/AME/ame_channel.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"
#include "computing_cluster/AME/ame_types.hh"
#include "computing_cluster/AME/p1_pipeline.hh"
#include "computing_cluster/AME/register_file.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

namespace
{

RegId
mreg(unsigned index)
{
    return RiscvISA::matRegClass[RiscvISA::ZttMatRegBase + index];
}

RegId
areg(unsigned index)
{
    return RiscvISA::matRegClass[RiscvISA::ZttAccRegBase + index];
}

AmeInstContext
context(AmeContextId id, InstSeqNum seq, std::vector<RegId> sources,
        std::vector<RegId> destinations)
{
    AmeInstContext result;
    result.desc.id = id;
    result.desc.key = {0, seq};
    result.desc.packet.key = result.desc.key;
    result.desc.packet.engine = InstQueueEngine::AME;
    result.desc.packet.mnemonic = "ztt_mmov_m_m";
    result.desc.packet.srcRegs = std::move(sources);
    result.desc.packet.destRegs = std::move(destinations);
    result.desc.config.nAme = 4;
    result.desc.config.unitBits = 32;
    result.desc.config.mdBits = 32;
    result.desc.config.kLogical = 4;
    result.desc.opcode = AmeOpcode::Move;
    result.desc.targets = TargetMptu;
    populateRegisterRegions(result.desc);
    return result;
}

InstQueuePacket
matrixPacket(InstSeqNum seq)
{
    InstQueuePacket packet;
    packet.key = {0, seq};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mmulacc_2d";
    packet.srcRegs = {areg(0), mreg(0), mreg(1)};
    packet.destRegs = {areg(0)};
    return packet;
}

InstQueuePacket
auxiliaryPacket(InstSeqNum seq, const std::string &mnemonic)
{
    InstQueuePacket packet;
    packet.key = {0, seq};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = mnemonic;
    packet.srcRegs = {mreg(0)};
    packet.destRegs = {mreg(1)};
    return packet;
}

uint64_t
runToCompletion(AmePipeline &pipeline, MinorInstQueue &queue,
                const InstQueueKey &key, uint64_t limit = 2000)
{
    for (uint64_t elapsed = 0; elapsed < limit; ++elapsed) {
        pipeline.evaluate();
        if (queue.hasCompletion(key))
            return elapsed + 1;
    }
    return limit;
}

} // anonymous namespace

TEST(AmeChannelTest, TransferOnlyAtBoundary)
{
    AmeChannel<unsigned> channel(2);
    channel.pushNext(7);
    EXPECT_TRUE(channel.empty());
    channel.advance();
    ASSERT_FALSE(channel.empty());
    EXPECT_EQ(channel.front(), 7U);
    channel.popNext();
    EXPECT_FALSE(channel.empty());
    channel.advance();
    EXPECT_TRUE(channel.empty());
}

TEST(AmeRegisterFileTest, FixedRawByteMappingUsesSingleBankPort)
{
    BankedRegisterFile mrf(RegisterFileKind::Mrf, 4, 64, 64, 2, 4, 1, 1);
    LogicalBankRequest first;
    first.file = RegisterFileKind::Mrf;
    first.access = BankAccess::Read;
    first.reg = 3;
    first.byteOffset = 0;
    first.byteCount = 8;
    EXPECT_TRUE(mrf.canAcceptNext(first));
    mrf.acceptNext(first);
    mrf.evaluate(0);
    mrf.advance();
    EXPECT_TRUE(mrf.responseChannel().empty());
    mrf.evaluate(1);
    mrf.advance();
    EXPECT_TRUE(mrf.responseChannel().empty());
    mrf.evaluate(2);
    mrf.advance();
    ASSERT_EQ(mrf.responseChannel().size(), 1U);
    EXPECT_EQ(mrf.responseChannel().front().bank, 0U);
}

TEST(AmeScoreboardTest, ByteCoverageBecomesVisibleNextCycle)
{
    AmeContextTable contexts(2);
    auto producer = context(1, 1, {}, {mreg(2)});
    auto consumer = context(2, 2, {mreg(2)}, {mreg(3)});
    contexts.allocate(producer);
    contexts.allocate(consumer);
    AmeScoreboard scoreboard;
    scoreboard.allocate(contexts.get(1));
    scoreboard.allocate(contexts.get(2));
    scoreboard.bindSlot(1, 1);
    scoreboard.bindSlot(2, 2);
    const auto region = contexts.get(1).desc.mDestinations.front();
    EXPECT_FALSE(scoreboard.mRegionReady(1, region));
    scoreboard.publishMWriteNext(1, region);
    EXPECT_FALSE(scoreboard.mRegionReady(1, region));
    scoreboard.advance();
    EXPECT_TRUE(scoreboard.mRegionReadyForConsumer(2, region));
}

TEST(AmeScoreboardTest, AccumulatorWawIsBlocked)
{
    AmeContextTable contexts(2);
    auto first = context(1, 1, {areg(0)}, {areg(0)});
    auto second = context(2, 2, {areg(0)}, {areg(0)});
    contexts.allocate(first);
    contexts.allocate(second);
    AmeScoreboard scoreboard;
    scoreboard.allocate(contexts.get(1));
    scoreboard.allocate(contexts.get(2));
    EXPECT_FALSE(scoreboard.canSequence(2));
}

TEST(AmeTypesTest, WideDatatypeSpansPhysicalMRegisters)
{
    auto wide = context(1, 1, {mreg(4)}, {});
    wide.desc.config.unitBits = 32;
    wide.desc.config.mdBits = 64;
    wide.desc.mSourceGroups.clear();
    wide.desc.mSources.clear();
    wide.desc.mBytesRead = 0;
    populateRegisterRegions(wide.desc);

    ASSERT_EQ(wide.desc.mSourceGroups.size(), 1U);
    ASSERT_EQ(wide.desc.mSourceGroups.front().size(), 2U);
    EXPECT_EQ(wide.desc.mSourceGroups.front()[0].reg, 4U);
    EXPECT_EQ(wide.desc.mSourceGroups.front()[1].reg, 5U);
    EXPECT_EQ(wide.desc.mBytesRead, 128U);
}

TEST(AmeTypesTest, EachOperandUsesItsOwnDatatypeSnapshot)
{
    auto mixed = context(1, 1, {mreg(4), mreg(8)}, {mreg(12)});
    mixed.desc.config.unitBits = 32;
    mixed.desc.config.mdBits = 32;
    mixed.desc.config.mRegBits[4] = 64;
    mixed.desc.config.mRegBits[8] = 8;
    mixed.desc.config.mRegBits[12] = 16;
    mixed.desc.mSourceGroups.clear();
    mixed.desc.mDestinationGroups.clear();
    mixed.desc.mSources.clear();
    mixed.desc.mDestinations.clear();
    mixed.desc.mBytesRead = 0;
    mixed.desc.mBytesWritten = 0;
    populateRegisterRegions(mixed.desc);

    ASSERT_EQ(mixed.desc.mSourceGroups.size(), 2U);
    EXPECT_EQ(mixed.desc.mSourceGroups[0].size(), 2U);
    EXPECT_EQ(mixed.desc.mSourceGroups[1].size(), 1U);
    ASSERT_EQ(mixed.desc.mDestinationGroups.size(), 1U);
    EXPECT_EQ(mixed.desc.mDestinationGroups[0].size(), 1U);
    EXPECT_EQ(mixed.desc.mBytesRead, 192U);
    EXPECT_EQ(mixed.desc.mBytesWritten, 64U);
}

TEST(AmePipelineTest, LoadPacketWritesEveryMrfFragment)
{
    MinorInstQueue queue("ame-test-queue", 4, false, true, [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mld_m";
    packet.isMemory = true;
    packet.destRegs = {mreg(0)};
    ASSERT_TRUE(queue.dispatch(packet));

    AmePipelineParams params;
    params.mrfRegisterBytes = 256;
    params.mlsu.packetBytes = 64;
    params.mlsu.responseLatency = 2;
    params.mlsu.reorderWindow = 1;
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0;
         cycle < 100 && !queue.hasCompletion(packet.key); ++cycle) {
        pipeline.evaluate();
    }

    ASSERT_TRUE(queue.hasCompletion(packet.key));
    EXPECT_EQ(pipeline.mrfGrants(), 8U);
}

TEST(AmePipelineTest, TimingSenderRetriesTheSameTransaction)
{
    MinorInstQueue queue("ame-mlsu-retry-test-queue", 4, false, true, [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mld_m";
    packet.isMemory = true;
    packet.destRegs = {mreg(0)};
    ASSERT_TRUE(queue.dispatch(packet));

    std::vector<MatrixLsuParams::TimingRequest> attempts;
    AmePipelineParams params;
    params.mlsu.packetBytes = 64;
    params.mlsu.sendTimingRequest = [&attempts](const auto &request) {
        attempts.push_back(request);
        return attempts.size() != 1;
    };
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0; cycle < 100 && attempts.size() < 2; ++cycle)
        pipeline.evaluate();
    ASSERT_EQ(attempts.size(), 2U);
    EXPECT_EQ(attempts[0].transactionId, attempts[1].transactionId);

    pipeline.completeTimingResponse(attempts.back().transactionId);
    EXPECT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);
    EXPECT_EQ(pipeline.mlsuOutstanding(), 0U);
}

TEST(AmePipelineTest, OutOfOrderColumnMajorPacketsKeepTheirMrfOffsets)
{
    MinorInstQueue queue("ame-mlsu-reorder-test-queue", 4, false, true,
                         [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mls_cm";
    packet.isMemory = true;
    packet.destRegs = {mreg(0)};
    ASSERT_TRUE(queue.dispatch(packet));

    std::vector<MatrixLsuParams::TimingRequest> requests;
    AmePipelineParams params;
    params.mlsu.packetBytes = 16;
    params.mlsu.responseWidth = 1;
    params.mlsu.sendTimingRequest = [&requests](const auto &request) {
        requests.push_back(request);
        return true;
    };
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0; cycle < 100 && requests.size() < 4; ++cycle)
        pipeline.evaluate();
    ASSERT_EQ(requests.size(), 4U);
    for (auto request = requests.rbegin(); request != requests.rend();
         ++request) {
        pipeline.completeTimingResponse(request->transactionId);
    }

    EXPECT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);
    EXPECT_EQ(pipeline.mrfGrants(), 16U);
    EXPECT_EQ(pipeline.mlsuOutstanding(), 0U);
}

TEST(AmePipelineTest, StridedDescriptorIssuesOneBurstPerLogicalRow)
{
    MinorInstQueue queue("ame-mlsu-stride-test-queue", 4, false, true,
                         [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mls_st";
    packet.isMemory = true;
    packet.destRegs = {mreg(0)};
    packet.scalarSrcs.resize(2);
    packet.scalarSrcs[0].value = 0x1000;
    packet.scalarSrcs[1].value = 32;
    ASSERT_TRUE(queue.dispatch(packet));

    std::vector<MatrixLsuParams::TimingRequest> requests;
    AmePipelineParams params;
    params.mlsu.packetBytes = 64;
    params.mlsu.sendTimingRequest = [&requests](const auto &request) {
        requests.push_back(request);
        return true;
    };
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0; cycle < 100 && requests.size() < 4; ++cycle)
        pipeline.evaluate();
    ASSERT_EQ(requests.size(), 4U);
    for (unsigned row = 0; row < requests.size(); ++row) {
        EXPECT_EQ(requests[row].address, 0x1000U + row * 32);
        EXPECT_EQ(requests[row].bytes, 16U);
        pipeline.completeTimingResponse(requests[row].transactionId);
    }
    EXPECT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);
}

TEST(AmePipelineTest, StridedStoreAcquiresEachLogicalRow)
{
    MinorInstQueue queue("ame-mlsu-store-test-queue", 4, false, true,
                         [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mss_st";
    packet.isMemory = true;
    packet.srcRegs = {mreg(0)};
    packet.scalarSrcs.resize(2);
    packet.scalarSrcs[0].value = 0x2000;
    packet.scalarSrcs[1].value = 32;
    ASSERT_TRUE(queue.dispatch(packet));

    std::vector<MatrixLsuParams::TimingRequest> requests;
    AmePipelineParams params;
    params.mlsu.packetBytes = 64;
    params.mlsu.sendTimingRequest = [&requests](const auto &request) {
        requests.push_back(request);
        return true;
    };
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0; cycle < 100 && requests.size() < 4; ++cycle)
        pipeline.evaluate();
    ASSERT_EQ(requests.size(), 4U);
    for (unsigned row = 0; row < requests.size(); ++row) {
        EXPECT_TRUE(requests[row].isStore);
        EXPECT_EQ(requests[row].address, 0x2000U + row * 32);
        EXPECT_EQ(requests[row].bytes, 16U);
        pipeline.completeTimingResponse(requests[row].transactionId);
    }
    EXPECT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);
}

TEST(AmePipelineTest, SquashedMlsuAbsorbsOutstandingResponses)
{
    MinorInstQueue queue("ame-mlsu-squash-test-queue", 4, false, true,
                         [] {});
    InstQueuePacket packet;
    packet.key = {0, 1};
    packet.engine = InstQueueEngine::AME;
    packet.mnemonic = "ztt_mls_rm";
    packet.isMemory = true;
    packet.destRegs = {mreg(0)};
    ASSERT_TRUE(queue.dispatch(packet));

    std::vector<MatrixLsuParams::TimingRequest> requests;
    AmePipelineParams params;
    params.mlsu.packetBytes = 16;
    params.mlsu.sendTimingRequest = [&requests](const auto &request) {
        requests.push_back(request);
        return true;
    };
    AmePipeline pipeline(queue, params);
    queue.addCancelNotify([&pipeline](const InstQueueKey &key) {
        pipeline.cancel(key);
    });
    for (unsigned cycle = 0; cycle < 100 && requests.size() < 4; ++cycle)
        pipeline.evaluate();
    ASSERT_EQ(requests.size(), 4U);

    queue.cancel(packet.key);
    EXPECT_TRUE(pipeline.active());
    for (const auto &request : requests)
        pipeline.completeTimingResponse(request.transactionId);
    for (unsigned cycle = 0; cycle < 20 && pipeline.active(); ++cycle)
        pipeline.evaluate();
    EXPECT_FALSE(pipeline.active());
    EXPECT_EQ(pipeline.mlsuOutstanding(), 0U);
}

TEST(AmePipelineTest, ConfigSnapshotsNextCommandBeforeRetirement)
{
    MinorInstQueue queue("ame-serial-test-queue", 4, false, true, [] {});
    InstQueuePacket config;
    config.key = {0, 1};
    config.engine = InstQueueEngine::AME;
    config.mnemonic = "ztt_msettype";
    InstQueuePacket move;
    move.key = {0, 2};
    move.engine = InstQueueEngine::AME;
    move.mnemonic = "ztt_mmov_m_m";
    ASSERT_TRUE(queue.dispatch(config));
    ASSERT_TRUE(queue.dispatch(move));

    AmePipelineParams params;
    params.controlLatency = 1;
    AmePipeline pipeline(queue, params);
    for (unsigned cycle = 0;
         cycle < 50 && !queue.hasCompletion(config.key); ++cycle) {
        pipeline.evaluate();
    }
    ASSERT_TRUE(queue.hasCompletion(config.key));
    for (unsigned cycle = 0;
         cycle < 50 && !queue.hasCompletion(move.key); ++cycle) {
        pipeline.evaluate();
    }
    EXPECT_EQ(queue.pendingSize(InstQueueEngine::AME), 0U);
    EXPECT_TRUE(queue.hasCompletion(move.key));

    // Minor/O3 may retire both commands later in architectural order; AME
    // no longer needs a global retirement barrier for datatype snapshots.
    queue.retire(config.key);
    queue.retire(move.key);
}

TEST(AmePipelineTest, TwoDimensionalGridExercisesEveryPeAndBoundary)
{
    MinorInstQueue queue("ame-grid-test-queue", 4, false, true, [] {});
    const auto packet = matrixPacket(1);
    ASSERT_TRUE(queue.dispatch(packet));

    AmePipelineParams params;
    params.systolic.peRows = 2;
    params.systolic.peCols = 2;
    params.systolic.tileRows = 2;
    params.systolic.tileCols = 2;
    params.systolic.kBlock = 4;
    params.operands.kBlock = 4;
    AmePipeline pipeline(queue, params);
    ASSERT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);

    EXPECT_EQ(pipeline.saMacIssues(), 64U);
    EXPECT_EQ(pipeline.saPeIssues(), 64U);
    EXPECT_EQ(pipeline.saPeResults(), 64U);
    EXPECT_EQ(pipeline.saTileCompletions(), 16U);
    EXPECT_EQ(pipeline.saEdgeTransfers(), 16U);
}

TEST(AmePipelineTest, PeCountersExcludeFutureIssuesAndCancelledResults)
{
    MinorInstQueue queue("ame-pe-accounting-test", 4, false, true, [] {});
    const auto packet = matrixPacket(1);
    ASSERT_TRUE(queue.dispatch(packet));
    AmePipelineParams params;
    AmePipeline pipeline(queue, params);
    EXPECT_FALSE(pipeline.cancel({0, 999}));
    for (unsigned cycle = 0; cycle < 500 && pipeline.saPeIssues() == 0; ++cycle)
        pipeline.evaluate();
    // Only PE(0,0) has reached its issue cycle; the other scheduled events
    // must not be credited yet, and cancellation must not credit them later.
    ASSERT_EQ(pipeline.saPeIssues(), 1U);
    EXPECT_EQ(pipeline.saPeResults(), 0U);
    EXPECT_TRUE(pipeline.cancel(packet.key));
    EXPECT_FALSE(pipeline.cancel(packet.key));
    for (unsigned cycle = 0; cycle < 100; ++cycle)
        pipeline.evaluate();
    EXPECT_EQ(pipeline.saPeIssues(), 1U);
    EXPECT_EQ(pipeline.saPeResults(), 0U);
}

TEST(AmePipelineTest, PartialOutputBlockDoesNotEnterInactiveTiles)
{
    MinorInstQueue queue("ame-partial-grid-test-queue", 4, false, true, [] {});
    const auto packet = matrixPacket(1);
    ASSERT_TRUE(queue.dispatch(packet));

    AmePipelineParams params;
    params.systolic.peRows = 2;
    params.systolic.peCols = 2;
    params.systolic.tileRows = 3;
    params.systolic.tileCols = 3;
    AmePipeline pipeline(queue, params);
    ASSERT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);

    EXPECT_EQ(pipeline.saMacIssues(), 64U);
    EXPECT_EQ(pipeline.saTileCompletions(), 16U);
    EXPECT_EQ(pipeline.saEdgeTransfers(), 16U);
}

TEST(AmePipelineTest, MacInitiationIntervalCreatesRealWavefrontStalls)
{
    const auto packet = matrixPacket(1);
    auto run = [&packet](unsigned mac_ii) {
        MinorInstQueue queue("ame-mac-ii-test-queue", 4, false, true, [] {});
        EXPECT_TRUE(queue.dispatch(packet));
        AmePipelineParams params;
        params.systolic.peRows = 2;
        params.systolic.peCols = 2;
        params.systolic.macII = mac_ii;
        AmePipeline pipeline(queue, params);
        return runToCompletion(pipeline, queue, packet.key);
    };

    const uint64_t ii_one = run(1);
    const uint64_t ii_three = run(3);
    EXPECT_LT(ii_one, 2000U);
    EXPECT_GT(ii_three, ii_one);
}

TEST(AmePipelineTest, RegisteredTileEdgesAffectCompletionCycle)
{
    const auto packet = matrixPacket(1);
    auto run = [&packet](unsigned boundary_latency) {
        MinorInstQueue queue("ame-edge-test-queue", 4, false, true, [] {});
        EXPECT_TRUE(queue.dispatch(packet));
        AmePipelineParams params;
        params.systolic.peRows = 2;
        params.systolic.peCols = 2;
        params.systolic.tileRows = 2;
        params.systolic.tileCols = 2;
        params.systolic.tileBoundaryLatency = boundary_latency;
        AmePipeline pipeline(queue, params);
        return runToCompletion(pipeline, queue, packet.key);
    };

    const uint64_t one_cycle = run(1);
    const uint64_t four_cycles = run(4);
    EXPECT_LT(one_cycle, 2000U);
    EXPECT_GT(four_cycles, one_cycle);
}

TEST(AmePipelineTest, ContinuousPssPreservesWorkAndHonorsFiniteCapacity)
{
    auto run = [](bool continuous, unsigned depth, unsigned ii,
                  unsigned grid, unsigned request_width) {
        MinorInstQueue queue("continuous-pss-test", 4, false, true, [] {});
        const auto packet = matrixPacket(1);
        EXPECT_TRUE(queue.dispatch(packet));
        AmePipelineParams params;
        params.systolic.pssContinuous = continuous;
        params.systolic.operationDepth = depth;
        params.systolic.macII = ii;
        params.systolic.peRows = grid == 1 ? 4 : 2;
        params.systolic.peCols = grid == 1 ? 4 : 2;
        params.systolic.tileRows = grid;
        params.systolic.tileCols = grid;
        params.operands.requestWidth = request_width;
        AmePipeline pipeline(queue, params, [](AmeConfig &cfg, const auto &) {
            cfg.mdBits = 8; // Four packed squares: logical K=16.
        });
        uint64_t cycles = runToCompletion(pipeline, queue, packet.key);
        EXPECT_LT(cycles, 2000U);
        EXPECT_EQ(pipeline.saPeIssues(), 256U);
        EXPECT_EQ(pipeline.saPeResults(), 256U);
        EXPECT_EQ(pipeline.saMacIssues(), 256U);
        return cycles;
    };
    const auto baseline = run(false, 2, 1, 1, 2);
    const auto continuous = run(true, 12, 1, 1, 2);
    EXPECT_LT(continuous, baseline);
    EXPECT_GT(run(true, 2, 1, 1, 2), continuous);
    EXPECT_GT(run(true, 12, 3, 1, 2), continuous);
    run(true, 12, 1, 2, 1); // Multi-tile forwarding and delayed supply.
}

TEST(AmePipelineTest, ContinuousPssCancellationDoesNotLeakIntoNextContext)
{
    MinorInstQueue queue("continuous-pss-cancel", 4, false, true, [] {});
    AmePipelineParams params;
    params.systolic.pssContinuous = true;
    params.systolic.operationDepth = 12;
    AmePipeline pipeline(queue, params, [](AmeConfig &cfg, const auto &) {
        cfg.mdBits = 8;
    });
    queue.addCancelNotify([&pipeline](const InstQueueKey &key) { pipeline.cancel(key); });
    auto packet = matrixPacket(1);
    ASSERT_TRUE(queue.dispatch(packet));
    for (unsigned i = 0; i < 500 && pipeline.saPeIssues() < 80; ++i)
        pipeline.evaluate();
    ASSERT_GE(pipeline.saPeIssues(), 80U);
    queue.cancel(packet.key);
    auto issued = pipeline.saPeIssues(), results = pipeline.saPeResults();
    for (unsigned i = 0; i < 50; ++i) pipeline.evaluate();
    EXPECT_EQ(pipeline.saPeIssues(), issued);
    EXPECT_EQ(pipeline.saPeResults(), results);
    EXPECT_FALSE(pipeline.active());
    packet = matrixPacket(2);
    ASSERT_TRUE(queue.dispatch(packet));
    EXPECT_LT(runToCompletion(pipeline, queue, packet.key), 2000U);
    EXPECT_EQ(pipeline.saPeIssues() - issued, 256U);
    EXPECT_EQ(pipeline.saPeResults() - results, 256U);
}

TEST(AmePipelineTest, PrefixCarriesRemainOrderedAcrossTiles)
{
    auto run = [](unsigned tile_cols) {
        MinorInstQueue queue("ame-mrsu-test-queue", 4, false, true, [] {});
        const auto packet = auxiliaryPacket(1, "ztt_mprefixadd_row");
        EXPECT_TRUE(queue.dispatch(packet));
        AmePipelineParams params;
        params.mrsu.tileRows = 1;
        params.mrsu.tileCols = tile_cols;
        params.mrsu.localLatency = 2;
        params.mrsu.crossTileLatency = 2;
        params.mrsu.resultWidth = 1;
        AmePipeline pipeline(queue, params);
        return runToCompletion(pipeline, queue, packet.key);
    };

    const uint64_t one_tile = run(1);
    const uint64_t four_tiles = run(4);
    EXPECT_LT(one_tile, 2000U);
    EXPECT_GT(four_tiles, one_tile);
}

TEST(AmePipelineTest, SquashClearsResidentPeAndAuxiliaryState)
{
    MinorInstQueue queue("ame-squash-test-queue", 4, false, true, [] {});
    const auto packet = matrixPacket(1);
    ASSERT_TRUE(queue.dispatch(packet));
    AmePipelineParams params;
    params.systolic.peRows = 2;
    params.systolic.peCols = 2;
    params.systolic.tileRows = 2;
    params.systolic.tileCols = 2;
    AmePipeline pipeline(queue, params);
    queue.addCancelNotify([&pipeline](const InstQueueKey &key) {
        pipeline.cancel(key);
    });

    for (unsigned cycle = 0; cycle < 20; ++cycle)
        pipeline.evaluate();
    EXPECT_TRUE(pipeline.active());
    queue.cancel(packet.key);
    for (unsigned cycle = 0; cycle < 4; ++cycle)
        pipeline.evaluate();
    EXPECT_FALSE(pipeline.active());
    EXPECT_EQ(pipeline.mlsuOutstanding(), 0U);
}

} // namespace ame
} // namespace minor
} // namespace gem5
