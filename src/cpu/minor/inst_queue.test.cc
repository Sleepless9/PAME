/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "cpu/minor/inst_queue.hh"

#include <gtest/gtest.h>

namespace gem5
{
namespace minor
{

namespace
{

InstQueuePacket
packet(InstQueueEngine engine, InstSeqNum seq, bool is_memory = false,
       ThreadID tid = 0)
{
    InstQueuePacket result;
    result.key = {tid, seq};
    result.engine = engine;
    result.mnemonic = engine == InstQueueEngine::RVV ? "vadd.vv" : "ztt_op";
    result.isMemory = is_memory;
    return result;
}

} // anonymous namespace

TEST(MinorInstQueueTest, PopsOnlyRequestedEngine)
{
    MinorInstQueue queue("queue", 4, true, true, {});
    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::AME, 1)));
    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::RVV, 2)));

    ASSERT_TRUE(queue.hasPending(InstQueueEngine::RVV));
    EXPECT_EQ(queue.pendingSize(InstQueueEngine::RVV), 1);
    EXPECT_EQ(queue.pop(InstQueueEngine::RVV).key.execSeqNum, 2);
    EXPECT_FALSE(queue.hasPending(InstQueueEngine::RVV));

    ASSERT_TRUE(queue.hasPending(InstQueueEngine::AME));
    EXPECT_EQ(queue.front().key.execSeqNum, 1);
}

TEST(MinorInstQueueTest, CompletesExactlyOnce)
{
    unsigned notifications = 0;
    MinorInstQueue queue(
        "queue", 2, true, false, [&notifications] { ++notifications; });
    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::RVV, 7)));
    queue.pop(InstQueueEngine::RVV);

    InstQueueCompletion completion;
    completion.key = {0, 7};
    EXPECT_TRUE(queue.complete(completion));
    EXPECT_FALSE(queue.complete(completion));
    EXPECT_EQ(notifications, 1);
    EXPECT_TRUE(queue.hasCompletion({0, 7}));

    queue.retire({0, 7});
    EXPECT_TRUE(queue.empty());
    EXPECT_FALSE(queue.complete(completion));
}

TEST(MinorInstQueueTest, NotifiesAllConsumers)
{
    unsigned first = 0;
    unsigned second = 0;
    MinorInstQueue queue("queue", 2, true, false, {});
    queue.addDispatchNotify([&first] { ++first; });
    queue.addDispatchNotify([&second] { ++second; });

    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::RVV, 3)));
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 1);
}

TEST(MinorInstQueueTest, CancelAllNotifiesWithKeys)
{
    std::vector<InstQueueKey> cancelled;
    MinorInstQueue queue("queue", 4, true, false, {});
    queue.addCancelNotify([&cancelled](const InstQueueKey &key) {
        cancelled.push_back(key);
    });

    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::RVV, 10)));
    ASSERT_TRUE(queue.dispatch(packet(InstQueueEngine::RVV, 11)));
    queue.cancelAll();
    queue.cancel({0, 10});

    EXPECT_TRUE(queue.empty());
    ASSERT_EQ(cancelled.size(), 2);
    EXPECT_EQ(cancelled[0].execSeqNum, 10);
    EXPECT_EQ(cancelled[1].execSeqNum, 11);
}

TEST(MinorInstQueueTest, TracksOlderMemoryUntilLSQIssue)
{
    MinorInstQueue queue("queue", 4, true, false, {});
    ASSERT_TRUE(queue.dispatch(
        packet(InstQueueEngine::RVV, 10, true, 0)));
    ASSERT_TRUE(queue.dispatch(
        packet(InstQueueEngine::RVV, 11, false, 0)));
    ASSERT_TRUE(queue.dispatch(
        packet(InstQueueEngine::RVV, 5, true, 1)));

    EXPECT_FALSE(queue.hasOlderUnissuedMemory(0, 10));
    EXPECT_TRUE(queue.hasOlderUnissuedMemory(0, 11));
    EXPECT_FALSE(queue.hasOlderUnissuedMemory(2, 100));

    queue.markMemoryIssued({0, 10});
    EXPECT_FALSE(queue.hasOlderUnissuedMemory(0, 11));
    EXPECT_TRUE(queue.hasOlderUnissuedMemory(1, 6));

    queue.cancel({1, 5});
    EXPECT_FALSE(queue.hasOlderUnissuedMemory(1, 6));
}

} // namespace minor
} // namespace gem5
