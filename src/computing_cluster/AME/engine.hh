/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_ENGINE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_ENGINE_HH__

#include "base/statistics.hh"
#include "computing_cluster/AME/p1_pipeline.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "params/AMEEngine.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class BaseCPU;

namespace minor
{
namespace ame
{

class AMEEngine : public ClockedObject
{
  private:
    class MemoryPort : public RequestPort
    {
      private:
        struct SenderState : public Packet::SenderState
        {
            const uint64_t transactionId;
            explicit SenderState(uint64_t id) : transactionId(id) {}
        };

        AMEEngine &engine;
        PacketPtr retryPacket = nullptr;

      protected:
        bool recvTimingResp(PacketPtr packet) override;
        void recvReqRetry() override;

      public:
        MemoryPort(const std::string &name, AMEEngine &engine_);
        ~MemoryPort() override;
        bool send(const MatrixLsuParams::TimingRequest &request);
        bool blocked() const { return retryPacket != nullptr; }
    };

    struct AMEStats : public statistics::Group
    {
        statistics::Scalar cycles;
        statistics::Scalar activeCycles;
        statistics::Scalar completedInstructions;
        statistics::Scalar squashes;
        statistics::Scalar mrfGrants;
        statistics::Scalar mrfConflictCycles;
        statistics::Scalar accGrants;
        statistics::Scalar accConflictCycles;
        statistics::Scalar saMacOperations;
        statistics::Scalar saPeIssues;
        statistics::Scalar saPeResults;
        statistics::Scalar saActiveCycles;
        statistics::Scalar mlsuActiveCycles;
        statistics::Scalar saMlsuOverlapCycles;
        statistics::Scalar saCommandQueueOverlapCycles;
        statistics::Scalar saTileCompletions;
        statistics::Scalar saEdgeTransfers;
        statistics::Scalar saEdgeStalls;
        statistics::Scalar saInjectionStalls;
        statistics::Scalar saDrainStalls;
        statistics::Scalar mrsuLocalCycles;
        statistics::Scalar mrsuCrossTileCycles;
        statistics::Scalar mrsuPrefixCycles;
        statistics::Scalar mrsuComputeItems;
        statistics::Scalar mptuComputeItems;
        statistics::Scalar memoryRequests;
        statistics::Scalar memoryAccepted;
        statistics::Scalar memoryRetries;
        statistics::Scalar memoryResponses;
        statistics::Scalar memoryOutstanding;

        AMEStats(statistics::Group *parent);
    } stats;

    BaseCPU &cpu;
    MinorInstQueue &instQueue;
    MemoryPort memoryPort;
    const RequestorID requestorId;
    AmePipeline pipeline;
    EventFunctionWrapper tickEvent;
    bool draining = false;

    void tick();
    void wakeup();
    void cancel(const InstQueueKey &key);

  public:
    using Params = AMEEngineParams;
    explicit AMEEngine(const Params &params);
    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void init() override;
    void startup() override;
    DrainState drain() override;
    void drainResume() override;

  private:
    bool sendTimingRequest(const MatrixLsuParams::TimingRequest &request);
    void timingResponse(uint64_t transaction_id);
    void memoryRetryAccepted();
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_ENGINE_HH__
