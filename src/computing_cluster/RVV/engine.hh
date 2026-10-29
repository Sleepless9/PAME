/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_ENGINE_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_ENGINE_HH__

#include "base/statistics.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "computing_cluster/RVV/pipeline.hh"
#include "params/RVVEngine.hh"
#include "sim/clocked_object.hh"
#include "sim/eventq.hh"

namespace gem5
{

class BaseCPU;

namespace minor
{
namespace rvv
{

class RVVEngine : public ClockedObject
{
  private:
    class MemoryPort : public RequestPort
    {
      private:
        struct SenderState : public Packet::SenderState
        {
            explicit SenderState(uint64_t id_) : transactionId(id_) {}
            uint64_t transactionId;
        };

        RVVEngine &engine;
        PacketPtr retryPacket = nullptr;

      protected:
        bool recvTimingResp(PacketPtr packet) override;
        void recvReqRetry() override;

      public:
        MemoryPort(const std::string &name, RVVEngine &engine_);
        ~MemoryPort() override;
        bool send(const VectorMemoryParams::TimingRequest &request);
        bool blocked() const { return retryPacket != nullptr; }
    };

    struct RVVStats : public statistics::Group
    {
        statistics::Scalar cycles, activeCycles, completedInstructions, squashes;
        statistics::Scalar memoryRequests, memoryAccepted, memoryRetries, memoryResponses;
        explicit RVVStats(statistics::Group *parent);
    } stats;

    BaseCPU &cpu;
    MinorInstQueue &instQueue;
    MemoryPort memoryPort;
    const bool useTimingMemory;
    const RequestorID requestorId;
    Pipeline pipeline;
    EventFunctionWrapper tickEvent;
    bool draining = false;

    void tick();
    void wakeup();
    void cancel(const InstQueueKey &key);
    bool sendTimingRequest(const VectorMemoryParams::TimingRequest &request);
    void timingResponse(uint64_t transaction_id);
    void memoryRetryAccepted();

  public:
    using Params = RVVEngineParams;
    explicit RVVEngine(const Params &params);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void init() override;

    void startup() override;
    DrainState drain() override;
    void drainResume() override;
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_ENGINE_HH__
