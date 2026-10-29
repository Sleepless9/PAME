/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __MEM_BANKED_SCRATCHPAD_HH__
#define __MEM_BANKED_SCRATCHPAD_HH__

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "base/statistics.hh"
#include "mem/abstract_mem.hh"
#include "mem/port.hh"
#include "params/BankedScratchpad.hh"
#include "sim/eventq.hh"

namespace gem5
{
namespace memory
{

/**
 * A byte-addressable scratchpad with striped banks and shared timing ports.
 * Requests to different banks overlap. Requests sharing a bank observe its
 * issue interval and finite queue, including requests which span bank stripes.
 */
class BankedScratchpad : public AbstractMemory
{
  private:
    class MemoryPort : public ResponsePort
    {
      private:
        BankedScratchpad &memory;
        const PortID id;
        bool needRetry = false;
        unsigned responseReservations = 0;
        std::deque<PacketPtr> responses;

      protected:
        Tick recvAtomic(PacketPtr pkt) override;
        Tick recvAtomicBackdoor(
            PacketPtr pkt, MemBackdoorPtr &backdoor) override;
        void recvFunctional(PacketPtr pkt) override;
        void recvMemBackdoorReq(
            const MemBackdoorReq &req, MemBackdoorPtr &backdoor) override;
        bool recvTimingReq(PacketPtr pkt) override;
        void recvRespRetry() override;
        AddrRangeList getAddrRanges() const override;

      public:
        MemoryPort(const std::string &name, PortID id_,
                   BankedScratchpad &memory_);
        bool responseFull() const;
        void reserveResponse() { ++responseReservations; }
        void releaseResponse();
        void queueResponse(PacketPtr pkt);
        void trySendResponses();
        void markRetry() { needRetry = true; }
        void trySendRetry();
        const std::deque<PacketPtr> &responseQueue() const { return responses; }
    };

    struct DeferredRequest
    {
        PacketPtr packet = nullptr;
        PortID portId = InvalidPortID;
        std::vector<unsigned> banks;
        Tick receivedAt = 0;
    };

    struct ScratchpadStats : public statistics::Group
    {
        statistics::Scalar timingRequests;
        statistics::Scalar requestRetries;
        statistics::Scalar bytesRead;
        statistics::Scalar bytesWritten;
        statistics::Scalar storeReservationBytes;
        statistics::Scalar functionalBytesRead;
        statistics::Scalar functionalBytesWritten;
        statistics::Vector bankAccesses;
        statistics::Vector bankQueueFull;
        statistics::Vector bankWaitTicks;
        statistics::Vector portReadBytes;
        statistics::Vector portWriteBytes;
        statistics::Scalar completedRequestTicks;

        ScratchpadStats(statistics::Group *parent, unsigned banks, unsigned ports);
    } stats;

    const unsigned bankCount;
    const Addr bankInterleave;
    const unsigned bankQueueDepth;
    const Cycles bankIssueLatency;
    const Cycles readLatency;
    const Cycles writeLatency;
    const unsigned responseQueueDepth;

    std::vector<std::unique_ptr<MemoryPort>> ports;
    std::vector<unsigned> bankOccupancy;
    std::vector<Tick> bankAvailableAt;
    std::multimap<Tick, DeferredRequest> deferredRequests;
    EventFunctionWrapper serviceEvent;

    std::vector<unsigned> touchedBanks(Addr address, unsigned size) const;
    Tick cyclesToTicks(Cycles cycles) const;
    Tick recvAtomic(PacketPtr pkt);
    void recvFunctional(PacketPtr pkt);
    bool recvTimingReq(PacketPtr pkt, PortID port_id);
    void serviceReady();
    void scheduleService();
    bool allIdle() const;

  public:
    PARAMS(BankedScratchpad);
    explicit BankedScratchpad(const Params &params);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void init() override;
    DrainState drain() override;
};

} // namespace memory
} // namespace gem5

#endif // __MEM_BANKED_SCRATCHPAD_HH__
