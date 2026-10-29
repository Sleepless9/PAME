/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __DEV_SPM_VIRT_DMA_HH__
#define __DEV_SPM_VIRT_DMA_HH__

#include <cstdint>
#include <array>
#include <map>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/statistics.hh"
#include "dev/dma_virt_device.hh"
#include "mem/port.hh"
#include "params/SPMVirtDma.hh"

namespace gem5
{

/**
 * A descriptor-driven, two-dimensional DMA between system memory and a
 * banked scratchpad. System accesses use DmaVirtDevice while scratchpad
 * accesses use a dedicated timing port, so both sides contribute timing and
 * backpressure to the transfer.
 */
class SPMVirtDma : public DmaVirtDevice
{
  private:
    enum Register : Addr
    {
        RegControl = 0x00,
        RegStatus = 0x08,
        RegSystemAddr = 0x10,
        RegSpmAddr = 0x18,
        RegRowBytes = 0x20,
        RegRows = 0x28,
        RegSystemStride = 0x30,
        RegSpmStride = 0x38,
        RegCompletedBytes = 0x40,
        RegCompletedRows = 0x48,
        RegDescriptorId = 0x50,
        RegQueuedDescriptors = 0x58,
        RegCompletedId = 0x60,
        RegSubmittedSequence = 0x68,
        RegCompletedSequence = 0x70,
        RegRingBase = 0x78,
        RegRingCapacity = 0x80,
        RegRingHead = 0x88,
        RegRingTail = 0x90,
        RegEvents = 0x98,
        RegEventSet = 0xa0,
        RegEventClear = 0xa8,
        RegActiveDescriptors = 0xb0,
        RegFreeDescriptors = 0xb8,
        RegAbiVersion = 0xc0,
    };

    enum ControlBits : uint64_t
    {
        ControlStart = 1ULL << 0,
        ControlSpmToSystem = 1ULL << 1,
        ControlClearStatus = 1ULL << 2,
    };

    enum StatusBits : uint64_t
    {
        StatusBusy = 1ULL << 0,
        StatusDone = 1ULL << 1,
        StatusError = 1ULL << 2,
    };

    enum class ErrorCode : uint8_t
    {
        None = 0,
        QueueFull = 1,
        EmptyTransfer = 2,
        AddressOverflow = 3,
        SpmOutOfRange = 4,
        SpmResponse = 5,
        InvalidRing = 6,
        InvalidDescriptor = 7,
        SystemOutOfRange = 8,
    };

    struct Descriptor
    {
        uint64_t id = 0;
        uint64_t sequence = 0;
        uint64_t waitSequence = 0;
        uint64_t waitEvents = 0;
        uint64_t signalEvents = 0;
        Tick startedAt = 0;
        Addr systemBase = 0;
        Addr spmBase = 0;
        uint64_t rowBytes = 0;
        uint64_t rows = 0;
        uint64_t systemStride = 0;
        uint64_t spmStride = 0;
        bool spmToSystem = false;

        uint64_t nextRow = 0;
        uint64_t nextOffset = 0;
        uint64_t outstanding = 0;
        uint64_t completedBytes = 0;
        uint64_t completedRows = 0;
        bool failed = false;
        std::unordered_map<uint64_t, unsigned> rowOutstanding;
        std::unordered_set<uint64_t> fullyIssuedRows;
    };

    struct Chunk
    {
        Descriptor *owner = nullptr;
        Tick issuedAt = 0;
        Tick spmSentAt = 0;
        uint64_t row = 0;
        Addr systemAddr = 0;
        Addr spmAddr = 0;
        unsigned bytes = 0;
        unsigned bank = 0;
        bool spmToSystem = false;
        std::vector<uint8_t> data;
    };

    class RingCompletion : public DmaCallback
    {
        SPMVirtDma &device;
        void process() override;
      public:
        explicit RingCompletion(SPMVirtDma &d) : device(d) {}
    };

    struct SpmSenderState : public Packet::SenderState
    {
        explicit SpmSenderState(Chunk *chunk_) : chunk(chunk_) {}
        Chunk *chunk;
    };

    class SpmPort : public RequestPort
    {
      private:
        SPMVirtDma &device;

      protected:
        bool recvTimingResp(PacketPtr pkt) override;
        void recvReqRetry() override;

      public:
        SpmPort(const std::string &name, SPMVirtDma &device_)
            : RequestPort(name), device(device_)
        {}
    };

    class DmaCompletion : public DmaCallback
    {
      private:
        SPMVirtDma &device;
        Chunk *chunk;
        bool wasRead;

        void process() override;

      public:
        DmaCompletion(SPMVirtDma &device_, Chunk *chunk_, bool was_read)
            : device(device_), chunk(chunk_), wasRead(was_read)
        {}
    };

    struct DmaStats : public statistics::Group
    {
        statistics::Scalar descriptorsStarted;
        statistics::Scalar descriptorsCompleted;
        statistics::Scalar descriptorErrors;
        statistics::Scalar completedDescriptorTicks;
        statistics::Scalar systemToSpmBytes;
        statistics::Scalar spmToSystemBytes;
        statistics::Scalar spmRequests;
        statistics::Scalar spmRetries;
        statistics::Vector bankChunks;
        statistics::Scalar ringFetches;
        statistics::Scalar ringFetchBytes;
        statistics::Scalar ringDoorbells;
        statistics::Scalar completionEvents;
        statistics::Scalar concurrentDescriptorTicks;
        statistics::Scalar bidirectionalTicks;
        statistics::Scalar activeDescriptorTicks;
        statistics::Scalar outstandingChunkTicks;
        statistics::Scalar maxActiveDescriptors;
        statistics::Scalar maxOutstandingChunks;
        statistics::Scalar maxSystemCredits;
        statistics::Scalar systemCreditTicks;

        DmaStats(statistics::Group *parent, unsigned banks);
        SPMVirtDma *device;
        void preDumpStats() override;
        void resetStats() override;
    } stats;

    SpmPort spmPort;
    EventFunctionWrapper issueEvent;
    const Addr pioAddr;
    const Addr pioSize;
    const Tick pioLatency;
    const AddrRange spmRange;
    const unsigned bankCount;
    const Addr bankInterleave;
    const unsigned burstBytes;
    const unsigned descriptorQueueDepth;
    const unsigned maxOutstandingPerBank;
    const unsigned maxOutstandingSystem;
    const unsigned maxOutstandingPerDescriptor;
    const unsigned maxActiveDescriptors;
    const unsigned issueWidth;
    const unsigned cacheLineBytes;
    const RequestorID requestorId;

    uint64_t controlReg = 0;
    Addr systemAddrReg = 0;
    Addr spmAddrReg = 0;
    uint64_t rowBytesReg = 0;
    uint64_t rowsReg = 0;
    uint64_t systemStrideReg = 0;
    uint64_t spmStrideReg = 0;
    uint64_t descriptorIdReg = 0;
    uint64_t completedBytesReg = 0;
    uint64_t completedRowsReg = 0;
    uint64_t lastDescriptorId = 0;
    uint64_t submittedSequence = 0;
    uint64_t completedSequence = 0;
    uint64_t completedId = 0;
    std::map<uint64_t, uint64_t> retired;
    Addr ringBase = 0;
    uint64_t ringCapacity = 0;
    uint64_t ringHead = 0;
    uint64_t ringTail = 0;
    uint64_t events = 0;
    std::array<uint32_t, 64> eventCredits{};
    bool ringFetching = false;
    static constexpr unsigned RingDescriptorBytes = 128;
    static constexpr unsigned RingFetchBatch = 4;
    unsigned ringFetchCount = 0;
    unsigned ringFetchBytes = 0;
    unsigned ringFetchSystemCredits = 0;
    std::array<uint8_t, RingDescriptorBytes * RingFetchBatch> ringData{};
    Tick accountingTick = 0;
    unsigned roundRobin = 0;
    unsigned activePeak = 0;
    unsigned outstandingPeak = 0;
    unsigned systemCreditPeak = 0;
    bool doneSticky = false;
    bool errorSticky = false;
    ErrorCode lastError = ErrorCode::None;

    std::deque<Descriptor> descriptorQueue;
    std::deque<std::unique_ptr<Descriptor>> active;
    std::unordered_set<Chunk *> inFlightChunks;
    std::vector<unsigned> bankOutstanding;
    unsigned systemOutstanding = 0;
    std::deque<PacketPtr> spmSendQueue;
    bool spmWaitingRetry = false;

    uint64_t readRegister(Addr offset) const;
    void writeRegister(Addr offset, uint64_t value);
    void submitDescriptor(bool spm_to_system);
    ErrorCode validateDescriptor(const Descriptor &descriptor) const;
    void startNextDescriptor();
    void issueTransfers();
    void enqueueSpmRequest(Chunk *chunk, bool write);
    void trySendSpmRequests();
    void dmaReadDone(Chunk *chunk);
    void dmaWriteDone(Chunk *chunk);
    void recvSpmResponse(PacketPtr pkt);
    void recvSpmRetry();
    void finishChunk(Chunk *chunk, bool success);
    void completeActive(Descriptor *descriptor);
    void failDescriptor(Descriptor *descriptor, ErrorCode code);
    void setError(ErrorCode code);
    bool enqueueDescriptor(Descriptor descriptor);
    void fetchRing();
    void ringReadDone();
    void kick();
    void account();
    bool issueOne(Descriptor &descriptor);
    bool localIdle() const;
    unsigned bankFor(Addr spm_addr) const;

  protected:
    AddrRangeList getAddrRanges() const override;
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;

  public:
    PARAMS(SPMVirtDma);
    explicit SPMVirtDma(const Params &params);

    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void init() override;
    DrainState drain() override;
    TranslationGenPtr translate(Addr vaddr, Addr size) override;
    // Device completion event interface; consumers must schedule their own
    // timing work and must not bypass architectural AME retirement.
  public:
    /** Hardware event sideband consumed by accelerator command queues. */
    bool eventsReady(uint64_t mask) const;
    void publishEvents(uint64_t mask);
    void consumeEvents(uint64_t mask);
    void clearEvents(uint64_t mask);

  private:
};

} // namespace gem5

#endif // __DEV_SPM_VIRT_DMA_HH__
