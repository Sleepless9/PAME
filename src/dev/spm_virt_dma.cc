/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "dev/spm_virt_dma.hh"

#include <algorithm>
#include <limits>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/SPMVirtDma.hh"
#include "mem/packet.hh"
#include "mem/packet_access.hh"
#include "mem/request.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

class IdentityTranslationGen : public TranslationGen
{
  protected:
    void
    translate(Range &range) const override
    {
        range.paddr = range.vaddr;
    }

  public:
    IdentityTranslationGen(Addr start, Addr size)
        : TranslationGen(start, size)
    {}
};

bool
spanOverflows(Addr base, uint64_t stride, uint64_t rows, uint64_t row_bytes,
              Addr &last_start)
{
    const Addr max_addr = std::numeric_limits<Addr>::max();
    const uint64_t row_index = rows - 1;
    if (row_index != 0 && stride > (max_addr - base) / row_index)
        return true;
    last_start = base + row_index * stride;
    return row_bytes - 1 > max_addr - last_start;
}

} // anonymous namespace

SPMVirtDma::DmaStats::DmaStats(statistics::Group *parent, unsigned banks)
    : statistics::Group(parent),
      ADD_STAT(descriptorsStarted, statistics::units::Count::get(),
               "DMA descriptors started"),
      ADD_STAT(descriptorsCompleted, statistics::units::Count::get(),
               "DMA descriptors completed successfully"),
      ADD_STAT(descriptorErrors, statistics::units::Count::get(),
               "DMA descriptors rejected or completed with an error"),
      ADD_STAT(completedDescriptorTicks, statistics::units::Tick::get(),
               "Start-to-completion ticks of retired descriptors, including failed transfers"),
      ADD_STAT(systemToSpmBytes, statistics::units::Byte::get(),
               "Bytes copied from system memory to the SPM"),
      ADD_STAT(spmToSystemBytes, statistics::units::Byte::get(),
               "Bytes copied from the SPM to system memory"),
      ADD_STAT(spmRequests, statistics::units::Count::get(),
               "SPM timing requests accepted"),
      ADD_STAT(spmRetries, statistics::units::Count::get(),
               "SPM timing requests rejected for retry"),
      ADD_STAT(bankChunks, statistics::units::Count::get(),
               "Transfer chunks assigned to each SPM bank"),
      ADD_STAT(ringFetches, statistics::units::Count::get(), "Timing descriptor fetches"),
      ADD_STAT(ringFetchBytes, statistics::units::Byte::get(), "Descriptor fetch traffic, separate from payload"),
      ADD_STAT(ringDoorbells, statistics::units::Count::get(), "Accepted ring tail updates"),
      ADD_STAT(completionEvents, statistics::units::Count::get(), "Successful descriptor event publications"),
      ADD_STAT(concurrentDescriptorTicks, statistics::units::Tick::get(), "Ticks with at least two active descriptors"),
      ADD_STAT(bidirectionalTicks, statistics::units::Tick::get(), "Ticks with active input and output descriptors"),
      ADD_STAT(activeDescriptorTicks, statistics::units::Tick::get(), "Integral of active descriptor count"),
      ADD_STAT(outstandingChunkTicks, statistics::units::Tick::get(), "Integral of payload chunk credits in use"),
      ADD_STAT(maxActiveDescriptors, statistics::units::Count::get(), "Peak active descriptors"),
      ADD_STAT(maxOutstandingChunks, statistics::units::Count::get(), "Peak payload chunk credits in use"),
      ADD_STAT(maxSystemCredits, statistics::units::Count::get(), "Peak shared credits including ring metadata"),
      ADD_STAT(systemCreditTicks, statistics::units::Tick::get(), "Integral of shared credits including ring metadata")
{
    device = static_cast<SPMVirtDma *>(parent);
    bankChunks.init(banks);
    for (unsigned bank = 0; bank < banks; ++bank)
        bankChunks.subname(bank, csprintf("bank%u", bank));
}

void
SPMVirtDma::DmaStats::preDumpStats()
{
    device->account();
    statistics::Group::preDumpStats();
}

void
SPMVirtDma::DmaStats::resetStats()
{
    statistics::Group::resetStats();
    device->accountingTick = curTick();
    device->activePeak = device->active.size();
    device->outstandingPeak = device->systemOutstanding;
    maxActiveDescriptors = device->activePeak;
    maxOutstandingChunks = device->outstandingPeak;
    device->systemCreditPeak = device->systemOutstanding +
        device->ringFetchSystemCredits;
    maxSystemCredits = device->systemCreditPeak;
}

SPMVirtDma::SPMVirtDma(const Params &params)
    : DmaVirtDevice(params), stats(this, params.spm_bank_count),
      spmPort(name() + ".spm_port", *this),
      issueEvent([this] { issueTransfers(); }, name() + ".issue"),
      pioAddr(params.pio_addr),
      pioSize(params.pio_size), pioLatency(params.pio_latency),
      spmRange(params.spm_range), bankCount(params.spm_bank_count),
      bankInterleave(params.spm_bank_interleave),
      burstBytes(params.burst_bytes),
      descriptorQueueDepth(params.descriptor_queue_depth),
      maxOutstandingPerBank(params.max_outstanding_per_bank),
      maxOutstandingSystem(params.max_outstanding_system),
      maxOutstandingPerDescriptor(params.max_outstanding_per_descriptor),
      maxActiveDescriptors(params.max_active_descriptors),
      issueWidth(params.issue_width),
      cacheLineBytes(params.system->cacheLineSize()),
      requestorId(params.system->getRequestorId(this)),
      bankOutstanding(bankCount, 0)
{
    fatal_if(pioSize < RegAbiVersion + sizeof(uint64_t),
             "%s PIO range is too small", name());
    fatal_if(bankCount == 0 || bankInterleave == 0 || burstBytes == 0 ||
             descriptorQueueDepth == 0 || maxOutstandingPerBank == 0 ||
             maxOutstandingSystem == 0 || maxActiveDescriptors == 0 ||
             maxOutstandingPerDescriptor == 0 ||
             maxActiveDescriptors > descriptorQueueDepth || issueWidth == 0,
             "%s has an invalid zero-valued capacity parameter", name());
    fatal_if(burstBytes > std::numeric_limits<unsigned>::max(),
             "%s burst_bytes exceeds the packet size representation", name());
}

Port &
SPMVirtDma::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "spm_port")
        return spmPort;
    return DmaVirtDevice::getPort(if_name, idx);
}

void
SPMVirtDma::init()
{
    DmaVirtDevice::init();
    panic_if(!spmPort.isConnected(),
             "SPM port of %s is not connected", name());
}

AddrRangeList
SPMVirtDma::getAddrRanges() const
{
    return {RangeSize(pioAddr, pioSize)};
}

Tick
SPMVirtDma::read(PacketPtr pkt)
{
    account();
    const Addr offset = pkt->getAddr() - pioAddr;
    if ((pkt->getSize() != 4 && pkt->getSize() != 8) ||
        (offset % sizeof(uint64_t)) != 0 ||
        offset + pkt->getSize() > pioSize) {
        pkt->setBadAddress();
        return pioLatency;
    }

    pkt->setUintX(readRegister(offset), ByteOrder::little);
    pkt->makeResponse();
    return pioLatency;
}

Tick
SPMVirtDma::write(PacketPtr pkt)
{
    const Addr offset = pkt->getAddr() - pioAddr;
    if ((pkt->getSize() != 4 && pkt->getSize() != 8) ||
        (offset % sizeof(uint64_t)) != 0 ||
        offset + pkt->getSize() > pioSize) {
        pkt->setBadAddress();
        return pioLatency;
    }

    writeRegister(offset, pkt->getUintX(ByteOrder::little));
    pkt->makeResponse();
    return pioLatency;
}

uint64_t
SPMVirtDma::readRegister(Addr offset) const
{
    switch (offset) {
      case RegControl:
        return controlReg;
      case RegStatus: {
        uint64_t status = !localIdle() ?
            static_cast<uint64_t>(StatusBusy) : 0;
        status |= doneSticky ? static_cast<uint64_t>(StatusDone) : 0;
        status |= errorSticky ? static_cast<uint64_t>(StatusError) : 0;
        status |= static_cast<uint64_t>(lastError) << 8;
        return status;
      }
      case RegSystemAddr:
        return systemAddrReg;
      case RegSpmAddr:
        return spmAddrReg;
      case RegRowBytes:
        return rowBytesReg;
      case RegRows:
        return rowsReg;
      case RegSystemStride:
        return systemStrideReg;
      case RegSpmStride:
        return spmStrideReg;
      case RegCompletedBytes:
        return !active.empty() ? active.front()->completedBytes : completedBytesReg;
      case RegCompletedRows:
        return !active.empty() ? active.front()->completedRows : completedRowsReg;
      case RegDescriptorId:
        return !active.empty() ? active.front()->id : lastDescriptorId;
      case RegQueuedDescriptors:
        return descriptorQueue.size();
      case RegCompletedId: return completedId;
      case RegSubmittedSequence: return submittedSequence;
      case RegCompletedSequence: return completedSequence;
      case RegRingBase: return ringBase;
      case RegRingCapacity: return ringCapacity;
      case RegRingHead: return ringHead;
      case RegRingTail: return ringTail;
      case RegEvents: return events;
      case RegActiveDescriptors: return active.size();
      case RegFreeDescriptors:
        return descriptorQueueDepth - active.size() - descriptorQueue.size() -
            retired.size() - ringFetchCount;
      case RegAbiVersion: return 1;
      default:
        return 0;
    }
}

void
SPMVirtDma::writeRegister(Addr offset, uint64_t value)
{
    switch (offset) {
      case RegControl:
        controlReg = value & ControlSpmToSystem;
        if (value & ControlClearStatus) {
            doneSticky = false;
            errorSticky = false;
            lastError = ErrorCode::None;
            kick();
        }
        if (value & ControlStart)
            submitDescriptor(value & ControlSpmToSystem);
        break;
      case RegSystemAddr:
        systemAddrReg = value;
        break;
      case RegSpmAddr:
        spmAddrReg = value;
        break;
      case RegRowBytes:
        rowBytesReg = value;
        break;
      case RegRows:
        rowsReg = value;
        break;
      case RegSystemStride:
        systemStrideReg = value;
        break;
      case RegSpmStride:
        spmStrideReg = value;
        break;
      case RegDescriptorId:
        descriptorIdReg = value;
        break;
      case RegRingBase:
        if (!localIdle() || value % 64 ||
            (ringCapacity && ringCapacity * 128 - 1 >
             std::numeric_limits<Addr>::max() - value) ||
            (ringCapacity && (!sys->isMemAddr(value) ||
             !sys->isMemAddr(value + ringCapacity * 128 - 1) ||
             RangeSize(value, ringCapacity * 128).intersects(spmRange)))) {
            setError(ErrorCode::InvalidRing); break;
        }
        ringBase = value;
        break;
      case RegRingCapacity:
        if (!localIdle() || value > 65536 ||
            (value && value * 128 - 1 > std::numeric_limits<Addr>::max() - ringBase) ||
            (value && (!sys->isMemAddr(ringBase) ||
             !sys->isMemAddr(ringBase + value * 128 - 1) ||
             RangeSize(ringBase, value * 128).intersects(spmRange)))) {
            setError(ErrorCode::InvalidRing); break;
        }
        ringCapacity = value;
        ringHead = ringTail = 0;
        break;
      case RegRingTail:
        if (!ringCapacity || value < ringTail || value < ringHead ||
            value - ringHead > ringCapacity) { setError(ErrorCode::InvalidRing); break; }
        ringTail = value;
        ++stats.ringDoorbells;
        kick();
        break;
      case RegEventSet:
        publishEvents(value);
        kick();
        break;
      case RegEventClear:
        clearEvents(value);
        break;
      default:
        warn("%s ignored write to read-only or reserved register %#x\n",
             name(), offset);
        break;
    }
}

void
SPMVirtDma::submitDescriptor(bool spm_to_system)
{
    // A ring batch exclusively owns the submission stream until drained.
    if (ringFetching || ringHead != ringTail) {
        setError(ErrorCode::InvalidRing);
        return;
    }

    Descriptor descriptor;
    descriptor.id = descriptorIdReg;
    descriptor.systemBase = systemAddrReg;
    descriptor.spmBase = spmAddrReg;
    descriptor.rowBytes = rowBytesReg;
    descriptor.rows = rowsReg;
    descriptor.systemStride = systemStrideReg;
    descriptor.spmStride = spmStrideReg;
    descriptor.spmToSystem = spm_to_system;

    enqueueDescriptor(std::move(descriptor));
}

void
SPMVirtDma::setError(ErrorCode code)
{
    errorSticky = true;
    lastError = code;
    ++stats.descriptorErrors;
}

bool
SPMVirtDma::enqueueDescriptor(Descriptor descriptor)
{
    if (descriptorQueue.size() + active.size() + retired.size() >= descriptorQueueDepth) {
        setError(ErrorCode::QueueFull);
        return false;
    }
    const ErrorCode error = validateDescriptor(descriptor);
    if (error != ErrorCode::None) {
        setError(error);
        return false;
    }
    if (descriptor.waitSequence > submittedSequence ||
        submittedSequence == std::numeric_limits<uint64_t>::max()) {
        setError(ErrorCode::InvalidDescriptor);
        return false;
    }
    doneSticky = false;
    descriptor.sequence = ++submittedSequence;
    descriptorQueue.push_back(std::move(descriptor));
    kick();
    return true;
}

SPMVirtDma::ErrorCode
SPMVirtDma::validateDescriptor(const Descriptor &descriptor) const
{
    if (descriptor.rowBytes == 0 || descriptor.rows == 0)
        return ErrorCode::EmptyTransfer;

    Addr system_last = 0;
    Addr spm_last = 0;
    if (spanOverflows(descriptor.systemBase, descriptor.systemStride,
                      descriptor.rows, descriptor.rowBytes, system_last) ||
        spanOverflows(descriptor.spmBase, descriptor.spmStride,
                      descriptor.rows, descriptor.rowBytes, spm_last)) {
        return ErrorCode::AddressOverflow;
    }

    const AddrRange last_row = RangeSize(spm_last, descriptor.rowBytes);
    if (!spmRange.contains(descriptor.spmBase) ||
        !last_row.isSubset(spmRange)) {
        return ErrorCode::SpmOutOfRange;
    }
    if (!sys->isMemAddr(descriptor.systemBase) ||
        !sys->isMemAddr(system_last + descriptor.rowBytes - 1) ||
        RangeSize(descriptor.systemBase,
                  system_last - descriptor.systemBase + descriptor.rowBytes).intersects(spmRange))
        return ErrorCode::SystemOutOfRange;
    return ErrorCode::None;
}

void
SPMVirtDma::startNextDescriptor()
{
    while (active.size() < maxActiveDescriptors) {
        auto next = std::find_if(descriptorQueue.begin(), descriptorQueue.end(),
            [this](const Descriptor &d) {
                return d.waitSequence <= completedSequence && eventsReady(d.waitEvents);
            });
        if (next == descriptorQueue.end()) break;
        account();
        auto descriptor = std::make_unique<Descriptor>(std::move(*next));
        descriptorQueue.erase(next);
        descriptor->startedAt = curTick();
        ++stats.descriptorsStarted;
        DPRINTF(SPMVirtDma,
                "start descriptor=%llu sequence=%llu direction=%s rows=%llu row_bytes=%llu\n",
                descriptor->id, descriptor->sequence,
                descriptor->spmToSystem ? "SPM-to-system" : "system-to-SPM",
                descriptor->rows, descriptor->rowBytes);
        active.push_back(std::move(descriptor));
        activePeak = std::max(activePeak, unsigned(active.size()));
        stats.maxActiveDescriptors = activePeak;
    }
}

unsigned
SPMVirtDma::bankFor(Addr spm_addr) const
{
    return ((spm_addr - spmRange.start()) / bankInterleave) % bankCount;
}

void
SPMVirtDma::issueTransfers()
{
    account();
    startNextDescriptor();
    fetchRing();
    unsigned issued = 0;
    unsigned missed = 0;
    while (!active.empty() && issued < issueWidth && missed < active.size()) {
        roundRobin %= active.size();
        if (issueOne(*active[roundRobin++])) {
            ++issued;
            missed = 0;
        } else {
            ++missed;
        }
    }
    trySendSpmRequests();
    if (issued == issueWidth) kick();
    if (localIdle() && drainState() == DrainState::Draining) signalDrainDone();
}

bool
SPMVirtDma::issueOne(Descriptor &descriptor)
{
    auto *active = &descriptor;
    if (!active->failed && active->nextRow < active->rows &&
        active->outstanding < maxOutstandingPerDescriptor &&
        systemOutstanding < maxOutstandingSystem) {
        const uint64_t row = active->nextRow;
        const uint64_t offset = active->nextOffset;
        const Addr system_addr = active->systemBase +
            row * active->systemStride + offset;
        const Addr spm_addr = active->spmBase +
            row * active->spmStride + offset;
        const unsigned bank = bankFor(spm_addr);

        if (bankOutstanding[bank] >= maxOutstandingPerBank) {
            DPRINTF(SPMVirtDma,
                    "STALL descriptor=%llu reason=bank_credit bank=%u outstanding=%u\n",
                    active->id, bank, bankOutstanding[bank]);
            return false;
        }

        const uint64_t row_remaining = active->rowBytes - offset;
        const uint64_t burst_remaining = burstBytes -
            (system_addr % burstBytes);
        const uint64_t stripe_offset =
            (spm_addr - spmRange.start()) % bankInterleave;
        const uint64_t stripe_remaining = bankInterleave - stripe_offset;
        const uint64_t line_remaining = cacheLineBytes - (system_addr % cacheLineBytes);
        const unsigned bytes = static_cast<unsigned>(std::min(
            {row_remaining, burst_remaining, stripe_remaining, line_remaining,
             static_cast<uint64_t>(burstBytes)}));

        auto *chunk = new Chunk;
        chunk->owner = active;
        chunk->issuedAt = curTick();
        chunk->row = row;
        chunk->systemAddr = system_addr;
        chunk->spmAddr = spm_addr;
        chunk->bytes = bytes;
        chunk->bank = bank;
        chunk->spmToSystem = active->spmToSystem;
        chunk->data.resize(bytes);
        inFlightChunks.insert(chunk);

        ++active->outstanding;
        ++active->rowOutstanding[row];
        ++bankOutstanding[bank];
        account();
        ++systemOutstanding;
        systemCreditPeak = std::max(systemCreditPeak, systemOutstanding + unsigned(ringFetching));
        stats.maxSystemCredits = systemCreditPeak;
        outstandingPeak = std::max(outstandingPeak, systemOutstanding);
        stats.maxOutstandingChunks = outstandingPeak;
        ++stats.bankChunks[bank];

        active->nextOffset += bytes;
        if (active->nextOffset == active->rowBytes) {
            active->fullyIssuedRows.insert(row);
            ++active->nextRow;
            active->nextOffset = 0;
        }

        DPRINTF(SPMVirtDma,
                "issue descriptor=%llu row=%llu system=%#x spm=%#x "
                "bytes=%u bank=%u\n",
                active->id, row, system_addr, spm_addr, bytes, bank);
        if (chunk->spmToSystem) {
            enqueueSpmRequest(chunk, false);
        } else {
            auto *callback = new DmaCompletion(*this, chunk, true);
            dmaReadVirt(chunk->systemAddr, chunk->bytes, callback,
                        chunk->data.data());
        }
        return true;
    }

    if (active->nextRow < active->rows &&
        active->outstanding >= maxOutstandingPerDescriptor) {
        DPRINTF(SPMVirtDma,
                "STALL descriptor=%llu reason=descriptor_credit outstanding=%llu\n",
                active->id, active->outstanding);
    } else if (active->nextRow < active->rows &&
               systemOutstanding >= maxOutstandingSystem) {
        DPRINTF(SPMVirtDma,
                "STALL descriptor=%llu reason=system_credit outstanding=%u\n",
                active->id, systemOutstanding);
    }
    return false;
}

void
SPMVirtDma::enqueueSpmRequest(Chunk *chunk, bool write)
{
    RequestPtr request = std::make_shared<Request>(
        chunk->spmAddr, chunk->bytes, Request::UNCACHEABLE, requestorId);
    request->taskId(context_switch_task_id::DMA);
    PacketPtr packet = write ? Packet::createWrite(request) :
                               Packet::createRead(request);
    packet->dataStatic(chunk->data.data());
    packet->pushSenderState(new SpmSenderState(chunk));
    spmSendQueue.push_back(packet);
}

void
SPMVirtDma::trySendSpmRequests()
{
    if (spmWaitingRetry)
        return;

    while (!spmSendQueue.empty()) {
        PacketPtr packet = spmSendQueue.front();
        if (!spmPort.sendTimingReq(packet)) {
            DPRINTF(SPMVirtDma, "STALL reason=spm_retry addr=%#x bytes=%u\n",
                    packet->getAddr(), packet->getSize());
            spmWaitingRetry = true;
            ++stats.spmRetries;
            return;
        }
        auto *state = static_cast<SpmSenderState *>(packet->senderState);
        state->chunk->spmSentAt = curTick();
        DPRINTF(SPMVirtDma,
                "SPM_REQUEST descriptor=%llu row=%llu addr=%#x bytes=%u bank=%u\n",
                state->chunk->owner->id, state->chunk->row, packet->getAddr(),
                packet->getSize(), state->chunk->bank);
        spmSendQueue.pop_front();
        ++stats.spmRequests;
    }
}

void
SPMVirtDma::dmaReadDone(Chunk *chunk)
{
    panic_if(!inFlightChunks.count(chunk),
             "%s completed an unknown system read", name());
    DPRINTF(SPMVirtDma,
            "SYSTEM_READ_DONE descriptor=%llu row=%llu addr=%#x bytes=%u latency=%llu\n",
            chunk->owner->id, chunk->row, chunk->systemAddr, chunk->bytes,
            curTick() - chunk->issuedAt);
    enqueueSpmRequest(chunk, true);
    trySendSpmRequests();
}

void
SPMVirtDma::dmaWriteDone(Chunk *chunk)
{
    panic_if(!inFlightChunks.count(chunk),
             "%s completed an unknown system write", name());
    finishChunk(chunk, true);
}

void
SPMVirtDma::recvSpmResponse(PacketPtr pkt)
{
    auto *state = dynamic_cast<SpmSenderState *>(pkt->popSenderState());
    panic_if(!state, "%s received an SPM response without sender state",
             name());
    Chunk *chunk = state->chunk;
    DPRINTF(SPMVirtDma,
            "SPM_RESPONSE descriptor=%llu row=%llu addr=%#x bytes=%u bank=%u latency=%llu error=%u\n",
            chunk->owner->id, chunk->row, chunk->spmAddr, chunk->bytes,
            chunk->bank, curTick() - chunk->spmSentAt, pkt->isError());
    delete state;

    const bool success = !pkt->isError();
    delete pkt;
    if (!success) {
        finishChunk(chunk, false);
    } else if (chunk->spmToSystem) {
        auto *callback = new DmaCompletion(*this, chunk, false);
        dmaWriteVirt(chunk->systemAddr, chunk->bytes, callback,
                     chunk->data.data());
    } else {
        finishChunk(chunk, true);
    }
}

void
SPMVirtDma::recvSpmRetry()
{
    panic_if(!spmWaitingRetry,
             "%s received an unexpected SPM retry", name());
    spmWaitingRetry = false;
    trySendSpmRequests();
}

void
SPMVirtDma::finishChunk(Chunk *chunk, bool success)
{
    auto *active = chunk->owner;
    panic_if(!active || !inFlightChunks.erase(chunk),
             "%s completed a chunk outside its descriptor", name());
    panic_if(active->outstanding == 0 || bankOutstanding[chunk->bank] == 0 ||
             systemOutstanding == 0,
             "%s transfer credit underflow", name());

    DPRINTF(SPMVirtDma,
            "CHUNK_DONE descriptor=%llu row=%llu system=%#x spm=%#x bytes=%u bank=%u latency=%llu success=%u\n",
            active->id, chunk->row, chunk->systemAddr, chunk->spmAddr,
            chunk->bytes, chunk->bank, curTick() - chunk->issuedAt, success);
    account();
    --active->outstanding;
    --bankOutstanding[chunk->bank];
    --systemOutstanding;

    auto row_it = active->rowOutstanding.find(chunk->row);
    panic_if(row_it == active->rowOutstanding.end() || row_it->second == 0,
             "%s row credit underflow", name());
    --row_it->second;
    if (row_it->second == 0 &&
        active->fullyIssuedRows.erase(chunk->row) != 0) {
        active->rowOutstanding.erase(row_it);
        ++active->completedRows;
    }

    if (success) {
        active->completedBytes += chunk->bytes;
        if (chunk->spmToSystem)
            stats.spmToSystemBytes += chunk->bytes;
        else
            stats.systemToSpmBytes += chunk->bytes;
    } else {
        failDescriptor(active, ErrorCode::SpmResponse);
    }
    delete chunk;

    if (active->outstanding == 0 &&
        (active->failed || active->nextRow == active->rows)) {
        completeActive(active);
    }
    kick();
}

void
SPMVirtDma::failDescriptor(Descriptor *active, ErrorCode code)
{
    panic_if(!active, "%s has no active descriptor to fail", name());
    active->failed = true;
    errorSticky = true;
    lastError = code;
}

void
SPMVirtDma::completeActive(Descriptor *descriptor)
{
    auto *active = descriptor;
    account();
    panic_if(!active || active->outstanding != 0,
             "%s attempted to complete a busy descriptor", name());
    completedBytesReg = active->completedBytes;
    completedRowsReg = active->completedRows;
    lastDescriptorId = active->id;
    stats.completedDescriptorTicks += curTick() - active->startedAt;
    doneSticky = true;
    if (active->failed) {
        ++stats.descriptorErrors;
    } else {
        ++stats.descriptorsCompleted;
        publishEvents(active->signalEvents);
        if (active->signalEvents) ++stats.completionEvents;
    }
    retired.emplace(active->sequence, active->id);
    while (retired.count(completedSequence + 1)) {
        ++completedSequence;
        completedId = retired.at(completedSequence);
        retired.erase(completedSequence);
    }
    DPRINTF(SPMVirtDma,
            "complete descriptor=%llu sequence=%llu fence=%llu bytes=%llu rows=%llu error=%u\n",
            active->id, active->sequence, completedSequence,
            active->completedBytes, active->completedRows,
            static_cast<unsigned>(lastError));
    auto found = std::find_if(this->active.begin(), this->active.end(),
        [descriptor](const auto &d) { return d.get() == descriptor; });
    panic_if(found == this->active.end(), "Unknown completed descriptor");
    this->active.erase(found);
    if (localIdle() && drainState() == DrainState::Draining) {
        DPRINTF(Drain, "%s completed drain\n", name());
        signalDrainDone();
    }
}

bool
SPMVirtDma::eventsReady(uint64_t mask) const
{
    for (unsigned bit = 0; bit < eventCredits.size(); ++bit) {
        if ((mask & (1ULL << bit)) && eventCredits[bit] == 0)
            return false;
    }
    return true;
}

void
SPMVirtDma::publishEvents(uint64_t mask)
{
    for (unsigned bit = 0; bit < eventCredits.size(); ++bit) {
        if (!(mask & (1ULL << bit)))
            continue;
        if (eventCredits[bit] != std::numeric_limits<uint32_t>::max())
            ++eventCredits[bit];
        events |= 1ULL << bit;
    }
}

void
SPMVirtDma::consumeEvents(uint64_t mask)
{
    for (unsigned bit = 0; bit < eventCredits.size(); ++bit) {
        if (!(mask & (1ULL << bit)) || eventCredits[bit] == 0)
            continue;
        if (--eventCredits[bit] == 0)
            events &= ~(1ULL << bit);
    }
}

void
SPMVirtDma::clearEvents(uint64_t mask)
{
    for (unsigned bit = 0; bit < eventCredits.size(); ++bit) {
        if (mask & (1ULL << bit))
            eventCredits[bit] = 0;
    }
    events &= ~mask;
}

bool
SPMVirtDma::localIdle() const
{
    return active.empty() && descriptorQueue.empty() && inFlightChunks.empty() &&
        spmSendQueue.empty() && !spmWaitingRetry && !ringFetching && ringHead == ringTail;
}

void
SPMVirtDma::kick()
{
    if (!issueEvent.scheduled()) schedule(issueEvent, clockEdge(Cycles(1)));
}

void
SPMVirtDma::account()
{
    const Tick elapsed = curTick() - accountingTick;
    stats.activeDescriptorTicks += elapsed * active.size();
    stats.outstandingChunkTicks += elapsed * systemOutstanding;
    stats.systemCreditTicks += elapsed *
        (systemOutstanding + ringFetchSystemCredits);
    if (active.size() > 1) stats.concurrentDescriptorTicks += elapsed;
    bool input = false, output = false;
    for (const auto &d : active) {
        input |= !d->spmToSystem;
        output |= d->spmToSystem;
    }
    if (input && output) stats.bidirectionalTicks += elapsed;
    accountingTick = curTick();
}

void
SPMVirtDma::fetchRing()
{
    if (ringFetching || ringHead == ringTail || errorSticky ||
        active.size() + descriptorQueue.size() + retired.size() >=
            descriptorQueueDepth) return;
    const unsigned free = descriptorQueueDepth - active.size() -
        descriptorQueue.size() - retired.size();
    const unsigned available = std::min<uint64_t>(ringTail - ringHead, free);
    const unsigned before_wrap = ringCapacity - (ringHead % ringCapacity);
    ringFetchCount = std::min({RingFetchBatch, available, before_wrap});
    if (ringFetchCount == 0)
        return;
    account();
    ringFetching = true;
    ringFetchBytes = ringFetchCount * RingDescriptorBytes;
    ringFetchSystemCredits =
        (ringFetchBytes + cacheLineBytes - 1) / cacheLineBytes;
    systemCreditPeak = std::max(
        systemCreditPeak, systemOutstanding + ringFetchSystemCredits);
    stats.maxSystemCredits = systemCreditPeak;
    stats.ringFetches += ringFetchCount;
    stats.ringFetchBytes += ringFetchBytes;
    const Addr addr = ringBase +
        (ringHead % ringCapacity) * RingDescriptorBytes;
    DPRINTF(SPMVirtDma,
            "RING_FETCH_BATCH head=%llu count=%u addr=%#x bytes=%u\n",
            ringHead, ringFetchCount, addr, ringFetchBytes);
    /* DmaPort splits the batch into cache-line bursts and keeps their
     * responses outstanding, eliminating one response-latency round trip per
     * descriptor while retaining the 128-byte software ABI. */
    dmaReadVirt(addr, ringFetchBytes, new RingCompletion(*this),
                ringData.data());
}

void
SPMVirtDma::ringReadDone()
{
    panic_if(!ringFetching, "Unexpected ring completion");
    account();
    ringFetching = false;
    const unsigned fetched = ringFetchCount;
    ringFetchSystemCredits = 0;
    ringFetchCount = 0;
    for (unsigned record = 0; record < fetched; ++record) {
        uint64_t fields[16] = {};
        const unsigned base = record * RingDescriptorBytes;
        for (unsigned i = 0; i < 16; ++i)
            for (unsigned b = 0; b < 8; ++b)
                fields[i] |= uint64_t(ringData[base + i * 8 + b]) <<
                    (8 * b);
        ++ringHead;
        bool invalid = fields[7] & ~1ULL;
        for (unsigned i = 11; i < 16; ++i)
            invalid |= fields[i] != 0;
        if (invalid) {
            setError(ErrorCode::InvalidDescriptor);
            continue;
        }
        Descriptor d;
        d.id = fields[0]; d.systemBase = fields[1]; d.spmBase = fields[2];
        d.rowBytes = fields[3]; d.rows = fields[4];
        d.systemStride = fields[5]; d.spmStride = fields[6];
        d.spmToSystem = fields[7] & 1;
        d.waitSequence = fields[8]; d.waitEvents = fields[9];
        d.signalEvents = fields[10];
        enqueueDescriptor(std::move(d));
    }
    kick();
}

void
SPMVirtDma::RingCompletion::process()
{
    device.ringReadDone();
}

DrainState
SPMVirtDma::drain()
{
    return localIdle() ? DrainState::Drained : DrainState::Draining;
}

TranslationGenPtr
SPMVirtDma::translate(Addr vaddr, Addr size)
{
    return std::make_unique<IdentityTranslationGen>(vaddr, size);
}

bool
SPMVirtDma::SpmPort::recvTimingResp(PacketPtr pkt)
{
    device.recvSpmResponse(pkt);
    return true;
}

void
SPMVirtDma::SpmPort::recvReqRetry()
{
    device.recvSpmRetry();
}

void
SPMVirtDma::DmaCompletion::process()
{
    if (wasRead)
        device.dmaReadDone(chunk);
    else
        device.dmaWriteDone(chunk);
}

} // namespace gem5
