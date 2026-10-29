/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "mem/banked_scratchpad.hh"

#include <algorithm>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/BankedSPM.hh"
#include "debug/Drain.hh"

namespace gem5
{
namespace memory
{

BankedScratchpad::ScratchpadStats::ScratchpadStats(
    statistics::Group *parent, unsigned banks, unsigned port_count)
    : statistics::Group(parent),
      ADD_STAT(timingRequests, statistics::units::Count::get(),
               "Accepted timing requests"),
      ADD_STAT(requestRetries, statistics::units::Count::get(),
               "Timing requests rejected by finite resources"),
      ADD_STAT(bytesRead, statistics::units::Byte::get(),
               "Bytes read through timing ports"),
      ADD_STAT(bytesWritten, statistics::units::Byte::get(),
               "Write bytes serviced through timing ports, including reservations"),
      ADD_STAT(storeReservationBytes, statistics::units::Byte::get(),
               "Speculative write bytes timed without publishing data"),
      ADD_STAT(functionalBytesRead, statistics::units::Byte::get(),
               "Functional read bytes, including architectural engine loads"),
      ADD_STAT(functionalBytesWritten, statistics::units::Byte::get(),
               "Functional write bytes, including committed engine stores"),
      ADD_STAT(bankAccesses, statistics::units::Count::get(),
               "Requests touching each SPM bank"),
      ADD_STAT(bankQueueFull, statistics::units::Count::get(),
               "Rejected requests targeting each full SPM bank"),
      ADD_STAT(bankWaitTicks, statistics::units::Tick::get(),
               "Sum of accepted request queue waits per touched bank; not stall union"),
      ADD_STAT(portReadBytes, statistics::units::Byte::get(),
               "Accepted timing read bytes by incoming port"),
      ADD_STAT(portWriteBytes, statistics::units::Byte::get(),
               "Accepted timing write bytes by incoming port, including reservations"),
      ADD_STAT(completedRequestTicks, statistics::units::Tick::get(),
               "Sum of acceptance-to-service-completion latency; not total stall")
{
    bankAccesses.init(banks);
    bankQueueFull.init(banks);
    bankWaitTicks.init(banks);
    portReadBytes.init(port_count);
    portWriteBytes.init(port_count);
    for (unsigned bank = 0; bank < banks; ++bank) {
        bankAccesses.subname(bank, csprintf("bank%u", bank));
        bankQueueFull.subname(bank, csprintf("bank%u", bank));
        bankWaitTicks.subname(bank, csprintf("bank%u", bank));
    }
    for (unsigned port = 0; port < port_count; ++port) {
        portReadBytes.subname(port, csprintf("port%u", port));
        portWriteBytes.subname(port, csprintf("port%u", port));
    }
}

BankedScratchpad::BankedScratchpad(const Params &params)
    : AbstractMemory(params), stats(this, params.bank_count, params.port_port_connection_count),
      bankCount(params.bank_count), bankInterleave(params.bank_interleave),
      bankQueueDepth(params.bank_queue_depth),
      bankIssueLatency(params.bank_issue_latency),
      readLatency(params.read_latency), writeLatency(params.write_latency),
      responseQueueDepth(params.response_queue_depth),
      bankOccupancy(bankCount, 0), bankAvailableAt(bankCount, 0),
      serviceEvent([this] { serviceReady(); }, name() + ".service")
{
    fatal_if(bankCount == 0 || bankInterleave == 0 || bankQueueDepth == 0 ||
             static_cast<uint64_t>(bankIssueLatency) == 0 ||
             static_cast<uint64_t>(readLatency) == 0 ||
             static_cast<uint64_t>(writeLatency) == 0 ||
             responseQueueDepth == 0,
             "%s has invalid bank or latency parameters", name());

    for (PortID id = 0; id < params.port_port_connection_count; ++id) {
        ports.push_back(std::make_unique<MemoryPort>(
            name() + csprintf(".port[%d]", id), id, *this));
    }
}

Tick
BankedScratchpad::cyclesToTicks(Cycles cycles) const
{
    return clockPeriod() * static_cast<Tick>(cycles);
}

std::vector<unsigned>
BankedScratchpad::touchedBanks(Addr address, unsigned size) const
{
    panic_if(size == 0, "%s received a zero-byte request", name());
    const AddrRange request_range = RangeSize(address, size);
    panic_if(!request_range.isSubset(getAddrRange()),
             "%s request range %s is outside SPM range %s", name(),
             request_range.to_string(), getAddrRange().to_string());

    std::vector<unsigned> banks;
    Addr cursor = address;
    const Addr end = address + size;
    while (cursor < end) {
        const Addr offset = cursor - start();
        const unsigned bank = (offset / bankInterleave) % bankCount;
        if (std::find(banks.begin(), banks.end(), bank) == banks.end())
            banks.push_back(bank);
        const Addr stripe_end = cursor + bankInterleave -
            (offset % bankInterleave);
        cursor = std::min(end, stripe_end);
    }
    return banks;
}

Tick
BankedScratchpad::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(),
             "%s must not receive cache-responding packets", name());
    touchedBanks(pkt->getAddr(), pkt->getSize());
    const Cycles latency = pkt->isRead() ? readLatency : writeLatency;
    access(pkt);
    return cyclesToTicks(latency);
}

void
BankedScratchpad::recvFunctional(PacketPtr pkt)
{
    touchedBanks(pkt->getAddr(), pkt->getSize());
    if (pkt->isRead())
        stats.functionalBytesRead += pkt->getSize();
    else if (pkt->isWrite())
        stats.functionalBytesWritten += pkt->getSize();
    pkt->pushLabel(name());
    functionalAccess(pkt);
    for (auto &[ready, request] : deferredRequests)
        if (!request.packet->req->getFlags().isSet(Request::NO_ACCESS))
            pkt->trySatisfyFunctional(request.packet);
    for (const auto &port : ports) {
        for (const auto response : port->responseQueue())
            if (!response->req->getFlags().isSet(Request::NO_ACCESS))
                pkt->trySatisfyFunctional(response);
    }
    pkt->popLabel();
}

bool
BankedScratchpad::recvTimingReq(PacketPtr pkt, PortID port_id)
{
    panic_if(pkt->cacheResponding(),
             "%s must not receive cache-responding packets", name());
    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "%s only accepts reads and writes, got %s", name(),
             pkt->cmdString());
    auto &port = *ports.at(port_id);
    if (pkt->needsResponse() && port.responseFull()) {
        DPRINTF(BankedSPM, "STALL port=%d reason=response_capacity addr=%#x\n",
                port_id, pkt->getAddr());
        port.markRetry();
        ++stats.requestRetries;
        return false;
    }

    const auto banks = touchedBanks(pkt->getAddr(), pkt->getSize());
    bool full = false;
    for (const unsigned bank : banks) {
        if (bankOccupancy[bank] >= bankQueueDepth) {
            DPRINTF(BankedSPM,
                    "STALL port=%d reason=bank_capacity bank=%u occupancy=%u addr=%#x\n",
                    port_id, bank, bankOccupancy[bank], pkt->getAddr());
            ++stats.bankQueueFull[bank];
            full = true;
        }
    }
    if (full) {
        port.markRetry();
        ++stats.requestRetries;
        return false;
    }

    const Tick receive_delay = pkt->headerDelay + pkt->payloadDelay;
    pkt->headerDelay = pkt->payloadDelay = 0;
    // A bank cannot begin servicing a packet before its payload arrives.
    const Tick arrived = clockEdge() + receive_delay;
    Tick service_start = arrived;
    for (const unsigned bank : banks)
        service_start = std::max(service_start, bankAvailableAt[bank]);
    for (const unsigned bank : banks) {
        ++bankOccupancy[bank];
        ++stats.bankAccesses[bank];
        stats.bankWaitTicks[bank] += service_start - arrived;
        bankAvailableAt[bank] = service_start +
            cyclesToTicks(bankIssueLatency);
        DPRINTF(BankedSPM,
                "BANK port=%d addr=%#x bank=%u occupancy=%u arrival=%llu start=%llu wait=%llu\n",
                port_id, pkt->getAddr(), bank, bankOccupancy[bank],
                arrived, service_start, service_start - arrived);
    }
    if (pkt->needsResponse())
        port.reserveResponse();

    const Cycles access_latency = pkt->isRead() ? readLatency : writeLatency;
    const Tick ready = service_start + cyclesToTicks(access_latency);
    deferredRequests.emplace(
        ready, DeferredRequest{pkt, port_id, std::move(banks), curTick()});
    ++stats.timingRequests;
    if (pkt->isRead()) {
        stats.bytesRead += pkt->getSize();
        stats.portReadBytes[port_id] += pkt->getSize();
    } else {
        stats.bytesWritten += pkt->getSize();
        stats.portWriteBytes[port_id] += pkt->getSize();
    }
    if (pkt->req->getFlags().isSet(Request::NO_ACCESS)) {
        panic_if(!pkt->isWrite(), "NO_ACCESS SPM timing request must be a write");
        stats.storeReservationBytes += pkt->getSize();
    }

    DPRINTF(BankedSPM,
            "accepted port=%d cmd=%s addr=%#x bytes=%u ready=%llu\n",
            port_id, pkt->cmdString(), pkt->getAddr(), pkt->getSize(), ready);
    scheduleService();
    return true;
}

void
BankedScratchpad::scheduleService()
{
    if (deferredRequests.empty())
        return;
    const Tick when = std::max(curTick(), deferredRequests.begin()->first);
    if (!serviceEvent.scheduled())
        schedule(serviceEvent, when);
    else if (when < serviceEvent.when())
        reschedule(serviceEvent, when, true);
}

void
BankedScratchpad::serviceReady()
{
    while (!deferredRequests.empty() &&
           deferredRequests.begin()->first <= curTick()) {
        auto it = deferredRequests.begin();
        DeferredRequest request = std::move(it->second);
        deferredRequests.erase(it);
        stats.completedRequestTicks += curTick() - request.receivedAt;

        const bool needs_response = request.packet->needsResponse();
        DPRINTF(BankedSPM,
                "COMPLETE port=%d addr=%#x bytes=%u cmd=%s\n",
                request.portId, request.packet->getAddr(),
                request.packet->getSize(), request.packet->cmdString());
        if (request.packet->req->getFlags().isSet(Request::NO_ACCESS)) {
            if (needs_response)
                request.packet->makeResponse();
        } else {
            access(request.packet);
        }
        for (const unsigned bank : request.banks) {
            panic_if(bankOccupancy[bank] == 0,
                     "%s bank occupancy underflow", name());
            --bankOccupancy[bank];
        }
        if (needs_response) {
            ports.at(request.portId)->releaseResponse();
            ports.at(request.portId)->queueResponse(request.packet);
        } else {
            delete request.packet;
        }
    }

    for (auto &port : ports) {
        port->trySendResponses();
        port->trySendRetry();
    }
    scheduleService();
    if (allIdle() && drainState() == DrainState::Draining) {
        DPRINTF(Drain, "%s completed drain\n", name());
        signalDrainDone();
    }
}

bool
BankedScratchpad::allIdle() const
{
    if (!deferredRequests.empty())
        return false;
    return std::all_of(ports.begin(), ports.end(), [](const auto &port) {
        return port->responseQueue().empty();
    });
}

Port &
BankedScratchpad::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "port" && idx >= 0 && idx < ports.size())
        return *ports[idx];
    return AbstractMemory::getPort(if_name, idx);
}

void
BankedScratchpad::init()
{
    AbstractMemory::init();
    for (auto &port : ports) {
        if (port->isConnected())
            port->sendRangeChange();
    }
}

DrainState
BankedScratchpad::drain()
{
    return allIdle() ? DrainState::Drained : DrainState::Draining;
}

BankedScratchpad::MemoryPort::MemoryPort(
    const std::string &name, PortID id_, BankedScratchpad &memory_)
    : ResponsePort(name, id_), memory(memory_), id(id_)
{
}

Tick
BankedScratchpad::MemoryPort::recvAtomic(PacketPtr pkt)
{
    return memory.recvAtomic(pkt);
}

Tick
BankedScratchpad::MemoryPort::recvAtomicBackdoor(
    PacketPtr pkt, MemBackdoorPtr &backdoor)
{
    backdoor = nullptr;
    return memory.recvAtomic(pkt);
}

void
BankedScratchpad::MemoryPort::recvFunctional(PacketPtr pkt)
{
    memory.recvFunctional(pkt);
}

void
BankedScratchpad::MemoryPort::recvMemBackdoorReq(
    const MemBackdoorReq &req, MemBackdoorPtr &backdoor)
{
    backdoor = nullptr;
}

bool
BankedScratchpad::MemoryPort::recvTimingReq(PacketPtr pkt)
{
    if (needRetry) {
        ++memory.stats.requestRetries;
        return false;
    }
    return memory.recvTimingReq(pkt, id);
}

void
BankedScratchpad::MemoryPort::recvRespRetry()
{
    trySendResponses();
    trySendRetry();
}

AddrRangeList
BankedScratchpad::MemoryPort::getAddrRanges() const
{
    return {memory.getAddrRange()};
}

bool
BankedScratchpad::MemoryPort::responseFull() const
{
    return responses.size() + responseReservations >=
        memory.responseQueueDepth;
}

void
BankedScratchpad::MemoryPort::releaseResponse()
{
    panic_if(responseReservations == 0,
             "%s response reservation underflow", name());
    --responseReservations;
}

void
BankedScratchpad::MemoryPort::queueResponse(PacketPtr pkt)
{
    panic_if(responseFull(), "%s response queue overflow", name());
    responses.push_back(pkt);
}

void
BankedScratchpad::MemoryPort::trySendResponses()
{
    while (!responses.empty()) {
        const Addr address = responses.front()->getAddr();
        if (!sendTimingResp(responses.front())) {
            DPRINTF(BankedSPM, "STALL port=%d reason=response_backpressure addr=%#x\n",
                    id, address);
            return;
        }
        DPRINTF(BankedSPM, "RESPONSE port=%d addr=%#x\n", id, address);
        responses.pop_front();
    }
}

void
BankedScratchpad::MemoryPort::trySendRetry()
{
    if (!needRetry || responseFull())
        return;
    needRetry = false;
    sendRetryReq();
}

} // namespace memory
} // namespace gem5
