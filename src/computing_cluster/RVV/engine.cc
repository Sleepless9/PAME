/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/engine.hh"

#include "arch/riscv/isa.hh"
#include "base/logging.hh"
#include "cpu/base.hh"
#include "cpu/compute_offload.hh"
#include "debug/RVVEngine.hh"
#include "mem/request.hh"
#include "sim/system.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

namespace
{

ComputeOffloadCPU &
offloadCPU(BaseCPU *cpu)
{
    auto *host = dynamic_cast<ComputeOffloadCPU *>(cpu);
    fatal_if(!host,
             "RVVEngine cpu must implement the compute offload boundary");
    return *host;
}

unsigned
cycles(Cycles value, const char *name)
{
    const auto count = static_cast<unsigned>(value);
    fatal_if(count == 0, "RVVEngine %s must be non-zero", name);
    return count;
}

PipelineParams
pipelineParams(const RVVEngine::Params &params,
               VectorMemoryParams::TimingRequestSender sender)
{
    PipelineParams result;
    result.numLanes = params.num_lanes;
    result.laneWidthBits = params.lane_width_bits;
    result.vlenBits = params.vlen_bits;
    result.maxPackedElementsPerLane = params.elements_per_lane_cycle;
    result.dispatchQueueDepth = params.dispatch_queue_depth;
    result.maxInflightInsts = params.max_inflight_insts;
    result.rvvInsnSlotCapacity = params.rvv_insn_slot_capacity;
    result.peRequestQueueDepth = params.pe_request_queue_depth;
    result.completionQueueDepth = params.completion_queue_depth;
    result.completionWidth = params.completion_width;
    result.externalRequestQueueDepth = params.external_request_queue_depth;
    result.externalResultQueueDepth = params.external_result_queue_depth;

    result.memory.numLanes = params.num_lanes;
    result.memory.maxOutstanding = params.vlsu_max_outstanding;
    result.memory.lineBytes = params.vlsu_cache_line_bytes;
    result.memory.responseLatency =
        cycles(params.memory_latency, "memory_latency");
    result.memory.reorderWindow = params.vlsu_reorder_window;
    result.memory.responseWidth = params.vlsu_response_width;
    result.memory.sendTimingRequest = std::move(sender);

    result.sldu.numLanes = params.num_lanes;
    result.sldu.maxInflightWords = params.sldu_max_inflight_words;
    result.sldu.slideLatency =
        cycles(params.sldu_slide_latency, "sldu_slide_latency");
    result.sldu.intraLaneReductionLatency = cycles(
        params.sldu_intra_lane_reduction_latency,
        "sldu_intra_lane_reduction_latency");
    result.sldu.reductionStageLatency = cycles(
        params.sldu_reduction_stage_latency,
        "sldu_reduction_stage_latency");
    result.sldu.resultWidth = params.sldu_result_width;

    result.lane.numLanes = params.num_lanes;
    result.lane.laneWidthBits = params.lane_width_bits;
    result.lane.vlenBits = params.vlen_bits;
    result.lane.banksPerLane = params.vrf_banks_per_lane;
    result.lane.vrfArbitrationPolicy = params.vrf_arbitration_policy;
    result.lane.laneScoreboardEntries = params.lane_queue_depth;
    result.lane.laneInputDepth = params.lane_queue_depth;
    result.lane.operandTaskDepth = params.operand_task_depth;
    result.lane.directFuBypass = params.direct_fu_bypass;
    result.lane.vrfRequestQueueDepth = params.vrf_request_queue_depth;
    result.lane.operandQueueDepth = params.operand_queue_depth;
    result.lane.resultQueueDepth = params.result_queue_depth;
    result.lane.aluLatency = cycles(params.alu_latency, "alu_latency");
    result.lane.aluII = params.alu_initiation_interval;
    result.lane.multiplyLatency =
        cycles(params.multiply_latency, "multiply_latency");
    result.lane.multiplyII = params.multiply_initiation_interval;
    result.lane.fpuLatency = cycles(params.fpu_latency, "fpu_latency");
    result.lane.fpuII = params.fpu_initiation_interval;
    result.lane.divideLatency =
        cycles(params.divide_latency, "divide_latency");
    result.lane.divideII = params.divide_initiation_interval;
    result.lane.memoryLatency = 1;
    result.lane.memoryII = params.memory_initiation_interval;
    result.lane.configLatency =
        cycles(params.config_latency, "config_latency");
    return result;
}

} // anonymous namespace

RVVEngine::RVVStats::RVVStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(cycles, statistics::units::Cycle::get(), "Scheduled RVV engine ticks"),
      ADD_STAT(activeCycles, statistics::units::Cycle::get(), "RVV ticks with active work; not lane utilization"),
      ADD_STAT(completedInstructions, statistics::units::Count::get(), "RVV instructions completed"),
      ADD_STAT(squashes, statistics::units::Count::get(), "RVV contexts actually cancelled"),
      ADD_STAT(memoryRequests, statistics::units::Count::get(), "Unique timing requests generated"),
      ADD_STAT(memoryAccepted, statistics::units::Count::get(), "Timing requests accepted by SPM"),
      ADD_STAT(memoryRetries, statistics::units::Count::get(), "Rejected timing send attempts"),
      ADD_STAT(memoryResponses, statistics::units::Count::get(), "Timing responses received")
{}

RVVEngine::RVVEngine(const Params &params)
    : ClockedObject(params), stats(this), cpu(*params.cpu),
      instQueue(offloadCPU(params.cpu).getInstQueue()),
      memoryPort(name() + ".memory_port", *this),
      useTimingMemory(params.use_timing_memory),
      requestorId(params.system->getRequestorId(this)),
      pipeline(instQueue, pipelineParams(
          params,
          params.use_timing_memory ?
              VectorMemoryParams::TimingRequestSender(
                  [this](const VectorMemoryParams::TimingRequest &request) {
                      return sendTimingRequest(request);
                  }) :
              VectorMemoryParams::TimingRequestSender{}),
          [this](RvvConfig &config, const InstQueuePacket &packet) {
              const auto *tc = cpu.getContext(packet.key.threadId);
              panic_if(!tc, "RVV config snapshot has no ThreadContext");
              config.vstart = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_VSTART);
              config.vxrm = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_VXRM) & 0x3;
              config.frm = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_FRM) & 0x7;
          }),
      tickEvent([this] { tick(); }, name() + ".tick",
                false, Event::CPU_Tick_Pri)
{
}

RVVEngine::MemoryPort::MemoryPort(const std::string &name, RVVEngine &engine_)
    : RequestPort(name), engine(engine_)
{
}

RVVEngine::MemoryPort::~MemoryPort()
{
    if (retryPacket) {
        delete retryPacket->popSenderState();
        delete retryPacket;
    }
}

bool
RVVEngine::MemoryPort::send(const VectorMemoryParams::TimingRequest &request)
{
    if (blocked())
        return false;

    auto memory_request = std::make_shared<Request>(
        request.address, request.bytes, Request::Flags{},
        engine.requestorId);
    // Reserve local SPM write timing; architectural data is committed by Minor.
    if (request.isStore)
        memory_request->setFlags(Request::NO_ACCESS);
    const MemCmd command = request.isStore ?
        MemCmd::WriteReq : MemCmd::ReadReq;
    auto *packet = new Packet(memory_request, command);
    packet->allocate();
    packet->pushSenderState(new SenderState(request.transactionId));
    ++engine.stats.memoryRequests;
    if (!sendTimingReq(packet)) {
        retryPacket = packet;
        ++engine.stats.memoryRetries;
    } else {
        ++engine.stats.memoryAccepted;
    }
    return true;
}

void
RVVEngine::MemoryPort::recvReqRetry()
{
    panic_if(!retryPacket, "RVV VLSU received retry without a packet");
    if (!sendTimingReq(retryPacket)) {
        ++engine.stats.memoryRetries;
        return;
    }
    ++engine.stats.memoryAccepted;
    retryPacket = nullptr;
    engine.memoryRetryAccepted();
}

bool
RVVEngine::MemoryPort::recvTimingResp(PacketPtr packet)
{
    panic_if(!packet->isResponse(), "RVV VLSU received a non-response");
    panic_if(packet->isError(), "RVV timing memory error at address %#x",
             packet->getAddr());
    auto *state = dynamic_cast<SenderState *>(packet->popSenderState());
    panic_if(!state, "RVV VLSU response has invalid sender state");
    const uint64_t transaction_id = state->transactionId;
    delete state;
    delete packet;
    ++engine.stats.memoryResponses;
    engine.timingResponse(transaction_id);
    return true;
}

Port &
RVVEngine::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "memory_port")
        return memoryPort;
    return ClockedObject::getPort(if_name, idx);
}

void
RVVEngine::init()
{
    ClockedObject::init();
    fatal_if(useTimingMemory && !memoryPort.isConnected(),
             "%s.memory_port must be connected when timing memory is enabled",
             name());
}

bool
RVVEngine::sendTimingRequest(
    const VectorMemoryParams::TimingRequest &request)
{
    return memoryPort.send(request);
}

void
RVVEngine::timingResponse(uint64_t transaction_id)
{
    pipeline.completeTimingResponse(transaction_id);
    wakeup();
}

void
RVVEngine::memoryRetryAccepted()
{
    wakeup();
}

void
RVVEngine::startup()
{
    ClockedObject::startup();
    instQueue.addDispatchNotify([this] { wakeup(); });
    instQueue.addCancelNotify(
        [this](const InstQueueKey &key) { cancel(key); });
    if (instQueue.hasPending(InstQueueEngine::RVV))
        wakeup();
}

void
RVVEngine::wakeup()
{
    if (!tickEvent.scheduled())
        schedule(tickEvent, clockEdge(Cycles(1)));
}

void
RVVEngine::cancel(const InstQueueKey &key)
{
    if (pipeline.cancel(key))
        ++stats.squashes;
    if (pipeline.active())
        return;

    if (tickEvent.scheduled())
        deschedule(tickEvent);
    if (draining) {
        draining = false;
        signalDrainDone();
    }
}

void
RVVEngine::tick()
{
    ++stats.cycles;
    if (pipeline.active())
        ++stats.activeCycles;
    pipeline.evaluate();
    stats.completedInstructions = pipeline.completionCount();

    if (pipeline.active()) {
        schedule(tickEvent, clockEdge(Cycles(1)));
    } else if (draining) {
        draining = false;
        signalDrainDone();
    }
}

DrainState
RVVEngine::drain()
{
    if (!pipeline.active()) {
        if (tickEvent.scheduled())
            deschedule(tickEvent);
        return DrainState::Drained;
    }

    draining = true;
    wakeup();
    return DrainState::Draining;
}

void
RVVEngine::drainResume()
{
    draining = false;
    if (pipeline.active())
        wakeup();
}

} // namespace rvv
} // namespace minor
} // namespace gem5
