/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/engine.hh"

#include <algorithm>

#include "arch/riscv/isa.hh"
#include "arch/riscv/regs/matrix.hh"
#include "base/logging.hh"
#include "cpu/base.hh"
#include "cpu/compute_offload.hh"
#include "cpu/thread_context.hh"
#include "debug/AMEFlow.hh"
#include "dev/spm_virt_dma.hh"
#include "mem/request.hh"
#include "sim/system.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

namespace
{

ComputeOffloadCPU &
offloadCPU(BaseCPU *cpu)
{
    auto *host = dynamic_cast<ComputeOffloadCPU *>(cpu);
    fatal_if(!host,
             "AMEEngine cpu must implement the compute offload boundary");
    return *host;
}

unsigned
cycles(Cycles value, const char *name)
{
    const auto count = static_cast<unsigned>(value);
    fatal_if(count == 0, "AMEEngine %s must be non-zero", name);
    return count;
}

MatrixDataflow
parseDataflow(const std::string &name)
{
    if (name == "pss")
        return MatrixDataflow::PartialSumStationary;
    if (name == "ws")
        return MatrixDataflow::WeightStationary;
    fatal("Unknown AME dataflow '%s' (expected pss or ws)", name);
}

uint32_t
matrixRegisterRaw(ThreadContext *tc, const RegId &reg)
{
    RiscvISA::MatRegContainer value;
    value.zero();
    tc->getReg(reg, &value);
    const auto rows = value.as<uint8_t>();
    uint32_t raw = 0;
    for (unsigned byte = 0; byte < sizeof(raw); ++byte)
        raw |= static_cast<uint32_t>(rows[0][byte]) << (byte * 8);
    return raw;
}

AmePipelineParams
pipelineParams(const AMEEngine::Params &p,
               MatrixLsuParams::TimingRequestSender sender)
{
    AmePipelineParams result;
    result.dispatchQueueDepth = p.dispatch_queue_depth;
    result.maxInflightInsts = p.max_inflight_insts;
    result.instructionSlots = p.instruction_slots;
    result.peRequestQueueDepth = p.pe_request_queue_depth;
    result.completionQueueDepth = p.completion_queue_depth;
    result.completionWidth = p.completion_width;
    result.controlLatency = cycles(p.control_latency, "control_latency");
    result.mrfBanks = p.mrf_banks;
    result.mrfWordBits = p.mrf_word_bits;
    result.mrfRegisterBytes = p.mrf_register_bytes;
    result.mrfBankSkewGroupRegs = p.mrf_bank_skew_group_regs;
    result.mrfBankSkewWords = p.mrf_bank_skew_words;
    result.mrfRequestQueueDepth = p.mrf_request_queue_depth;
    result.mrfResponseQueueDepth = p.mrf_response_queue_depth;
    result.mrfReadLatency = cycles(p.mrf_read_latency, "mrf_read_latency");
    result.mrfWriteVisibility = cycles(
        p.mrf_write_visibility, "mrf_write_visibility");
    result.accBanks = p.acc_banks;
    result.accWordBits = p.acc_word_bits;
    result.accRegisterBytes = p.acc_register_bytes;
    result.accRequestQueueDepth = p.acc_request_queue_depth;
    result.accResponseQueueDepth = p.acc_response_queue_depth;
    result.accReadLatency = cycles(p.acc_read_latency, "acc_read_latency");
    result.accWriteVisibility = cycles(
        p.acc_write_visibility, "acc_write_visibility");
    result.operands.kBlock = p.k_block;
    result.operands.requestWidth = p.operand_request_width;
    result.systolic.peRows = p.pe_rows;
    result.systolic.peCols = p.pe_cols;
    result.systolic.tileRows = p.tile_rows;
    result.systolic.tileCols = p.tile_cols;
    result.systolic.kBlock = p.k_block;
    result.systolic.commandQueueDepth = p.tile_command_queue_depth;
    result.systolic.tileIngressDepth = p.tile_ingress_depth;
    result.systolic.pssContinuous = p.pss_continuous;
    result.systolic.operationDepth = p.tile_operation_depth;
    result.systolic.edgeFifoDepth = p.edge_fifo_depth;
    result.systolic.drainQueueDepth = p.drain_queue_depth;
    result.systolic.peHopLatency = cycles(p.pe_hop_latency, "pe_hop_latency");
    result.systolic.tileBoundaryLatency = cycles(
        p.tile_boundary_latency, "tile_boundary_latency");
    result.systolic.macLatency = cycles(p.mac_latency, "mac_latency");
    result.systolic.macII = p.mac_initiation_interval;
    result.mlsu.descriptorQueueDepth = p.mlsu_descriptor_queue_depth;
    result.mlsu.packetBytes = p.mlsu_packet_bytes;
    result.mlsu.maxOutstanding = p.mlsu_max_outstanding;
    result.mlsu.responseLatency = cycles(
        p.mlsu_response_latency, "mlsu_response_latency");
    result.mlsu.reorderWindow = p.mlsu_reorder_window;
    result.mlsu.responseWidth = p.mlsu_response_width;
    result.mlsu.sendTimingRequest = std::move(sender);
    result.mrsu.queueDepth = p.auxiliary_queue_depth;
    result.mrsu.localLatency = cycles(
        p.mrsu_local_latency, "mrsu_local_latency");
    result.mrsu.crossTileLatency = cycles(
        p.mrsu_cross_tile_latency, "mrsu_cross_tile_latency");
    result.mrsu.tileRows = p.tile_rows;
    result.mrsu.tileCols = p.tile_cols;
    result.mrsu.resultWidth = p.auxiliary_result_width;
    result.mptu.queueDepth = p.auxiliary_queue_depth;
    result.mptu.localLatency = cycles(p.mptu_latency, "mptu_latency");
    result.mptu.crossTileLatency = 1;
    result.mptu.tileRows = p.tile_rows;
    result.mptu.tileCols = p.tile_cols;
    result.mptu.resultWidth = p.auxiliary_result_width;
    return result;
}

} // anonymous namespace

AMEEngine::AMEStats::AMEStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(cycles, statistics::units::Cycle::get(), "AME engine cycles"),
      ADD_STAT(activeCycles, statistics::units::Cycle::get(),
               "AME cycles with active work"),
      ADD_STAT(completedInstructions, statistics::units::Count::get(),
               "AME instructions completed"),
      ADD_STAT(squashes, statistics::units::Count::get(),
               "AME contexts squashed"),
      ADD_STAT(mrfGrants, statistics::units::Count::get(), "MRF bank grants"),
      ADD_STAT(mrfConflictCycles, statistics::units::Count::get(),
               "MRF queued bank conflicts"),
      ADD_STAT(accGrants, statistics::units::Count::get(), "Acc bank grants"),
      ADD_STAT(accConflictCycles, statistics::units::Count::get(),
               "Acc queued bank conflicts"),
      ADD_STAT(saMacOperations, statistics::units::Count::get(),
               "Legacy MAC count credited at tile completion (not PE issue time)"),
      ADD_STAT(saPeIssues, statistics::units::Count::get(),
               "PE MAC issues whose scheduled issue cycle has arrived"),
      ADD_STAT(saPeResults, statistics::units::Count::get(),
               "PE MAC results at actual pipeline completion"),
      ADD_STAT(saActiveCycles, statistics::units::Cycle::get(),
               "Engine ticks with active SA work, including supply/drain waits"),
      ADD_STAT(mlsuActiveCycles, statistics::units::Cycle::get(),
               "Engine ticks with active M-LSU work"),
      ADD_STAT(saMlsuOverlapCycles, statistics::units::Cycle::get(),
               "Cycles with an active SA work item and active M-LSU work"),
      ADD_STAT(saCommandQueueOverlapCycles, statistics::units::Cycle::get(),
               "Cycles with active SA work and a following queued SA command"),
      ADD_STAT(saTileCompletions, statistics::units::Count::get(),
               "K-step tile operations completed"),
      ADD_STAT(saEdgeTransfers, statistics::units::Count::get(),
               "Horizontal and vertical tile-edge transfers"),
      ADD_STAT(saEdgeStalls, statistics::units::Count::get(),
               "Tile link-cycles blocked by a full edge FIFO"),
      ADD_STAT(saInjectionStalls, statistics::units::Count::get(),
               "SA cycles blocked at a west or north ingress"),
      ADD_STAT(saDrainStalls, statistics::units::Count::get(),
               "SA cycles blocked by a full drain queue"),
      ADD_STAT(mrsuLocalCycles, statistics::units::Cycle::get(),
               "MRSU local reduction or scan cycles"),
      ADD_STAT(mrsuCrossTileCycles, statistics::units::Cycle::get(),
               "MRSU cross-tile reduction merge cycles"),
      ADD_STAT(mrsuPrefixCycles, statistics::units::Cycle::get(),
               "MRSU ordered cross-tile prefix cycles"),
      ADD_STAT(mrsuComputeItems, statistics::units::Count::get(),
               "MRSU local elements processed"),
      ADD_STAT(mptuComputeItems, statistics::units::Count::get(),
               "MPTU fragments processed"),
      ADD_STAT(memoryRequests, statistics::units::Count::get(),
               "M-LSU unique timing requests generated, including pending retry"),
      ADD_STAT(memoryAccepted, statistics::units::Count::get(),
               "M-LSU timing requests accepted by SPM, excluding retries"),
      ADD_STAT(memoryRetries, statistics::units::Count::get(),
               "M-LSU timing requests rejected and retried"),
      ADD_STAT(memoryResponses, statistics::units::Count::get(),
               "M-LSU timing responses received"),
      ADD_STAT(memoryOutstanding, statistics::units::Count::get(),
               "Current M-LSU transactions awaiting timing completion")
{
}

AMEEngine::MemoryPort::MemoryPort(
    const std::string &name, AMEEngine &engine_)
    : RequestPort(name), engine(engine_)
{
}

AMEEngine::MemoryPort::~MemoryPort()
{
    if (retryPacket) {
        delete retryPacket->popSenderState();
        delete retryPacket;
    }
}

bool
AMEEngine::MemoryPort::send(const MatrixLsuParams::TimingRequest &request)
{
    if (blocked())
        return false;
    auto memory_request = std::make_shared<Request>(
        request.address, request.bytes, Request::Flags{}, engine.requestorId);
    memory_request->setContext(request.contextId);

    /* Local SPM stores reserve write resources without publishing speculative
     * data. BankedScratchpad acknowledges NO_ACCESS writes; the host CPU
     * publishes the architectural payload at in-order commit. */
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
AMEEngine::MemoryPort::recvReqRetry()
{
    panic_if(!retryPacket, "AME M-LSU received retry without a packet");
    if (!sendTimingReq(retryPacket)) {
        ++engine.stats.memoryRetries;
        return;
    }
    retryPacket = nullptr;
    ++engine.stats.memoryAccepted;
    engine.memoryRetryAccepted();
}

bool
AMEEngine::MemoryPort::recvTimingResp(PacketPtr packet)
{
    panic_if(packet->isError(), "AME timing memory error at address %#x",
             packet->getAddr());
    panic_if(!packet->isResponse(), "AME M-LSU received a non-response");
    auto *state = dynamic_cast<SenderState *>(packet->popSenderState());
    panic_if(!state, "AME M-LSU response has invalid sender state");
    const uint64_t transaction_id = state->transactionId;
    delete state;
    delete packet;
    ++engine.stats.memoryResponses;
    engine.timingResponse(transaction_id);
    return true;
}

AMEEngine::AMEEngine(const Params &params)
    : ClockedObject(params), stats(this), cpu(*params.cpu),
      instQueue(offloadCPU(params.cpu).getInstQueue()),
      memoryPort(name() + ".memory_port", *this),
      requestorId(params.system->getRequestorId(this)),
      pipeline(instQueue,
          pipelineParams(params,
              [this](const MatrixLsuParams::TimingRequest &request) {
                  return sendTimingRequest(request);
              }),
          [this, dataflow = parseDataflow(params.dataflow)](
              AmeConfig &config, const InstQueuePacket &packet) {
              auto *tc = cpu.getContext(packet.key.threadId);
              panic_if(!tc, "AME config snapshot has no ThreadContext");
              config.nAme = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_AMENLEN);
              config.unitBits = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_AMEUDSZ);
              config.ameType = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_AMESTYPE);
              config.exceptionFlags = tc->readMiscRegNoEffect(
                  RiscvISA::MISCREG_AMEFFLAGS);
              config.dataflow = dataflow;
              config.mdBits = config.unitBits;
              config.adBits = config.unitBits;
              for (unsigned operand = 0; operand < packet.srcRegs.size();
                   ++operand) {
                  const auto &reg = packet.srcRegs[operand];
                  if (!reg.is(MatRegClass))
                      continue;
                  auto captured = std::find_if(
                      packet.matrixTypeSrcs.begin(),
                      packet.matrixTypeSrcs.end(),
                      [operand](const auto &entry) {
                          return entry.operandIndex == operand;
                      });
                  const uint32_t raw =
                      captured == packet.matrixTypeSrcs.end() ?
                          matrixRegisterRaw(tc, reg) : captured->value;
                  if (reg.index() >= RiscvISA::ZttMatDataRegBase &&
                      reg.index() < RiscvISA::ZttMatDataRegBase +
                          RiscvISA::NumZttMatDataRegs) {
                      const unsigned bits = raw & 0xff;
                      const unsigned index = reg.index() -
                          RiscvISA::ZttMatDataRegBase;
                      if (bits != 0) {
                          config.mRegBits[index] = bits;
                          config.mdBits = bits;
                      }
                  } else if (reg.index() >= RiscvISA::ZttAccDataRegBase &&
                      reg.index() < RiscvISA::ZttAccDataRegBase +
                          RiscvISA::NumZttAccDataRegs) {
                      const unsigned bits = raw & 0xff;
                      const unsigned index = reg.index() -
                          RiscvISA::ZttAccDataRegBase;
                      if (bits != 0) {
                          config.accRegBits[index] = bits;
                          config.adBits = bits;
                      }
                  }
              }
          },
          [source = params.dma_event_source](uint64_t mask) {
              return !source || source->eventsReady(mask);
          },
          [source = params.dma_event_source](uint64_t mask) {
              if (source) source->consumeEvents(mask);
          }),
      tickEvent([this] { tick(); }, name() + ".tick",
                false, Event::CPU_Tick_Pri)
{
}

Port &
AMEEngine::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "memory_port")
        return memoryPort;
    return ClockedObject::getPort(if_name, idx);
}

void
AMEEngine::init()
{
    ClockedObject::init();
    fatal_if(!memoryPort.isConnected(),
             "%s.memory_port must be connected to a timing memory system",
             name());
}

bool
AMEEngine::sendTimingRequest(
    const MatrixLsuParams::TimingRequest &request)
{
    return memoryPort.send(request);
}

void
AMEEngine::timingResponse(uint64_t transaction_id)
{
    pipeline.completeTimingResponse(transaction_id);
    wakeup();
}

void
AMEEngine::memoryRetryAccepted()
{
    wakeup();
}

void
AMEEngine::startup()
{
    ClockedObject::startup();
    instQueue.addDispatchNotify([this] { wakeup(); });
    instQueue.addCancelNotify(
        [this](const InstQueueKey &key) { cancel(key); });
    if (instQueue.hasPending(InstQueueEngine::AME))
        wakeup();
}

void
AMEEngine::wakeup()
{
    if (!tickEvent.scheduled())
        schedule(tickEvent, clockEdge(Cycles(1)));
}

void
AMEEngine::cancel(const InstQueueKey &key)
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
AMEEngine::tick()
{
    ++stats.cycles;
    if (pipeline.active())
        ++stats.activeCycles;
    if (pipeline.saActive())
        ++stats.saActiveCycles;
    if (pipeline.mlsuActive())
        ++stats.mlsuActiveCycles;
    if (pipeline.saWorkActive() && pipeline.mlsuActive())
        ++stats.saMlsuOverlapCycles;
    if (pipeline.saWorkActive() && pipeline.saCommandQueued())
        ++stats.saCommandQueueOverlapCycles;
    pipeline.evaluate();
    stats.completedInstructions = pipeline.completionCount();
    stats.mrfGrants = pipeline.mrfGrants();
    stats.mrfConflictCycles = pipeline.mrfConflicts();
    stats.accGrants = pipeline.accGrants();
    stats.accConflictCycles = pipeline.accConflicts();
    stats.saMacOperations = pipeline.saMacIssues();
    stats.saPeIssues = pipeline.saPeIssues();
    stats.saPeResults = pipeline.saPeResults();
    stats.saTileCompletions = pipeline.saTileCompletions();
    stats.saEdgeTransfers = pipeline.saEdgeTransfers();
    stats.saEdgeStalls = pipeline.saEdgeStalls();
    stats.saInjectionStalls = pipeline.saInjectionStalls();
    stats.saDrainStalls = pipeline.saDrainStalls();
    stats.mrsuLocalCycles = pipeline.mrsuLocalCycles();
    stats.mrsuCrossTileCycles = pipeline.mrsuCrossTileCycles();
    stats.mrsuPrefixCycles = pipeline.mrsuPrefixCycles();
    stats.mrsuComputeItems = pipeline.mrsuComputeItems();
    stats.mptuComputeItems = pipeline.mptuComputeItems();
    stats.memoryOutstanding = pipeline.mlsuOutstanding();
    if (pipeline.active()) {
        schedule(tickEvent, clockEdge(Cycles(1)));
    } else if (draining) {
        draining = false;
        signalDrainDone();
    }
}

DrainState
AMEEngine::drain()
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
AMEEngine::drainResume()
{
    draining = false;
    if (pipeline.active())
        wakeup();
}

} // namespace ame
} // namespace minor
} // namespace gem5
