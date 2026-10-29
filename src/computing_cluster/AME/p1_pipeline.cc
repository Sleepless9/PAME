/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/p1_pipeline.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

AmePipeline::AmePipeline(
    MinorInstQueue &inst_queue, const AmePipelineParams &params_,
    ConfigSnapshot config_snapshot, DmaEventReady dma_event_ready,
    DmaEventConsume dma_event_consume)
    : instQueue(inst_queue), params(params_),
      contexts(params.maxInflightInsts),
      dispatchQueue(params.dispatchQueueDepth),
      peRequests(params.peRequestQueueDepth),
      completions(params.completionQueueDepth),
      mrf(RegisterFileKind::Mrf, params.mrfBanks, params.mrfWordBits,
          params.mrfRegisterBytes, params.mrfRequestQueueDepth,
          params.mrfResponseQueueDepth, params.mrfReadLatency,
          params.mrfWriteVisibility, params.mrfBankSkewGroupRegs,
          params.mrfBankSkewWords),
      accRf(RegisterFileKind::Acc, params.accBanks, params.accWordBits,
            params.accRegisterBytes, params.accRequestQueueDepth,
            params.accResponseQueueDepth, params.accReadLatency,
            params.accWriteVisibility),
      operands(contexts, scoreboard, mrf, params.operands),
      accumulator(contexts, scoreboard, accRf),
      systolic(contexts, operands, accumulator, params.systolic),
      mlsu(contexts, scoreboard, mrf, params.mlsu),
      mrsu(TargetMrsu, contexts, scoreboard, mrf, accRf, params.mrsu),
      mptu(TargetMptu, contexts, scoreboard, mrf, accRf, params.mptu),
      dispatcher(instQueue, contexts, scoreboard, dispatchQueue,
                 std::move(config_snapshot), std::move(dma_event_ready),
                 std::move(dma_event_consume)),
      sequencer(contexts, scoreboard, dispatchQueue, peRequests,
                params.instructionSlots),
      completionTracker(contexts, completions)
{
    fatal_if(params.completionWidth == 0 || params.controlLatency == 0,
             "Invalid AME pipeline parameters");
}

void
AmePipeline::evaluateMinorCompletion()
{
    /* AmeChannel pops take effect at advance().  Index the current entries
     * explicitly when retiring more than one completion in a cycle; repeatedly
     * reading front() would otherwise return the same command. */
    const size_t available = std::min<size_t>(
        completions.size(), params.completionWidth);
    for (size_t index = 0; index < available; ++index) {
        const auto completion = completions.at(index);
        InstQueueCompletion result;
        result.key = completion.key;
        result.fault = completion.fault;
        const bool accepted = instQueue.complete(std::move(result));
        panic_if(!accepted,
                 "AME completion rejected by MinorInstQueue id=%llu",
                 completion.contextId);
        const auto &context = contexts.get(completion.contextId);
        const AmeInsnId slot = context.desc.ameId;
        const bool allocated = context.slotAllocated;
        completions.popNext();
        dispatcher.release(completion.contextId);
        scoreboard.release(completion.contextId);
        contexts.release(completion.contextId);
        if (allocated)
            sequencer.releaseSlot(slot);
        ++completedInstructions;
    }
}

void
AmePipeline::evaluateRegisterResponses()
{
    const auto route = [this](AmeChannel<BankResponse> &responses) {
        const size_t available = responses.size();
        for (size_t index = 0; index < available; ++index) {
            const auto response = responses.at(index);
            switch (response.client) {
              case BankClient::SaA:
              case BankClient::SaB:
                operands.acceptResponse(response);
                break;
              case BankClient::SaAcc:
              case BankClient::SaWriteback:
                accumulator.acceptResponse(response);
                break;
              case BankClient::Mlsu:
                mlsu.acceptResponse(response);
                break;
              case BankClient::Mrsu:
                mrsu.acceptResponse(response);
                break;
              case BankClient::Mptu:
                mptu.acceptResponse(response);
                break;
            }
            responses.popNext();
        }
    };
    route(mrf.responseChannel());
    route(accRf.responseChannel());
}

void
AmePipeline::evaluateTargetDone()
{
    const auto consume = [this](AmeChannel<TargetDone> &channel) {
        const size_t available = channel.size();
        for (size_t index = 0; index < available; ++index) {
            completionTracker.targetDone(channel.at(index));
            channel.popNext();
        }
    };
    consume(systolic.doneChannel());
    consume(mlsu.doneChannel());
    consume(mrsu.doneChannel());
    consume(mptu.doneChannel());
}

void
AmePipeline::evaluateControl()
{
    if (controls.empty() || controls.front().readyCycle > currentCycle)
        return;
    completionTracker.targetDone(
        TargetDone{controls.front().contextId, TargetConfig});
    controls.pop_front();
}

void
AmePipeline::evaluateFanout()
{
    if (peRequests.empty())
        return;
    const auto request = peRequests.front();
    if ((request.targets & TargetSa) && !systolic.canAcceptNext())
        return;
    if ((request.targets & TargetMlsu) && !mlsu.canAcceptNext())
        return;
    if ((request.targets & TargetMrsu) && !mrsu.canAcceptNext())
        return;
    if ((request.targets & TargetMptu) && !mptu.canAcceptNext())
        return;
    if ((request.targets & TargetConfig) &&
        controls.size() >= params.peRequestQueueDepth) {
        return;
    }
    if (request.targets & TargetSa)
        systolic.acceptNext(request);
    if (request.targets & TargetMlsu)
        mlsu.acceptNext(request);
    if (request.targets & TargetMrsu)
        mrsu.acceptNext(request);
    if (request.targets & TargetMptu)
        mptu.acceptNext(request);
    if (request.targets & TargetConfig) {
        controls.push_back(
            ControlToken{currentCycle + params.controlLatency,
                         request.contextId});
    }
    sequencer.targetAccepted(request.contextId, request.targets);
    peRequests.popNext();
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u TARGET_ACCEPT mask=%#x\n",
            currentCycle, request.contextId, request.ameId, request.targets);
}

void
AmePipeline::evaluate()
{
    evaluateMinorCompletion();
    completionTracker.evaluate(currentCycle);
    evaluateRegisterResponses();
    evaluateTargetDone();
    evaluateControl();
    systolic.evaluate(currentCycle);
    mlsu.evaluate(currentCycle);
    mrsu.evaluate(currentCycle);
    mptu.evaluate(currentCycle);
    operands.evaluate(currentCycle);
    mrf.evaluate(currentCycle);
    accRf.evaluate(currentCycle);
    evaluateFanout();
    sequencer.evaluate(currentCycle);
    dispatcher.evaluate(currentCycle);
    advance();
    ++currentCycle;
}

void
AmePipeline::advance()
{
    scoreboard.advance();
    dispatchQueue.advance();
    peRequests.advance();
    completions.advance();
    systolic.advance();
    mlsu.advance();
    mrsu.advance();
    mptu.advance();
    mrf.advance();
    accRf.advance();
}

bool
AmePipeline::cancel(const InstQueueKey &key)
{
    const auto id = dispatcher.idForKey(key);
    if (!id)
        return false;
    sequencer.cancel(*id);
    systolic.cancel(*id);
    mlsu.cancel(*id);
    mrsu.cancel(*id);
    mptu.cancel(*id);
    operands.cancel(*id);
    accumulator.cancel(*id);
    mrf.cancel(*id);
    accRf.cancel(*id);
    completionTracker.cancel(*id);
    controls.erase(std::remove_if(controls.begin(), controls.end(),
        [id](const ControlToken &token) {
            return token.contextId == *id;
        }), controls.end());
    dispatcher.cancel(key);
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu SQUASH exec=%llu\n",
            currentCycle, *id, key.execSeqNum);
    return true;
}

bool
AmePipeline::active() const
{
    return !contexts.empty() || !scoreboard.empty() || dispatcher.active() ||
        sequencer.active() || completionTracker.active() ||
        dispatchQueue.active() || peRequests.active() || completions.active() ||
        operands.active() || accumulator.active() || systolic.active() ||
        mlsu.active() || mrsu.active() || mptu.active() || mrf.active() ||
        accRf.active() || !controls.empty() ||
        instQueue.hasPending(InstQueueEngine::AME);
}

} // namespace ame
} // namespace minor
} // namespace gem5
