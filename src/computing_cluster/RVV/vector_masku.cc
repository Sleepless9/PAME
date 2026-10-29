/* Copyright (c) 2026 */
#include "computing_cluster/RVV/vector_masku.hh"

#include "base/logging.hh"

namespace gem5::minor::rvv
{

VectorMasku::VectorMasku(unsigned queue_depth, unsigned latency_)
    : requests(queue_depth), completions(queue_depth), latency(latency_)
{
    fatal_if(queue_depth == 0 || latency == 0,
             "RVV mask unit queue and latency must be non-zero");
}

void
VectorMasku::evaluate(uint64_t cycle)
{
    if (!activeRequest && !requests.empty()) {
        activeRequest = requests.front();
        requests.popNext();
        readyCycle = cycle + latency;
    }
    if (activeRequest && readyCycle <= cycle && completions.canPushNext()) {
        completions.pushNext(TargetDone{activeRequest->contextId,
                                        activeRequest->rvvSlotId, TargetMasku});
        activeRequest.reset();
    }
}

void
VectorMasku::advance()
{
    requests.advance();
    completions.advance();
}

void
VectorMasku::cancel(RvvInstId id)
{
    requests.eraseIf([id](const PeRequest &request) {
        return request.contextId == id;
    });
    completions.eraseIf([id](const TargetDone &done) {
        return done.contextId == id;
    });
    if (activeRequest && activeRequest->contextId == id)
        activeRequest.reset();
}

} // namespace gem5::minor::rvv
