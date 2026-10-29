/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/RVV/vector_fu.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/RVVEngine.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

VectorFuPipeline::VectorFuPipeline(
    VecFuType type_, const VectorFuParams &params_)
    : type(type_), params(params_), input(params.inputQueueDepth)
{
    fatal_if(params.latency == 0 || params.initiationInterval == 0 ||
             params.maxInflight == 0,
             "RVV FU latency, II and capacity must be non-zero");
}

void
VectorFuPipeline::acceptInputNext(VectorFuReq request)
{
    panic_if(request.fuType != type, "RVV request routed to wrong FU");
    input.pushNext(std::move(request));
}

void
VectorFuPipeline::evaluate(
    uint64_t cycle, LaneId lane_id, RvvChannel<VectorFuResp> &result_queue)
{
    auto ready = std::find_if(inflight.begin(), inflight.end(),
        [cycle](const Inflight &entry) { return entry.readyCycle <= cycle; });
    if (ready != inflight.end() && result_queue.canPushNext()) {
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u FU RESULT type=%u\n",
                cycle, ready->response.tag.instId,
                ready->response.tag.beatId, lane_id,
                static_cast<unsigned>(type));
        result_queue.pushNext(std::move(ready->response));
        inflight.erase(ready);
    }

    const bool latency_one_blocked = params.latency == 1 &&
        !result_queue.canPushNext();
    if (input.empty() || cycle < nextAcceptCycle || latency_one_blocked ||
        inflight.size() + pendingInflight.size() >= params.maxInflight) {
        return;
    }

    const VectorFuReq &request = input.front();
    VectorFuResp response;
    response.tag = request.tag;
    response.fuType = request.fuType;
    response.destReg = request.destReg;
    response.groupByteOffset = request.groupByteOffset;
    response.byteEnable = request.byteEnable;
    input.popNext();
    nextAcceptCycle = cycle + params.initiationInterval;

    DPRINTF(RVVEngine,
            "cycle=%llu inst=%llu beat=%u lane=%u FU ISSUE type=%u "
            "latency=%u ii=%u\n",
            cycle, request.tag.instId, request.tag.beatId, lane_id,
            static_cast<unsigned>(type), params.latency,
            params.initiationInterval);

    if (params.latency == 1) {
        DPRINTF(RVVEngine,
                "cycle=%llu inst=%llu beat=%u lane=%u FU RESULT type=%u\n",
                cycle, response.tag.instId, response.tag.beatId, lane_id,
                static_cast<unsigned>(type));
        result_queue.pushNext(std::move(response));
    } else {
        pendingInflight.push_back(
            Inflight{std::move(response), cycle + params.latency - 1});
    }
}

void
VectorFuPipeline::advance()
{
    input.advance();
    for (auto &entry : pendingInflight)
        inflight.push_back(std::move(entry));
    pendingInflight.clear();
}

void
VectorFuPipeline::cancel(RvvInstId id)
{
    input.eraseIf([id](const VectorFuReq &request) {
        return request.tag.instId == id;
    });
    inflight.erase(std::remove_if(inflight.begin(), inflight.end(),
        [id](const Inflight &entry) {
            return entry.response.tag.instId == id;
        }), inflight.end());
    pendingInflight.erase(
        std::remove_if(pendingInflight.begin(), pendingInflight.end(),
            [id](const Inflight &entry) {
                return entry.response.tag.instId == id;
            }), pendingInflight.end());
}

bool
VectorFuPipeline::active() const
{
    return input.active() || !inflight.empty() || !pendingInflight.empty();
}

} // namespace rvv
} // namespace minor
} // namespace gem5
