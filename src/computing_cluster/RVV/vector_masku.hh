/* Copyright (c) 2026 */
#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_MASKU_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_VECTOR_MASKU_HH__

#include <optional>
#include <utility>

#include "computing_cluster/RVV/rvv_channel.hh"
#include "computing_cluster/RVV/rvv_types.hh"

namespace gem5::minor::rvv
{

/** Finite mask/scalar/gather control target of the global sequencer. */
class VectorMasku
{
  private:
    RvvChannel<PeRequest> requests;
    RvvChannel<TargetDone> completions;
    std::optional<PeRequest> activeRequest;
    uint64_t readyCycle = 0;
    const unsigned latency;

  public:
    VectorMasku(unsigned queue_depth = 4, unsigned latency_ = 1);
    bool canAcceptNext() const { return requests.canPushNext(); }
    void acceptNext(PeRequest request) { requests.pushNext(std::move(request)); }
    RvvChannel<TargetDone> &doneChannel() { return completions; }
    void evaluate(uint64_t cycle);
    void advance();
    void cancel(RvvInstId id);
    bool activeState() const { return requests.active() || activeRequest.has_value() ||
        completions.active(); }
};

} // namespace gem5::minor::rvv

#endif
