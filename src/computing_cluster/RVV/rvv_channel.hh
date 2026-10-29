/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CHANNEL_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CHANNEL_HH__

#include <algorithm>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace rvv
{

/** Finite FIFO whose scheduled updates become visible only after advance(). */
template <class T>
class RvvChannel
{
  private:
    const size_t depth;
    std::deque<T> current;
    std::vector<T> pendingPushes;
    size_t pendingPops = 0;

  public:
    explicit RvvChannel(size_t depth_) : depth(depth_)
    {
        fatal_if(depth == 0, "RVV channel depth must be non-zero");
    }

    bool empty() const { return current.empty(); }
    bool active() const { return !current.empty() || !pendingPushes.empty(); }
    size_t size() const { return current.size(); }
    size_t capacity() const { return depth; }

    const T &front() const
    {
        panic_if(current.empty(), "Reading an empty RVV channel");
        return current.front();
    }

    const T &at(size_t index) const
    {
        panic_if(index >= current.size(), "RVV channel index out of range");
        return current[index];
    }

    bool canPushNext(size_t count = 1) const
    {
        return current.size() - pendingPops + pendingPushes.size() + count <=
            depth;
    }

    void pushNext(const T &value)
    {
        panic_if(!canPushNext(), "RVV channel overflow");
        pendingPushes.push_back(value);
    }

    void pushNext(T &&value)
    {
        panic_if(!canPushNext(), "RVV channel overflow");
        pendingPushes.push_back(std::move(value));
    }

    void popNext(size_t count = 1)
    {
        panic_if(pendingPops + count > current.size(),
                 "RVV channel underflow");
        pendingPops += count;
    }

    template <class Predicate>
    void eraseIf(Predicate predicate)
    {
        panic_if(pendingPops != 0,
                 "Cannot cancel an RVV channel during evaluation");
        current.erase(std::remove_if(current.begin(), current.end(), predicate),
                      current.end());
        pendingPushes.erase(
            std::remove_if(pendingPushes.begin(), pendingPushes.end(), predicate),
            pendingPushes.end());
    }

    void advance()
    {
        while (pendingPops != 0) {
            current.pop_front();
            --pendingPops;
        }
        for (auto &value : pendingPushes)
            current.push_back(std::move(value));
        pendingPushes.clear();
        panic_if(current.size() > depth, "RVV channel capacity invariant");
    }
};

} // namespace rvv
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_RVV_RVV_CHANNEL_HH__
