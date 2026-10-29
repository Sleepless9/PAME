/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CHANNEL_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CHANNEL_HH__

#include <algorithm>
#include <cstddef>
#include <deque>
#include <utility>

#include "base/logging.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

/** Finite current/next FIFO. Writes and pops become visible at advance(). */
template <class T>
class AmeChannel
{
  private:
    const size_t capacity;
    std::deque<T> current;
    std::deque<T> pendingPushes;
    size_t pendingPops = 0;

  public:
    explicit AmeChannel(size_t capacity_) : capacity(capacity_)
    {
        fatal_if(capacity == 0, "AME channel capacity must be non-zero");
    }

    bool empty() const { return current.empty(); }
    size_t size() const { return current.size(); }
    bool active() const
    {
        return !current.empty() || !pendingPushes.empty() || pendingPops != 0;
    }
    bool canPushNext(size_t count = 1) const
    {
        return current.size() + pendingPushes.size() + count <=
            capacity + pendingPops;
    }
    const T &front() const
    {
        panic_if(current.empty(), "Reading an empty AME channel");
        return current.front();
    }
    const T &at(size_t index) const { return current.at(index); }
    void eraseAt(size_t index)
    {
        panic_if(index >= current.size(), "Erasing outside an AME channel");
        current.erase(current.begin() + index);
        pendingPops = std::min(pendingPops, current.size());
    }
    void pushNext(const T &value)
    {
        panic_if(!canPushNext(), "Overflowing an AME channel");
        pendingPushes.push_back(value);
    }
    void pushNext(T &&value)
    {
        panic_if(!canPushNext(), "Overflowing an AME channel");
        pendingPushes.push_back(std::move(value));
    }
    void popNext()
    {
        panic_if(pendingPops >= current.size(),
                 "Popping an empty AME channel");
        ++pendingPops;
    }
    template <class Predicate>
    void eraseIf(Predicate predicate)
    {
        current.erase(std::remove_if(current.begin(), current.end(), predicate),
                      current.end());
        pendingPushes.erase(std::remove_if(pendingPushes.begin(),
                                           pendingPushes.end(), predicate),
                            pendingPushes.end());
        pendingPops = std::min(pendingPops, current.size());
    }
    void advance()
    {
        while (pendingPops != 0) {
            current.pop_front();
            --pendingPops;
        }
        while (!pendingPushes.empty()) {
            current.push_back(std::move(pendingPushes.front()));
            pendingPushes.pop_front();
        }
        panic_if(current.size() > capacity, "AME channel overflow at advance");
    }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_AME_CHANNEL_HH__
