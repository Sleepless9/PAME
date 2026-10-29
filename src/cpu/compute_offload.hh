/* Copyright (c) 2026 */

#ifndef __CPU_COMPUTE_OFFLOAD_HH__
#define __CPU_COMPUTE_OFFLOAD_HH__

namespace gem5
{

namespace minor
{
class MinorInstQueue;
}

/** CPU-side architectural boundary shared by Minor and O3 compute hosts. */
class ComputeOffloadCPU
{
  public:
    virtual ~ComputeOffloadCPU() = default;
    virtual minor::MinorInstQueue &getInstQueue() = 0;
};

} // namespace gem5

#endif // __CPU_COMPUTE_OFFLOAD_HH__
