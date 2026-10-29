/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#ifndef __CPU_MINOR_COMPUTING_CLUSTER_AME_OPERAND_SUBSYSTEM_HH__
#define __CPU_MINOR_COMPUTING_CLUSTER_AME_OPERAND_SUBSYSTEM_HH__

#include <array>
#include <map>

#include "computing_cluster/AME/ame_context.hh"
#include "computing_cluster/AME/ame_scoreboard.hh"
#include "computing_cluster/AME/register_file.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

struct OperandSubsystemParams
{
    unsigned kBlock = 4;
    unsigned requestWidth = 2;
};

/** MOFU requesters plus finite A/B ping-pong buffers. */
class MatrixOperandSubsystem
{
  private:
    struct Buffer
    {
        bool valid = false;
        uint32_t block = 0;
        unsigned aExpected = 0;
        unsigned bExpected = 0;
        unsigned aIssued = 0;
        unsigned bIssued = 0;
        unsigned aReceived = 0;
        unsigned bReceived = 0;
    };

    struct Work
    {
        AmeContextId contextId = 0;
        AmeInsnId ameId = 0;
        AmeConfig config;
        std::vector<MRegion> aRegions;
        std::vector<MRegion> bRegions;
        bool hasA = false;
        bool hasB = false;
        uint32_t totalBlocks = 0;
        uint32_t nextBlock = 0;
        std::array<Buffer, 2> buffers;
    };

    AmeContextTable &contexts;
    AmeScoreboard &scoreboard;
    BankedRegisterFile &mrf;
    const OperandSubsystemParams params;
    std::map<AmeContextId, Work> works;
    size_t roundRobin = 0;

    void fillFreeBuffers(Work &work);
    bool issueOne(Work &work, Buffer &buffer, bool operand_a);
    std::vector<MRegion> formFragments(
        const std::vector<MRegion> &regions, const AmeConfig &config,
        bool transpose, unsigned element_bits) const;
    MRegion fragment(const Work &work, const Buffer &buffer,
                     bool operand_a, unsigned index) const;

  public:
    MatrixOperandSubsystem(AmeContextTable &contexts_,
                           AmeScoreboard &scoreboard_,
                           BankedRegisterFile &mrf_,
                           const OperandSubsystemParams &params_);

    void start(AmeContextId id, uint32_t total_blocks);
    bool blockReady(AmeContextId id, uint32_t block) const;
    void consumeBlock(AmeContextId id, uint32_t block);
    void acceptResponse(const BankResponse &response);
    void evaluate(uint64_t cycle);
    void cancel(AmeContextId id);
    bool active() const { return !works.empty(); }
    bool contains(AmeContextId id) const
    { return works.find(id) != works.end(); }
};

} // namespace ame
} // namespace minor
} // namespace gem5

#endif // __CPU_MINOR_COMPUTING_CLUSTER_AME_OPERAND_SUBSYSTEM_HH__
