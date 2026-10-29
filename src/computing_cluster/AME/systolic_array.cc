/*
 * Copyright (c) 2026
 * All rights reserved.
 */

#include "computing_cluster/AME/systolic_array.hh"

#include <algorithm>

#include "base/logging.hh"
#include "debug/AMEFlow.hh"
#include "debug/AMEPE.hh"

namespace gem5
{
namespace minor
{
namespace ame
{

namespace
{

bool
sameWave(const WaveToken &left, const WaveToken &right)
{
    return left.contextId == right.contextId &&
        left.ameId == right.ameId &&
        left.outputBlock == right.outputBlock &&
        left.kBlock == right.kBlock && left.kStep == right.kStep &&
        left.dataflow == right.dataflow;
}

} // anonymous namespace

MatrixTile::MatrixTile(
    unsigned id_, unsigned row_, unsigned col_, unsigned ingress_depth,
    unsigned edge_depth, unsigned pe_rows, unsigned pe_cols,
    unsigned pe_hop_latency, unsigned mac_latency, unsigned mac_ii,
    bool has_east, bool has_south, unsigned operation_depth)
    : id(id_), row(row_), col(col_), peRows(pe_rows), peCols(pe_cols),
      peHopLatency(pe_hop_latency), macLatency(mac_latency), macII(mac_ii),
      operationDepth(operation_depth ? operation_depth : ingress_depth),
      hasEast(has_east), hasSouth(has_south),
      westIngress(ingress_depth), northIngress(ingress_depth),
      eastOutput(edge_depth), southOutput(edge_depth),
      completions(ingress_depth + edge_depth), pes(pe_rows * pe_cols)
{
    fatal_if(peRows == 0 || peCols == 0 || peHopLatency == 0 ||
             macLatency == 0 || macII == 0,
             "Invalid AME matrix tile parameters");
}

void
MatrixTile::beginWeightPreload(
    AmeContextId context_id, uint32_t k_block, unsigned valid_rows,
    unsigned valid_cols, uint64_t cycle)
{
    for (unsigned pe_row = 0; pe_row < std::min(peRows, valid_rows);
         ++pe_row) {
        for (unsigned pe_col = 0; pe_col < std::min(peCols, valid_cols);
             ++pe_col) {
            auto &state = pe(pe_row, pe_col);
            panic_if(!state.macPipe.empty(),
                     "AME weight preload overlaps an active PE pipeline");
            state.weightPending = true;
            state.pendingWeightContext = context_id;
            state.pendingWeightKBlock = k_block;
            state.weightReadyCycle = cycle +
                (pe_row + pe_col + 1) * peHopLatency;
        }
    }
}

bool
MatrixTile::weightsReady(
    AmeContextId context_id, uint32_t k_block, unsigned valid_rows,
    unsigned valid_cols) const
{
    for (unsigned pe_row = 0; pe_row < std::min(peRows, valid_rows);
         ++pe_row) {
        for (unsigned pe_col = 0; pe_col < std::min(peCols, valid_cols);
             ++pe_col) {
            const auto &state = pe(pe_row, pe_col);
            if (!state.weightValid || state.weightContext != context_id ||
                state.weightKBlock != k_block) {
                return false;
            }
        }
    }
    return true;
}

void
MatrixTile::beginPsumSeed(
    AmeContextId context_id, unsigned valid_rows, unsigned valid_cols,
    uint64_t cycle)
{
    for (unsigned pe_row = 0; pe_row < std::min(peRows, valid_rows);
         ++pe_row) {
        for (unsigned pe_col = 0; pe_col < std::min(peCols, valid_cols);
             ++pe_col) {
            auto &state = pe(pe_row, pe_col);
            panic_if(!state.macPipe.empty(),
                     "AME psum seed overlaps an active PE pipeline");
            state.psumPending = true;
            state.pendingPsumContext = context_id;
            state.psumReadyCycle = cycle +
                (pe_row + pe_col + 1) * peHopLatency;
        }
    }
}

bool
MatrixTile::psumsReady(
    AmeContextId context_id, unsigned valid_rows, unsigned valid_cols) const
{
    for (unsigned pe_row = 0; pe_row < std::min(peRows, valid_rows);
         ++pe_row) {
        for (unsigned pe_col = 0; pe_col < std::min(peCols, valid_cols);
             ++pe_col) {
            const auto &state = pe(pe_row, pe_col);
            if (!state.psumValid || state.psumContext != context_id)
                return false;
        }
    }
    return true;
}

void
MatrixTile::updateResidentState(uint64_t cycle)
{
    for (auto &state : pes) {
        if (state.weightPending && state.weightReadyCycle <= cycle) {
            state.weightPending = false;
            state.weightValid = true;
            state.weightContext = state.pendingWeightContext;
            state.weightKBlock = state.pendingWeightKBlock;
        }
        if (state.psumPending && state.psumReadyCycle <= cycle) {
            state.psumPending = false;
            state.psumValid = true;
            state.psumContext = state.pendingPsumContext;
        }
    }
}

void
MatrixTile::startOperation(const WaveToken &wave, uint64_t cycle)
{
    TileOperation operation;
    operation.id = nextOperationId++;
    operation.wave = wave;
    operation.expectedPes = wave.validRows * wave.validCols;
    operation.eastReadyCycle = cycle + wave.validCols * peHopLatency;
    operation.southReadyCycle = wave.dataflow ==
        MatrixDataflow::PartialSumStationary ?
            cycle + wave.validRows * peHopLatency :
            cycle + (wave.validRows + wave.validCols - 2) * peHopLatency +
                macLatency;
    operation.eastSent = !wave.forwardEast;
    operation.southSent = !wave.forwardSouth;
    panic_if(operation.expectedPes == 0,
             "AME tile accepted a wave with no valid PEs");

    for (unsigned pe_row = 0; pe_row < wave.validRows; ++pe_row) {
        for (unsigned pe_col = 0; pe_col < wave.validCols; ++pe_col) {
            auto &state = pe(pe_row, pe_col);
            if (wave.dataflow == MatrixDataflow::WeightStationary) {
                panic_if(!state.weightValid ||
                         state.weightContext != wave.contextId ||
                         state.weightKBlock != wave.kBlock,
                         "AME WS MAC fired without matching resident weight");
            } else {
                panic_if(!state.psumValid ||
                         state.psumContext != wave.contextId,
                         "AME PSS MAC fired without matching resident psum");
            }
            const uint64_t issue_cycle = cycle +
                (pe_row + pe_col) * peHopLatency;
            state.macPipe.push_back(MacToken{
                issue_cycle + macLatency, operation.id, wave,
                issue_cycle == cycle});
            if (issue_cycle == cycle)
                ++peIssueCount;
            DPRINTF(AMEPE,
                    "cycle=%llu inst=%llu slot=%u PE_MAC_ISSUE tile=%u "
                    "pe=(%u,%u) block=%u kblock=%u kstep=%u issue=%llu\n",
                    cycle, wave.contextId, wave.ameId, id, pe_row, pe_col,
                    wave.outputBlock, wave.kBlock, wave.kStep, issue_cycle);
        }
    }
    operations.push_back(std::move(operation));
    nextIssueCycle = cycle + macII;
}

void
MatrixTile::updateMacPipelines(uint64_t cycle)
{
    for (unsigned pe_row = 0; pe_row < peRows; ++pe_row) {
        for (unsigned pe_col = 0; pe_col < peCols; ++pe_col) {
            auto &state = pe(pe_row, pe_col);
            for (auto &token : state.macPipe) {
                if (!token.issueCounted && token.readyCycle - macLatency <= cycle) {
                    ++peIssueCount;
                    token.issueCounted = true;
                }
            }
            while (!state.macPipe.empty() &&
                   state.macPipe.front().readyCycle <= cycle) {
                const auto token = state.macPipe.front();
                const auto operation = std::find_if(
                    operations.begin(), operations.end(),
                    [&token](const TileOperation &candidate) {
                        return candidate.id == token.operation;
                    });
                panic_if(operation == operations.end(),
                         "AME PE result has no owning tile operation");
                ++operation->completedPes;
                ++peResultCount;
                panic_if(operation->completedPes > operation->expectedPes,
                         "AME tile operation received duplicate PE result");
                if (token.wave.dataflow ==
                        MatrixDataflow::PartialSumStationary) {
                    state.psumValid = true;
                    state.psumContext = token.wave.contextId;
                }
                DPRINTF(AMEPE,
                        "cycle=%llu inst=%llu slot=%u PE_MAC_RESULT tile=%u "
                        "pe=(%u,%u) block=%u kblock=%u kstep=%u\n",
                        cycle, token.wave.contextId, token.wave.ameId, id,
                        pe_row, pe_col, token.wave.outputBlock,
                        token.wave.kBlock, token.wave.kStep);
                state.macPipe.pop_front();
            }
        }
    }
}

void
MatrixTile::evaluate(uint64_t cycle)
{
    updateResidentState(cycle);
    updateMacPipelines(cycle);

    if (!operations.empty()) {
        auto &operation = operations.front();
        auto token = operation.wave;
        token.tile = id;
        panic_if(token.forwardEast && !hasEast,
                     "AME wave forwards east outside the physical grid");
        panic_if(token.forwardSouth && !hasSouth,
                     "AME wave forwards south outside the physical grid");
        if (!operation.eastSent && cycle >= operation.eastReadyCycle &&
            eastOutput.canPushNext()) {
            eastOutput.pushNext(token);
            operation.eastSent = true;
        }
        if (!operation.southSent && cycle >= operation.southReadyCycle &&
            (operation.wave.dataflow ==
                 MatrixDataflow::PartialSumStationary ||
             operation.completedPes == operation.expectedPes) &&
            southOutput.canPushNext()) {
            southOutput.pushNext(token);
            operation.southSent = true;
        }
        if (operation.completedPes == operation.expectedPes &&
            operation.eastSent && operation.southSent &&
            completions.canPushNext()) {
            completions.pushNext(token);
            operations.pop_front();
        }
    }

    if (westIngress.empty() || northIngress.empty() ||
        cycle < nextIssueCycle || operations.size() >= operationDepth) {
        return;
    }
    const auto west = westIngress.front();
    const auto north = northIngress.front();
    panic_if(!sameWave(west, north),
             "AME tile north/west operands have mismatched tags");
    auto wave = west;
    wave.tile = id;
    wave.validRows = std::min(west.validRows, north.validRows);
    wave.validCols = std::min(west.validCols, north.validCols);
    startOperation(wave, cycle);
    westIngress.popNext();
    northIngress.popNext();
}

void
MatrixTile::advance()
{
    westIngress.advance();
    northIngress.advance();
    eastOutput.advance();
    southOutput.advance();
    completions.advance();
}

void
MatrixTile::clearResident(AmeContextId context_id)
{
    for (auto &state : pes) {
        if (state.weightContext == context_id ||
            state.pendingWeightContext == context_id) {
            state.weightValid = false;
            state.weightPending = false;
        }
        if (state.psumContext == context_id ||
            state.pendingPsumContext == context_id) {
            state.psumValid = false;
            state.psumPending = false;
        }
    }
}

void
MatrixTile::cancel(AmeContextId context_id)
{
    const auto matches = [context_id](const WaveToken &token) {
        return token.contextId == context_id;
    };
    westIngress.eraseIf(matches);
    northIngress.eraseIf(matches);
    eastOutput.eraseIf(matches);
    southOutput.eraseIf(matches);
    completions.eraseIf(matches);
    operations.erase(std::remove_if(operations.begin(), operations.end(),
        [context_id](const TileOperation &operation) {
            return operation.wave.contextId == context_id;
        }), operations.end());
    for (auto &state : pes) {
        state.macPipe.erase(std::remove_if(
            state.macPipe.begin(), state.macPipe.end(),
            [context_id](const MacToken &token) {
                return token.wave.contextId == context_id;
            }), state.macPipe.end());
    }
    clearResident(context_id);
}

bool
MatrixTile::transientFor(AmeContextId context_id) const
{
    const auto matching_channel = [context_id](const auto &channel) {
        for (size_t index = 0; index < channel.size(); ++index) {
            if (channel.at(index).contextId == context_id)
                return true;
        }
        return false;
    };
    if (matching_channel(westIngress) || matching_channel(northIngress) ||
        matching_channel(eastOutput) || matching_channel(southOutput) ||
        matching_channel(completions)) {
        return true;
    }
    if (std::any_of(operations.begin(), operations.end(),
            [context_id](const TileOperation &operation) {
                return operation.wave.contextId == context_id;
            })) {
        return true;
    }
    return std::any_of(pes.begin(), pes.end(),
        [context_id](const PeState &state) {
            return std::any_of(state.macPipe.begin(), state.macPipe.end(),
                [context_id](const MacToken &token) {
                    return token.wave.contextId == context_id;
                });
        });
}

bool
MatrixTile::active() const
{
    return westIngress.active() || northIngress.active() ||
        eastOutput.active() || southOutput.active() || completions.active() ||
        !operations.empty() ||
        std::any_of(pes.begin(), pes.end(), [](const PeState &state) {
            return state.weightPending || state.psumPending ||
                !state.macPipe.empty();
        });
}

SystolicBackend::SystolicBackend(
    AmeContextTable &contexts_, MatrixOperandSubsystem &operands_,
    AccumulatorSubsystem &accumulator_, const SystolicArrayParams &params_)
    : contexts(contexts_), operands(operands_), accumulator(accumulator_),
      params(params_), commands(params.commandQueueDepth),
      done(params.commandQueueDepth),
      horizontalBoundaries(params.tileRows *
          (params.tileCols > 0 ? params.tileCols - 1 : 0)),
      verticalBoundaries((params.tileRows > 0 ? params.tileRows - 1 : 0) *
          params.tileCols)
{
    fatal_if(params.peRows == 0 || params.peCols == 0 ||
             params.tileRows == 0 || params.tileCols == 0 ||
             params.kBlock == 0 || params.tileBoundaryLatency == 0 ||
             params.edgeFifoDepth == 0 || params.drainQueueDepth == 0,
             "Invalid AME systolic array parameters");
    const unsigned tile_count = params.tileRows * params.tileCols;
    tiles.reserve(tile_count);
    for (unsigned row = 0; row < params.tileRows; ++row) {
        for (unsigned col = 0; col < params.tileCols; ++col) {
            tiles.emplace_back(tileId(row, col), row, col,
                params.tileIngressDepth, params.edgeFifoDepth, params.peRows,
                params.peCols, params.peHopLatency, params.macLatency,
                params.macII, col + 1 < params.tileCols,
                row + 1 < params.tileRows,
                params.pssContinuous ? params.operationDepth : 0);
        }
    }
}

unsigned
SystolicBackend::validRows(const Work &state, unsigned tile_row) const
{
    const unsigned block_row = state.outputBlock / state.outputBlocksN;
    const unsigned logical_row = block_row * params.tileRows * params.peRows +
        tile_row * params.peRows;
    if (logical_row >= state.request.config.nAme)
        return 0;
    return std::min<unsigned>(params.peRows,
        state.request.config.nAme - logical_row);
}

unsigned
SystolicBackend::validCols(const Work &state, unsigned tile_col) const
{
    const unsigned block_col = state.outputBlock % state.outputBlocksN;
    const unsigned logical_col = block_col * params.tileCols * params.peCols +
        tile_col * params.peCols;
    if (logical_col >= state.request.config.nAme)
        return 0;
    return std::min<unsigned>(params.peCols,
        state.request.config.nAme - logical_col);
}

WaveToken
SystolicBackend::tokenForTile(const WaveToken &token, unsigned tile) const
{
    auto result = token;
    result.tile = tile;
    if (work) {
        const unsigned row = tile / params.tileCols;
        const unsigned col = tile % params.tileCols;
        result.validRows = validRows(*work, row);
        result.validCols = validCols(*work, col);
        result.forwardEast = col + 1 < work->activeTileCols;
        result.forwardSouth = row + 1 < work->activeTileRows;
    }
    return result;
}

void
SystolicBackend::configureOutputBlock(Work &state)
{
    state.activeTileRows = 0;
    state.activeTileCols = 0;
    for (unsigned row = 0; row < params.tileRows; ++row)
        state.activeTileRows += validRows(state, row) != 0;
    for (unsigned col = 0; col < params.tileCols; ++col)
        state.activeTileCols += validCols(state, col) != 0;
    panic_if(state.activeTileRows == 0 || state.activeTileCols == 0,
             "AME output block has no active physical tiles");
    state.injected = 0;
    state.completed = 0;
    state.expected = state.activeTileRows * state.activeTileCols *
        (params.pssContinuous && state.request.config.dataflow ==
             MatrixDataflow::PartialSumStationary ?
             state.request.config.kLogical : state.kSteps);
    state.drainsExpected = state.request.config.dataflow ==
        MatrixDataflow::WeightStationary ? state.activeTileCols :
        state.activeTileRows * state.activeTileCols;
    state.drainsIssued = 0;
    state.drainsCompleted = 0;
    state.weightsStarted = false;
    state.seedStarted = false;
    state.outputSeeded = false;
}

void
SystolicBackend::startWork(const AmePeRequest &request, uint64_t cycle)
{
    Work next;
    next.request = request;
    next.phase = MatrixPhase::FormOperands;
    const unsigned sa_rows = params.peRows * params.tileRows;
    const unsigned sa_cols = params.peCols * params.tileCols;
    next.outputBlocksM = std::max(1U,
        (request.config.nAme + sa_rows - 1) / sa_rows);
    next.outputBlocksN = std::max(1U,
        (request.config.nAme + sa_cols - 1) / sa_cols);
    next.outputBlocks = next.outputBlocksM * next.outputBlocksN;
    next.kBlocks = std::max(1U,
        static_cast<unsigned>((request.config.kLogical + params.kBlock - 1) /
                              params.kBlock));
    next.kSteps = std::min<unsigned>(params.kBlock, request.config.kLogical);
    next.phaseStartCycle = cycle;
    configureOutputBlock(next);
    work = next;
    operands.start(request.contextId, next.outputBlocks * next.kBlocks);
    accumulator.start(request.contextId);
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u TILE_CMD_ACCEPT outputs=%u "
            "kblocks=%u grid=%ux%u dataflow=%s\n",
            cycle, request.contextId, request.ameId, next.outputBlocks,
            next.kBlocks, params.tileRows, params.tileCols,
            request.config.dataflow == MatrixDataflow::WeightStationary ?
                "WS" : "PSS");
}

void
SystolicBackend::evaluateGrid(uint64_t cycle)
{
    for (auto &tile : tiles)
        tile.evaluate(cycle);

    for (auto &tile : tiles) {
        if (tile.completionReady()) {
            const auto token = tile.completion();
            if (work && token.contextId == work->request.contextId) {
                ++work->completed;
                ++tileCompletionCount;
                macIssueCount += token.validRows * token.validCols;
            }
            tile.popCompletionNext();
        }
    }

    for (unsigned row = 0; row < params.tileRows; ++row) {
        for (unsigned col = 0; col < params.tileCols; ++col) {
            auto &tile = tiles[tileId(row, col)];
            if (col + 1 < params.tileCols && tile.eastReady()) {
                auto &boundary = horizontalBoundaries[horizontalId(row, col)];
                if (boundary.size() < params.edgeFifoDepth) {
                    const auto token = tokenForTile(
                        tile.east(), tileId(row, col + 1));
                    boundary.push_back(BoundaryToken{
                        cycle + params.tileBoundaryLatency, token});
                    tile.popEastNext();
                    ++edgeTransferCount;
                    DPRINTF(AMEFlow,
                            "cycle=%llu inst=%llu slot=%u EDGE_FIFO_FIRE "
                            "east from=%u to=%u\n", cycle, token.contextId,
                            token.ameId, tileId(row, col),
                            tileId(row, col + 1));
                } else {
                    ++edgeStallCount;
                }
            }
            if (row + 1 < params.tileRows && tile.southReady()) {
                auto &boundary = verticalBoundaries[verticalId(row, col)];
                if (boundary.size() < params.edgeFifoDepth) {
                    const auto token = tokenForTile(
                        tile.south(), tileId(row + 1, col));
                    boundary.push_back(BoundaryToken{
                        cycle + params.tileBoundaryLatency, token});
                    tile.popSouthNext();
                    ++edgeTransferCount;
                    DPRINTF(AMEFlow,
                            "cycle=%llu inst=%llu slot=%u EDGE_FIFO_FIRE "
                            "south from=%u to=%u\n", cycle, token.contextId,
                            token.ameId, tileId(row, col),
                            tileId(row + 1, col));
                } else {
                    ++edgeStallCount;
                }
            }
        }
    }

    for (unsigned row = 0; row < params.tileRows; ++row) {
        for (unsigned col = 0; col + 1 < params.tileCols; ++col) {
            auto &boundary = horizontalBoundaries[horizontalId(row, col)];
            auto &destination = tiles[tileId(row, col + 1)];
            if (!boundary.empty() && boundary.front().readyCycle <= cycle &&
                destination.canAcceptWestNext()) {
                destination.acceptWestNext(boundary.front().token);
                boundary.pop_front();
            }
        }
    }
    for (unsigned row = 0; row + 1 < params.tileRows; ++row) {
        for (unsigned col = 0; col < params.tileCols; ++col) {
            auto &boundary = verticalBoundaries[verticalId(row, col)];
            auto &destination = tiles[tileId(row + 1, col)];
            if (!boundary.empty() && boundary.front().readyCycle <= cycle &&
                destination.canAcceptNorthNext()) {
                destination.acceptNorthNext(boundary.front().token);
                boundary.pop_front();
            }
        }
    }
}

bool
SystolicBackend::injectWavefront(Work &state, uint64_t cycle)
{
    if (state.injected >= state.kSteps)
        return false;
    for (unsigned row = 0; row < state.activeTileRows; ++row) {
        if (!tiles[tileId(row, 0)].canAcceptWestNext()) {
            ++injectionStallCount;
            return false;
        }
    }
    for (unsigned col = 0; col < state.activeTileCols; ++col) {
        if (!tiles[tileId(0, col)].canAcceptNorthNext()) {
            ++injectionStallCount;
            return false;
        }
    }

    WaveToken base{state.request.contextId, state.request.ameId,
        state.outputBlock, state.kBlock,
        static_cast<uint16_t>(state.injected), 0, 0, 0,
        state.request.config.dataflow, false, false};
    for (unsigned row = 0; row < state.activeTileRows; ++row) {
        const unsigned tile = tileId(row, 0);
        tiles[tile].acceptWestNext(tokenForTile(base, tile));
    }
    for (unsigned col = 0; col < state.activeTileCols; ++col) {
        const unsigned tile = tileId(0, col);
        tiles[tile].acceptNorthNext(tokenForTile(base, tile));
    }
    ++state.injected;
    DPRINTF(AMEFlow,
            "cycle=%llu inst=%llu slot=%u KSTEP_INJECT output=%u "
            "kblock=%u kstep=%u rows=%u cols=%u\n",
            cycle, state.request.contextId, state.request.ameId,
            state.outputBlock, state.kBlock, state.injected - 1,
            state.activeTileRows, state.activeTileCols);
    return true;
}

bool
SystolicBackend::gridIdleFor(AmeContextId id) const
{
    const auto boundary_active = [id](const auto &boundaries) {
        for (const auto &boundary : boundaries) {
            if (std::any_of(boundary.begin(), boundary.end(),
                    [id](const BoundaryToken &token) {
                        return token.token.contextId == id;
                    })) {
                return true;
            }
        }
        return false;
    };
    if (boundary_active(horizontalBoundaries) ||
        boundary_active(verticalBoundaries)) {
        return false;
    }
    return std::none_of(tiles.begin(), tiles.end(),
        [id](const MatrixTile &tile) { return tile.transientFor(id); });
}

void
SystolicBackend::beginTileDrain(Work &state)
{
    state.drainsIssued = 0;
    state.drainsCompleted = 0;
}

bool
SystolicBackend::evaluateTileDrain(Work &state, uint64_t cycle)
{
    if (!drainQueue.empty() && drainQueue.front().readyCycle <= cycle) {
        if (drainQueue.front().token.contextId == state.request.contextId)
            ++state.drainsCompleted;
        drainQueue.pop_front();
    }
    if (state.drainsIssued < state.drainsExpected &&
        drainQueue.size() < params.drainQueueDepth) {
        unsigned tile = 0;
        if (state.request.config.dataflow ==
                MatrixDataflow::WeightStationary) {
            tile = tileId(state.activeTileRows - 1, state.drainsIssued);
        } else {
            const unsigned row = state.drainsIssued / state.activeTileCols;
            const unsigned col = state.drainsIssued % state.activeTileCols;
            tile = tileId(row, col);
        }
        WaveToken token{state.request.contextId, state.request.ameId,
            state.outputBlock, state.kBlock, 0,
            static_cast<uint16_t>(tile),
            static_cast<uint16_t>(validRows(state, tile / params.tileCols)),
            static_cast<uint16_t>(validCols(state, tile % params.tileCols)),
            state.request.config.dataflow, false, false};
        drainQueue.push_back(BoundaryToken{
            cycle + params.peHopLatency, token});
        ++state.drainsIssued;
        DPRINTF(AMEFlow,
                "cycle=%llu inst=%llu slot=%u TILE_DRAIN_ENQUEUE tile=%u "
                "output=%u\n", cycle, token.contextId, token.ameId, tile,
                token.outputBlock);
    } else if (state.drainsIssued < state.drainsExpected) {
        ++drainStallCount;
    }
    return state.drainsCompleted == state.drainsExpected &&
        std::none_of(drainQueue.begin(), drainQueue.end(),
            [&state](const BoundaryToken &token) {
                return token.token.contextId == state.request.contextId;
            });
}

void
SystolicBackend::finishOutputBlock(Work &state, uint64_t cycle)
{
    for (auto &tile : tiles)
        tile.clearResident(state.request.contextId);
    if (state.outputBlock + 1 < state.outputBlocks) {
        ++state.outputBlock;
        state.kBlock = 0;
        state.kSteps = std::min<unsigned>(
            params.kBlock, state.request.config.kLogical);
        configureOutputBlock(state);
        state.phase = state.request.config.dataflow ==
            MatrixDataflow::WeightStationary ? MatrixPhase::LoadWeight :
                                               MatrixPhase::LoadAcc;
        state.phaseStartCycle = cycle;
        return;
    }
    if (!state.accumulatorDrainStarted) {
        accumulator.beginDrain(state.request.contextId);
        state.accumulatorDrainStarted = true;
    }
}

void
SystolicBackend::evaluate(uint64_t cycle)
{
    evaluateGrid(cycle);
    if (!work && !commands.empty()) {
        startWork(commands.front(), cycle);
        commands.popNext();
    }
    if (!work)
        return;

    auto &state = *work;
    const AmeContextId id = state.request.contextId;
    const auto operands_ready = [&](uint32_t block) {
        const bool ready = operands.blockReady(id, block);
        if (!ready && !state.supplyStalled) {
            state.supplyStalled = true;
            state.supplyStallStart = cycle;
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu SUPPLY_STALL_BEGIN block=%u\n",
                    cycle, id, block);
        } else if (ready && state.supplyStalled) {
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu SUPPLY_STALL_END block=%u cycles=%llu\n",
                    cycle, id, block, cycle - state.supplyStallStart);
            state.supplyStalled = false;
        }
        return ready;
    };
    switch (state.phase) {
      case MatrixPhase::FormOperands:
        state.phase = state.request.config.dataflow ==
            MatrixDataflow::WeightStationary ? MatrixPhase::LoadWeight :
                                               MatrixPhase::LoadAcc;
        state.phaseStartCycle = cycle;
        break;
      case MatrixPhase::LoadWeight: {
        const uint32_t operand_block =
            state.outputBlock * state.kBlocks + state.kBlock;
        if (!operands_ready(operand_block))
            break;
        if (!state.weightsStarted) {
            for (unsigned row = 0; row < state.activeTileRows; ++row) {
                for (unsigned col = 0; col < state.activeTileCols; ++col) {
                    tiles[tileId(row, col)].beginWeightPreload(
                        id, state.kBlock, validRows(state, row),
                        validCols(state, col), cycle);
                }
            }
            state.weightsStarted = true;
        }
        bool ready = true;
        for (unsigned row = 0; row < state.activeTileRows; ++row) {
            for (unsigned col = 0; col < state.activeTileCols; ++col) {
                ready = ready && tiles[tileId(row, col)].weightsReady(
                    id, state.kBlock, validRows(state, row),
                    validCols(state, col));
            }
        }
        if (ready) {
            state.phase = state.outputSeeded ? MatrixPhase::Wavefront :
                                               MatrixPhase::LoadAcc;
            state.phaseStartCycle = cycle;
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu slot=%u WEIGHT_PRELOAD block=%u\n",
                    cycle, id, state.request.ameId, state.kBlock);
        }
        break;
      }
      case MatrixPhase::LoadAcc:
        if (!state.accumulatorResident) {
            accumulator.evaluateSeed(id);
            if (!accumulator.seedReady(id))
                break;
            accumulator.markResident(id);
            state.accumulatorResident = true;
        }
        if (state.request.config.dataflow ==
                MatrixDataflow::PartialSumStationary && !state.seedStarted) {
            for (unsigned row = 0; row < state.activeTileRows; ++row) {
                for (unsigned col = 0; col < state.activeTileCols; ++col) {
                    tiles[tileId(row, col)].beginPsumSeed(
                        id, validRows(state, row), validCols(state, col),
                        cycle);
                }
            }
            state.seedStarted = true;
        }
        if (state.request.config.dataflow ==
                MatrixDataflow::PartialSumStationary) {
            for (unsigned row = 0; row < state.activeTileRows; ++row) {
                for (unsigned col = 0; col < state.activeTileCols; ++col) {
                    if (!tiles[tileId(row, col)].psumsReady(
                            id, validRows(state, row),
                            validCols(state, col))) {
                        return;
                    }
                }
            }
        }
        state.outputSeeded = true;
        state.phase = MatrixPhase::Wavefront;
        state.phaseStartCycle = cycle;
        DPRINTF(AMEFlow,
                "cycle=%llu inst=%llu slot=%u ACC_SEED_LOAD output=%u\n",
                cycle, id, state.request.ameId, state.outputBlock);
        break;
      case MatrixPhase::Wavefront: {
        if (params.pssContinuous && state.request.config.dataflow ==
                MatrixDataflow::PartialSumStationary) {
            if (state.kBlock < state.kBlocks) {
                const uint32_t operand_block =
                    state.outputBlock * state.kBlocks + state.kBlock;
                if (operands_ready(operand_block))
                    injectWavefront(state, cycle);
                if (state.injected == state.kSteps) {
                    // Ingress owns the operands after the final handshake.
                    // Tagged in-flight waves outlive this finite MOFU buffer.
                    operands.consumeBlock(id, operand_block);
                    ++state.kBlock;
                    state.injected = 0;
                    if (state.kBlock < state.kBlocks)
                        state.kSteps = std::min<unsigned>(params.kBlock,
                            state.request.config.kLogical -
                            state.kBlock * params.kBlock);
                }
            }
            // Preserve psums and completion counts across K blocks. Drain
            // only after the final wave leaves the grid.
            if (state.kBlock == state.kBlocks &&
                state.completed == state.expected && gridIdleFor(id)) {
                state.kBlock = state.kBlocks - 1;
                state.phase = MatrixPhase::Drain;
                beginTileDrain(state);
                state.phaseStartCycle = cycle;
            }
            break;
        }
        const uint32_t operand_block =
            state.outputBlock * state.kBlocks + state.kBlock;
        if (operands_ready(operand_block))
            injectWavefront(state, cycle);
        if (state.completed != state.expected || !gridIdleFor(id))
            break;
        operands.consumeBlock(id, operand_block);
        state.injected = 0;
        state.completed = 0;
        ++state.kBlock;
        if (state.kBlock < state.kBlocks) {
            state.kSteps = std::min<unsigned>(params.kBlock,
                state.request.config.kLogical - state.kBlock * params.kBlock);
            state.expected = state.activeTileRows * state.activeTileCols *
                state.kSteps;
            state.weightsStarted = false;
            state.phase = state.request.config.dataflow ==
                MatrixDataflow::WeightStationary ? MatrixPhase::LoadWeight :
                                                   MatrixPhase::NextKBlock;
        } else {
            state.kBlock = state.kBlocks - 1;
            state.phase = MatrixPhase::Drain;
            beginTileDrain(state);
        }
        state.phaseStartCycle = cycle;
        break;
      }
      case MatrixPhase::NextKBlock:
        state.phase = MatrixPhase::Wavefront;
        state.phaseStartCycle = cycle;
        break;
      case MatrixPhase::Drain:
        if (!evaluateTileDrain(state, cycle))
            break;
        if (!state.accumulatorDrainStarted) {
            finishOutputBlock(state, cycle);
            if (!state.accumulatorDrainStarted)
                break;
        }
        accumulator.evaluateDrain(id);
        if (accumulator.committed(id)) {
            state.phase = MatrixPhase::AccCommit;
            state.phaseStartCycle = cycle;
        }
        break;
      case MatrixPhase::AccCommit:
        if (done.canPushNext()) {
            done.pushNext(TargetDone{id, TargetSa});
            state.phase = MatrixPhase::Done;
            DPRINTF(AMEFlow,
                    "cycle=%llu inst=%llu slot=%u ACC_FINAL_GRANT\n",
                    cycle, id, state.request.ameId);
        }
        break;
      case MatrixPhase::Done:
        accumulator.release(id);
        work.reset();
        break;
      case MatrixPhase::Allocate:
      case MatrixPhase::Prefetch:
      case MatrixPhase::Killed:
        break;
    }
}

void
SystolicBackend::advance()
{
    commands.advance();
    done.advance();
    for (auto &tile : tiles)
        tile.advance();
}

void
SystolicBackend::cancel(AmeContextId id)
{
    commands.eraseIf([id](const AmePeRequest &request) {
        return request.contextId == id;
    });
    done.eraseIf([id](const TargetDone &completion) {
        return completion.contextId == id;
    });
    for (auto &tile : tiles)
        tile.cancel(id);
    const auto erase_boundaries = [id](auto &boundaries) {
        for (auto &boundary : boundaries) {
            boundary.erase(std::remove_if(boundary.begin(), boundary.end(),
                [id](const BoundaryToken &token) {
                    return token.token.contextId == id;
                }), boundary.end());
        }
    };
    erase_boundaries(horizontalBoundaries);
    erase_boundaries(verticalBoundaries);
    drainQueue.erase(std::remove_if(drainQueue.begin(), drainQueue.end(),
        [id](const BoundaryToken &token) {
            return token.token.contextId == id;
        }), drainQueue.end());
    if (work && work->request.contextId == id)
        work.reset();
    operands.cancel(id);
    accumulator.cancel(id);
}

bool
SystolicBackend::active() const
{
    const auto boundaries_active = [](const auto &boundaries) {
        return std::any_of(boundaries.begin(), boundaries.end(),
            [](const auto &boundary) { return !boundary.empty(); });
    };
    return work.has_value() || commands.active() || done.active() ||
        std::any_of(tiles.begin(), tiles.end(),
            [](const MatrixTile &tile) { return tile.active(); }) ||
        boundaries_active(horizontalBoundaries) ||
        boundaries_active(verticalBoundaries) || !drainQueue.empty();
}

} // namespace ame
} // namespace minor
} // namespace gem5
