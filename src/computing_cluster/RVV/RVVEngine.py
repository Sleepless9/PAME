# Copyright (c) 2026
# All rights reserved.

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *


class RVVEngine(ClockedObject):
    type = "RVVEngine"
    cxx_header = "computing_cluster/RVV/engine.hh"
    cxx_class = "gem5::minor::rvv::RVVEngine"

    cpu = Param.BaseCPU(Parent.any, "CPU owning the instruction queue")
    system = Param.System(Parent.any, "System containing this RVV engine")
    use_timing_memory = Param.Bool(
        False,
        "Issue VLSU line requests through memory_port instead of synthetic "
        "latency",
    )
    memory_port = RequestPort("VLSU timing request port")

    # Architectural snapshot and physical topology are independent.
    vlen_bits = Param.Unsigned(256, "Architectural vector register width")
    num_lanes = Param.Unsigned(4, "Number of physical vector lanes")
    lane_width_bits = Param.Unsigned(64, "Lane datapath and VRF word width")
    vrf_banks_per_lane = Param.Unsigned(8, "Single-port VRF banks per lane")
    vrf_arbitration_policy = Param.String(
        "two_level_priority",
        "VRF bank arbitration: two_level_priority, write_priority, "
        "round_robin, or oldest",
    )
    elements_per_lane_cycle = Param.Unsigned(
        8, "Maximum packed elements carried by one lane beat"
    )

    # Instruction, beat, lane, operand, result and completion capacities.
    dispatch_queue_depth = Param.Unsigned(4, "RVV instruction dispatch FIFO")
    max_inflight_insts = Param.Unsigned(8, "Instruction context entries")
    rvv_insn_slot_capacity = Param.Unsigned(
        8, "Finite in-flight RVV instruction slots"
    )
    pe_request_queue_depth = Param.Unsigned(
        2, "Global sequencer instruction-level PE request FIFO"
    )
    lane_queue_depth = Param.Unsigned(
        4, "Compatibility alias for lane scoreboard and input capacity"
    )
    operand_task_depth = Param.Unsigned(4, "Lane operand requester tasks")
    direct_fu_bypass = Param.Bool(
        False, "Experimental direct FU bypass (disabled by default)"
    )
    vrf_request_queue_depth = Param.Unsigned(2, "Requests buffered per VRF bank")
    operand_queue_depth = Param.Unsigned(2, "Input queue depth per lane FU")
    result_queue_depth = Param.Unsigned(4, "Result queue depth per lane")
    completion_queue_depth = Param.Unsigned(4, "Instruction completion FIFO")
    completion_width = Param.Unsigned(
        1, "Completions handed to Minor per engine cycle"
    )
    external_request_queue_depth = Param.Unsigned(
        16, "Per-unit VLSU/SLDU request queue depth"
    )
    external_result_queue_depth = Param.Unsigned(
        16, "Per-unit VLSU/SLDU result queue depth"
    )

    # P1 shared VLSU topology and response timing.
    vlsu_max_outstanding = Param.Unsigned(16, "Maximum timing memory transactions")
    vlsu_cache_line_bytes = Param.Unsigned(64, "VLSU coalescing line size")
    vlsu_reorder_window = Param.Unsigned(4, "Deterministic response reorder window")
    vlsu_response_width = Param.Unsigned(1, "Completed lane memory ops per cycle")

    # Shared slide and reduction unit connected to all lanes.
    sldu_max_inflight_words = Param.Unsigned(
        4, "SLDU lane words resident in the datapath"
    )
    sldu_result_width = Param.Unsigned(1, "SLDU lane results per cycle")
    sldu_slide_latency = Param.Cycles(2, "SLDU slide exchange latency")
    sldu_intra_lane_reduction_latency = Param.Cycles(
        2, "SLDU intra-lane reduction phase latency"
    )
    sldu_reduction_stage_latency = Param.Cycles(
        1, "SLDU latency per inter-lane reduction stage"
    )

    # Latency and initiation interval are separate microarchitectural knobs.
    alu_latency = Param.Cycles(1, "Vector ALU result latency")
    alu_initiation_interval = Param.Unsigned(1, "Vector ALU initiation interval")
    multiply_latency = Param.Cycles(3, "Vector multiply result latency")
    multiply_initiation_interval = Param.Unsigned(
        1, "Vector multiply initiation interval"
    )
    fpu_latency = Param.Cycles(4, "Vector floating-point result latency")
    fpu_initiation_interval = Param.Unsigned(1, "Vector FPU initiation interval")
    divide_latency = Param.Cycles(8, "Vector divide/sqrt result latency")
    divide_initiation_interval = Param.Unsigned(
        8, "Vector divide/sqrt initiation interval"
    )
    memory_latency = Param.Cycles(8, "VLSU timing response base latency")
    memory_initiation_interval = Param.Unsigned(1, "VLSU lane initiation interval")
    config_latency = Param.Cycles(1, "Vector configuration latency")
