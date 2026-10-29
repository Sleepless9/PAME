# Copyright (c) 2026
# All rights reserved.

from m5.objects.ClockedObject import ClockedObject
from m5.params import *
from m5.proxy import *
from m5.objects.SPMVirtDma import SPMVirtDma


class AMEEngine(ClockedObject):
    type = "AMEEngine"
    cxx_header = "computing_cluster/AME/engine.hh"
    cxx_class = "gem5::minor::ame::AMEEngine"

    cpu = Param.BaseCPU(Parent.any, "CPU owning the AME instruction queue")
    system = Param.System(Parent.any, "System containing this AME engine")
    memory_port = RequestPort("M-LSU timing request port")
    dma_event_source = Param.SPMVirtDma(
        NULL, "Optional DMA event source for queue-to-queue dependencies"
    )

    dispatch_queue_depth = Param.Unsigned(16, "AME asynchronous command FIFO depth")
    max_inflight_insts = Param.Unsigned(32, "AME context table entries")
    instruction_slots = Param.Unsigned(16, "Finite AME instruction IDs")
    pe_request_queue_depth = Param.Unsigned(8, "Instruction broadcast depth")
    completion_queue_depth = Param.Unsigned(16, "AME completion FIFO depth")
    completion_width = Param.Unsigned(4, "Host completions per AME cycle")
    control_latency = Param.Cycles(1, "Configuration/control target latency")

    dataflow = Param.String("pss", "Systolic dataflow: pss or ws")
    k_block = Param.Unsigned(4, "Prefetched K elements per block")
    operand_request_width = Param.Unsigned(2, "MOFU MRF requests per cycle")

    mrf_banks = Param.Unsigned(8, "Single-port MRF banks")
    mrf_word_bits = Param.Unsigned(64, "MRF bank word width")
    mrf_register_bytes = Param.Unsigned(
        256, "Maximum raw bytes reserved for one M register"
    )
    mrf_bank_skew_group_regs = Param.Unsigned(
        8, "Register-group size used by cyclic MRF bank skew"
    )
    mrf_bank_skew_words = Param.Unsigned(
        1, "Cyclic bank offset added per MRF register group"
    )
    mrf_request_queue_depth = Param.Unsigned(4, "Requests buffered per MRF bank")
    mrf_response_queue_depth = Param.Unsigned(16, "MRF response FIFO depth")
    mrf_read_latency = Param.Cycles(1, "MRF grant-to-response latency")
    mrf_write_visibility = Param.Cycles(1, "MRF write visibility latency")

    acc_banks = Param.Unsigned(4, "Single-port accumulator RF banks")
    acc_word_bits = Param.Unsigned(128, "Accumulator bank word width")
    acc_register_bytes = Param.Unsigned(256, "Raw bytes in one Acc register")
    acc_request_queue_depth = Param.Unsigned(4, "Requests buffered per Acc bank")
    acc_response_queue_depth = Param.Unsigned(16, "Accumulator response FIFO depth")
    acc_read_latency = Param.Cycles(1, "Accumulator read latency")
    acc_write_visibility = Param.Cycles(1, "Accumulator write visibility latency")

    pe_rows = Param.Unsigned(4, "PE rows in one tile")
    pe_cols = Param.Unsigned(4, "PE columns in one tile")
    tile_rows = Param.Unsigned(1, "Physical tile rows")
    tile_cols = Param.Unsigned(1, "Physical tile columns")
    tile_command_queue_depth = Param.Unsigned(2, "SA command FIFO depth")
    tile_ingress_depth = Param.Unsigned(2, "Per-tile ingress FIFO depth")
    pss_continuous = Param.Bool(False, "Inject PSS K blocks without intermediate grid drain")
    tile_operation_depth = Param.Unsigned(2, "Finite in-flight tile waves for continuous PSS")
    edge_fifo_depth = Param.Unsigned(2, "Registered tile-edge FIFO depth")
    drain_queue_depth = Param.Unsigned(2, "Tile drain queue depth")
    pe_hop_latency = Param.Cycles(1, "Latency of one PE hop")
    tile_boundary_latency = Param.Cycles(1, "Latency of a tile boundary")
    mac_latency = Param.Cycles(3, "PE MAC pipeline latency")
    mac_initiation_interval = Param.Unsigned(1, "PE MAC initiation interval")

    mlsu_descriptor_queue_depth = Param.Unsigned(4, "M-LSU descriptor queue")
    mlsu_packet_bytes = Param.Unsigned(64, "M-LSU packet payload bytes")
    mlsu_max_outstanding = Param.Unsigned(16, "M-LSU outstanding transactions")
    mlsu_response_latency = Param.Cycles(
        8, "Synthetic M-LSU latency used only by unit-level pipeline tests"
    )
    mlsu_reorder_window = Param.Unsigned(4, "M-LSU response reorder window")
    mlsu_response_width = Param.Unsigned(1, "M-LSU responses per cycle")

    auxiliary_queue_depth = Param.Unsigned(2, "MRSU/MPTU request queue depth")
    mrsu_local_latency = Param.Cycles(2, "MRSU local reduction/scan latency")
    mrsu_cross_tile_latency = Param.Cycles(1, "MRSU merge stage latency")
    mptu_latency = Param.Cycles(2, "MPTU permutation network latency")
    auxiliary_result_width = Param.Unsigned(1, "Auxiliary results per cycle")
