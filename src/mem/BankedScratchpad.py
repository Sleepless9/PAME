# Copyright (c) 2026
# All rights reserved.

from m5.objects.AbstractMemory import AbstractMemory
from m5.params import Param, VectorResponsePort


class BankedScratchpad(AbstractMemory):
    type = "BankedScratchpad"
    cxx_header = "mem/banked_scratchpad.hh"
    cxx_class = "gem5::memory::BankedScratchpad"

    port = VectorResponsePort("Scratchpad request and response ports")
    bank_count = Param.Unsigned(8, "Number of interleaved banks")
    bank_interleave = Param.MemorySize("64B", "Bytes per bank stripe")
    bank_queue_depth = Param.Unsigned(8, "Outstanding requests per bank")
    bank_issue_latency = Param.Cycles(1, "Bank request initiation interval")
    read_latency = Param.Cycles(1, "Read service latency")
    write_latency = Param.Cycles(1, "Write service latency")
    response_queue_depth = Param.Unsigned(16, "Reserved responses per port")

    def controller(self):
        return self
