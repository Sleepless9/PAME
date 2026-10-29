# Copyright (c) 2026
# All rights reserved.

from m5.objects.Device import DmaVirtDevice
from m5.params import *


class SPMVirtDma(DmaVirtDevice):
    type = "SPMVirtDma"
    cxx_header = "dev/spm_virt_dma.hh"
    cxx_class = "gem5::SPMVirtDma"

    spm_port = RequestPort("Direct timing port to the shared scratchpad")

    pio_addr = Param.Addr("Base address of the DMA register file")
    pio_size = Param.Addr(0x100, "Size of the DMA register file")
    pio_latency = Param.Latency("10ns", "PIO access latency")

    spm_range = Param.AddrRange("Physical address range of the scratchpad")
    spm_bank_count = Param.Unsigned(8, "Number of scratchpad banks")
    spm_bank_interleave = Param.MemorySize(
        "64B", "Bytes in a bank stripe before rotating to the next bank"
    )
    burst_bytes = Param.MemorySize(
        "64B", "Maximum system-side DMA burst and chunk size"
    )
    descriptor_queue_depth = Param.Unsigned(
        16, "Descriptor credits shared by queued, active, fetching and unretired completions"
    )
    max_outstanding_per_bank = Param.Unsigned(
        4, "Maximum chunks in flight for each scratchpad bank"
    )
    max_outstanding_system = Param.Unsigned(
        32, "End-to-end payload chunk credits; descriptor fetch has an independent credit"
    )
    max_outstanding_per_descriptor = Param.Unsigned(
        12, "Payload credits available to one active descriptor"
    )
    max_active_descriptors = Param.Unsigned(4, "Concurrent data descriptors")
    issue_width = Param.Unsigned(16, "Maximum data chunks issued per DMA clock")
