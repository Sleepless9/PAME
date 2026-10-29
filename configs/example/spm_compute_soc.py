# Copyright (c) 2026
# All rights reserved.

"""Cycle-accurate RISC-V SoC with shared banked SPM and SPMVirtDma.

The MinorCPU or O3CPU command queue is the architectural dispatch boundary for
the RVV and AME engines. Both engines and SPMVirtDma use the same banked SPM
timing model, while the DMA's system-side port is attached to SystemXBar.
"""

import argparse
import os
import sys

import m5
from m5.objects import (
    AddrRange,
    BadAddr,
    BankedScratchpad,
    Bridge,
    DDR3_1600_8x8,
    HiFive,
    IOXBar,
    MemCtrl,
    PMAChecker,
    RiscvBareMetal,
    RiscvMinorCPU,
    RiscvO3CPU,
    RiscvRTC,
    RiscvSemihosting,
    RiscvSystem,
    Root,
    SPMVirtDma,
    SrcClockDomain,
    SystemXBar,
    VoltageDomain,
)
from m5.util.convert import toMemorySize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a RISC-V bare-metal binary on the shared compute SoC"
    )
    parser.add_argument("binary", help="RISC-V bare-metal ELF image")
    parser.add_argument("--cpu-type", choices=("minor", "o3"),
                        default="minor")
    parser.add_argument("--clock", default="1GHz")
    parser.add_argument("--memory-size", default="128MiB")
    parser.add_argument("--spm-size", default="1MiB")
    parser.add_argument("--spm-banks", type=int, default=8)
    parser.add_argument("--spm-interleave", default="64B")
    parser.add_argument("--dma-addr", type=lambda value: int(value, 0),
                        default=0x40010000)
    parser.add_argument("--spm-addr", type=lambda value: int(value, 0),
                        default=0x90000000)
    parser.add_argument("--dataflow", choices=("pss", "ws"), default="pss")
    parser.add_argument("--pe-rows", type=int, default=4)
    parser.add_argument("--pe-cols", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    binary = os.path.abspath(args.binary)
    if not os.path.isfile(binary):
        raise FileNotFoundError(binary)

    memory_start = 0x7FFFF000
    reset_vector_size = 0x1000
    memory_range = AddrRange(
        start=memory_start,
        size=toMemorySize(args.memory_size) + reset_vector_size,
    )
    spm_range = AddrRange(start=args.spm_addr, size=toMemorySize(args.spm_size))

    system = RiscvSystem()
    system.clk_domain = SrcClockDomain(
        clock=args.clock, voltage_domain=VoltageDomain()
    )
    system.mem_mode = "timing"
    system.mem_ranges = [memory_range]
    system.workload = RiscvBareMetal(
        bootloader=binary,
        semihosting=RiscvSemihosting(cmd_line=binary),
    )

    system.voltage_domain = VoltageDomain(voltage="1V")
    system.cpu_clk_domain = SrcClockDomain(
        clock=args.clock, voltage_domain=VoltageDomain()
    )
    cpu_class = RiscvMinorCPU if args.cpu_type == "minor" else RiscvO3CPU
    system.cpu = cpu_class(clk_domain=system.cpu_clk_domain, cpu_id=0)
    if args.cpu_type == "minor":
        system.cpu.executeInstQueueSize = 32
        system.cpu.executeOffloadLocalMemoryBase = args.spm_addr
        system.cpu.executeOffloadLocalMemorySize = args.spm_size
    else:
        system.cpu.computeInstQueueSize = 192
        system.cpu.computeOffloadLocalMemoryBase = args.spm_addr
        system.cpu.computeOffloadLocalMemorySize = args.spm_size
    system.cpu.addRvvEngine(
        use_timing_memory=True,
    )
    system.cpu.addAmeEngine(
        dataflow=args.dataflow,
        pe_rows=args.pe_rows,
        pe_cols=args.pe_cols,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    )

    system.spm = BankedScratchpad(
        range=spm_range,
        bank_count=args.spm_banks,
        bank_interleave=args.spm_interleave,
        bank_queue_depth=8,
        bank_issue_latency=1,
        read_latency=2,
        write_latency=1,
        response_queue_depth=32,
        conf_table_reported=False,
        kvm_map=False,
    )
    system.spm_dma = SPMVirtDma(
        pio_addr=args.dma_addr,
        spm_range=spm_range,
        spm_bank_count=args.spm_banks,
        spm_bank_interleave=args.spm_interleave,
        burst_bytes="64B",
        descriptor_queue_depth=8,
        max_outstanding_per_bank=4,
        max_outstanding_system=16,
    )

    system.iobus = IOXBar()
    system.membus = SystemXBar()
    system.membus.badaddr_responder = BadAddr(warn_access="warn")
    system.membus.default = system.membus.badaddr_responder.pio
    system.cpu.icache_port = system.membus.cpu_side_ports
    system.cpu.dcache_port = system.membus.cpu_side_ports
    # Accelerator LSUs issue architectural addresses.  Route them through
    # the XBar so DRAM and the mapped SPM window are both reachable.
    system.cpu.rvv_engine.memory_port = system.membus.cpu_side_ports
    system.cpu.ame_engine.memory_port = system.membus.cpu_side_ports
    system.spm_dma.spm_port = system.spm.port
    system.spm.port = system.membus.mem_side_ports
    system.spm_dma.dma = system.membus.cpu_side_ports
    system.spm_dma.pio = system.membus.mem_side_ports
    system.cpu.createInterruptController()
    system.cpu.createThreads()

    system.platform = HiFive()
    system.platform.rtc = RiscvRTC(frequency="100MHz")
    system.platform.clint.int_pin = system.platform.rtc.int_pin
    system.platform.pci_host.internal_connect()
    system.platform.pci_host.connect_upper_bus(system.iobus, True)
    system.platform.attachOnChipIO(system.membus)
    system.platform.attachOffChipIO(system.iobus)
    system.platform.attachPlic()
    system.platform.setNumCores(1)

    system.bridge = Bridge(delay="50ns")
    system.bridge.mem_side_port = system.iobus.cpu_side_ports
    system.bridge.cpu_side_port = system.membus.mem_side_ports
    system.bridge.ranges = system.platform._off_chip_ranges()

    system.iobridge = Bridge(delay="50ns", ranges=system.mem_ranges)
    system.iobridge.cpu_side_port = system.iobus.mem_side_ports
    system.iobridge.mem_side_port = system.membus.cpu_side_ports

    uncacheable = [
        *system.platform._on_chip_ranges(),
        *system.platform._off_chip_ranges(),
        spm_range,
    ]
    system.cpu.mmu.pma_checker = PMAChecker(uncacheable=uncacheable)

    system.mem_ctrl = MemCtrl()
    system.mem_ctrl.dram = DDR3_1600_8x8()
    system.mem_ctrl.dram.range = memory_range
    system.mem_ctrl.port = system.membus.mem_side_ports
    system.system_port = system.membus.cpu_side_ports

    root = Root(full_system=True, system=system)
    m5.instantiate()
    event = m5.simulate()
    print(
        f"Exiting @ tick {m5.curTick()} because {event.getCause()} "
        f"(code {event.getCode()})"
    )
    sys.exit(event.getCode())


if __name__ == "__m5_main__":
    main()
