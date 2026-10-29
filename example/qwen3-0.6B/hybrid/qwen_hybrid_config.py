"""Qwen3 hybrid SoC: MinorCPU + RVV + AME/Ztt + shared SPM + DMA."""

import argparse
import json
import os
import sys

import m5
from m5.objects import (
    AddrRange,
    BadAddr,
    BankedScratchpad,
    Bridge,
    Cache,
    DDR3_1600_8x8,
    HiFive,
    IOXBar,
    MemCtrl,
    PMAChecker,
    RiscvBareMetal,
    RiscvMinorCPU,
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


PHASES = {
    1: "tokenizer_load",
    2: "prompt_tokenize",
    3: "model_load",
    4: "state_build",
    5: "prefill",
    6: "generate",
    7: "output",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd", required=True, help="hybrid bare-metal ELF")
    parser.add_argument("--mem-size", default="1GiB")
    parser.add_argument("--clock", default="1GHz")
    parser.add_argument("--vlen", type=int, default=512)
    parser.add_argument("--num-lanes", type=int, default=4)
    parser.add_argument("--lane-width-bits", type=int, default=64)
    parser.add_argument("--vrf-banks-per-lane", type=int, default=8)
    parser.add_argument("--inst-queue-size", type=int, default=32)
    parser.add_argument("--spm-size", default="1MiB")
    parser.add_argument("--spm-banks", type=int, default=8)
    parser.add_argument("--spm-interleave", default="64B")
    parser.add_argument(
        "--spm-addr", type=lambda value: int(value, 0), default=0x70000000
    )
    parser.add_argument(
        "--dma-addr", type=lambda value: int(value, 0), default=0x40010000
    )
    parser.add_argument(
        "--detail-phase", default="prefill", choices=[*PHASES.values(), "none"]
    )
    parser.add_argument("--detail-cycles", type=int, default=250000)
    parser.add_argument("--dataflow", choices=("pss", "ws"), default="pss")
    parser.add_argument("--pe-rows", type=int, default=4)
    parser.add_argument("--pe-cols", type=int, default=4)
    args = parser.parse_args()

    if args.vlen <= 0 or args.vlen % args.lane_width_bits:
        parser.error("--vlen must be a positive multiple of --lane-width-bits")
    if args.spm_banks <= 0:
        parser.error("--spm-banks must be positive")
    if toMemorySize(args.spm_size) < 128 * 1024:
        parser.error("--spm-size must be at least 128KiB for hybrid workspaces")
    if not os.path.isfile(args.cmd):
        parser.error(f"ELF does not exist: {args.cmd}")
    return args


def clock_period_ticks(clock):
    units = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1.0}
    for suffix, scale in units.items():
        if clock.endswith(suffix):
            return max(1, round(1e12 / (float(clock[: -len(suffix)]) * scale)))
    raise ValueError(f"unsupported --clock value: {clock}")


args = parse_args()
memory_start = 0x80000000
memory_range = AddrRange(start=memory_start, size=args.mem_size)
spm_range = AddrRange(start=args.spm_addr, size=toMemorySize(args.spm_size))
dma_range = AddrRange(start=args.dma_addr, size=0x100)

system = RiscvSystem()
system.mem_mode = "timing"
system.mem_ranges = [memory_range]
system.exit_on_work_items = True
system.workload = RiscvBareMetal(
    bootloader=os.path.abspath(args.cmd),
    semihosting=RiscvSemihosting(
        cmd_line=os.path.abspath(args.cmd), files_root_dir="/"
    ),
)
system.cache_line_size = 64
system.voltage_domain = VoltageDomain(voltage="1V")
system.clk_domain = SrcClockDomain(
    clock=args.clock, voltage_domain=system.voltage_domain
)
system.cpu_clk_domain = SrcClockDomain(
    clock=args.clock, voltage_domain=VoltageDomain()
)

system.iobus = IOXBar()
system.membus = SystemXBar()
system.membus.badaddr_responder = BadAddr(warn_access="warn")
system.membus.default = system.membus.badaddr_responder.pio
system.system_port = system.membus.cpu_side_ports

system.cpu = RiscvMinorCPU(clk_domain=system.cpu_clk_domain, cpu_id=0)
system.cpu.executeInstQueueSize = args.inst_queue_size
system.cpu.addRvvEngine(
    use_timing_memory=True,
    vlen_bits=args.vlen,
    num_lanes=args.num_lanes,
    lane_width_bits=args.lane_width_bits,
    vrf_banks_per_lane=args.vrf_banks_per_lane,
)
system.cpu.addAmeEngine(
    dataflow=args.dataflow,
    pe_rows=args.pe_rows,
    pe_cols=args.pe_cols,
)

# Keep only the instruction cache. CPU data, both engines, and DMA all observe
# the same XBar-visible state, avoiding a non-coherent write-back D-cache path.
system.icache = Cache(
    size="32KiB",
    assoc=2,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.cpu.icache_port = system.icache.cpu_side
system.icache.mem_side = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.rvv_engine.memory_port = system.membus.cpu_side_ports
system.cpu.ame_engine.memory_port = system.membus.cpu_side_ports

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

# SPM has independent response ports backed by one banked byte store. The XBar
# port serves CPU/RVV/AME addresses; the private port serves DMA timing traffic.
system.spm.port = system.membus.mem_side_ports
system.spm_dma.spm_port = system.spm.port
system.spm_dma.dma = system.membus.cpu_side_ports
system.spm_dma.pio = system.membus.mem_side_ports

system.cpu.createInterruptController()
system.cpu.createThreads()
system.cpu.isa[0].vlen = args.vlen
system.cpu.isa[0].elen = 64

system.platform = HiFive()
system.platform.terminal.outfile = "file"
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

system.cpu.mmu.pma_checker = PMAChecker(
    uncacheable=[
        *system.platform._on_chip_ranges(),
        *system.platform._off_chip_ranges(),
        spm_range,
        dma_range,
    ]
)
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8(range=memory_range)
system.mem_ctrl.port = system.membus.mem_side_ports

root = Root(full_system=True, system=system)
m5.instantiate()

outdir = m5.options.outdir
events_path = os.path.join(outdir, "phase_events.json")
events = []
active_phase = None
detail_active = False
detail_ticks_left = 0
period_ticks = clock_period_ticks(args.clock)


def set_detail(enabled):
    method = "enable" if enabled else "disable"
    for flag in ("RVVEngine", "AMEEngine"):
        getattr(m5.debug.flags[flag], method)()


def save_events(final_event=None):
    report = {
        "backend": "hybrid",
        "clock": args.clock,
        "clock_period_ticks": period_ticks,
        "vlen_bits": args.vlen,
        "num_lanes": args.num_lanes,
        "lane_width_bits": args.lane_width_bits,
        "inst_queue_size": args.inst_queue_size,
        "spm": {
            "address": hex(args.spm_addr),
            "size": args.spm_size,
            "banks": args.spm_banks,
            "interleave": args.spm_interleave,
        },
        "dma": {"pio_address": hex(args.dma_addr), "burst_bytes": 64},
        "topology": "MinorCPU+RVV+AME+SPMVirtDma->SystemXBar->DDR; DMA->SPM",
        "detail_phase": args.detail_phase,
        "detail_cycles": args.detail_cycles,
        "phases": events,
    }
    if final_event is not None:
        report["final_event"] = final_event
    with open(events_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


while True:
    before_tick = int(m5.curTick())
    event = m5.simulate(detail_ticks_left if detail_active else m5.MaxTick)
    cause = event.getCause()
    tick = int(m5.curTick())

    if detail_active:
        detail_ticks_left = max(0, detail_ticks_left - (tick - before_tick))
        if detail_ticks_left == 0:
            set_detail(False)
            detail_active = False
            if cause == "simulate() limit reached":
                continue

    if cause == "workbegin":
        phase_id = int(event.getCode())
        if phase_id not in PHASES or active_phase is not None:
            raise RuntimeError(f"invalid workbegin phase {phase_id} at tick {tick}")
        active_phase = {
            "id": phase_id,
            "name": PHASES[phase_id],
            "begin_tick": tick,
        }
        m5.stats.reset()
        if args.detail_cycles > 0 and args.detail_phase == PHASES[phase_id]:
            detail_ticks_left = args.detail_cycles * period_ticks
            set_detail(True)
            detail_active = True
        save_events()
        print(f"Qwen hybrid phase begin: {PHASES[phase_id]} at tick {tick}")
        continue

    if cause == "workend":
        phase_id = int(event.getCode())
        if active_phase is None or active_phase["id"] != phase_id:
            raise RuntimeError(f"unmatched workend phase {phase_id} at tick {tick}")
        active_phase["end_tick"] = tick
        active_phase["ticks"] = tick - active_phase["begin_tick"]
        if detail_active:
            set_detail(False)
            detail_active = False
            detail_ticks_left = 0
        m5.stats.dump(message=f"QWEN_PHASE {PHASES[phase_id]}")
        m5.stats.reset()
        events.append(active_phase)
        print(
            f"Qwen hybrid phase end: {PHASES[phase_id]} at tick {tick} "
            f"({active_phase['ticks']} ticks)"
        )
        active_phase = None
        save_events()
        continue

    final_event = {"cause": cause, "code": int(event.getCode()), "tick": tick}
    save_events(final_event)
    print(f"gem5 exit: {cause} at tick {tick}")
    if active_phase is not None:
        print(f"error: simulation exited inside phase {active_phase['name']}")
        sys.exit(2)
    sys.exit(event.getCode())
