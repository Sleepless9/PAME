"""Cycle-accurate MinorCPU + RVV/AME accelerator runner for Qwen3-0.6B W4A8."""

import argparse
import json
import os
import sys

import m5
from m5.objects import (
    AddrRange,
    BadAddr,
    Bridge,
    Cache,
    DDR3_1600_8x8,
    HiFive,
    IOXBar,
    MemCtrl,
    PMAChecker,
    RiscvBareMetal,
    RiscvRTC,
    RiscvSemihosting,
    RiscvSystem,
    Root,
    SrcClockDomain,
    SystemXBar,
    VoltageDomain,
)
from m5.objects.RiscvCPU import RiscvMinorCPU


PHASES = {
    1: "tokenizer_load",
    2: "prompt_tokenize",
    3: "model_load",
    4: "state_build",
    5: "prefill",
    6: "generate",
    7: "output",
}


parser = argparse.ArgumentParser()
parser.add_argument("--cmd", required=True, help="Qwen bare-metal ELF")
parser.add_argument("--mem-size", default="1GiB")
parser.add_argument("--clock", default="1GHz")
parser.add_argument("--vlen", type=int, default=512)
parser.add_argument("--num-lanes", type=int, default=4)
parser.add_argument("--lane-width-bits", type=int, default=64)
parser.add_argument("--vrf-banks-per-lane", type=int, default=8)
parser.add_argument("--inst-queue-size", type=int, default=32)
parser.add_argument("--rvv-detail-phase", default="prefill", choices=[*PHASES.values(), "none"])
parser.add_argument("--rvv-detail-cycles", type=int, default=250000)
parser.add_argument("--backend", choices=["rvv", "ame"], default="rvv")
args = parser.parse_args()

if args.vlen % args.lane_width_bits:
    parser.error("--vlen must be a multiple of --lane-width-bits")
if not os.path.isfile(args.cmd):
    parser.error(f"ELF does not exist: {args.cmd}")

memory_start = 0x80000000

system = RiscvSystem()
system.mem_mode = "timing"
system.mem_ranges = [AddrRange(start=memory_start, size=args.mem_size)]
system.exit_on_work_items = True
system.workload = RiscvBareMetal(
    bootloader=os.path.abspath(args.cmd),
    semihosting=RiscvSemihosting(
        cmd_line=os.path.abspath(args.cmd),
        files_root_dir="/",
    ),
)

system.iobus = IOXBar()
system.membus = SystemXBar()
system.membus.badaddr_responder = BadAddr(warn_access="warn")
system.membus.default = system.membus.badaddr_responder.pio
system.system_port = system.membus.cpu_side_ports

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

system.cache_line_size = 64
system.voltage_domain = VoltageDomain(voltage="1V")
system.clk_domain = SrcClockDomain(
    clock=args.clock, voltage_domain=system.voltage_domain
)
system.cpu_clk_domain = SrcClockDomain(
    clock=args.clock, voltage_domain=VoltageDomain()
)

system.cpu = RiscvMinorCPU(clk_domain=system.cpu_clk_domain, cpu_id=0)
system.cpu.executeInstQueueSize = args.inst_queue_size
if args.backend == "ame":
    system.cpu.addAmeEngine()
    system.cpu.ame_engine.memory_port = system.membus.cpu_side_ports
else:
    system.cpu.executeOffloadAME = False
    system.cpu.addRvvEngine(
        vlen_bits=args.vlen,
        num_lanes=args.num_lanes,
        lane_width_bits=args.lane_width_bits,
        vrf_banks_per_lane=args.vrf_banks_per_lane,
    )
system.icache = Cache(
    size="32KiB",
    assoc=2,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=4,
    tgts_per_mshr=20,
)
system.dcache = Cache(
    size="64KiB",
    assoc=4,
    tag_latency=2,
    data_latency=2,
    response_latency=2,
    mshrs=8,
    tgts_per_mshr=20,
)
system.cpu.icache_port = system.icache.cpu_side
system.cpu.dcache_port = system.dcache.cpu_side
system.icache.mem_side = system.membus.cpu_side_ports
system.dcache.mem_side = system.membus.cpu_side_ports
system.cpu.createInterruptController()
system.cpu.createThreads()
system.cpu.isa[0].vlen = args.vlen
system.cpu.isa[0].elen = 64
system.cpu.mmu.pma_checker = PMAChecker(
    uncacheable=[
        *system.platform._on_chip_ranges(),
        *system.platform._off_chip_ranges(),
    ]
)

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8(range=system.mem_ranges[0])
system.mem_ctrl.port = system.membus.mem_side_ports

root = Root(full_system=True, system=system)
m5.instantiate()

outdir = m5.options.outdir
phase_events_path = os.path.join(outdir, "phase_events.json")
phase_events = []
active_phase = None
detail_active = False
detail_ticks_left = 0


def set_detail_trace(enabled):
    method = "enable" if enabled else "disable"
    flag = "AMEEngine" if args.backend == "ame" else "RVVEngine"
    getattr(m5.debug.flags[flag], method)()


def clock_period_ticks(clock):
    units = {"GHz": 1e9, "MHz": 1e6, "kHz": 1e3, "Hz": 1.0}
    for suffix, scale in units.items():
        if clock.endswith(suffix):
            frequency = float(clock[: -len(suffix)]) * scale
            return max(1, round(1e12 / frequency))
    raise ValueError(f"unsupported --clock value: {clock}")


cpu_clock_period = clock_period_ticks(args.clock)


def save_phase_events(final_event=None):
    report = {
        "clock": args.clock,
        "vlen_bits": args.vlen,
        "num_lanes": args.num_lanes,
        "lane_width_bits": args.lane_width_bits,
        "clock_period_ticks": cpu_clock_period,
        "rvv_detail_phase": args.rvv_detail_phase,
        "rvv_detail_cycles": args.rvv_detail_cycles,
        "phases": phase_events,
    }
    if final_event is not None:
        report["final_event"] = final_event
    with open(phase_events_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


while True:
    before_tick = int(m5.curTick())
    exit_event = m5.simulate(detail_ticks_left if detail_active else m5.MaxTick)
    cause = exit_event.getCause()
    tick = int(m5.curTick())

    if detail_active:
        detail_ticks_left = max(0, detail_ticks_left - (tick - before_tick))
        if detail_ticks_left == 0:
            set_detail_trace(False)
            detail_active = False
            print(f"Qwen {args.backend.upper()} detail window end at tick {tick}")
            if cause == "simulate() limit reached":
                continue

    if cause == "workbegin":
        phase_id = int(exit_event.getCode())
        if phase_id not in PHASES:
            raise RuntimeError(f"unknown Qwen workbegin phase id {phase_id}")
        if active_phase is not None:
            raise RuntimeError(f"nested Qwen phase at tick {tick}: {active_phase}")
        active_phase = {
            "id": phase_id,
            "name": PHASES[phase_id],
            "begin_tick": tick,
        }
        m5.stats.reset()
        if (
            args.rvv_detail_cycles > 0
            and args.rvv_detail_phase == PHASES[phase_id]
        ):
            detail_ticks_left = args.rvv_detail_cycles * cpu_clock_period
            set_detail_trace(True)
            detail_active = True
            print(
                f"Qwen {args.backend.upper()} detail window begin: {PHASES[phase_id]} "
                f"for {args.rvv_detail_cycles} cycles"
            )
        save_phase_events()
        print(f"Qwen phase begin: {PHASES[phase_id]} at tick {tick}")
        continue

    if cause == "workend":
        phase_id = int(exit_event.getCode())
        if active_phase is None or active_phase["id"] != phase_id:
            raise RuntimeError(
                f"unmatched Qwen workend phase id {phase_id} at tick {tick}"
            )
        active_phase["end_tick"] = tick
        active_phase["ticks"] = tick - active_phase["begin_tick"]
        if detail_active:
            set_detail_trace(False)
            detail_active = False
            detail_ticks_left = 0
        m5.stats.dump(message=f"QWEN_PHASE {PHASES[phase_id]}")
        m5.stats.reset()
        phase_events.append(active_phase)
        print(
            f"Qwen phase end: {PHASES[phase_id]} at tick {tick} "
            f"({active_phase['ticks']} ticks)"
        )
        active_phase = None
        save_phase_events()
        continue

    final_event = {
        "cause": cause,
        "code": int(exit_event.getCode()),
        "tick": tick,
    }
    save_phase_events(final_event)
    print(f"gem5 exit: {cause} at tick {tick}")
    if active_phase is not None:
        print(f"error: simulation exited inside phase {active_phase['name']}")
        sys.exit(2)
    sys.exit(exit_event.getCode())
