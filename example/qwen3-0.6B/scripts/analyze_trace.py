#!/usr/bin/env python3
"""Create phase, instruction, lane, and stall reports from a Qwen accelerator run."""

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


EVENT_PATTERNS = (
    # AMEEngine P1/Ztt events.
    ("ame_dispatch_accept", "AME_DISPATCH_ACCEPT"),
    ("ame_id_alloc", "AME_ID_ALLOC"),
    ("ame_global_broadcast", "AME_GLOBAL_BROADCAST"),
    ("ame_target_accept", "TARGET_ACCEPT"),
    ("ame_mlsu_req", "MLSU_REQ"),
    ("ame_mlsu_resp", "MLSU_RESP"),
    ("ame_pe_mac_issue", "PE_MAC_ISSUE"),
    ("ame_pe_mac_result", "PE_MAC_RESULT"),
    ("ame_tile_accept", "TILE_CMD_ACCEPT"),
    ("ame_kstep", "KSTEP_INJECT"),
    ("ame_target_done", "TARGET_DONE"),
    ("ame_complete", "AME_COMPLETE"),
    ("dispatcher_accept", "Dispatcher ACCEPT"),
    ("global_alloc", "GlobalSeq ALLOC"),
    ("broadcast_start", "PE_BROADCAST_START"),
    ("broadcast_complete", "PE_BROADCAST_COMPLETE"),
    ("fanout_fire", "PE_FANOUT FIRE"),
    ("lane_request", "LaneSeq REQUEST"),
    ("lane_alloc", "LaneSeq ALLOC"),
    ("operand_stall", "Operand STALL"),
    ("operand_request", "Operand REQUEST"),
    ("vrf_read", "VRF READ_FIRE"),
    ("vrf_write", "VRF WRITE_FIRE"),
    ("fu_issue", "FU ISSUE"),
    ("fu_result", "FU RESULT"),
    ("vlsu_response", "VLSU RESP"),
    ("vlsu_result", "VLSU RESULT"),
    ("sldu", "SLDU "),
    ("writeback", "WB COMPLETE"),
    ("beat_done", "Completion BEAT_DONE"),
    ("inst_done", "Completion INST_DONE"),
    ("completion_handshake", "Completion HANDSHAKE"),
    ("queue_dispatch", "Dispatch tid="),
    ("queue_complete", "Complete tid="),
    ("queue_retire", "Retire tid="),
    ("queue_cancel", "Cancel tid="),
)

STAT_KEYS = (
    "simSeconds",
    "simTicks",
    "simInsts",
    "simOps",
    "hostSeconds",
    "hostMemory",
    "system.cpu.numCycles",
    "system.cpu.cpi",
    "system.cpu.ipc",
)

TICK_RE = re.compile(r"^(\d+):")
CYCLE_RE = re.compile(r"\bcycle=(\d+)\b")
INST_RE = re.compile(r"\binst=(\d+)\b")
QUEUE_EXEC_RE = re.compile(r"\bexec=(\d+)\b")
QUEUE_ENGINE_RE = re.compile(r"\bengine=(RVV|AME)\b")
LANE_RE = re.compile(r"\blane=(\d+)\b")
STALL_RE = re.compile(r"Operand STALL\s*(.*)$")
STAT_RE = re.compile(r"^([A-Za-z0-9_.:]+)\s+([-+]?\d+(?:\.\d+)?)\s", re.M)
SECTION_RE = re.compile(
    r"^-+ Begin Simulation Statistics(?: : QWEN_PHASE ([a-z_]+))? -+\n"
    r"(.*?)"
    r"^-+ End Simulation Statistics\s+-+$",
    re.M | re.S,
)


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def phase_for_tick(phases, tick):
    for phase in phases:
        if phase["begin_tick"] <= tick <= phase.get("end_tick", tick):
            return phase["name"]
    return "outside"


def parse_stats(run_dir):
    stats_path = run_dir / "stats.txt"
    results = {}
    if not stats_path.is_file():
        return results
    text = stats_path.read_text(encoding="utf-8", errors="replace")
    for match in SECTION_RE.finditer(text):
        name, body = match.groups()
        if not name:
            continue
        section = match.group(0) + "\n"
        (run_dir / f"stats_{name}.txt").write_text(section, encoding="utf-8")
        values = {}
        for stat_match in STAT_RE.finditer(body):
            key, value = stat_match.groups()
            if key in STAT_KEYS:
                values[key] = value
        results[name] = values
    return results


def write_markdown(path, rows, event_names):
    accelerator = rows[0].get("accelerator", "accelerator") if rows else "accelerator"
    columns = [
        "phase",
        "ticks",
        "cpu_cycles",
        "sim_insts",
        "ipc",
        "accelerator_insts",
        "accelerator_latency_avg",
        "operand_stalls",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# Qwen3 accelerator phase profile\n\n")
        stream.write("| " + " | ".join(columns) + " |\n")
        stream.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            stream.write(
                "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n"
            )
        stream.write(f"\n## {accelerator.upper()} pipeline events\n\n")
        stream.write("| phase | " + " | ".join(event_names) + " |\n")
        stream.write("|---|" + "|".join("---:" for _ in event_names) + "|\n")
        for row in rows:
            stream.write(
                f"| {row['phase']} | "
                + " | ".join(str(row.get(name, 0)) for name in event_names)
                + " |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--backend", choices=["rvv", "ame"], default="rvv")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    events_path = run_dir / "phase_events.json"
    if not events_path.is_file():
        raise SystemExit(f"missing phase metadata: {events_path}")
    metadata = json.loads(events_path.read_text(encoding="utf-8"))
    phases = metadata.get("phases", [])
    trace_path = args.trace or run_dir / f"{args.backend}_trace.log.gz"
    if not trace_path.is_file() and args.trace is None:
        trace_path = run_dir / f"{args.backend}_trace.log"
    stats = parse_stats(run_dir)

    counts = defaultdict(Counter)
    lane_counts = defaultdict(Counter)
    stall_counts = defaultdict(Counter)
    latency_sums = Counter()
    latency_counts = Counter()
    latency_mins = {}
    latency_maxes = Counter()
    inflight = {}
    unmatched_completions = 0

    latency_path = run_dir / f"{args.backend}_instruction_latency.csv"
    with latency_path.open("w", newline="", encoding="utf-8") as latency_stream:
        latency_writer = csv.writer(latency_stream)
        latency_writer.writerow(
            ["phase", "exec_seq", "dispatch_tick", "completion_tick", "latency_cycles"]
        )
        if trace_path.is_file():
            with open_text(trace_path) as trace:
                for line in trace:
                    tick_match = TICK_RE.match(line)
                    if not tick_match:
                        continue
                    tick = int(tick_match.group(1))
                    phase = phase_for_tick(phases, tick)
                    event = None
                    for event_name, marker in EVENT_PATTERNS:
                        if marker in line:
                            event = event_name
                            counts[phase][event] += 1
                            break
                    if event is None:
                        continue

                    lane_match = LANE_RE.search(line)
                    if lane_match:
                        lane_counts[(phase, int(lane_match.group(1)))][event] += 1
                    if event == "operand_stall":
                        reason_match = STALL_RE.search(line)
                        reason = reason_match.group(1).strip() if reason_match else "unspecified"
                        stall_counts[phase][reason] += 1

                    engine_match = QUEUE_ENGINE_RE.search(line)
                    if (
                        event == "queue_dispatch"
                        and engine_match
                        and engine_match.group(1).lower() == args.backend
                    ):
                        exec_match = QUEUE_EXEC_RE.search(line)
                        if exec_match:
                            inflight[int(exec_match.group(1))] = (phase, tick)
                        continue
                    if event == "queue_complete":
                        exec_match = QUEUE_EXEC_RE.search(line)
                        if not exec_match:
                            continue
                        exec_seq = int(exec_match.group(1))
                        dispatch = inflight.pop(exec_seq, None)
                        if dispatch is None:
                            unmatched_completions += 1
                            continue
                        dispatch_phase, dispatch_tick = dispatch
                        latency = (tick - dispatch_tick) // metadata.get(
                            "clock_period_ticks", 1000
                        )
                        latency_writer.writerow(
                            [dispatch_phase, exec_seq, dispatch_tick, tick, latency]
                        )
                        latency_sums[dispatch_phase] += latency
                        latency_counts[dispatch_phase] += 1
                        latency_mins[dispatch_phase] = min(
                            latency_mins.get(dispatch_phase, latency), latency
                        )
                        latency_maxes[dispatch_phase] = max(
                            latency_maxes[dispatch_phase], latency
                        )
                        continue

                    inst_match = INST_RE.search(line)
                    cycle_match = CYCLE_RE.search(line)
                    if not inst_match or not cycle_match:
                        continue
                    inst_id = int(inst_match.group(1))
                    cycle = int(cycle_match.group(1))

    event_names = [name for name, _ in EVENT_PATTERNS]
    phase_order = [phase["name"] for phase in phases]
    rows = []
    for phase in phases:
        name = phase["name"]
        phase_stats = stats.get(name, {})
        latency_count = latency_counts[name]
        row = {
            "phase": name,
            "begin_tick": phase["begin_tick"],
            "end_tick": phase.get("end_tick", ""),
            "ticks": phase.get("ticks", ""),
            "cpu_cycles": phase_stats.get("system.cpu.numCycles", ""),
            "sim_insts": phase_stats.get("simInsts", ""),
            "ipc": phase_stats.get("system.cpu.ipc", ""),
            "host_seconds": phase_stats.get("hostSeconds", ""),
            "accelerator": args.backend,
            "accelerator_insts": counts[name]["queue_dispatch"],
            "accelerator_latency_avg": (
                f"{latency_sums[name] / latency_count:.2f}" if latency_count else ""
            ),
            "accelerator_latency_min": latency_mins.get(name, ""),
            "accelerator_latency_max": latency_maxes.get(name, ""),
            "operand_stalls": counts[name]["operand_stall"],
        }
        row.update({event: counts[name][event] for event in event_names})
        rows.append(row)

    profile_columns = list(rows[0].keys()) if rows else ["phase"]
    with (run_dir / "phase_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=profile_columns)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(run_dir / "phase_profile.md", rows, event_names)

    with (run_dir / "lane_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "lane", *event_names])
        for phase, lane in sorted(
            lane_counts, key=lambda item: (phase_order.index(item[0]) if item[0] in phase_order else 999, item[1])
        ):
            writer.writerow(
                [phase, lane, *(lane_counts[(phase, lane)][event] for event in event_names)]
            )

    with (run_dir / "stall_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "reason", "count"])
        for phase in phase_order + ["outside"]:
            for reason, count in stall_counts[phase].most_common():
                writer.writerow([phase, reason, count])

    final_event = metadata.get("final_event", {})
    complete = (
        len(phases) == 7
        and final_event.get("cause") in {"simulate() limit reached", "exited normally"}
    )
    summary = {
        "trace": str(trace_path),
        "complete": complete,
        "status": "complete" if complete else "partial",
        "final_cause": final_event.get("cause", "running"),
        "unmatched_inflight": len(inflight),
        "unmatched_completions": unmatched_completions,
        "phase_count": len(phases),
        "accelerator": args.backend,
        "accelerator_instruction_count": sum(latency_counts.values()),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote Qwen {args.backend.upper()} profiling reports under {run_dir}")


if __name__ == "__main__":
    main()
