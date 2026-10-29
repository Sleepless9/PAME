#!/usr/bin/env python3
"""Analyze joint RVV/AME InstQueue activity and SPM DMA descriptors."""

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TICK_RE = re.compile(r"^(\d+):")
EXEC_RE = re.compile(r"\bexec=(\d+)\b")
ENGINE_RE = re.compile(r"\bengine=(RVV|AME)\b")
DMA_START_RE = re.compile(r"start descriptor=(\d+)")
DMA_COMPLETE_RE = re.compile(
    r"complete descriptor=(\d+) bytes=(\d+) rows=(\d+) error=(\d+)"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    events_path = run_dir / "phase_events.json"
    if not events_path.is_file():
        raise SystemExit(f"missing phase metadata: {events_path}")
    metadata = json.loads(events_path.read_text(encoding="utf-8"))
    phases = metadata.get("phases", [])
    trace_path = args.trace or run_dir / "hybrid_trace.log"
    if not trace_path.is_file() and args.trace is None:
        trace_path = run_dir / "hybrid_trace.log.gz"

    dispatch_counts = defaultdict(Counter)
    completion_counts = defaultdict(Counter)
    latency_sum = defaultdict(Counter)
    latency_min = {}
    latency_max = {}
    inflight = {}
    dma_started = Counter()
    dma_completed = Counter()
    dma_errors = Counter()
    dma_bytes = Counter()
    unmatched_completions = 0
    period = metadata.get("clock_period_ticks", 1000)

    latency_path = run_dir / "hybrid_instruction_latency.csv"
    with latency_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["phase", "engine", "exec_seq", "dispatch_tick",
             "completion_tick", "latency_cycles"]
        )
        if trace_path.is_file():
            with open_text(trace_path) as trace:
                for line in trace:
                    tick_match = TICK_RE.match(line)
                    if not tick_match:
                        continue
                    tick = int(tick_match.group(1))
                    phase = phase_for_tick(phases, tick)
                    if "Dispatch tid=" in line:
                        exec_match = EXEC_RE.search(line)
                        engine_match = ENGINE_RE.search(line)
                        if exec_match and engine_match:
                            sequence = int(exec_match.group(1))
                            engine = engine_match.group(1).lower()
                            inflight[sequence] = (phase, tick, engine)
                            dispatch_counts[phase][engine] += 1
                        continue
                    if "Complete tid=" in line:
                        exec_match = EXEC_RE.search(line)
                        if not exec_match:
                            continue
                        sequence = int(exec_match.group(1))
                        dispatch = inflight.pop(sequence, None)
                        if dispatch is None:
                            unmatched_completions += 1
                            continue
                        dispatch_phase, dispatch_tick, engine = dispatch
                        latency = (tick - dispatch_tick) // period
                        writer.writerow(
                            [dispatch_phase, engine, sequence, dispatch_tick,
                             tick, latency]
                        )
                        completion_counts[dispatch_phase][engine] += 1
                        latency_sum[dispatch_phase][engine] += latency
                        key = (dispatch_phase, engine)
                        latency_min[key] = min(latency_min.get(key, latency), latency)
                        latency_max[key] = max(latency_max.get(key, latency), latency)
                        continue
                    start_match = DMA_START_RE.search(line)
                    if start_match:
                        dma_started[phase] += 1
                        continue
                    complete_match = DMA_COMPLETE_RE.search(line)
                    if complete_match:
                        _, byte_count, _, error = complete_match.groups()
                        dma_completed[phase] += 1
                        dma_bytes[phase] += int(byte_count)
                        if int(error):
                            dma_errors[phase] += 1

    rows = []
    for phase in phases:
        name = phase["name"]
        row = {
            "phase": name,
            "ticks": phase.get("ticks", ""),
            "rvv_dispatched": dispatch_counts[name]["rvv"],
            "rvv_completed": completion_counts[name]["rvv"],
            "ame_dispatched": dispatch_counts[name]["ame"],
            "ame_completed": completion_counts[name]["ame"],
            "dma_started": dma_started[name],
            "dma_completed": dma_completed[name],
            "dma_bytes": dma_bytes[name],
            "dma_errors": dma_errors[name],
        }
        for engine in ("rvv", "ame"):
            count = completion_counts[name][engine]
            key = (name, engine)
            row[f"{engine}_latency_avg"] = (
                f"{latency_sum[name][engine] / count:.2f}" if count else ""
            )
            row[f"{engine}_latency_min"] = latency_min.get(key, "")
            row[f"{engine}_latency_max"] = latency_max.get(key, "")
        rows.append(row)

    columns = list(rows[0]) if rows else ["phase"]
    with (run_dir / "hybrid_phase_profile.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    final_event = metadata.get("final_event", {})
    summary = {
        "trace": str(trace_path),
        "complete": len(phases) == 7 and not inflight,
        "final_cause": final_event.get("cause", "running"),
        "phase_count": len(phases),
        "accelerator_instruction_counts": {
            "rvv": sum(row["rvv_completed"] for row in rows),
            "ame": sum(row["ame_completed"] for row in rows),
        },
        "dma": {
            "started": sum(dma_started.values()),
            "completed": sum(dma_completed.values()),
            "bytes": sum(dma_bytes.values()),
            "errors": sum(dma_errors.values()),
        },
        "unmatched_inflight": len(inflight),
        "unmatched_completions": unmatched_completions,
    }
    (run_dir / "hybrid_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote hybrid profiling reports under {run_dir}")


if __name__ == "__main__":
    main()
