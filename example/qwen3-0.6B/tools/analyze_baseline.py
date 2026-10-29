#!/usr/bin/env python3
"""Extract prefill/decode cumulative-stat deltas; never substitute progress for completion."""
import argparse
import json
from pathlib import Path


def snapshots(path):
    result, current = [], None
    with path.open() as stream:
        for line in stream:
            if "Begin Simulation Statistics" in line:
                current = {}
            elif "End Simulation Statistics" in line and current is not None:
                result.append(current)
                current = None
            elif current is not None:
                fields = line.split()
                if len(fields) > 1:
                    try:
                        current[fields[0]] = float(fields[1])
                    except ValueError:
                        pass
    return result


COUNTERS = {
    "cpu_instructions": "system.cpu.committedInsts",
    "ame_active_cycles": "system.cpu.ame_engine.activeCycles",
    "rvv_active_cycles": "system.cpu.rvv_engine.activeCycles",
    "rvv_completed": "system.cpu.rvv_engine.completedInstructions",
    "rvv_context_squashes": "system.cpu.rvv_engine.squashes",
    "rvv_memory_accepted": "system.cpu.rvv_engine.memoryAccepted",
    "rvv_memory_retries": "system.cpu.rvv_engine.memoryRetries",
    "sa_active_cycles": "system.cpu.ame_engine.saActiveCycles",
    "pe_mac_issues": "system.cpu.ame_engine.saPeIssues",
    "pe_mac_results": "system.cpu.ame_engine.saPeResults",
    "ame_completed": "system.cpu.ame_engine.completedInstructions",
    "ame_context_squashes": "system.cpu.ame_engine.squashes",
    "ame_memory_generated": "system.cpu.ame_engine.memoryRequests",
    "ame_memory_accepted": "system.cpu.ame_engine.memoryAccepted",
    "ame_memory_responses": "system.cpu.ame_engine.memoryResponses",
    "ame_memory_retries": "system.cpu.ame_engine.memoryRetries",
    "spm_timing_requests": "system.spm.timingRequests",
    "spm_timing_read_bytes": "system.spm.bytesRead",
    "spm_timing_write_bytes": "system.spm.bytesWritten",
    "spm_store_reservation_bytes": "system.spm.storeReservationBytes",
    "spm_functional_read_bytes": "system.spm.functionalBytesRead",
    "spm_functional_write_bytes": "system.spm.functionalBytesWritten",
    "spm_retries": "system.spm.requestRetries",
    "spm_completed_request_ticks": "system.spm.completedRequestTicks",
    "dma_descriptors": "system.spm_dma.descriptorsCompleted",
    "dma_errors": "system.spm_dma.descriptorErrors",
    "dma_completed_descriptor_ticks": "system.spm_dma.completedDescriptorTicks",
    "dma_to_spm_bytes": "system.spm_dma.systemToSpmBytes",
    "dma_to_ddr_bytes": "system.spm_dma.spmToSystemBytes",
    "dma_spm_retries": "system.spm_dma.spmRetries",
    "host_seconds": "hostSeconds",
}
for suffix in ("offloadQueueFullStalls", "offloadSourceStalls",
               "offloadSerializeBeforeStalls", "offloadSerializeAfterStalls",
               "offloadCompletionWaits"):
    COUNTERS[suffix] = "system.cpu." + suffix


def analyze(out):
    events_path, stats_path = out / "phase_events.json", out / "stats.txt"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    stats = snapshots(stats_path) if stats_path.exists() else []
    config = json.loads((out / "config.json").read_text())["system"]
    engine = config["cpu"]["ame_engine"]
    pes = engine["pe_rows"] * engine["pe_cols"] * engine["tile_rows"] * engine["tile_cols"]
    ii = engine["mac_initiation_interval"]
    report = dict(statistics_version="local-spm-baseline-v1", phases={},
                  physical_pes=pes, mac_ii=ii, dataflow=engine["dataflow"],
                  stats_reset=False, phase_counter_method="cumulative end minus begin")
    for phase, name in [(5, "prefill"), (6, "decode")]:
        starts = [e for e in events if e["phase"] == phase and e["event"] == "workbegin"]
        ends = [e for e in events if e["phase"] == phase and e["event"] == "workend"]
        data = dict(status="not_started" if not starts else "incomplete")
        report["phases"][name] = data
        if len(starts) != 1 or len(ends) != 1:
            continue
        start, end = starts[0], ends[0]
        if max(start["stats_dump_index"], end["stats_dump_index"]) >= len(stats):
            continue
        before, after = stats[start["stats_dump_index"]], stats[end["stats_dump_index"]]
        # Detect stale/misaligned event indexes instead of producing plausible deltas.
        assert before["finalTick"] == start["tick"] and after["finalTick"] == end["tick"]
        ticks = end["tick"] - start["tick"]
        assert ticks > 0
        cycles = ticks / after["system.clk_domain.clock"]
        data.update(status="phase_complete", start_tick=start["tick"], end_tick=end["tick"],
                    ticks=ticks, cycles=cycles, seconds=ticks / after["simFreq"])
        for label, key in COUNTERS.items():
            data[label] = after[key] - before[key] if key in before and key in after else None
            if data[label] is not None:
                assert data[label] >= 0, f"counter reset or incompatible statistics: {key}"
        macs = data["pe_mac_issues"]
        data["pe_issue_utilization"] = macs / (cycles * pes / ii) if macs is not None else None
        data["sa_busy_fraction"] = data["sa_active_cycles"] / cycles
        data["ame_busy_fraction"] = data["ame_active_cycles"] / cycles
        data["algorithm_useful_mac_utilization"] = None
        counters0, counters1 = start.get("guest_counters", {}), end.get("guest_counters", {})
        data["guest_counter_deltas"] = {key: value - counters0[key] for key, value in counters1.items() if key in counters0}
        manifest_path = out / "artifacts/guest_build.json"
        if manifest_path.exists() and "completed_ztt_tiles" in data["guest_counter_deltas"]:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("effective_k") in (4, 16):
                tiles = data["guest_counter_deltas"]["completed_ztt_tiles"]
                data["useful_ame_projection_macs"] = tiles * 4 * manifest["effective_k"]
                data["algorithm_useful_mac_utilization"] = data["useful_ame_projection_macs"] / (cycles * pes / ii)
        data["bank_counters"] = {key: value - before.get(key, 0)
            for key, value in after.items() if key.startswith(("system.spm.bankAccesses", "system.spm.bankQueueFull", "system.spm.bankWaitTicks", "system.spm.portReadBytes", "system.spm.portWriteBytes"))}
    report["both_phases_complete"] = all(p["status"] == "phase_complete" for p in report["phases"].values())
    report["correctness_validated"] = False  # Phase completion alone does not validate numerical results.
    return report


def write_report(outdir, report):
    (outdir / "phase_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 4×4 baseline 阶段统计", "", "统计采用 workbegin/workend 边界累计差分；完成阶段不等于数值正确性通过。", "",
             "| 指标 | prefill | decode |", "|---|---:|---:|"]
    for metric in ("status", "cycles", "seconds", "pe_mac_issues", "pe_mac_results",
                   "pe_issue_utilization", "algorithm_useful_mac_utilization", "sa_busy_fraction", "ame_busy_fraction",
                   "ame_context_squashes", "dma_descriptors", "dma_to_spm_bytes",
                   "dma_to_ddr_bytes", "spm_timing_read_bytes", "spm_timing_write_bytes",
                   "spm_store_reservation_bytes", "host_seconds"):
        values = [report["phases"][p].get(metric) for p in ("prefill", "decode")]
        lines.append("| " + metric + " | " + " | ".join("—" if v is None else str(v) for v in values) + " |")
    lines += ["", "PE 利用率 = 实际到达发射周期的 MAC 数 /（阶段时钟周期 × PE 数 / MAC II）。",
              "包含 padding/零值运算；没有用 AME activeCycles 充当 PE 利用率。SA busy 包括供数与排空等待。",
              "SPM timing 写字节包含无投机副作用的写资源预留；functional 字节是另一层口径，不能与 timing 相加。",
              "下发阻塞与提交等待可以重叠，不能直接相加作为 CPI 分解。",
              "宿主秒数包含仿真器和日志开销，不代表芯片时延。", ""]
    lines.insert(3, "数值参考校验通过：" + str(report["correctness_validated"]) + "。")
    (outdir / "phase_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    report = analyze(args.outdir)
    write_report(args.outdir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
