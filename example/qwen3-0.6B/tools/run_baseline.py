#!/usr/bin/env python3
"""Run a frozen 4x4 baseline, export both phases, and check native-reference logits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time
import numpy as np
from analyze_baseline import analyze, write_report
from build_baseline import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--simulator", type=Path)
    parser.add_argument("--phase-trace-ticks", type=int, default=50_000_000)
    parser.add_argument("--max-ticks", type=int, default=(1 << 63) - 1)
    parser.add_argument("--dma-burst", default="64B")
    parser.add_argument("--dma-outstanding", type=int, default=16)
    parser.add_argument("--dma-active-descriptors", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    binary, out = args.binary.resolve(), args.outdir.resolve()
    manifest = json.loads(binary.with_suffix(".json").read_text())
    assert manifest["local_spm"] and not manifest["operator_test"]
    features = manifest["enabled_optimizations"]
    assert features in ([], list("A"), list("AB"), list("ABC"), list("ABCD"))
    assert digest(binary) == manifest["output_sha256"]
    if (out / "run.json").exists():
        parser.error("This output directory already contains a run; choose a new one")
    out.mkdir(parents=True, exist_ok=True)
    artifacts = out / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for phase in ("prefill", "decode"):
        reference = Path(str(args.reference_prefix.resolve()) + "." + phase + ".f32")
        values = np.fromfile(reference, dtype="<f4")
        assert len(values) == 151936 and np.isfinite(values).all() and np.any(values != 0)
        shutil.copy2(reference, artifacts / (phase + ".reference.f32"))
    image = artifacts / "guest.elf"
    simulator = artifacts / "gem5.opt"
    shutil.copy2(binary, image)
    simulator_source = args.simulator.resolve() if args.simulator else root / "build/RISCV/gem5.opt"
    shutil.copy2(simulator_source, simulator)
    shutil.copy2(binary.with_suffix(".json"), artifacts / "guest_build.json")
    config = artifacts / "spm_compute_soc.py"
    shutil.copy2(root / "configs/example/spm_compute_soc.py", config)
    build = binary.parent / (binary.stem + "_build")
    for name in ("progress_symbols.json", "result_symbols.json"):
        shutil.copy2(build / name, artifacts / name)
    command = [str(simulator), "--listener-mode=off", "-d", str(out),
        "--debug-flags=PseudoInst", "--debug-file=trace.out", str(config), str(image),
        "--clock=1GHz", "--cpu-mode=minor", "--dataflow=pss", "--pe-rows=4", "--pe-cols=4",
        "--tile-rows=1", "--tile-cols=1", "--progress-ticks=10000000000",
        "--no-qwen-fpga-compat",
        "--progress-symbols=" + str(artifacts / "progress_symbols.json"),
        "--result-symbols=" + str(artifacts / "result_symbols.json"),
        "--phase-trace-ticks=" + str(args.phase_trace_ticks), "--max-ticks=" + str(args.max_ticks)]
    if "C" in features:
        command += ["--pss-continuous", "--tile-operation-depth=12"]
    command += ["--dma-burst=" + args.dma_burst,
                "--dma-outstanding=" + str(args.dma_outstanding),
                "--dma-active-descriptors=" + str(args.dma_active_descriptors)]
    report = dict(status="starting", started_unix=time.time(), command=command,
                  prefill_tokens=manifest["prefill_tokens"], decode_steps=manifest["decode_steps"],
                  seed=manifest["seed"], enabled_optimizations=features, effective_k=manifest["effective_k"],
                  variant=manifest.get("variant", "baseline"), simulator_build=str(simulator_source),
                  physical_array=[4, 4], continuous_injection="C" in features, dma_overlap="D" in features,
                  numerical_acceptance="bitwise identical phase logits to matching native algorithm reference",
                  simulator_sha256=digest(simulator), guest_sha256=digest(image),
                  config_sha256=digest(config), model_sha256=manifest["model_sha256"],
                  trace_format="plain", trace_window_ticks=args.phase_trace_ticks)
    report["dma"] = dict(submission=manifest.get("dma_submission", "mmio"),
                         burst=args.dma_burst, outstanding=args.dma_outstanding,
                         active_descriptors=args.dma_active_descriptors)
    def save():
        report["updated_unix"] = time.time()
        temporary = out / "run.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "run.json")
    save()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    last_event_count = 0
    with (out / "run.log").open("w") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        report.update(status="running", pid=child.pid)
        save()
        while child.poll() is None:
            time.sleep(30)
            with (out / "run.log").open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 32768))
                tail = stream.read().decode(errors="replace")
            progress = re.findall(r"Progress tick=(\d+)([^\n]*)", tail)
            if progress:
                report["last_progress_tick"] = int(progress[-1][0])
                report["guest_progress"] = progress[-1][1].strip()
            events = out / "phase_events.json"
            if events.exists():
                report["phase_events"] = json.loads(events.read_text())
                if len(report["phase_events"]) != last_event_count:
                    live_phases = analyze(out)
                    write_report(out, live_phases)
                    last_event_count = len(report["phase_events"])
            save()
        report["returncode"] = child.returncode
    uart_path = out / "system.platform.terminal"
    uart = uart_path.read_text(errors="replace") if uart_path.exists() else ""
    report.update(ended_unix=time.time(), guest_pass="QWEN_BASELINE PASS" in uart)
    comparisons = {}
    for phase in ("prefill", "decode"):
        path = out / (phase + ".logits.f32")
        if not path.exists():
            comparisons[phase] = dict(status="missing")
            continue
        actual = np.fromfile(path, dtype="<f4")
        expected = np.fromfile(artifacts / (phase + ".reference.f32"), dtype="<f4")
        same = actual.shape == expected.shape
        bitwise = same and np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        comparisons[phase] = dict(status="passed" if bitwise else "failed", bitwise_equal=bitwise,
            max_abs_error=float(np.max(np.abs(actual.astype(np.float64) - expected))) if same else None,
            actual_sha256=digest(path), reference_sha256=digest(artifacts / (phase + ".reference.f32")))
    report["numerical_comparison"] = comparisons
    try:
        phase_report = analyze(out)
        phase_report["correctness_validated"] = all(c["status"] == "passed" for c in comparisons.values())
        report["work_counts_match"] = all(phase_report["phases"][phase].get("guest_counter_deltas", {}).get(
            "completed_ztt_tiles") == manifest["expected_ame_tiles"][phase] for phase in ("prefill", "decode"))
        write_report(out, phase_report)
        report["both_phases_complete"] = phase_report["both_phases_complete"]
    except (KeyError, ValueError, AssertionError, FileNotFoundError) as error:
        report["analysis_error"] = str(error)
        report["both_phases_complete"] = False
    report["status"] = "passed" if (report["returncode"] == 0 and report["guest_pass"] and
        report["both_phases_complete"] and report.get("work_counts_match", False) and
        all(c["status"] == "passed" for c in comparisons.values())) else "failed_or_incomplete"
    save()
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
