#!/usr/bin/env python3
"""Plot observed partial prefill progress and a clearly labelled decode forecast."""
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/qwen3_optimizations_20260908/partial_performance"
GROUPS = [("A", "full_model"), ("AB", "full_model"),
          ("ABC", "full_model_v2"), ("ABCD", "full_model_v2")]

rows = []
for name, run_dir in GROUPS:
    base = ROOT / "runs/qwen3_optimizations_20260908" / name
    run = json.loads((base / run_dir / "run.json").read_text())
    manifest = json.loads((base / "guest.json").read_text())
    begin = run["phase_events"][0]["tick"]
    tick = run["last_progress_tick"]
    text = run.get("guest_progress", "")
    tiles = int(text.split("completed_ztt_tiles=")[1].split()[0])
    layers = int(text.split("completed_layer_passes=")[1].split()[0])
    cycles = (tick - begin) / 1000.0
    rate = tiles / cycles if cycles else 0.0
    total_pre = manifest["expected_ame_tiles"]["prefill"]
    total_dec = manifest["expected_ame_tiles"]["decode"]
    remain_pre = max(0, total_pre - tiles)
    forecast_pre = remain_pre / rate if rate else None
    forecast_dec = total_dec / rate if rate else None
    rows.append(dict(group=name, tick=tick, cycles=cycles, tiles=tiles,
                     layers=layers, prefill_expected=total_pre,
                     decode_expected=total_dec, tile_rate=rate,
                     prefill_progress=tiles / total_pre,
                     prefill_remaining_cycles=forecast_pre,
                     decode_forecast_cycles=forecast_dec))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "partial_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

names = [r["group"] for r in rows]
rate = [r["tile_rate"] for r in rows]
progress = [100*r["prefill_progress"] for r in rows]
forecast = [r["decode_forecast_cycles"]/1e9 for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(names, rate, color="#4472c4")
ax[0].set_title("Observed prefill tile throughput")
ax[0].set_ylabel("tiles / simulated cycle")
ax[0].grid(axis="y", alpha=.3)
ax[1].bar(names, progress, color="#70ad47")
ax[1].set_title("Observed prefill completion")
ax[1].set_ylabel("completed tiles / expected prefill (%)")
ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "observed_prefill.png", dpi=180); fig.savefig(OUT / "observed_prefill.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(names, forecast, color="#ed7d31")
ax.set_title("Decode latency forecast from current prefill rate")
ax.set_ylabel("forecast simulated cycles (×10⁹)")
ax.grid(axis="y", alpha=.3)
fig.text(.5, .01, "Forecast only: assumes decode tile rate equals observed prefill rate; must be replaced by completed decode phase.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(OUT / "decode_forecast.png", dpi=180); fig.savefig(OUT / "decode_forecast.pdf")
plt.close(fig)

lines = ["# 当前递进优化阶段性能报告", "", "生成时间：2026-09-08。四组均处于 prefill，尚未产生完整 workend，也没有真实 decode 统计。图中观测值来自运行时 `run.json` 的 progress counter；预测值不能替代 decode 仿真。", "", "## 当前观测", "", "| 组别 | 已完成 tile | layer pass | 观测 tile/cycle | prefill 完成度 |", "|---|---:|---:|---:|---:|"]
for r in rows:
    lines.append(f"| {r['group']} | {r['tiles']:,} | {r['layers']} | {r['tile_rate']:.6g} | {100*r['prefill_progress']:.3f}% |")
lines += ["", "![观测 prefill](observed_prefill.png)", "", "## Decode 预测", "", "预测使用每组当前 prefill 的 `completed_ztt_tiles / (last_progress_tick-workbegin_tick)` 作为 tile 吞吐，并乘以该组 manifest 中的 decode expected tile 数。它没有建模 decode 的 M=1、KV cache、RVV/AME 比例变化，因此只能用于运行期间的粗略排序。", "", "| 组别 | decode expected tile | 预测 decode cycles |", "|---|---:|---:|"]
for r in rows:
    lines.append(f"| {r['group']} | {r['decode_expected']:,} | {r['decode_forecast_cycles']/1e9:.3f}e9 |")
lines += ["", "![decode forecast](decode_forecast.png)", "", "## 结论边界", "", "- 当前不能宣布 A/AB/ABC/ABCD 的真实加速比，因为阶段边界尚未完成。", "- A/AB 的 tile 进度较快，反映的是当前 guest 计数吞吐，不能直接等同于端到端时延。", "- ABC/ABCD 已确认连续注入和 DMA overlap 开关生效；其收益必须等待完整 prefill/decode 的 cycles、PE issue、SPM 和 DMA 统计。", "- 完成后应重新运行 `analyze_baseline.py`，用 workbegin/workend 差分替换本报告中的预测列。", ""]
(OUT / "partial_performance_report.md").write_text("\n".join(lines))
print(OUT / "partial_performance_report.md")
