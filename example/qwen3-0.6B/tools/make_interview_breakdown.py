#!/usr/bin/env python3
"""Create English interview figures from measured live metrics plus labeled scenarios."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]
src=root/'runs/qwen3_optimizations_20260908/live_analysis/live_metrics.csv'
out=root/'runs/qwen3_optimizations_20260908/interview_breakdown'; out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(src.open()))
labels=['Packed K16','K16 + Group Acc','Continuous PSS Injection','Continuous PSS + DMA Double Buffering']
measured=[float(r['tile_rate']) for r in rows]
fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(labels,measured,color=['#4472c4','#70ad47','#ed7d31','#a64d79']); ax.set_title('Measured Live Prefill Throughput'); ax.set_ylabel('Completed tiles / simulated cycle'); ax.tick_params(axis='x',rotation=18); ax.grid(axis='y',alpha=.3); fig.tight_layout(); fig.savefig(out/'measured_prefill_throughput.png',dpi=180); fig.savefig(out/'measured_prefill_throughput.pdf'); plt.close(fig)

# A normalized architecture scenario, not a claim about completed simulation.
components=['AME compute','RVV compute','DDR↔SPM DMA','SPM arbitration','Dispatch & sync','Runtime overhead']
scenario={
 'Packed K16':[38,12,20,5,15,10],
 'K16 + Group Acc':[48,12,16,4,12,8],
 'Continuous PSS Injection':[53,12,15,4,9,7],
 'Continuous PSS + DMA Double Buffering':[56,12,10,4,9,9],
}
fig,ax=plt.subplots(figsize=(9,5)); bottom=[0]*len(labels)
colors=['#1f4e79','#70ad47','#ed7d31','#a5a5a5','#ffc000','#8064a2']
for i,c in enumerate(components):
 vals=[scenario[n][i] for n in labels]; ax.bar(labels,vals,bottom=bottom,label=c,color=colors[i]); bottom=[x+y for x,y in zip(bottom,vals)]
ax.set_ylim(0,100); ax.set_ylabel('Normalized end-to-end latency (%)'); ax.set_title('Scenario Breakdown: Descriptor DMA + Double Buffering'); ax.tick_params(axis='x',rotation=18); ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(out/'scenario_latency_breakdown.png',dpi=180); fig.savefig(out/'scenario_latency_breakdown.pdf'); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.5)); base=[100/measured[0]*x for x in measured]; projected=[1.0,1.25,1.38,1.52]
ax.plot(labels,base,'o-',label='Measured normalized throughput',color='#4472c4'); ax.plot(labels,projected,'s--',label='Scenario throughput index',color='#ed7d31'); ax.set_ylabel('Index (first configuration = 1.0)'); ax.set_title('Measured Trend vs Runtime Scenario'); ax.legend(); ax.grid(alpha=.3); ax.tick_params(axis='x',rotation=18); fig.tight_layout(); fig.savefig(out/'measured_vs_scenario.png',dpi=180); fig.savefig(out/'measured_vs_scenario.pdf'); plt.close(fig)

report=['# Interview Performance Breakdown', '', 'All figures use English labels. `Measured` values come from the current live prefill statistics. `Scenario` values are normalized architectural assumptions for a descriptor-ring DMA runtime; they are not completed-model simulation results.', '', '## Measured live prefill', '', '| Configuration | Tile throughput (tiles/cycle) | PE issue utilization | SA busy fraction | SPM bandwidth (B/cycle) | DMA bandwidth (B/cycle) |', '|---|---:|---:|---:|---:|---:|']
for r in rows: report.append(f"| {r['name']} | {float(r['tile_rate']):.6g} | {float(r['pe_issue_util']):.3f} | {float(r['sa_busy']):.3f} | {float(r['spm_bw']):.3f} | {float(r['dma_bw']):.3f} |")
report += ['', '![Measured prefill throughput](measured_prefill_throughput.png)', '', '## Runtime scenario breakdown', '', 'The scenario assumes batched DMA descriptors, two SPM input buffers, one output buffer, completion events instead of per-tile polling, and overlap whenever the next DMA transfer is no longer than the current compute window. Percentages are normalized to 100% for each configuration and are intended for architecture discussion, not measurement.', '', '![Scenario latency breakdown](scenario_latency_breakdown.png)', '', '## Interview framing', '', '- Measured: current gem5 cumulative snapshots from `workbegin` to the latest available statistics point.', '- Projected: an architecture-level estimate after descriptor batching and event-driven DMA/compute overlap.', '- The current measured data shows zero SPM bank wait and zero request retries; therefore the first optimization target is control and overlap, not blindly increasing bank count.', '- The low measured PE issue utilization indicates that end-to-end latency is dominated by supply, synchronization, and runtime control around the AME pipeline.', '- Decode must be measured separately because M=1 and KV-cache traffic change the balance between AME, RVV, DMA, and runtime overhead.', '', '![Measured versus scenario trend](measured_vs_scenario.png)', '', 'The scenario curve must be replaced by completed prefill/decode phase measurements before claiming a speedup.']
(out/'interview_breakdown.md').write_text('\n'.join(report)+'\n')
print(out/'interview_breakdown.md')
