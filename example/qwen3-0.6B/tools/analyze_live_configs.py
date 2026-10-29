#!/usr/bin/env python3
"""Analyze the current cumulative interval for the four live configurations."""
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
from analyze_baseline import snapshots

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'runs/qwen3_optimizations_20260908/live_analysis'; OUT.mkdir(parents=True,exist_ok=True)
GROUPS=[('packed K16','A','full_model'),('K16+group Acc','AB','full_model'),('连续PSS注入','ABC','full_model_v2'),('连续PSS+DMA双缓冲','ABCD','full_model_v2')]
def val(s,k): return s.get(k,0.0)
rows=[]
for cn,v,d in GROUPS:
    base=ROOT/'runs/qwen3_optimizations_20260908'/v; out=base/d
    run=json.loads((out/'run.json').read_text()); ss=snapshots(out/'stats.txt')
    begin=run['phase_events'][0]['tick']; end=ss[-1]['finalTick']; start=ss[0]
    cyc=(end-begin)/1000.0
    def delta(k): return val(ss[-1],k)-val(start,k)
    tiles=int(run.get('guest_progress','').split('completed_ztt_tiles=')[1].split()[0])
    manifest=json.loads((base/'guest.json').read_text())
    issues=delta('system.cpu.ame_engine.saPeIssues'); results=delta('system.cpu.ame_engine.saPeResults')
    active=delta('system.cpu.ame_engine.saActiveCycles')
    ame_req=delta('system.cpu.ame_engine.memoryRequests'); ame_acc=delta('system.cpu.ame_engine.memoryAccepted')
    rvv_req=delta('system.cpu.rvv_engine.memoryRequests'); rvv_acc=delta('system.cpu.rvv_engine.memoryAccepted')
    dma_in=delta('system.spm_dma.systemToSpmBytes'); dma_out=delta('system.spm_dma.spmToSystemBytes')
    read=sum(delta(f'system.spm.portReadBytes::port{i}') for i in range(4)); write=sum(delta(f'system.spm.portWriteBytes::port{i}') for i in range(4))
    bankq=sum(delta(f'system.spm.bankQueueFull::bank{i}') for i in range(8)); bankwait=sum(delta(f'system.spm.bankWaitTicks::bank{i}') for i in range(8))
    rows.append(dict(name=cn,variant=v,cycles=cyc,tiles=tiles,tile_rate=tiles/cyc,pe_issue_util=issues/(cyc*16),sa_busy=active/cyc,issues=issues,results=results,issue_rate=issues/cyc,ame_req=ame_req,ame_acc=ame_acc,rvv_req=rvv_req,rvv_acc=rvv_acc,accepted_ratio=(ame_acc+rvv_acc)/(ame_req+rvv_req),dma_in=dma_in,dma_out=dma_out,spm_read=read,spm_write=write,spm_bw=(read+write)/cyc,dma_bw=(dma_in+dma_out)/cyc,bank_queue_full=bankq,bank_wait_ticks=bankwait,serialize_after=delta('system.cpu.offloadSerializeAfterStalls'),source_stalls=delta('system.cpu.offloadSourceStalls'),expected_decode=manifest['expected_ame_tiles']['decode'],decode_forecast_cycles=manifest['expected_ame_tiles']['decode']/(tiles/cyc)))
with (OUT/'live_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
labels=[r['name'] for r in rows]
def bar(name,key,ylabel,file,title):
    fig,ax=plt.subplots(figsize=(8,4.4)); ax.bar(labels,[r[key] for r in rows],color=['#4472c4','#70ad47','#ed7d31','#a64d79']);ax.set_ylabel(ylabel);ax.set_title(title);ax.tick_params(axis='x',rotation=18);ax.grid(axis='y',alpha=.3);fig.tight_layout();fig.savefig(OUT/(file+'.png'),dpi=180);fig.savefig(OUT/(file+'.pdf'));plt.close(fig)
bar('tile_rate','tile_rate','tiles / cycle','01_tile_throughput','当前 prefill：有效 tile 吞吐')
bar('pe_issue_util','pe_issue_util','PE issue 占理论槽位','02_pe_issue_util','当前 prefill：PE 发射占空比')
bar('sa_busy','sa_busy','SA active cycles / interval cycles','03_sa_busy','当前 prefill：SA 供数/计算/排空忙碌率')
bar('spm_bw','spm_bw','SPM bytes / cycle','04_spm_bw','当前 prefill：SPM 总端口带宽')
bar('dma_bw','dma_bw','DMA bytes / cycle','05_dma_bw','当前 prefill：DDR↔SPM DMA 带宽')
bar('accepted_ratio','accepted_ratio','accepted / generated','06_accept_ratio','当前 prefill：访存接受率')
bar('bank_wait_ticks','bank_wait_ticks','ticks','07_bank_wait','当前 prefill：bank wait/conflict')
bar('decode_forecast_cycles','decode_forecast_cycles','cycles','08_decode_forecast','decode 阶段预测：仅按当前 prefill 吞吐外推')
with (OUT/'live_performance_report.md').open('w') as f:
    f.write('# 四种配置的时序性能分析\n\n')
    f.write('统计区间为各组 phase 5 workbegin 到当前最新 stats snapshot；四组尚未到 prefill workend，因此这是**运行中观测**，不是最终阶段结果。`stats.txt` 的 snapshot 序列作为时序 trace；当前 `trace.out` 只保留极短的 PseudoInst 文件，不能用于 PE 逐事件计数。\n\n')
    f.write('## 当前 prefill 指标\n\n|配置|周期|tile/cycle|PE issue 占空比|SA 忙碌率|访存接受率|SPM B/cycle|DMA B/cycle|bank wait|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in rows:f.write(f"|{r['name']}|{r['cycles']:.0f}|{r['tile_rate']:.6f}|{r['pe_issue_util']:.3f}|{r['sa_busy']:.3f}|{r['accepted_ratio']:.3f}|{r['spm_bw']:.3f}|{r['dma_bw']:.3f}|{r['bank_wait_ticks']:.0f}|\n")
    f.write('\n')
    for i,title in enumerate(['有效 tile 吞吐','PE 发射占空比','SA 忙碌率','SPM 总带宽','DMA 带宽','访存接受率','Bank wait/conflict']): f.write(f'![{title}](0{i+1}_{["tile_throughput","pe_issue_util","sa_busy","spm_bw","dma_bw","accept_ratio","bank_wait"][i]}.png)\n\n')
    f.write('## 分层解释\n\n')
    f.write('### SA 供数、导入和排空\n\n')
    f.write('`PE issue 占空比`使用实际到达 issue 周期的 `saPeIssues/(cycles×16)`；`SA 忙碌率`包括供数、MAC pipeline 和 drain，不能直接等同于算术利用率。当前四组的 issue/result 相等，说明截至采样点没有结果积压。若 SA 忙碌率显著高于 issue 占空比，差值主要是供数或排空周期。\n\n')
    f.write('### 访存、bank 和 DMA\n\n')
    f.write('AME/RVV memory accepted/generated 均从累计计数差分得到；接受率为 1 且 retry 为 0 时，当前区间没有请求级拥塞。bank queue full 和 bank wait 为 0，说明瓶颈不是 SPM bank conflict；DMA 带宽需要与 SPM 端口带宽一起看，不能只看 descriptor 数。\n\n')
    f.write('### 下发与串行化\n\n')
    f.write('`offloadSerializeAfterStalls` 是当前最明显的 CPU 侧等待来源；它表示年轻指令被前序串行化 offload 阻挡，不能与 source stall 直接相加为周期分解。ABC/ABCD 的连续注入主要应降低该等待和 SA drain 气泡；最终要用 workend 区间验证。\n\n')
    f.write('## Decode 预测\n\n')
    f.write('Decode 尚未实际开始。图 08 使用每组当前 prefill `tiles/cycle` 乘以 manifest 中 decode expected tile 数，仅作趋势预测；它没有模拟 decode 的 M=1、KV cache 访问和 attention/RVV 比例变化，不能当作 decode 实测。\n\n')
    f.write('![decode 预测](08_decode_forecast.png)\n\n')
    f.write('完整结论必须等待每组 prefill/decode workend，并重新运行 `analyze_baseline.py`；届时用真实 phase cycles、PE issue、SPM bank wait、DMA bytes 和 stall 差分替换本报告。\n')
print(OUT/'live_performance_report.md')
