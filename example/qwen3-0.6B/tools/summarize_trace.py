#!/usr/bin/env python3
"""Stream AME/DMA/SPM trace into latency summaries and a compact event CSV."""
import argparse
import collections
import csv
import gzip
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all-events', action='store_true',
                        help='Export every parsed event, including AME instruction flow')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hist = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    fields_re = re.compile(r'(\w+)=([^\s]+)')
    complete = True
    lines = 0
    last_tick = 0
    compressed = args.trace.suffix == '.gz'
    open_trace = gzip.open if compressed else open
    with open_trace(args.trace, 'rt', errors='replace') as source, \
         (args.output / 'events.csv').open('w') as dest:
        writer = csv.writer(dest)
        writer.writerow(['tick', 'component', 'event', 'fields'])
        try:
            for line in source:
                lines += 1
                match = re.match(r'\s*(\d+): ([^:]+): (.*)', line)
                if not match:
                    continue
                tick, component, message = match.groups()
                last_tick = int(tick)
                fields = dict(fields_re.findall(message))
                event_match = re.search(r'\b(SUPPLY_STALL_BEGIN|SUPPLY_STALL_END|'
                    r'CHUNK_DONE|SYSTEM_READ_DONE|SPM_REQUEST|SPM_RESPONSE|'
                    r'BANK|COMPLETE|RESPONSE|STALL|KSTEP_INJECT|ACC_FINAL_GRANT)\b', message)
                words = [word for word in message.split() if '=' not in word]
                event = event_match.group(1) if event_match else (
                    words[0] if words else 'fields_only')
                if args.all_events or 'spm_dma' in component or \
                        component == 'system.spm' or component.startswith('system.spm.') or \
                        event in ('SUPPLY_STALL_BEGIN', 'SUPPLY_STALL_END', 'ACC_FINAL_GRANT'):
                    writer.writerow([tick, component, event, json.dumps(fields, sort_keys=True)])
                counts[component + ':' + event] += 1
                for selected, field, label in [
                    ('AME_COMPLETE', 'latency', 'ame_instruction_cycles'),
                    ('CHUNK_DONE', 'latency', 'dma_chunk_ticks'),
                    ('SYSTEM_READ_DONE', 'latency', 'dma_system_read_ticks'),
                    ('SPM_RESPONSE', 'latency', 'dma_spm_ticks'),
                    ('BANK', 'wait', 'spm_bank_wait_ticks'),
                    ('SUPPLY_STALL_END', 'cycles', 'ame_supply_stall_cycles')]:
                    if event == selected and field in fields:
                        hist[label][int(fields[field], 0)] += 1
        except (EOFError, gzip.BadGzipFile):
            complete = False
    summaries = {}
    for label, values in hist.items():
        count = sum(values.values())
        def percentile(p):
            target = max(1, (count * p + 99) // 100)
            cumulative = 0
            for value, freq in sorted(values.items()):
                cumulative += freq
                if cumulative >= target:
                    return value
        summaries[label] = dict(count=count, min=min(values), max=max(values),
            mean=sum(v * n for v, n in values.items()) / count,
            p50=percentile(50), p95=percentile(95), p99=percentile(99))
    report = dict(trace=str(args.trace.resolve()),
                  gzip_complete=complete if compressed else None,
                  trace_read_complete=complete, all_events=args.all_events,
                  lines=lines, last_tick=last_tick, latency=summaries,
                  events=dict(counts))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
