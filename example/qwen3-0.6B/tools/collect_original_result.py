#!/usr/bin/env python3
"""Collect completion evidence from an already running original-ELF simulation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', type=Path)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    while True:
        log = args.log.read_text(errors='replace')
        exit_event = re.search(r'Exiting @ tick (\d+) because (.+) \(code (\d+)\)', log)
        error = re.search(r'Program aborted|\n(?:IndexError|RuntimeError|ValueError|TypeError):', log)
        uart_file = args.outdir / 'system.platform.terminal'
        uart = uart_file.read_text(errors='replace') if uart_file.exists() else ''
        layers = re.search(r'layer_passes=(\d+)', uart)
        report = dict(source_log=str(args.log.resolve()), updated_unix=time.time(),
                      status='running_or_incomplete', guest_done='=== done ===' in uart,
                      layer_passes=int(layers.group(1)) if layers else None,
                      last_uart_lines=uart.splitlines()[-8:])
        if exit_event or error:
            report['exit_event'] = exit_event.groups() if exit_event else None
            plain_trace = args.outdir / 'trace.out'
            if plain_trace.is_file():
                report['trace_format'] = 'plain'
                report['trace_available'] = True
            else:
                report['trace_format'] = 'gzip_legacy'
                try:
                    with gzip.open(args.outdir / 'trace.out.gz', 'rb') as stream:
                        while stream.read(1024 * 1024):
                            pass
                    report['trace_available'] = True
                except (OSError, EOFError):
                    report['trace_available'] = False
            binary = Path(__file__).resolve().parents[3] / 'example/qwen3_chat_w4a8_ame_fpga'
            digest = hashlib.sha256()
            with binary.open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            report['input_sha256'] = digest.hexdigest()
            report['input_matches_original'] = digest.hexdigest() == \
                '39b8e434f8d48ef57c1623f9920bba2b0adf03d84139b01da165abef89cba67c'
            passed = (exit_event and exit_event.group(2) == 'm5_exit instruction encountered'
                      and exit_event.group(3) == '0' and report['guest_done']
                      and report['layer_passes'] == 28 and report['trace_available']
                      and report['input_matches_original'])
            report['status'] = 'passed' if passed else 'failed_or_incomplete'
        (args.outdir / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
        if not args.follow or exit_event or error:
            print(json.dumps(report, indent=2))
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
