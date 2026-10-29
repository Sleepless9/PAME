#!/usr/bin/env python3
"""Run the supplied ELF and record completion evidence (never infer PASS from ticks)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--cpu-mode', choices=('minor', 'atomic'), default='minor')
    parser.add_argument('--restore-checkpoint', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    binary = root / 'example/qwen3_chat_w4a8_ame_fpga'
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run.json').exists():
        parser.error('outdir already contains a run; use a new directory')
    flags = 'PseudoInst' if args.cpu_mode == 'atomic' else 'AMEFlow,PseudoInst,SPMVirtDma,BankedSPM'
    cmd = [str(root / 'build/RISCV/gem5.opt'), '--listener-mode=off', '-d', str(out),
           '--debug-flags=' + flags, '--debug-file=trace.out',
           str(root / 'configs/example/spm_compute_soc.py'), str(binary),
           '--cpu-mode=' + args.cpu_mode,
           '--checkpoint-before-prefill=' + str(out / 'checkpoint.prefill')]
    if args.restore_checkpoint:
        cmd.append('--restore-checkpoint=' + str(args.restore_checkpoint.resolve()))
    report = dict(command=cmd, binary=str(binary), input_sha256=digest(binary),
                  cpu_mode=args.cpu_mode, status='running', started_unix=time.time())
    def save():
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    with (out / 'run.log').open('w') as log:
        child = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        report['pid'] = child.pid
        save()
        result = child.wait()
    uart_path = out / 'system.platform.terminal'
    uart = uart_path.read_text(errors='replace') if uart_path.exists() else ''
    log = (out / 'run.log').read_text(errors='replace')
    layers = re.search(r'layer_passes=(\d+)', uart)
    report.update(returncode=result, ended_unix=time.time(),
                  input_unchanged=digest(binary) == report['input_sha256'],
                  guest_done='=== done ===' in uart,
                  layer_passes=int(layers.group(1)) if layers else None,
                  exit_instruction='m5_exit instruction encountered (code 0)' in log)
    report['trace_exists'] = (out / 'trace.out').is_file()
    passed = (result == 0 and report['input_unchanged'] and report['guest_done']
              and report['layer_passes'] == 28 and report['exit_instruction']
              and report['trace_exists'])
    report['status'] = 'passed' if passed else 'failed_or_incomplete'
    save()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
