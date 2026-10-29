#!/usr/bin/env python3
"""Build a random-ID prefill/decode entry using the supplied ELF's kernels.

Missing kernel/tokenizer sources are not reconstructed. Original executable
segments and weights are retained, and a newly compiled model frontend is
added below the original image. The generated linker map records every ABI
symbol resolved from the original ELF. Requires stock RV64GCV clang and lld.
"""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path


def elf(path):
    with path.open('rb') as f:
        header = bytearray(f.read(64))
        assert header[:6] == b'\x7fELF\x02\x01'
        phoff = struct.unpack_from('<Q', header, 32)[0]
        entsize, count = struct.unpack_from('<HH', header, 54)
        segments = []
        for i in range(count):
            f.seek(phoff + i * entsize)
            segments.append(struct.unpack('<IIQQQQQQ', f.read(56)))
    return header, segments


def run(args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--clang', default='/usr/bin/clang-20')
    parser.add_argument('--lld', default='/home/xiangyy/111_research/install/bin/ld.lld')
    parser.add_argument('--dma', action='store_true', help='Stage every Ztt tile through DMA/SPM')
    parser.add_argument('--tile-test', action='store_true')
    parser.add_argument('--packed', action='store_true', help='Use all four packed squares and accumulate whole quantization groups')
    parser.add_argument('--backend-test', action='store_true')
    args = parser.parse_args()
    if args.packed or args.backend_test:
        args.dma = True
        args.packed = True
    args.input, args.output = args.input.resolve(), args.output.resolve()
    if args.input == args.output:
        parser.error('input and output must differ')
    here = Path(__file__).resolve().parent
    core, boot = here.parent / 'core', here.parent / 'linker_boot'
    build = args.output.parent / (args.output.stem + '_build')
    build.mkdir(parents=True, exist_ok=True)
    source = (core / 'qwen3_bare_w4a8_prefill_all.c').read_text()
    source = source[:source.index('\n#ifndef CHAT_BUILD\nenum Gem5ProfilePhase')]
    for inc in ('../platform/uart_helper.c', 'backends/ops_lib.h',
                'w4a8/matmul_quant_scalar_w4.h'):
        source = source.replace('#include "' + inc + '"', '')
    start = source.index('#include "../tokenizer/tokenizer_bare.h"')
    end = source.index('static float silu(float x)')
    source = source[:start] + source[end:]
    source = source.replace('g_timing.n_layers++;',
        'g_timing.n_layers++; print_uart("DECODE layer_complete="); '
        'print_uart_int_dec(l + 1); print_uart("\\r\\n");')
    source = source.replace('g_timing.n_layers += n_tokens;',
        'g_timing.n_layers += n_tokens; print_uart("PREFILL layer_complete="); '
        'print_uart_int_dec(l + 1); print_uart("\\r\\n");')
    source = '#include "gem5_kernel_abi.h"\n' + source
    source = source.replace('for (;;) { __asm__ volatile ("" ::: "memory"); }',
                            'm5_fail(0, 10); for (;;) __asm__ volatile("wfi");')
    source += '\n#include "gem5_random_main.c"\n'
    (build / 'frontend.c').write_text(source)
    flags = [args.clang, '--target=riscv64-unknown-elf', '-march=rv64gcv',
             '-mabi=lp64d', '-mcmodel=medany', '-O3', '-ffreestanding',
             '-fno-builtin', '-fno-vectorize', '-fno-slp-vectorize',
             '-fno-pie', '-ffunction-sections', '-fdata-sections',
             '-DUSE_AME=1', '-DFPGA_CLOCK_HZ=1000000000',
             '-I' + str(core), '-I' + str(boot), '-I' + str(here)]
    if args.tile_test:
        flags.append('-DQWEN_TILE_TEST=1')
    if args.backend_test:
        flags.append('-DQWEN_BACKEND_TEST=1')
    run(flags + ['-c', build / 'frontend.c', '-o', build / 'frontend.o'])
    run(flags + ['-c', boot / 'crt_uart.S', '-o', build / 'crt.o'])
    objects = [build / 'crt.o', build / 'frontend.o']
    if args.dma:
        _, original_segments = elf(args.input)
        for kind, perms, offset, va, pa, size, memsize, align in original_segments:
            if kind == 1 and va <= 0x8640c61c < va + size:
                with args.input.open('rb') as f:
                    f.seek(offset + 0x8640c61c - va)
                    kernel = f.read(68)
        assert kernel[:8] == bytes.fromhex('1303c003ab0393a4')
        (build / 'direct.S').write_text(
            '.section .text.qwen_ztt_direct,"ax"\n.globl qwen_ztt_direct\n'
            'qwen_ztt_direct:\n.byte ' + ','.join(str(b) for b in kernel) + '\n')
        for src, obj in [(here / 'gem5_dma_tile.c', build / 'dma.o'),
                         (build / 'direct.S', build / 'direct.o')]:
            run(flags + ['-c', src, '-o', obj])
            objects.append(obj)
        if args.packed:
            for name in ('gem5_packed_backend.c', 'gem5_ztt_group.S'):
                obj = build / (Path(name).stem + '.o')
                run(flags + ['-c', here / name, '-o', obj])
                objects.append(obj)
    nm = subprocess.check_output(['llvm-nm', args.input], text=True)
    symbols = {}
    for line in nm.splitlines():
        fields = line.split()
        if len(fields) == 3:
            symbols[fields[2]] = int(fields[0], 16)
    provided = '\n'.join('PROVIDE(%s = %#x);' % (name, address)
                         for name, address in symbols.items()
                         if name not in ('_start', '_init', 'exit', '_exit',
                                         '__global_pointer$', '__stack_bottom',
                                         '__stack_top'))
    script = '''OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); }
SECTIONS {
  . = 0x86000000;
  .text : { *(.text.init) *(.text .text.*) } :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } :text
  . = ALIGN(4096);
  .data : { *(.data .data.* .sdata .sdata.*) } :data
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } :data
  __global_pointer$ = ADDR(.data) + 0x800;
  __stack_bottom = 0x86300000;
  __stack_top = 0x86400000;
  ASSERT(. < __stack_bottom, "Frontend overlaps stack")
}
'''
    (build / 'frontend.ld').write_text(script + provided)
    run([args.lld, '-m', 'elf64lriscv', '--gc-sections', '--no-relax',
         '-T', build / 'frontend.ld', '-Map=' + str(build / 'frontend.map'),
         *(['--undefined=qwen_dma_tile'] if args.dma else []),
         *objects, '-o', build / 'frontend.elf'])
    header, segments = elf(args.input)
    new_header, additions = elf(build / 'frontend.elf')
    shutil.copyfile(args.input, args.output)
    with args.output.open('r+b') as output, (build / 'frontend.elf').open('rb') as front:
        if args.dma:
            nm_front = subprocess.check_output(['llvm-nm', build / 'frontend.elf'], text=True)
            target = next(int(line.split()[0], 16) for line in nm_front.splitlines()
                          if line.endswith(' T qwen_dma_tile'))
            delta = target - 0x8640c61c
            upper = (delta + 0x800) >> 12
            lower = delta - (upper << 12)
            # AUIPC t0; JALR zero,t0: tail call preserves the original ra.
            patch = struct.pack('<II', ((upper & 0xfffff) << 12) | (5 << 7) | 0x17,
                                ((lower & 0xfff) << 20) | (5 << 15) | 0x67)
            for kind, perms, offset, va, pa, size, memsize, align in segments:
                if kind == 1 and va <= 0x8640c61c < va + size:
                    output.seek(offset + 0x8640c61c - va)
                    output.write(patch)
        output.seek(0, 2)
        for segment in additions:
            kind, perms, offset, va, pa, size, memsize, align = segment
            if kind != 1:
                continue
            position = (output.tell() + 4095) & -4096
            output.seek(position)
            front.seek(offset)
            output.write(front.read(size))
            segments.append((kind, perms, position, va, pa, size, memsize, 4096))
        phoff = (output.tell() + 7) & -8
        output.seek(phoff)
        for segment in segments:
            output.write(struct.pack('<IIQQQQQQ', *segment))
        header[24:32] = new_header[24:32]
        struct.pack_into('<Q', header, 32, phoff)
        struct.pack_into('<HH', header, 54, 56, len(segments))
        output.seek(0)
        output.write(header)
    manifest = dict(input=str(args.input), output=str(args.output),
                    frontend=str(build / 'frontend.elf'),
                    prefill_tokens=0 if args.tile_test or args.backend_test else 1,
                    decode_forwards=0 if args.tile_test or args.backend_test else 1, seed=20260907,
                    tile_test=args.tile_test,
                    packed_backend=args.packed, backend_test=args.backend_test,
                    original_load_segments_preserved=not args.dma,
                    original_weights_preserved=True, dma_staging=args.dma)
    for key, path in [('input_sha256', args.input), ('output_sha256', args.output)]:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(1048576): digest.update(block)
        manifest[key] = digest.hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
