#!/usr/bin/env python3
"""Relocate the supplied FPGA ELF's stack without rebuilding its kernels.

Preserves every model byte. The matching source fix is in linker_boot/.
Only accepts the known startup instruction pair, and keeps the input intact.
"""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must differ")
    with args.input.open("rb") as src:
        header = src.read(64)
        if header[:6] != b"\x7fELF\x02\x01":
            raise ValueError("expected little-endian ELF64")
        phoff = struct.unpack_from("<Q", header, 32)[0]
        entsize, count = struct.unpack_from("<HH", header, 54)
        address = 0x864000F2
        offset = None
        for i in range(count):
            src.seek(phoff + i * entsize)
            kind, flags, off, va, pa, size, memsize, align = struct.unpack(
                "<IIQQQQQQ", src.read(56)
            )
            if kind == 1 and va <= address and address + 8 <= va + size:
                offset = off + address - va
        if offset is None:
            raise ValueError("startup not in a LOAD segment")
        src.seek(offset)
        expected = struct.pack("<II", 0x005FE217, 0x68D20213)
        if src.read(8) != expected:
            raise ValueError("unknown startup; refusing to patch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)
    # AUIPC tp + ADDI tp: tp=0x86300000, sp=tp+1MiB=0x86400000.
    replacement = struct.pack("<II", 0xFFF00217, 0xF0E20213)
    with args.output.open("r+b") as dst:
        dst.seek(offset)
        dst.write(replacement)
    manifest = dict(input=str(args.input.resolve()), output=str(args.output.resolve()),
                    patch_address=hex(address), before=expected.hex(),
                    after=replacement.hex(), stack_bottom="0x86300000",
                    stack_top="0x86400000")
    for name, path in (("input_sha256", args.input), ("output_sha256", args.output)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        manifest[name] = digest.hexdigest()
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
