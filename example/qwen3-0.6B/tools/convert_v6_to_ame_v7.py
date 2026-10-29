#!/usr/bin/env python3
"""Repack a Qwen W4A8 v6 blob into the DMA-ready AME v7 weight layout.

The numerical int8 weights and all scale tables are unchanged.  Only matrix
payloads move from v6 [out/64][group][out%64][K] to
v7 [out/64][out4][group][K][lane].  A 4-output-row, K=128 packet is therefore
one contiguous 512-byte DMA transfer in the exact order consumed by AME.
"""
import argparse
from pathlib import Path
import shutil
import struct

import numpy as np


HEADER_BYTES = 68
ALIGNED_HEADER_BYTES = 128


def copy_bytes(src, dst, count, chunk=16 * 1024 * 1024):
    while count:
        data = src.read(min(count, chunk))
        if not data:
            raise EOFError("truncated source model")
        dst.write(data)
        count -= len(data)


def repack_matrix(src, dst, output_rows, input_cols, group_size, label):
    if output_rows % 64 or input_cols % group_size or group_size != 128:
        raise ValueError(f"{label}: v7 requires output%64=0 and K-group=128")
    groups = input_cols // group_size
    block_bytes = groups * 64 * group_size
    for _ in range(output_rows // 64):
        raw = src.read(block_bytes)
        if len(raw) != block_bytes:
            raise EOFError(f"truncated {label}")
        v6 = np.frombuffer(raw, dtype=np.int8).reshape(groups, 64, 128)
        # [group][row4][lane][K] -> [row4][group][K][lane]
        v7 = v6.reshape(groups, 16, 4, 128).transpose(1, 0, 3, 2)
        dst.write(np.ascontiguousarray(v7).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must differ")

    with args.source.open("rb") as src, args.output.open("wb") as dst:
        header = bytearray(src.read(HEADER_BYTES))
        if len(header) != HEADER_BYTES:
            raise EOFError("truncated header")
        values = struct.unpack_from("<15i", header, 8)
        (version, hidden, layers, vocab, _max_seq, _qh, _kvh, _hd,
         qdim, kvdim, intermediate, has_qk_norm, group_size,
         weight_bits, act_bits) = values
        if version != 6 or weight_bits != 4 or act_bits != 8:
            raise ValueError(f"expected W4A8 v6, got version={version}")
        struct.pack_into("<i", header, 8, 7)
        dst.write(header)
        copy_bytes(src, dst, ALIGNED_HEADER_BYTES - HEADER_BYTES)

        ng_h = hidden // group_size
        ng_qd = qdim // group_size
        ng_inter = intermediate // group_size
        qkv_dim = qdim + 2 * kvdim

        # Embedding remains packed int4 and is not an AME matrix payload.
        copy_bytes(src, dst, vocab * ng_h * 4 + vocab * (hidden // 2))
        for layer in range(layers):
            copy_bytes(src, dst, hidden * 4)  # attention RMS
            copy_bytes(src, dst, qkv_dim * ng_h * 4)
            repack_matrix(src, dst, qkv_dim, hidden, group_size,
                          f"layer{layer}.wqkv")
            if has_qk_norm:
                copy_bytes(src, dst, 2 * 128 * 4)
            copy_bytes(src, dst, hidden * ng_qd * 4)
            repack_matrix(src, dst, hidden, qdim, group_size,
                          f"layer{layer}.wo")
            copy_bytes(src, dst, hidden * 4)  # FFN RMS
            for name in ("w1", "w3"):
                copy_bytes(src, dst, intermediate * ng_h * 4)
                repack_matrix(src, dst, intermediate, hidden, group_size,
                              f"layer{layer}.{name}")
            copy_bytes(src, dst, hidden * ng_inter * 4)
            repack_matrix(src, dst, hidden, intermediate, group_size,
                          f"layer{layer}.w2")
            print(f"repacked layer {layer + 1}/{layers}", flush=True)

        copy_bytes(src, dst, hidden * 4)  # final RMS
        copy_bytes(src, dst, vocab * ng_h * 4)
        repack_matrix(src, dst, vocab, hidden, group_size, "lm_head")
        if src.read(1):
            raise ValueError("unexpected trailing bytes")

    if args.output.stat().st_size != args.source.stat().st_size:
        raise AssertionError("layout conversion changed file size")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
