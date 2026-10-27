#!/usr/bin/env python3
"""Repack W4A8 row-major blob into tile-major layout for AME offline backend.

Input  : qwen3-0.6b-int4.bin from convert_to_int4.py
Output : tile-major int8 weights + group-major scales (version 6)

v6 stores int8 tiles directly so the runtime skips per-mlbe8 int4 unpack.
Embed/lm_head-adjacent row weights that are not AME-tiled remain int4-packed.
"""

import argparse
import json
import struct
import numpy as np

from w4_i4_codec import unpack_i4_rowmajor

MAGIC = 0x716F776E3362696E
HEADER_FMT_V5 = "<Q15iff"
HEADER_SIZE = struct.calcsize(HEADER_FMT_V5)
TILE_OUTBLK = 64
FORMAT_VERSION = 6
PAYLOAD_ALIGN = 64


def tile_pack_i8(arr_i8, m, n, gs):
    """Tile-major int8 layout: [m/64][n/gs][64][gs]."""
    assert m % TILE_OUTBLK == 0
    assert n % gs == 0
    m_blocks = m // TILE_OUTBLK
    n_groups = n // gs
    tiled = (arr_i8.reshape(m_blocks, TILE_OUTBLK, n_groups, gs)
                .transpose(0, 2, 1, 3)
                .copy()
                .reshape(m, n))
    return tiled.tobytes()


def scale_pack_group_major(buf_f32, m, n, gs):
    """Pack scales as [n/gs][m], contiguous over output rows for RVV accum."""
    assert m % TILE_OUTBLK == 0
    assert n % gs == 0
    n_groups = n // gs
    arr = np.frombuffer(buf_f32, dtype=np.float32).reshape(m, n_groups)
    packed = arr.transpose(1, 0).copy().reshape(m * n_groups)
    assert (m * 4) % PAYLOAD_ALIGN == 0
    return packed.tobytes()


def align_up(value, align):
    return (value + align - 1) // align * align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    q_heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // q_heads)
    intermediate = cfg["intermediate_size"]
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    with open(args.input, "rb") as f:
        blob = f.read()

    header = blob[:HEADER_SIZE]
    unpacked = list(struct.unpack(HEADER_FMT_V5, header))
    assert unpacked[0] == MAGIC
    assert unpacked[1] in (5, 6), "input header must be v5"
    assert unpacked[14] == 4, "expected weight_bits=4"
    has_qk_norm = bool(unpacked[12])
    gs = unpacked[13]

    out = bytearray()
    hdr = list(unpacked)
    hdr[1] = FORMAT_VERSION
    out += struct.pack(HEADER_FMT_V5, *hdr)
    out += b"\0" * (align_up(HEADER_SIZE, PAYLOAD_ALIGN) - HEADER_SIZE)
    p = align_up(HEADER_SIZE, PAYLOAD_ALIGN)

    def take_q4(m, n, name, do_tile):
        nonlocal p
        ng = n // gs
        scales_sz = m * ng * 4
        data_sz = m * (n // 2)
        scales = blob[p:p + scales_sz]
        p += scales_sz
        data = blob[p:p + data_sz]
        p += data_sz
        if do_tile:
            arr = unpack_i4_rowmajor(data, m, n)
            out.extend(scale_pack_group_major(scales, m, n, gs))
            out.extend(tile_pack_i8(arr, m, n, gs))
            assert len(out) % PAYLOAD_ALIGN == 0
            tag = "i8tile+gscale-gm"
            data_out = m * n
        else:
            out.extend(scales)
            out.extend(data)
            assert len(out) % PAYLOAD_ALIGN == 0
            tag = "i4row "
            data_out = data_sz
        print(f"  q4 {tag} {name}: [{m}, {n}] scales={scales_sz:,}B data={data_out:,}B")

    def take_f32(m, name):
        nonlocal p
        sz = m * 4
        out.extend(blob[p:p + sz])
        p += sz
        assert len(out) % PAYLOAD_ALIGN == 0
        print(f"  f32 {name}: [{m}]")

    take_q4(vocab, hidden, "embed_tokens", do_tile=False)
    qkv_dim = q_dim + 2 * kv_dim
    for l in range(n_layers):
        take_f32(hidden, f"rms_att.{l}")
        take_q4(qkv_dim, hidden, f"wqkv.{l}", do_tile=True)
        if has_qk_norm:
            take_f32(head_dim, f"q_norm.{l}")
            take_f32(head_dim, f"k_norm.{l}")
        take_q4(hidden, q_dim, f"wo.{l}", do_tile=True)
        take_f32(hidden, f"rms_ffn.{l}")
        take_q4(intermediate, hidden, f"w1_gate.{l}", do_tile=True)
        take_q4(intermediate, hidden, f"w3_up.{l}", do_tile=True)
        take_q4(hidden, intermediate, f"w2_down.{l}", do_tile=True)
    take_f32(hidden, "rms_final")
    take_q4(vocab, hidden, "lm_head", do_tile=True)

    if p != len(blob):
        raise RuntimeError(f"blob walk inconsistent: parsed {p}, expected {len(blob)}")

    with open(args.output, "wb") as f:
        f.write(out)
    print(f"\nwrote {args.output} ({len(out):,} bytes, v{FORMAT_VERSION} int8 tiles)")


if __name__ == "__main__":
    main()
