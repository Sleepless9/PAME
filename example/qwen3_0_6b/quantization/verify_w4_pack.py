#!/usr/bin/env python3
"""Round-trip and tile tests for W4A8 int4 pack/unpack (matches w4_common.h)."""

import numpy as np

from w4_i4_codec import I4_MAX, pack_i4_row, unpack_i4_row, unpack_i4_rowmajor

TILE_OUTBLK = 64


def tile_pack_i8(arr_i8, m, n, gs):
    m_blocks = m // TILE_OUTBLK
    n_groups = n // gs
    tiled = (arr_i8.reshape(m_blocks, TILE_OUTBLK, n_groups, gs)
             .transpose(0, 2, 1, 3)
             .copy()
             .reshape(m, n))
    return tiled


def tile_unpack_i8(tiled_flat, m, n, gs):
    """Inverse of tile_pack_i8 for verification."""
    m_blocks = m // TILE_OUTBLK
    n_groups = n // gs
    arr = tiled_flat.reshape(m_blocks, n_groups, TILE_OUTBLK, gs)
    return arr.transpose(0, 2, 1, 3).reshape(m, n)


def test_all_nibbles():
    for v in range(-I4_MAX, I4_MAX + 1):
        lo = np.array([[v, 0]], dtype=np.int8)
        back = unpack_i4_row(pack_i4_row(lo), 2)
        assert back[0, 0] == v, f"lo nibble failed for {v}: got {back[0, 0]}"
        hi = np.array([[0, v]], dtype=np.int8)
        back = unpack_i4_row(pack_i4_row(hi), 2)
        assert back[0, 1] == v, f"hi nibble failed for {v}: got {back[0, 1]}"
    print("PASS all nibbles [-7, 7]")


def test_random_roundtrip():
    rng = np.random.default_rng(0)
    for gs in (64, 128):
        arr = rng.integers(-I4_MAX, I4_MAX + 1, size=(4, gs), dtype=np.int8)
        back = unpack_i4_row(pack_i4_row(arr), gs)
        assert np.array_equal(arr, back), f"round-trip failed gs={gs}\n{arr}\n{back}"
        print(f"PASS random round-trip gs={gs}")


def test_rowmajor_bytes():
    rng = np.random.default_rng(1)
    m, n, gs = 128, 64, 64
    arr = rng.integers(-I4_MAX, I4_MAX + 1, size=(m, n), dtype=np.int8)
    raw = pack_i4_row(arr).tobytes()
    back = unpack_i4_rowmajor(raw, m, n)
    assert np.array_equal(arr, back), "rowmajor bytes round-trip failed"
    print("PASS rowmajor bytes")


def test_i8_tile_roundtrip():
    rng = np.random.default_rng(2)
    m, n, gs = 128, 64, 64
    arr = rng.integers(-I4_MAX, I4_MAX + 1, size=(m, n), dtype=np.int8)
    tiled = tile_pack_i8(arr, m, n, gs)
    back = tile_unpack_i8(tiled, m, n, gs)
    assert np.array_equal(arr, back), "i8 tile layout round-trip failed"
    print("PASS i8 tile layout")


def main():
    test_all_nibbles()
    test_random_roundtrip()
    test_rowmajor_bytes()
    test_i8_tile_roundtrip()
    print("all pack/unpack checks passed")


if __name__ == "__main__":
    main()
