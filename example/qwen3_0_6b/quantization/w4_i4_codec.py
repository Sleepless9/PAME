"""Int4 row pack/unpack — must match core/w4_common.h w4_unpack_nibble()."""

import numpy as np

I4_MAX = 7


def pack_i4_row(arr: np.ndarray) -> np.ndarray:
    """Pack int8 [m, n] (even n, values in [-7, 7]) -> uint8 [m, n // 2]."""
    m, n = arr.shape
    packed = np.zeros((m, n // 2), dtype=np.uint8)
    for k in range(0, n, 2):
        lo = arr[:, k].astype(np.int16) & 0x0F
        hi = arr[:, k + 1].astype(np.int16) & 0x0F
        packed[:, k // 2] = (lo | (hi << 4)).astype(np.uint8)
    return packed


def unpack_i4_row(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack uint8 [m, n // 2] -> int8 [m, n] (C w4_unpack_nibble semantics)."""
    m, _ = packed.shape
    out = np.zeros((m, n), dtype=np.int8)
    bi = packed.astype(np.uint8).astype(np.int8)
    for k in range(0, n, 2):
        b = bi[:, k // 2]
        out[:, k] = (b.astype(np.int16) << 4).astype(np.int8) >> 4
        out[:, k + 1] = b >> 4
    return out


def unpack_i4_rowmajor(buf_u8: bytes, m: int, n: int) -> np.ndarray:
    """Row-major packed bytes -> int8 [m, n]."""
    half_n = n // 2
    packed = np.frombuffer(buf_u8, dtype=np.uint8).reshape(m, half_n)
    return unpack_i4_row(packed, n)
