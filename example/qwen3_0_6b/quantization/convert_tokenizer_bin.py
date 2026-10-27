#!/usr/bin/env python3
"""Convert a Hugging Face tokenizer.json into the bare-metal TOKB image.

Usage:
    convert_tokenizer_bin.py INPUT_TOKENIZER_JSON OUTPUT_BIN

Binary format (little endian):
  magic[4] = "TOKB"
  uint32 vocab_size, merges_count, vocab_data_size, merges_data_size
  vocab_size records of uint32 token_id, string_offset, string_length
  NUL-terminated vocabulary strings
  merges_count records of uint32 string_offset, string_length
  NUL-terminated merge-pair strings
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source tokenizer.json")
    parser.add_argument("output", type=Path, help="destination TOKB binary")
    return parser.parse_args()


def merge_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(piece, str) for piece in value)
    ):
        return f"{value[0]} {value[1]}"
    raise ValueError(f"unsupported tokenizer merge record: {value!r}")


def append_string(blob: bytearray, value: str) -> tuple[int, int]:
    encoded = value.encode("utf-8")
    offset = len(blob)
    blob.extend(encoded)
    blob.append(0)
    return offset, len(encoded)


def main() -> int:
    args = parse_args()
    with args.input.open(encoding="utf-8") as stream:
        tokenizer = json.load(stream)

    model = tokenizer.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise SystemExit("tokenizer.json must contain model.vocab and model.merges")

    vocab_entries = sorted(vocab.items(), key=lambda item: item[1])
    ids = [token_id for _, token_id in vocab_entries]
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in ids):
        raise SystemExit("tokenizer vocabulary IDs must be non-negative integers")
    if len(set(ids)) != len(ids):
        raise SystemExit("tokenizer vocabulary contains duplicate token IDs")

    vocab_strings = bytearray()
    vocab_records: list[tuple[int, int, int]] = []
    for token, token_id in vocab_entries:
        if not isinstance(token, str):
            raise SystemExit("tokenizer vocabulary keys must be strings")
        offset, length = append_string(vocab_strings, token)
        vocab_records.append((token_id, offset, length))

    merge_strings = bytearray()
    merge_records: list[tuple[int, int]] = []
    for record in merges:
        offset, length = append_string(merge_strings, merge_text(record))
        merge_records.append((offset, length))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(b"TOKB")
        stream.write(
            struct.pack(
                "<IIII",
                len(vocab_records),
                len(merge_records),
                len(vocab_strings),
                len(merge_strings),
            )
        )
        for record in vocab_records:
            stream.write(struct.pack("<III", *record))
        stream.write(vocab_strings)
        for record in merge_records:
            stream.write(struct.pack("<II", *record))
        stream.write(merge_strings)

    size = args.output.stat().st_size
    print(f"Vocab: {len(vocab_records)} entries")
    print(f"Merges: {len(merge_records)} entries")
    print(f"Output: {args.output} ({size} bytes, {size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
