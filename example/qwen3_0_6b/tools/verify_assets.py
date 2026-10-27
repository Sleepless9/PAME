#!/usr/bin/env python3
import hashlib
import json
import struct
import sys
from pathlib import Path

MAGIC = 0x716F776E3362696E
EXPECTED = {
    "model/model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "model/qwen3-0.6b-int4-v5.bin": "eaf7e265ce47d6991946658a2c0f3b19c5775a40a37c081d97ab33747f6924de",
    "model/qwen3-0.6b-int4-tile-v6.bin": "2be6a748111407a0f3c6e351083ef24385641d4026d8517bd74f69c98822274a",
    "tokenizer/data/tokenizer.bin": "db52e8ce00466e4d34a187f47fe13a891a4f27ba25df4505a0706493c2afb402",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    for relative, expected in EXPECTED.items():
        path = root / relative
        actual = digest(path)
        if actual != expected:
            raise SystemExit(f"checksum mismatch: {relative}: {actual}")

    with (root / "model/qwen3-0.6b-int4-tile-v6.bin").open("rb") as stream:
        header = stream.read(struct.calcsize("<Q15iff"))
    values = struct.unpack("<Q15iff", header)
    magic, version = values[:2]
    hidden, layers, vocab, max_seq = values[2:6]
    q_heads, kv_heads, head_dim = values[6:9]
    intermediate = values[11]
    group_size, weight_bits, act_bits = values[13:16]
    if (magic, version, hidden, layers, vocab, q_heads, kv_heads, head_dim,
            intermediate, group_size, weight_bits, act_bits) != (
            MAGIC, 6, 1024, 28, 151936, 16, 8, 128, 3072, 128, 4, 8):
        raise SystemExit("unexpected Qwen3 W4A8 header")
    if max_seq != 40960:
        raise SystemExit(f"unexpected max_seq: {max_seq}")

    with (root / "tokenizer/data/tokenizer.bin").open("rb") as stream:
        tok_header = stream.read(20)
    if tok_header[:4] != b"TOKB":
        raise SystemExit("bad tokenizer magic")
    vocab_size, merges_count = struct.unpack_from("<II", tok_header, 4)
    # The binary BPE table contains normal vocabulary records. Qwen special
    # token IDs occupy the remaining model ID space up to config vocab_size.
    if vocab_size != 151643:
        raise SystemExit(f"unexpected tokenizer vocab: {vocab_size}")
    if merges_count != 151387:
        raise SystemExit(f"unexpected tokenizer merges: {merges_count}")

    with (root / "model/config.json").open() as stream:
        config = json.load(stream)
    if config["num_hidden_layers"] != 28 or config["vocab_size"] != 151936:
        raise SystemExit("config.json does not describe Qwen3-0.6B")

    with (root / "tokenizer/data/vocab.json").open() as stream:
        vocabulary = json.load(stream)
    if len(vocabulary) != 151643 or vocabulary.get("Hello") != 9707:
        raise SystemExit("tokenizer vocabulary does not map Hello to token 9707")
    with (root / "tokenizer/data/tokenizer.json").open() as stream:
        tokenizer = json.load(stream)
    if tokenizer.get("model", {}).get("vocab") != vocabulary:
        raise SystemExit("vocab.json does not match tokenizer.json model.vocab")
    if len(tokenizer.get("model", {}).get("merges", [])) != merges_count:
        raise SystemExit("tokenizer.bin does not contain every tokenizer.json merge")
    print("PASS: real safetensors, W4A8 v5/v6, tokenizer, and geometry verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
