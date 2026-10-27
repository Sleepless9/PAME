#!/usr/bin/env python3
"""Convert Qwen3-0.6B from safetensors to W4A8 quantized bin (version 5).

Weights: per-group symmetric INT4 in [-7, 7], packed 2 per byte.
Activations: runtime int8 (unchanged from W8A8 path).

Layout per quantized matrix [m, n] (n_groups = n / group_size):
  float32 scales[m * n_groups]
  uint8   data[m * n / 2]   (packed int4 along K)
"""

import argparse
import json
import os
import struct
import numpy as np
import torch
from safetensors import safe_open

MAGIC = 0x716F776E3362696E
DEFAULT_GROUP_SIZE = 128
HEADER_FMT_V5 = "<Q15iff"
FORMAT_VERSION = 5
WEIGHT_BITS = 4
ACT_BITS = 8
PAYLOAD_ALIGN = 64
I4_MAX = 7


def _pack_i4_rows(quantized):
    m, n = quantized.shape
    q = quantized.astype(np.int16) & 0x0F
    return (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8).reshape(m, n // 2)


def _quantize_grouped_max(grouped, group_absmax):
    quantized = np.clip(
        np.round(grouped / group_absmax[:, :, None] * float(I4_MAX)),
        -I4_MAX, I4_MAX,
    ).astype(np.int8)
    scales = (group_absmax / float(I4_MAX)).astype(np.float32)
    return scales, quantized


def _quantize_grouped_mse(grouped, group_absmax, mse_grid):
    best_clip = group_absmax.copy()
    best_err = np.full(group_absmax.shape, np.inf, dtype=np.float32)

    for shrink in np.linspace(1.0, 0.5, int(mse_grid), dtype=np.float32):
        clip = np.maximum(group_absmax * shrink, 1e-10)
        scale = clip / float(I4_MAX)
        q = np.clip(np.round(grouped / scale[:, :, None]), -I4_MAX, I4_MAX)
        deq = q * scale[:, :, None]
        err = np.mean((deq - grouped) * (deq - grouped), axis=2)
        better = err < best_err
        best_err[better] = err[better]
        best_clip[better] = clip[better]

    scales = (best_clip / float(I4_MAX)).astype(np.float32)
    quantized = np.clip(
        np.round(grouped / scales[:, :, None]),
        -I4_MAX, I4_MAX,
    ).astype(np.int8)
    return scales, quantized


def quantize_per_group_int4(tensor, group_size, scale_method="mse",
                            mse_grid=16, chunk_rows=1024):
    arr = tensor.to(torch.float32).numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    m, n = arr.shape
    assert n % group_size == 0, f"n={n} not divisible by group_size={group_size}"
    assert group_size % 2 == 0, "group_size must be even for int4 packing"
    n_groups = n // group_size
    if scale_method not in ("max", "mse"):
        raise ValueError(f"unsupported scale_method={scale_method}")
    if mse_grid < 2:
        raise ValueError("mse_grid must be >= 2")

    scales = np.empty((m, n_groups), dtype=np.float32)
    packed = np.empty((m, n // 2), dtype=np.uint8)

    for r0 in range(0, m, chunk_rows):
        r1 = min(r0 + chunk_rows, m)
        grouped = arr[r0:r1].reshape(r1 - r0, n_groups, group_size)
        group_absmax = np.abs(grouped).max(axis=2)
        group_absmax = np.maximum(group_absmax, 1e-10)
        if scale_method == "mse":
            sc, q = _quantize_grouped_mse(grouped, group_absmax, mse_grid)
        else:
            sc, q = _quantize_grouped_max(grouped, group_absmax)
        scales[r0:r1] = sc
        packed[r0:r1] = _pack_i4_rows(q.reshape(r1 - r0, n))

    return scales.reshape(-1), packed


def smooth_linear_inputs(norm_weight, tensors, alpha, max_scale):
    if alpha <= 0.0:
        return norm_weight, tensors, None
    col_abs = None
    for tensor in tensors:
        cur = tensor.to(torch.float32).abs().amax(dim=0)
        col_abs = cur if col_abs is None else torch.maximum(col_abs, cur)
    act_abs = norm_weight.to(torch.float32).abs()
    col_abs = torch.clamp(col_abs, min=1e-6)
    act_abs = torch.clamp(act_abs, min=1e-6)

    scale = act_abs.pow(alpha) / col_abs.pow(1.0 - alpha)
    scale = scale / torch.exp(torch.mean(torch.log(torch.clamp(scale, min=1e-6))))
    scale = torch.clamp(scale, min=1.0 / max_scale, max=max_scale)

    norm_out = norm_weight.to(torch.float32) / scale
    tensors_out = [tensor.to(torch.float32) * scale.reshape(1, -1)
                   for tensor in tensors]
    return norm_out, tensors_out, scale


def parse_args():
    ap = argparse.ArgumentParser(description="Convert Qwen3 safetensors to W4A8 bin.")
    ap.add_argument("--input", default="model.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--output", default="qwen3-0.6b-int4.bin")
    ap.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    ap.add_argument("--scale-method", choices=("max", "mse"), default="mse")
    ap.add_argument("--mse-grid", type=int, default=16)
    ap.add_argument("--chunk-rows", type=int, default=1024)
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--smooth-max-scale", type=float, default=8.0)
    return ap.parse_args()


def align_up(value, align):
    return (value + align - 1) // align * align


def main():
    args = parse_args()
    group_size = args.group_size
    if group_size <= 0 or group_size % 2 != 0:
        raise ValueError("group_size must be > 0 and even")
    if args.chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    if args.smooth_alpha < 0.0 or args.smooth_alpha > 1.0:
        raise ValueError("smooth_alpha must be in [0, 1]")
    if args.smooth_max_scale < 1.0:
        raise ValueError("smooth_max_scale must be >= 1")

    with open(args.config) as f:
        cfg = json.load(f)

    hidden = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    max_seq = cfg["max_position_embeddings"]
    q_heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // q_heads)
    intermediate = cfg["intermediate_size"]
    rms_eps = float(cfg["rms_norm_eps"])
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    tie_word_embeddings = cfg.get("tie_word_embeddings", False)

    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    print(f"Model: hidden={hidden} layers={n_layers} vocab={vocab}")
    print(f"  W4A8 group_size={group_size} scale_method={args.scale_method}")
    if args.scale_method == "mse":
        print(f"  mse_grid={args.mse_grid} chunk_rows={args.chunk_rows}")
    if args.smooth_alpha > 0.0:
        print(f"  smooth_alpha={args.smooth_alpha} smooth_max_scale={args.smooth_max_scale}")

    for name, dim in [("hidden", hidden), ("q_dim", q_dim), ("kv_dim", kv_dim),
                       ("intermediate", intermediate), ("qkv_dim", q_dim + 2 * kv_dim)]:
        assert dim % group_size == 0, f"{name}={dim} not divisible by {group_size}"

    st = safe_open(args.input, framework="pt")
    keys = st.keys()
    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in keys

    with open(args.output, "wb") as f:
        header = struct.pack(
            HEADER_FMT_V5,
            MAGIC,
            FORMAT_VERSION,
            hidden,
            n_layers,
            vocab,
            max_seq,
            q_heads,
            kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            intermediate,
            1 if has_qk_norm else 0,
            group_size,
            WEIGHT_BITS,
            ACT_BITS,
            rms_eps,
            rope_theta,
        )
        f.write(header)
        header_pad = align_up(len(header), PAYLOAD_ALIGN) - len(header)
        if header_pad:
            f.write(b"\0" * header_pad)

        def write_f32(tensor, name=""):
            arr = tensor.to(torch.float32).numpy()
            f.write(arr.tobytes())
            if name:
                print(f"  f32  {name}: {list(arr.shape)}")

        def write_q4(tensor, name=""):
            scales, qdata = quantize_per_group_int4(
                tensor,
                group_size=group_size,
                scale_method=args.scale_method,
                mse_grid=args.mse_grid,
                chunk_rows=args.chunk_rows,
            )
            f.write(scales.tobytes())
            f.write(qdata.tobytes())
            m, half_n = qdata.shape
            n = half_n * 2
            ng = n // group_size
            if name:
                print(f"  int4 {name}: [{m}, {n}] scales:[{m}x{ng}] "
                      f"data:{m * half_n:,}B packed")

        def smooth_and_log(norm_tensor, tensors, name):
            norm_out, tensors_out, scale = smooth_linear_inputs(
                norm_tensor, tensors, args.smooth_alpha, args.smooth_max_scale)
            if scale is not None:
                print(f"  smooth {name}: scale min={float(scale.min()):.6g} "
                      f"max={float(scale.max()):.6g}")
            return norm_out, tensors_out

        embed = st.get_tensor("model.embed_tokens.weight")
        write_q4(embed, "embed_tokens")

        for l in range(n_layers):
            p = f"model.layers.{l}."
            rms_att = st.get_tensor(p + "input_layernorm.weight")
            wq = st.get_tensor(p + "self_attn.q_proj.weight")
            wk = st.get_tensor(p + "self_attn.k_proj.weight")
            wv = st.get_tensor(p + "self_attn.v_proj.weight")
            wqkv = torch.cat([wq, wk, wv], dim=0)
            rms_att, (wqkv,) = smooth_and_log(rms_att, [wqkv], f"qkv.{l}")
            write_f32(rms_att, f"rms_att.{l}")
            write_q4(wqkv, f"wqkv.{l}")

            if has_qk_norm:
                write_f32(st.get_tensor(p + "self_attn.q_norm.weight"), f"q_norm.{l}")
                write_f32(st.get_tensor(p + "self_attn.k_norm.weight"), f"k_norm.{l}")

            write_q4(st.get_tensor(p + "self_attn.o_proj.weight"), f"wo.{l}")
            rms_ffn = st.get_tensor(p + "post_attention_layernorm.weight")
            w1 = st.get_tensor(p + "mlp.gate_proj.weight")
            w3 = st.get_tensor(p + "mlp.up_proj.weight")
            rms_ffn, (w1, w3) = smooth_and_log(rms_ffn, [w1, w3], f"w1w3.{l}")
            write_f32(rms_ffn, f"rms_ffn.{l}")
            write_q4(w1, f"w1_gate.{l}")
            write_q4(w3, f"w3_up.{l}")
            write_q4(st.get_tensor(p + "mlp.down_proj.weight"), f"w2_down.{l}")

        rms_final = st.get_tensor("model.norm.weight")
        if "lm_head.weight" in keys:
            lm_head = st.get_tensor("lm_head.weight")
        elif tie_word_embeddings:
            print("  lm_head tied to embed_tokens")
            lm_head = embed
        else:
            raise RuntimeError("lm_head.weight not found")
        rms_final, (lm_head,) = smooth_and_log(rms_final, [lm_head], "lm_head")
        write_f32(rms_final, "rms_final")
        write_q4(lm_head, "lm_head")

    size = os.path.getsize(args.output)
    print(f"\nDone! Wrote {args.output} ({size:,} bytes, {size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
