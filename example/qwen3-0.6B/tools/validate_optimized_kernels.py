#!/usr/bin/env python3
"""Check packed kernels against independently unpacked integer dot products."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    kernel = here.parent / "kernel"
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    functions = {}
    for name, packed, group in (("baseline", 0, 0), ("A", 1, 0), ("AB", 1, 1)):
        lib = out / (name + ".so")
        subprocess.run(["/home/xiangyy/001_intern/llvm-project_riscv/build/bin/clang-21",
            "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-I" + str(kernel / "include"),
            f"-DQWEN_PACKED_K16={packed}", f"-DQWEN_GROUP_ACC={group}",
            str(kernel / "src/ztt_matmul.c"), "-o", str(lib)], check=True)
        fn = ctypes.CDLL(str(lib)).ztt_matmul_quant_batch_i8_i8_f32
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
        fn.restype = None
        functions[name] = fn
    rng = np.random.default_rng(20260907)
    results = []
    for n, d, gs, stride in ((13, 7, 7, 21), (65, 66, 32, 96), (129, 8, 129, 129),
                            (1024, 68, 128, 1152), (2048, 64, 128, 2048), (3072, 64, 128, 3072)):
        groups, stride_groups = (n+gs-1)//gs, (stride+gs-1)//gs
        x = rng.integers(-127, 128, (2, n), dtype=np.int8)
        w = rng.integers(-7, 8, ((d+63)//64, stride_groups, 64, gs), dtype=np.int8)
        xs = rng.uniform(.01, 1, (2, groups)).astype(np.float32)
        ws = rng.uniform(.01, 1, (groups, d)).astype(np.float32)
        dense_w = np.zeros((groups, d, gs), np.int32)
        dense_x = np.zeros((2, groups, gs), np.int32)
        for j in range(d): dense_w[:, j] = w[j//64, :groups, j%64]
        dense_x.reshape(2, -1)[:, :n] = x
        outputs = {}
        for name, fn in functions.items():
            actual = np.zeros((2, d), np.float32)
            fn(*[a.ctypes.data for a in (actual, x, xs, w, ws)], 2, n, d, gs, stride)
            expected = np.zeros_like(actual)
            width = gs if name == "AB" else 16 if name == "A" else 4
            for g in range(groups):
                for k in range(0, gs, width):
                    dot = dense_x[:, g, k:k+width] @ dense_w[g, :, k:k+width].T
                    expected += (dot.astype(np.float32) * xs[:, g, None]) * ws[g]
            # Baseline is defined only for groups divisible by four; the new
            # packed path deliberately handles all partial-square tails.
            if name == "baseline" and gs % 4: continue
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), (name, n, d, gs)
            outputs[name] = actual
            results.append(dict(variant=name, n=n, d=d, gs=gs, stride=stride,
                                outputs=actual.size, bitwise_to_independent_reference=True))
        if "baseline" in outputs:
            for name in ("A", "AB"):
                delta = outputs[name].astype(float)-outputs["baseline"]
                results.append(dict(variant=name, n=n, d=d,
                    max_abs_vs_baseline=float(np.max(np.abs(delta))),
                    relative_l2_vs_baseline=float(np.linalg.norm(delta)/np.linalg.norm(outputs["baseline"]))))
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PACKED_KERNEL_REFERENCE PASS", sum(r.get("outputs", 0) for r in results), "checked outputs")


if __name__ == "__main__":
    main()
