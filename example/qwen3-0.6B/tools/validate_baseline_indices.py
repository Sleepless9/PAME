#!/usr/bin/env python3
"""Compare pre-change and hoisted K=4 kernels bit-for-bit on native scalar tiles."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    kernel = here.parent / "kernel"
    args.outdir.mkdir(parents=True, exist_ok=True)
    functions = []
    for name, source in [("original", args.original), ("hoisted", kernel / "src/ztt_matmul.c")]:
        library = args.outdir.resolve() / (name + ".so")
        subprocess.run(["/home/xiangyy/001_intern/llvm-project_riscv/build/bin/clang-21",
            "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-I" + str(kernel / "include"),
            str(source), "-o", str(library)], check=True)
        fn = ctypes.CDLL(str(library)).ztt_matmul_quant_batch_i8_i8_f32
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5
        fn.restype = None
        functions.append(fn)
    rng = np.random.default_rng(20260907)
    results = []
    for n, d, gs, stride in [(4, 4, 4, 4), (65, 66, 32, 96), (128, 8, 128, 128),
                             (1024, 68, 128, 1152), (2048, 64, 128, 2048), (3072, 64, 128, 3072)]:
        groups, stride_groups = (n + gs - 1) // gs, (stride + gs - 1) // gs
        x = rng.integers(-127, 128, size=2*n, dtype=np.int8)
        w = rng.integers(-7, 8, size=((d+63)//64)*stride_groups*64*gs, dtype=np.int8)
        xs = rng.uniform(0.01, 1, size=2*groups).astype(np.float32)
        ws = rng.uniform(0.01, 1, size=groups*d).astype(np.float32)
        outputs = [np.zeros(2*d, np.float32), np.zeros(2*d, np.float32)]
        for fn, out in zip(functions, outputs):
            fn(*[a.ctypes.data for a in (out, x, xs, w, ws)], 2, n, d, gs, stride)
        assert np.isfinite(outputs[0]).all()
        assert outputs[0].tobytes() == outputs[1].tobytes(), (n, d, gs, stride)
        results.append(dict(n=n, d=d, gs=gs, stride=stride, outputs=2*d, bitwise_equal=True))
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("INDEX_HOIST PASS", len(results), "shapes", sum(r["outputs"] for r in results), "outputs")


if __name__ == "__main__":
    main()
