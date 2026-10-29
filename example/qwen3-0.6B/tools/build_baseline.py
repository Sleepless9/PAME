#!/usr/bin/env python3
"""Build a source-only local-SPM K=4 baseline or matching native reference.

Uses the user-provided Ztt Clang. No original ELF helpers are retained.
Guest scalar code targets rv64gc so automatic memory lowering cannot emit
hidden DDR RVV accesses; explicit RVV and Ztt routines are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def replace_between(source, first, last, replacement):
    start = source.index(first)
    stop = source.index(last, start)
    return source[:start] + replacement + source[stop:]


def frontend(core, host):
    source = (core / "qwen3_bare_w4a8_prefill_all.c").read_text()
    source = source[:source.index("\n#ifndef CHAT_BUILD\nenum Gem5ProfilePhase")]
    source = replace_between(source, '#include "../tokenizer/tokenizer_bare.h"',
                             "static float silu(float x)", "")
    source = "volatile uint64_t baseline_layers_done;\n" + source
    source = "#include <stdint.h>\n" + source
    checkpoint_helper = r'''
#ifndef QWEN_SAFE_CHECKPOINT_LAYERS
#define QWEN_SAFE_CHECKPOINT_LAYERS 0
#endif
static inline void baseline_layer_safe_checkpoint(int completed_layer) {
#if QWEN_SAFE_CHECKPOINT_LAYERS > 0 && !defined(BASELINE_HOST_REFERENCE)
    /* Every matrix call has returned through its scalar AME completion token,
     * and ztt_matmul has fenced its last output DMA descriptor here.  No
     * dependency-blocked descriptor can be stranded by O3 checkpoint drain. */
    if ((completed_layer % QWEN_SAFE_CHECKPOINT_LAYERS) == 0)
        m5_checkpoint(0, 0);
#else
    (void)completed_layer;
#endif
}
'''
    source = source.replace('#include "gem5_m5ops.h"',
                            '#include "gem5_m5ops.h"\n' + checkpoint_helper)
    source = source.replace("g_timing.n_layers++;", "g_timing.n_layers++; baseline_layers_done = g_timing.n_layers; "
                            "baseline_layer_safe_checkpoint(l + 1); "
                            'print_uart("DECODE layer_complete="); print_uart_int_dec(l + 1); print_uart("\\n");')
    source = source.replace("g_timing.n_layers += n_tokens;", "g_timing.n_layers += n_tokens; baseline_layers_done = g_timing.n_layers; "
                            "baseline_layer_safe_checkpoint(l + 1); "
                            'print_uart("PREFILL layer_complete="); print_uart_int_dec(l + 1); print_uart("\\n");')
    source = source.replace('for (;;) { __asm__ volatile ("" ::: "memory"); }', 'm5_fail(0, 10); for (;;) {}')
    if host:
        source = source.replace('#include "../platform/uart_helper.c"', '#include "gem5_baseline_host.h"')
        source = replace_between(source, "static inline uint64_t rdcycle(void)",
            '\n__attribute__((unused))\nstatic void print_cycles', 'static inline uint64_t rdcycle(void) { return 0; }\n')
        source = replace_between(source, "static inline void enable_accel_state(void)",
            "uintptr_t handle_trap", "static inline void enable_accel_state(void) {}\n\n")
        first = "#if (defined(USE_AME) && USE_AME) || (defined(USE_RVV) && USE_RVV)\n#ifndef QK_RVV_MAX_HEAD_DIM"
        source = replace_between(source, first, "__attribute__((unused))\nstatic inline void attn_av_axpy_scalar", """
static float qk_dot_vfmul_rvv(const float *q, const float *k, int n) {
    float sum = 0;
    if (n == 128) {
        float lanes[16] = {0}; // VLEN=512, e32,m1: preserve vector lane reduction order.
        for (int base = 0; base < 128; base += 16)
            for (int lane = 0; lane < 16; ++lane) lanes[lane] += q[base + lane] * k[base + lane];
        for (int lane = 0; lane < 16; ++lane) sum += lanes[lane];
    } else {
        for (int i = 0; i < n; ++i) sum += q[i] * k[i];
    }
    return sum;
}
#define QK_DOT(q,k,n) qk_dot_vfmul_rvv(q,k,n)
static void attn_av_axpy_rvv(float *out, const float *v, float a, int n) {
    for (int i = 0; i < n; ++i) out[i] += a * v[i];
}
""")
        for name in ("malloc", "calloc", "free"):
            source = re.sub(r"\b" + name + r"\b", "qwen_" + name, source)
        source = source.replace("return (uintptr_t)__heap_start;", "return (uintptr_t)host_heap;")
        source = source.replace("return (uintptr_t)__heap_end;", "return (uintptr_t)(host_heap + sizeof(host_heap));")
        source = source.replace("#define MODEL_ADDR ((const char *)_model_blob_start)", "#define MODEL_ADDR ((const char *)host_model)")
    return source + '\n#include "gem5_baseline_main.c"\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--toolchain", type=Path, default=Path("/home/xiangyy/001_intern/llvm-project_riscv/build/bin"))
    parser.add_argument("--prefill-tokens", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--operator-test", action="store_true")
    parser.add_argument("--host-reference", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "A", "AB", "ABC", "ABCD"), default="baseline")
    parser.add_argument("--decode-tokens-file", type=Path)
    parser.add_argument("--dma-ring", action="store_true", help="Use the ABI v1 timing descriptor ring")
    parser.add_argument("--offline-ame-layout", action="store_true",
                        help="Use v7 DMA-ready 4x4 K128 AME weight packets")
    parser.add_argument("--ame-operand-prefetch", action="store_true",
                        help="Double-buffer K128 operands in the 32-entry AME MRF")
    parser.add_argument("--safe-checkpoint-layers", type=int, default=0,
                        help="Request a guest checkpoint after every N completed transformer layers")
    args = parser.parse_args()
    features = [] if args.variant == "baseline" else list(args.variant)
    decode_tokens = json.loads(args.decode_tokens_file.read_text()) if args.decode_tokens_file else None
    if decode_tokens is not None and (len(decode_tokens) != args.decode_steps or
            any(not isinstance(t, int) or t < 0 or t >= 151936 for t in decode_tokens)):
        parser.error("decode tokens must contain exactly decode_steps valid token IDs")
    if args.prefill_tokens < 1 or args.decode_steps < 1:
        parser.error("Both phases require a positive token count")
    if args.safe_checkpoint_layers < 0:
        parser.error("safe checkpoint layer interval must be non-negative")
    here = Path(__file__).resolve().parent
    core, kernel, boot = here.parent / "core", here.parent / "kernel", here.parent / "linker_boot"
    output = args.output.resolve()
    build = output.parent / (output.stem + "_build")
    build.mkdir(parents=True, exist_ok=True)
    commands = []
    def run(command):
        command = list(map(str, command))
        commands.append(command)
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    (build / "frontend.c").write_text(frontend(core, args.host_reference))
    cc, lld = args.toolchain / "clang-21", args.toolchain / "ld.lld"
    common = [cc, "-O3", "-ffp-contract=off", "-fno-builtin", "-fno-vectorize", "-fno-slp-vectorize",
              "-ffunction-sections", "-fdata-sections", "-DUSE_AME=1",
              f"-DPREFILL_TOKENS={args.prefill_tokens}", f"-DDECODE_STEPS={args.decode_steps}",
              f"-DQWEN_SAFE_CHECKPOINT_LAYERS={args.safe_checkpoint_layers}",
              "-DFPGA_CLOCK_HZ=1000000000", "-I" + str(here), "-I" + str(core),
              "-I" + str(kernel / "include"), "-I" + str(boot)]
    common += [f"-DQWEN_PACKED_K16={int('A' in features)}",
               f"-DQWEN_DMA_RING={int(args.dma_ring)}",
               f"-DQWEN_GROUP_ACC={int('B' in features)}",
               f"-DQWEN_DMA_OVERLAP={int('D' in features)}",
               f"-DQWEN_OFFLINE_AME_LAYOUT={int(args.offline_ame_layout)}",
               f"-DQWEN_AME_OPERAND_PREFETCH={int(args.ame_operand_prefetch)}"]
    if args.ame_operand_prefetch and not args.offline_ame_layout:
        parser.error("--ame-operand-prefetch requires --offline-ame-layout")
    if decode_tokens is not None:
        common += ["-DQWEN_DECODE_TOKENS={" + ",".join(map(str, decode_tokens)) + "}"]
    if args.operator_test:
        common += ["-DBASELINE_OPERATOR_TEST=1"]
    if args.host_reference:
        common += ["-DBASELINE_HOST_REFERENCE=1", "-DZTT_SCALAR_FALLBACK=1"]
    else:
        common += ["--target=riscv64-unknown-elf", "-mabi=lp64d", "-mcmodel=medany", "-ffreestanding", "-fno-pie", "-DQWEN_LOCAL_SPM=1"]
    sources = [(build / "frontend.c", "rv64gc", []),
               (core / "ztt_qwen_backend.c", "rv64gc", []),
               (kernel / "src/ztt_matmul.c", "rv64gc", [])]
    if not args.host_reference:
        sources += [(here / "gem5_baseline_start.S", "rv64gc", []),
                    (boot / "gem5_m5ops.S", "rv64gc", []),
                    (here.parent / "platform/uart.c", "rv64gc", []),
                    (here.parent / "platform/bare_syscalls_uart.c", "rv64gc", []),
                    (here / "gem5_dma_tile.c", "rv64gc", []),
                    (here / "gem5_local_spm.c", "rv64gcv", []),
                    (kernel / "src/ztt_matmul.S", "rv64gcv_xxiangshanztt", ["-Dztt_i8_i8_i32_4x4=qwen_ztt_direct"])]
        if "B" in features:
            sources += [(here / "gem5_group_acc.S", "rv64gcv_xxiangshanztt", [])]
    objects = []
    for index, (source, march, extra) in enumerate(sources):
        obj = build / f"{index}_{source.stem}.o"
        run(common + ([] if args.host_reference else ["-march=" + march]) + extra + ["-c", source, "-o", obj])
        objects.append(obj)
    model = core / "model_data" / ("qwen3-0.6b-ame-v7.bin" if args.offline_ame_layout
                                    else "qwen3-0.6b-int4-tile.bin")
    if args.offline_ame_layout and args.variant != "ABCD":
        parser.error("--offline-ame-layout currently requires --variant ABCD")
    if not args.host_reference and not args.operator_test:
        # Cache only the unchanged large model object across guest builds.
        model_tag = "ame_v7" if args.offline_ame_layout else "tile_v6"
        model_obj = output.parent / f"baseline_model_{model_tag}.o"
        model_asm = output.parent / f"baseline_model_{model_tag}.S"
        model_asm.write_text('.section .model_blob,"a",@progbits\n.balign 64\n.globl _model_blob_start\n_model_blob_start:\n.incbin "' + str(model) + '"\n.globl _model_blob_end\n_model_blob_end:\n')
        if not model_obj.exists() or model_obj.stat().st_mtime < model.stat().st_mtime:
            run([cc, "--target=riscv64-unknown-elf", "-march=rv64gc", "-c", model_asm, "-o", model_obj])
        objects.append(model_obj)
    if args.host_reference:
        run([cc, *objects, "-Wl,--gc-sections", "-o", output])
    else:
        run([lld, "-m", "elf64lriscv", "--gc-sections", "--no-relax", "-T", boot / "test_compact.ld",
             "-Map=" + str(build / "link.map"), *objects, "-o", output])
        nm = subprocess.check_output([args.toolchain / "llvm-nm", output], text=True)
        symbols = {line.split()[2]: int(line.split()[0], 16) for line in nm.splitlines() if len(line.split()) == 3}
        progress = {name: dict(address=symbols[symbol], bytes=8) for name, symbol in
                    [("completed_ztt_tiles", "g_tiles"), ("completed_layer_passes", "baseline_layers_done")] if symbol in symbols}
        (build / "progress_symbols.json").write_text(json.dumps(progress, indent=2) + "\n")
        if not args.operator_test:
            (build / "result_symbols.json").write_text(json.dumps({
                "logits_pointer": symbols["baseline_logits_address"],
                "vocab_size": symbols["baseline_vocab_size"]}, indent=2) + "\n")
    manifest = dict(output=str(output), output_sha256=digest(output), commands=commands,
                    compiler_version=subprocess.check_output([cc, "--version"], text=True),
                    local_spm=not args.host_reference, host_reference=args.host_reference,
                    operator_test=args.operator_test, prefill_tokens=args.prefill_tokens,
                    decode_steps=args.decode_steps, seed=20260907,
                    safe_checkpoint_layers=args.safe_checkpoint_layers,
                    effective_k=128 if "B" in features else (16 if features else 4),
                    variant=args.variant, enabled_optimizations=features,
                    decode_input_tokens=decode_tokens,
                    continuous_injection="C" in features, dma_overlap="D" in features,
                    dma_submission="memory_ring" if args.dma_ring else "mmio",
                    group_accumulation="B" in features,
                    true_k128_command="B" in features,
                    offline_ame_layout=args.offline_ame_layout,
                    ame_operand_prefetch=args.ame_operand_prefetch,
                    floating_point_contract="off in all comparison builds",
                    model=str(model), model_sha256=digest(model),
                    source_sha256={str(source): digest(source) for source in
                        set([source for source, _, _ in sources] +
                            list(here.glob("gem5_baseline*")) + [here / "gem5_local_spm.h", here / "gem5_dma_ring.h", Path(__file__)])})
    with model.open("rb") as stream:
        dims = struct.unpack("<15i", stream.read(68)[8:])
    _, h, layers, vocab, _, _, _, _, q, kv, intermediate, _, _, _, _ = dims
    layer_macs = layers * (h * (q + 2 * kv) + q * h + 3 * h * intermediate)
    lm_macs = h * vocab
    assert all(d % 4 == 0 for d in (h, q, kv, intermediate, vocab))
    macs_per_tile = 64 if features else 16
    manifest["expected_ame_tiles"] = dict(prefill=(args.prefill_tokens * layer_macs + lm_macs) // macs_per_tile,
                                          decode=args.decode_steps * (layer_macs + lm_macs) // macs_per_tile)
    manifest["useful_macs_per_tile"] = macs_per_tile
    manifest["expected_layer_passes"] = dict(prefill=layers * args.prefill_tokens, decode=layers * args.decode_steps)
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("BUILT", output, flush=True)


if __name__ == "__main__":
    main()
