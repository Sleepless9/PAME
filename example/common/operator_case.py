#!/usr/bin/env python3
"""Generate AME operator case headers, references, and error metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import struct
from pathlib import Path
from typing import Any

TILE = 4
MAX_HEADS = 2


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def load_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict):
        raise SystemExit(f"operator config must be an object: {path}")
    operator = suite.get("operator")
    if operator not in {"softmax", "rmsnorm", "rope", "mha", "ffn"}:
        raise SystemExit(f"unsupported operator: {operator!r}")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("operator config needs a non-empty cases array")
    if len(cases) > 24:
        raise SystemExit("operator config supports at most 24 cases")
    return suite


def positive_dim(value: Any, label: str, maximum: int = TILE) -> int:
    if not isinstance(value, int) or value <= 0 or value > maximum:
        raise SystemExit(f"{label} must be an integer in [1, {maximum}]")
    return value


def pattern_values(spec: Any, rows: int, cols: int, label: str) -> list[float]:
    count = rows * cols
    if isinstance(spec, list):
        values = spec
    elif isinstance(spec, dict) and "values" in spec:
        values = spec["values"]
    elif isinstance(spec, dict):
        pattern = spec.get("pattern")
        if pattern == "zeros":
            values = [0.0] * count
        elif pattern == "constant":
            values = [float(spec.get("value", 1.0))] * count
        elif pattern == "identity":
            values = [
                float(spec.get("value", 1.0)) if row == col else 0.0
                for row in range(rows)
                for col in range(cols)
            ]
        elif pattern == "mixed":
            base = [-2.5, -0.75, 0.0, 0.25, 1.5, -1.25, 3.0, -0.125]
            values = [base[i % len(base)] for i in range(count)]
        elif pattern == "extreme":
            low = float(spec.get("low", -16.0))
            high = float(spec.get("high", 16.0))
            middle = float(spec.get("middle", 0.125))
            base = [low, high, middle, -middle]
            values = [base[i % len(base)] for i in range(count)]
        elif pattern == "random":
            seed = spec.get("seed")
            if not isinstance(seed, int):
                raise SystemExit(f"{label}.seed must be an integer")
            low = float(spec.get("low", -2.0))
            high = float(spec.get("high", 2.0))
            generator = random.Random(seed)
            values = [generator.uniform(low, high) for _ in range(count)]
        else:
            raise SystemExit(f"{label} has unsupported pattern {pattern!r}")
    else:
        raise SystemExit(f"{label} must be an array or pattern object")
    if not isinstance(values, list) or len(values) != count:
        raise SystemExit(f"{label} must contain {count} values")
    result = [f32(item) for item in values]
    if not all(math.isfinite(item) for item in result):
        raise SystemExit(f"{label} currently accepts only finite FP32 values")
    return result


def vector_values(spec: Any, count: int, label: str) -> list[float]:
    return pattern_values(spec, 1, count, label)


def pad_matrix(values: list[float], rows: int, cols: int,
               fill: float = 0.0) -> list[float]:
    output = [f32(fill)] * (TILE * TILE)
    for row in range(rows):
        for col in range(cols):
            output[row * TILE + col] = values[row * cols + col]
    return output


def head_matrices(spec: Any, heads: int, rows: int, cols: int,
                  label: str) -> list[list[float]]:
    if isinstance(spec, list) and len(spec) == heads and all(
        isinstance(item, (dict, list)) for item in spec
    ):
        specs = spec
    else:
        specs = []
        for head in range(heads):
            if isinstance(spec, dict) and spec.get("pattern") == "random":
                item = dict(spec)
                item["seed"] = int(item["seed"]) + head
                specs.append(item)
            else:
                specs.append(spec)
    return [
        pattern_values(item, rows, cols, f"{label}[{head}]")
        for head, item in enumerate(specs)
    ]


def case_name(case: dict[str, Any], index: int) -> str:
    name = case.get("name", f"case-{index}")
    if not isinstance(name, str) or not name or "\n" in name or "\r" in name:
        raise SystemExit("case name must be a non-empty single line")
    return name


def normalize_case(operator: str, case: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise SystemExit(f"case[{index}] must be an object")
    result: dict[str, Any] = {"name": case_name(case, index)}
    if operator in {"softmax", "rmsnorm", "rope"}:
        shape = case.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise SystemExit(f"case[{index}].shape must be [rows, columns]")
        rows = positive_dim(shape[0], f"case[{index}].rows")
        cols = positive_dim(shape[1], f"case[{index}].columns")
        result.update(rows=rows, cols=cols)
        result["input"] = pattern_values(
            case.get("input"), rows, cols, f"case[{index}].input"
        )
        if operator == "rmsnorm":
            result["epsilon"] = f32(case.get("epsilon", 1.0e-5))
            if result["epsilon"] <= 0.0:
                raise SystemExit("RMSNorm epsilon must be positive")
            result["gamma"] = vector_values(
                case.get("gamma", {"pattern": "constant", "value": 1.0}),
                cols,
                f"case[{index}].gamma",
            )
        elif operator == "rope":
            if cols % 2 != 0:
                raise SystemExit("RoPE head dimension must be even")
            result["angles"] = pattern_values(
                case.get("angles"), rows, cols, f"case[{index}].angles"
            )
    elif operator == "mha":
        dimensions = case.get("dimensions")
        if not isinstance(dimensions, dict):
            raise SystemExit(f"case[{index}].dimensions must be an object")
        sequence = positive_dim(dimensions.get("sequence"), "sequence")
        model_dim = positive_dim(dimensions.get("model_dim"), "model_dim")
        head_dim = positive_dim(dimensions.get("head_dim"), "head_dim")
        heads = positive_dim(dimensions.get("heads"), "heads", MAX_HEADS)
        result.update(
            sequence=sequence,
            model_dim=model_dim,
            head_dim=head_dim,
            heads=heads,
            softmax_scale=f32(case.get("softmax_scale", 1.0)),
        )
        result["input"] = pattern_values(
            case.get("input"), sequence, model_dim, f"case[{index}].input"
        )
        for weight in ("wq", "wk", "wv"):
            result[weight] = head_matrices(
                case.get(weight), heads, model_dim, head_dim,
                f"case[{index}].{weight}"
            )
    elif operator == "ffn":
        dimensions = case.get("dimensions")
        if not isinstance(dimensions, dict):
            raise SystemExit(f"case[{index}].dimensions must be an object")
        tokens = positive_dim(dimensions.get("tokens"), "tokens")
        input_dim = positive_dim(dimensions.get("input_dim"), "input_dim")
        hidden_dim = positive_dim(dimensions.get("hidden_dim"), "hidden_dim")
        output_dim = positive_dim(dimensions.get("output_dim"), "output_dim")
        result.update(
            tokens=tokens,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        result["input"] = pattern_values(
            case.get("input"), tokens, input_dim, f"case[{index}].input"
        )
        result["w1"] = pattern_values(
            case.get("w1"), input_dim, hidden_dim, f"case[{index}].w1"
        )
        result["w2"] = pattern_values(
            case.get("w2"), hidden_dim, output_dim, f"case[{index}].w2"
        )
    return result


def normalize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    operator = suite["operator"]
    return {
        "operator": operator,
        "atol": float(suite.get("atol", 1.0e-5)),
        "rtol": float(suite.get("rtol", 1.0e-5)),
        "cases": [
            normalize_case(operator, case, index)
            for index, case in enumerate(suite["cases"])
        ],
    }


def c_float(value: float) -> str:
    return f"{f32(value).hex()}f"


def c_string(value: str) -> str:
    return json.dumps(value)


def c_array(name: str, values: list[float]) -> str:
    rows = []
    for offset in range(0, len(values), TILE):
        rows.append("    " + ", ".join(c_float(item) for item in values[offset:offset + TILE]))
    return (
        f"static const float {name}[{len(values)}] __attribute__((unused)) = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )


def render_header(suite: dict[str, Any]) -> str:
    operator = suite["operator"]
    cases = suite["cases"]
    prefix = "ztt_generated"
    declarations: list[str] = []
    entries: list[str] = []
    max_output = 0

    if operator == "softmax":
        structure = """struct ztt_generated_softmax_case {
    const char *name;
    unsigned rows;
    unsigned cols;
    const float *input;
};
"""
        for i, case in enumerate(cases):
            name = f"{prefix}_case_{i}_input"
            padded = [0.0] * (TILE * TILE)
            for row in range(TILE):
                for col in range(TILE):
                    if row < case["rows"] and col < case["cols"]:
                        padded[row * TILE + col] = case["input"][row * case["cols"] + col]
                    elif row < case["rows"]:
                        padded[row * TILE + col] = -200.0
            declarations.append(c_array(name, padded))
            entries.append(
                f"    {{{c_string(case['name'])}, {case['rows']}, {case['cols']}, {name}}}"
            )
            max_output = max(max_output, case["rows"] * case["cols"])
        table_type = "ztt_generated_softmax_case"
    elif operator == "rmsnorm":
        structure = """struct ztt_generated_rmsnorm_case {
    const char *name;
    unsigned rows;
    unsigned cols;
    float epsilon;
    const float *input;
    const float *gamma;
};
"""
        for i, case in enumerate(cases):
            input_name = f"{prefix}_case_{i}_input"
            gamma_name = f"{prefix}_case_{i}_gamma"
            declarations.append(c_array(input_name, pad_matrix(case["input"], case["rows"], case["cols"])))
            gamma_tile = [1.0] * (TILE * TILE)
            for row in range(TILE):
                for col in range(case["cols"]):
                    gamma_tile[row * TILE + col] = case["gamma"][col]
            declarations.append(c_array(gamma_name, gamma_tile))
            entries.append(
                f"    {{{c_string(case['name'])}, {case['rows']}, {case['cols']}, "
                f"{c_float(case['epsilon'])}, {input_name}, {gamma_name}}}"
            )
            max_output = max(max_output, case["rows"] * case["cols"])
        table_type = "ztt_generated_rmsnorm_case"
    elif operator == "rope":
        structure = """struct ztt_generated_rope_case {
    const char *name;
    unsigned rows;
    unsigned head_dim;
    const float *input;
    const float *angles;
};
"""
        for i, case in enumerate(cases):
            input_name = f"{prefix}_case_{i}_input"
            angle_name = f"{prefix}_case_{i}_angles"
            declarations.append(c_array(input_name, pad_matrix(case["input"], case["rows"], case["cols"])))
            declarations.append(c_array(angle_name, pad_matrix(case["angles"], case["rows"], case["cols"])))
            entries.append(
                f"    {{{c_string(case['name'])}, {case['rows']}, {case['cols']}, {input_name}, {angle_name}}}"
            )
            max_output = max(max_output, case["rows"] * case["cols"])
        table_type = "ztt_generated_rope_case"
    elif operator == "mha":
        structure = """struct ztt_generated_mha_case {
    const char *name;
    unsigned sequence;
    unsigned model_dim;
    unsigned head_dim;
    unsigned heads;
    float softmax_scale;
    const float *input;
    const float *wq[2];
    const float *wk[2];
    const float *wv[2];
};
"""
        for i, case in enumerate(cases):
            input_name = f"{prefix}_case_{i}_input"
            declarations.append(c_array(input_name, pad_matrix(case["input"], case["sequence"], case["model_dim"])))
            weight_names: dict[str, list[str]] = {}
            for weight in ("wq", "wk", "wv"):
                names = []
                for head in range(MAX_HEADS):
                    name = f"{prefix}_case_{i}_{weight}_{head}"
                    values = case[weight][head] if head < case["heads"] else [0.0] * (case["model_dim"] * case["head_dim"])
                    declarations.append(c_array(name, pad_matrix(values, case["model_dim"], case["head_dim"])))
                    names.append(name)
                weight_names[weight] = names
            entries.append(
                "    {"
                f"{c_string(case['name'])}, {case['sequence']}, {case['model_dim']}, "
                f"{case['head_dim']}, {case['heads']}, {c_float(case['softmax_scale'])}, "
                f"{input_name}, "
                f"{{{', '.join(weight_names['wq'])}}}, "
                f"{{{', '.join(weight_names['wk'])}}}, "
                f"{{{', '.join(weight_names['wv'])}}}"
                "}"
            )
            max_output = max(max_output, case["heads"] * case["sequence"] * case["head_dim"])
        table_type = "ztt_generated_mha_case"
    else:
        structure = """struct ztt_generated_ffn_case {
    const char *name;
    unsigned tokens;
    unsigned input_dim;
    unsigned hidden_dim;
    unsigned output_dim;
    const float *input;
    const float *w1;
    const float *w2;
};
"""
        for i, case in enumerate(cases):
            input_name = f"{prefix}_case_{i}_input"
            w1_name = f"{prefix}_case_{i}_w1"
            w2_name = f"{prefix}_case_{i}_w2"
            declarations.append(c_array(input_name, pad_matrix(case["input"], case["tokens"], case["input_dim"])))
            declarations.append(c_array(w1_name, pad_matrix(case["w1"], case["input_dim"], case["hidden_dim"])))
            declarations.append(c_array(w2_name, pad_matrix(case["w2"], case["hidden_dim"], case["output_dim"])))
            entries.append(
                f"    {{{c_string(case['name'])}, {case['tokens']}, {case['input_dim']}, "
                f"{case['hidden_dim']}, {case['output_dim']}, {input_name}, {w1_name}, {w2_name}}}"
            )
            max_output = max(max_output, case["tokens"] * case["output_dim"])
        table_type = "ztt_generated_ffn_case"

    return "".join([
        "/* Generated by common/operator_case.py; do not edit. */\n",
        "#ifndef ZTT_OPERATOR_CASES_GENERATED_H\n#define ZTT_OPERATOR_CASES_GENERATED_H\n\n",
        f"#define ZTT_OPERATOR_CASE_COUNT {len(cases)}\n",
        f"#define ZTT_OPERATOR_MAX_OUTPUT_ELEMENTS {max_output}\n\n",
        "#if !defined(__ASSEMBLER__)\n\n",
        structure,
        "\n",
        "\n".join(declarations),
        f"static const struct {table_type} ztt_operator_cases[ZTT_OPERATOR_CASE_COUNT] "
        "__attribute__((unused)) = {\n",
        ",\n".join(entries),
        "\n};\n\n#endif /* !__ASSEMBLER__ */\n\n",
        "#endif /* ZTT_OPERATOR_CASES_GENERATED_H */\n",
    ])


def matmul(a: list[float], a_rows: int, depth: int,
           b: list[float], b_cols: int) -> list[float]:
    return [
        sum(a[row * depth + k] * b[k * b_cols + col] for k in range(depth))
        for row in range(a_rows)
        for col in range(b_cols)
    ]


def base2_softmax(values: list[float], rows: int, cols: int) -> list[float]:
    output = []
    for row in range(rows):
        current = values[row * cols:(row + 1) * cols]
        maximum = max(current)
        exponentials = [2.0 ** (item - maximum) for item in current]
        total = sum(exponentials)
        output.extend(item / total for item in exponentials)
    return output


def reference_case(operator: str, case: dict[str, Any]) -> tuple[list[int], list[float]]:
    if operator == "softmax":
        shape = [case["rows"], case["cols"]]
        return shape, base2_softmax(case["input"], *shape)
    if operator == "rmsnorm":
        output = []
        for row in range(case["rows"]):
            values = case["input"][row * case["cols"]:(row + 1) * case["cols"]]
            mean = sum(item * item for item in values) / case["cols"]
            inverse = 1.0 / math.sqrt(mean + case["epsilon"])
            output.extend(
                values[col] * inverse * case["gamma"][col]
                for col in range(case["cols"])
            )
        return [case["rows"], case["cols"]], output
    if operator == "rope":
        output = []
        for row in range(case["rows"]):
            for col in range(0, case["cols"], 2):
                offset = row * case["cols"] + col
                even, odd = case["input"][offset:offset + 2]
                cosine = math.cos(case["angles"][offset])
                sine = math.sin(case["angles"][offset])
                output.extend([even * cosine - odd * sine,
                               odd * cosine + even * sine])
        return [case["rows"], case["cols"]], output
    if operator == "mha":
        output = []
        for head in range(case["heads"]):
            q = matmul(case["input"], case["sequence"], case["model_dim"],
                       case["wq"][head], case["head_dim"])
            k = matmul(case["input"], case["sequence"], case["model_dim"],
                       case["wk"][head], case["head_dim"])
            v = matmul(case["input"], case["sequence"], case["model_dim"],
                       case["wv"][head], case["head_dim"])
            scores = []
            for row in range(case["sequence"]):
                for col in range(case["sequence"]):
                    scores.append(
                        sum(q[row * case["head_dim"] + depth] *
                            k[col * case["head_dim"] + depth]
                            for depth in range(case["head_dim"]))
                        * case["softmax_scale"]
                    )
            probabilities = base2_softmax(scores, case["sequence"], case["sequence"])
            output.extend(matmul(probabilities, case["sequence"], case["sequence"],
                                 v, case["head_dim"]))
        return [case["heads"], case["sequence"], case["head_dim"]], output
    hidden = matmul(case["input"], case["tokens"], case["input_dim"],
                    case["w1"], case["hidden_dim"])
    activated = [item / (1.0 + 2.0 ** (-2.0 * item)) for item in hidden]
    output = matmul(activated, case["tokens"], case["hidden_dim"],
                    case["w2"], case["output_dim"])
    return [case["tokens"], case["output_dim"]], output


def build_reference(suite: dict[str, Any]) -> dict[str, Any]:
    cases = []
    for case in suite["cases"]:
        shape, values = reference_case(suite["operator"], case)
        cases.append({"name": case["name"], "shape": shape, "values": values})
    return {
        "operator": suite["operator"],
        "atol": suite["atol"],
        "rtol": suite["rtol"],
        "cases": cases,
    }


def parse_report(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^case\[(\d+)\]: (.+)$", text))
    result = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        values = [int(item, 16) for item in re.findall(r"0x([0-9a-fA-F]{16})", block)]
        if len(values) < 2:
            raise SystemExit(f"invalid operator report block in {path}")
        count = values[1]
        if len(values) != count + 2:
            raise SystemExit(
                f"report case {match.group(2)!r} has {len(values) - 2} values, expected {count}"
            )
        floats = [
            struct.unpack("<f", struct.pack("<I", item & 0xFFFFFFFF))[0]
            for item in values[2:]
        ]
        result.append({"name": match.group(2), "status": values[0], "values": floats})
    return result


def coordinate(index: int, shape: list[int]) -> list[int]:
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = index % shape[axis]
        index //= shape[axis]
    return result


def compare_report(reference: dict[str, Any], report_path: Path,
                   label: str) -> dict[str, Any]:
    actual_cases = parse_report(report_path)
    metrics: list[dict[str, Any]] = []
    all_pass = len(actual_cases) == len(reference["cases"])
    for index, expected in enumerate(reference["cases"]):
        if index >= len(actual_cases):
            metrics.append({"name": expected["name"], "pass": False,
                            "error": "missing report case"})
            continue
        actual = actual_cases[index]
        case_pass = actual["name"] == expected["name"] and actual["status"] == 0
        case_pass = case_pass and len(actual["values"]) == len(expected["values"])
        max_abs = 0.0
        max_rel = 0.0
        max_index = 0
        first_failure = None
        for element, (observed, wanted) in enumerate(
            zip(actual["values"], expected["values"])
        ):
            if math.isfinite(observed) and math.isfinite(wanted):
                absolute = abs(observed - wanted)
                relative = absolute / max(abs(wanted), 1.0e-30)
                passed = absolute <= reference["atol"] + reference["rtol"] * abs(wanted)
            else:
                absolute = math.inf
                relative = math.inf
                passed = observed == wanted
            if absolute > max_abs:
                max_abs = absolute
                max_index = element
            max_rel = max(max_rel, relative)
            if not passed and first_failure is None:
                first_failure = {
                    "index": element,
                    "coordinate": coordinate(element, expected["shape"]),
                    "expected": wanted,
                    "actual": observed,
                    "absolute_error": absolute,
                    "relative_error": relative,
                }
            case_pass = case_pass and passed
        metrics.append({
            "name": expected["name"],
            "shape": expected["shape"],
            "pass": case_pass,
            "status": actual["status"],
            "max_absolute_error": max_abs,
            "max_relative_error": max_rel,
            "max_error_coordinate": coordinate(max_index, expected["shape"]),
            "first_failure": first_failure,
        })
        all_pass = all_pass and case_pass
    return {
        "label": label,
        "operator": reference["operator"],
        "report": str(report_path),
        "atol": reference["atol"],
        "rtol": reference["rtol"],
        "pass": all_pass,
        "cases": metrics,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_header(args: argparse.Namespace) -> None:
    suite = normalize_suite(load_suite(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = render_header(suite)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != text:
        args.output.write_text(text, encoding="utf-8")


def write_reference(args: argparse.Namespace) -> None:
    suite = normalize_suite(load_suite(args.config))
    write_json(args.output, build_reference(suite))


def compare(args: argparse.Namespace) -> None:
    suite = normalize_suite(load_suite(args.config))
    reference = build_reference(suite)
    metrics = compare_report(reference, args.report, args.label)
    write_json(args.output, metrics)
    result = "PASS" if metrics["pass"] else "FAIL"
    print(f"[{result}] {args.label}: {suite['operator']} independent reference")
    if not metrics["pass"]:
        for case in metrics["cases"]:
            if not case["pass"]:
                print(f"  {case['name']}: {case.get('first_failure') or case.get('error')}")
        raise SystemExit(1)


def metric_by_name(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in metrics["cases"]}


def markdown_report(args: argparse.Namespace) -> None:
    host = json.loads(args.host_metrics.read_text(encoding="utf-8"))
    minor = json.loads(args.minor_metrics.read_text(encoding="utf-8"))
    host_cases = metric_by_name(host)
    rows = []
    for index, case in enumerate(minor["cases"], 1):
        host_case = host_cases.get(case["name"], {"pass": False})
        failure = case.get("first_failure")
        first = "-" if failure is None else str(failure["coordinate"])
        max_coordinate = case.get("max_error_coordinate", "-")
        rows.append(
            f"| {index} | `{case['name']}` | {case.get('shape', '-')} | "
            f"{'PASS' if host_case['pass'] else 'FAIL'} | "
            f"{'PASS' if case['pass'] else 'FAIL'} | "
            f"{case.get('max_absolute_error', math.inf):.6g} | "
            f"{case.get('max_relative_error', math.inf):.6g} | "
            f"{max_coordinate} | {first} |"
        )
    overall = host["pass"] and minor["pass"]
    lines = [
        f"# `{args.name}` Example Report",
        "",
        args.description,
        "",
        f"Independent reference tolerance: `atol={minor['atol']}`, "
        f"`rtol={minor['rtol']}`.",
        "",
        "| # | Operator case | Shape | Host/reference | AME/reference | "
        "Max abs error | Max rel error | Max-error coordinate | First failure |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        *rows,
        "",
        f"Overall report comparison: **{'PASS' if overall else 'FAIL'}**.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[RESULT] {args.name}: {'PASS' if overall else 'FAIL'}")
    if not overall:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    header = subparsers.add_parser("header")
    header.add_argument("--config", required=True, type=Path)
    header.add_argument("--output", required=True, type=Path)
    header.set_defaults(function=write_header)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--config", required=True, type=Path)
    reference.add_argument("--output", required=True, type=Path)
    reference.set_defaults(function=write_reference)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--config", required=True, type=Path)
    comparison.add_argument("--report", required=True, type=Path)
    comparison.add_argument("--label", required=True)
    comparison.add_argument("--output", required=True, type=Path)
    comparison.set_defaults(function=compare)

    report = subparsers.add_parser("markdown")
    report.add_argument("--name", required=True)
    report.add_argument("--description", required=True)
    report.add_argument("--host-metrics", required=True, type=Path)
    report.add_argument("--minor-metrics", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.set_defaults(function=markdown_report)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
