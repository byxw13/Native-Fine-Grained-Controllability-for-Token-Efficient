#!/usr/bin/env python3
"""Aggregate the 8B numerical/mechanism smoke evidence into one release gate."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--dynamic-stats-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = read(args.comparison)
    correction = read(args.correction)
    errors: list[str] = []
    k0 = comparison.get("vllm", {}).get("k0_identity", {})
    effect = comparison.get("vllm", {}).get("suppression_effect", {})
    if int(k0.get("num_items", 0)) <= 0:
        errors.append("missing K=0 comparison items")
    if int(k0.get("first_step_top1_equal", -1)) != int(k0.get("num_items", 0)):
        errors.append("K=0 first-step top1 identity failed")
    if int(effect.get("num_items", 0)) <= 0:
        errors.append("missing K=0.1 comparison items")
    effect_delta = float(effect.get("first_step_max_logprob_delta") or 0.0)
    if effect_delta <= 1e-5 and int(effect.get("first_step_top1_equal", 0)) == int(effect.get("num_items", 0)):
        errors.append("K=0.1 did not change any observed decision surface")
    if not bool(correction.get("passed")):
        errors.append("correction kernel smoke failed")
    if float(correction.get("grouped_max_abs_error", math.inf)) > 0.015625:
        errors.append("correction grouped max error exceeds tolerance")
    if float(correction.get("grouped_vs_scalar_max_abs_error", math.inf)) != 0.0:
        errors.append("grouped/scalar correction mismatch")

    stats_paths = [Path(name) for name in glob.glob(args.dynamic_stats_glob)]
    stats = [read(path) for path in stats_paths if path.is_file()]
    if not stats:
        errors.append("no dynamic stats files")
    first_assertions = sum(int(item.get("first_forward_budget_assertions", 0)) for item in stats)
    assertion_failures = sum(int(item.get("first_forward_budget_assertion_failures", 0)) for item in stats)
    shape_failures = sum(int(item.get("shape_failures", 0)) for item in stats)
    states_created = sum(int(item.get("states_created", 0)) for item in stats)
    states_finished = sum(int(item.get("states_finished", 0)) for item in stats)
    if first_assertions <= 0:
        errors.append("no first-forward K assertions")
    if assertion_failures != 0:
        errors.append(f"first-forward assertion failures={assertion_failures}")
    if shape_failures != 0:
        errors.append(f"shape failures={shape_failures}")
    if states_created != states_finished:
        errors.append(f"state leak created={states_created} finished={states_finished}")

    result = {
        "pass": not errors,
        "errors": errors,
        "comparison": {
            "k0_items": k0.get("num_items"),
            "k0_top1_equal": k0.get("first_step_top1_equal"),
            "k01_items": effect.get("num_items"),
            "k01_first_step_max_logprob_delta": effect_delta,
        },
        "correction": correction,
        "dynamic_stats_files": [str(path) for path in stats_paths],
        "first_forward_budget_assertions": first_assertions,
        "first_forward_budget_assertion_failures": assertion_failures,
        "shape_failures": shape_failures,
        "states_created": states_created,
        "states_finished": states_finished,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
