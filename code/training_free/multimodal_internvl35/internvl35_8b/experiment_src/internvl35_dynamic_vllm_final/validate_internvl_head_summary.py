#!/usr/bin/env python3
"""Strict structural validation for the InternVL head-summary artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=100)
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-heads", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    summary = json.loads((root / "head_scores_summary.json").read_text())
    merge = json.loads((root / "head_summary_merge_report.json").read_text())
    manifest = json.loads((root / "head_summary_manifest.json").read_text())
    errors: list[str] = []
    if not merge.get("complete") or merge.get("failures"):
        errors.append("merge incomplete or contains failures")
    for key in ("expected", "found", "unique"):
        if int(merge.get(key, -1)) != args.expected_items:
            errors.append(f"merge.{key} != {args.expected_items}")
    if int(summary.get("num_total_items", -1)) != args.expected_items:
        errors.append("summary total item count mismatch")
    if int(summary.get("num_valid_items_with_end_think", -1)) != args.expected_items:
        errors.append("not all items have valid end-think head scores")
    layers = summary.get("summary_by_layer", {})
    if sorted(int(k) for k in layers) != list(range(args.expected_layers)):
        errors.append("language layer coverage mismatch")
    mass = 0.0
    for layer, record in layers.items():
        if int(record.get("num_items", -1)) != args.expected_items:
            errors.append(f"layer {layer} item count mismatch")
        for field in ("attn_mean_over_items", "contrib_mean_over_items"):
            values = record.get(field, [])
            if len(values) != args.expected_heads:
                errors.append(f"layer {layer} {field} head count mismatch")
            if not all(math.isfinite(float(value)) and float(value) >= 0 for value in values):
                errors.append(f"layer {layer} {field} contains invalid values")
            if field == "attn_mean_over_items":
                mass += sum(float(value) for value in values)
    if mass <= 0:
        errors.append("attention mass is not positive")
    expected_prefix = "<think>\n\n</think>\n\n"
    if manifest.get("prefix") != expected_prefix:
        errors.append("prefix is not the required closed-think scaffold without a cue word")
    if manifest.get("system_prompt") is not None:
        errors.append("R1 system prompt unexpectedly present")
    for field, value in {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": "model_default",
        "repetition_penalty": args.repetition_penalty,
    }.items():
        if manifest.get(field) != value:
            errors.append(f"manifest.{field} != {value}")
    indices = [int(value) for value in manifest.get("global_indices", [])]
    if indices != list(range(args.expected_items)):
        errors.append("manifest indices are not MMMUVal 0..N-1")
    result = {"pass": not errors, "errors": errors, "attention_score_mass": mass, "expected_items": args.expected_items, "expected_layers": args.expected_layers, "expected_heads": args.expected_heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
