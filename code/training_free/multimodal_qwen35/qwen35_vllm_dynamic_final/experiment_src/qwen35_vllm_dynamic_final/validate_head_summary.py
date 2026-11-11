from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PREFIX_BY_PROFILE = {
    "closed_think": "<think>\n\n</think>\n\n",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=100)
    parser.add_argument("--expected-indices", default="")
    parser.add_argument("--expected-max-new-tokens", type=int, default=4000)
    parser.add_argument("--expected-temperature", type=float, default=0.7)
    parser.add_argument("--expected-top-p", type=float, default=1.0)
    parser.add_argument("--expected-repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--prefix-profile",
        choices=tuple(PREFIX_BY_PROFILE),
        default="closed_think",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.run_dir / "head_scores_summary.json"
    merge_path = args.run_dir / "head_summary_merge_report.json"
    manifest_path = args.run_dir / "head_summary_manifest.json"
    signal_path = args.run_dir / "signal_calibration.json"
    question_path = args.run_dir / "question_lambda_calibration.json"
    summary, merge, manifest = map(read_json, (summary_path, merge_path, manifest_path))
    model_config = read_json(args.model_config)["text_config"]
    expected_layers = [
        index
        for index, layer_type in enumerate(model_config.get("layer_types", []))
        if layer_type == "full_attention"
    ]
    if expected_layers != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise RuntimeError(f"unexpected Qwen3.5 full-attention layers: {expected_layers}")
    expected_heads = int(model_config["num_attention_heads"])

    errors: list[str] = []
    if not merge.get("complete") or merge.get("failures"):
        errors.append("merge is incomplete or contains failures")
    for field in ("expected", "found", "unique"):
        if int(merge.get(field, -1)) != args.expected_items:
            errors.append(f"merge.{field} != {args.expected_items}")
    if int(summary.get("num_total_items", -1)) != args.expected_items:
        errors.append("summary item count mismatch")
    if int(summary.get("num_valid_items_with_end_think", -1)) != args.expected_items:
        errors.append("not every item produced valid head scores")

    layers = summary.get("summary_by_layer", {})
    if sorted(map(int, layers)) != expected_layers:
        errors.append("language full-attention layer coverage mismatch")
    total_score = 0.0
    for layer, record in layers.items():
        if int(record.get("num_items", -1)) != args.expected_items:
            errors.append(f"layer {layer} item count mismatch")
        for field in ("attn_mean_over_items", "contrib_mean_over_items"):
            values = record.get(field, [])
            if len(values) != expected_heads:
                errors.append(f"layer {layer} {field} head count mismatch")
                continue
            if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values):
                errors.append(f"layer {layer} {field} has invalid values")
            if field == "attn_mean_over_items":
                total_score += sum(float(value) for value in values)
    if total_score <= 0.0:
        errors.append("attention score mass is not positive")

    indices = [int(value) for value in manifest.get("global_indices", [])]
    expected_indices = (
        [int(value) for value in args.expected_indices.split(",") if value.strip()]
        if args.expected_indices
        else list(range(args.expected_items))
    )
    if indices != expected_indices:
        errors.append(f"manifest indices mismatch: {indices} != {expected_indices}")
    if manifest.get("failures"):
        errors.append("manifest contains failures")
    expected_generation = {
        "max_new_tokens": args.expected_max_new_tokens,
        "temperature": args.expected_temperature,
        "top_p": args.expected_top_p,
        "repetition_penalty": args.expected_repetition_penalty,
    }
    for field, expected in expected_generation.items():
        if manifest.get(field) != expected:
            errors.append(f"manifest.{field} != {expected}")
    if manifest.get("seed_base") != 0 or manifest.get("seed_policy") != "seed_base + global_index":
        errors.append("manifest seed schedule is not seed=0+global_index")
    items = manifest.get("items", [])
    if len(items) != args.expected_items:
        errors.append("manifest item record count mismatch")
    expected_prefix = PREFIX_BY_PROFILE[args.prefix_profile]
    if manifest.get("prefix_profile") != args.prefix_profile:
        errors.append("manifest prefix profile mismatch")
    if manifest.get("generation_prefix") != expected_prefix:
        errors.append("manifest generation prefix mismatch")
    for item in items:
        if item.get("error"):
            errors.append(f"item {item.get('global_index')} has an error")
        if int(item.get("seed", -1)) != int(item.get("global_index", -2)):
            errors.append(f"item {item.get('global_index')} seed schedule mismatch")
        prompt = str(item.get("prompt") or "")
        if prompt.count(expected_prefix) != 1 or not prompt.endswith(expected_prefix):
            errors.append(
                f"item {item.get('global_index')} requested prefix is not exactly once at the prompt tail"
            )

    for path in (signal_path, question_path):
        if not path.is_file():
            errors.append(f"missing {path.name}")
    if signal_path.is_file():
        signal = read_json(signal_path)
        if int(signal.get("selected_sample_count", -1)) != args.expected_items:
            errors.append("signal calibration sample count mismatch")
        if int(signal.get("checkpoint_count", 0)) <= 0:
            errors.append("signal calibration captured no checkpoints")
    if question_path.is_file():
        question = read_json(question_path)
        if int(question.get("source_sample_count", -1)) != args.expected_items:
            errors.append("question-lambda sample count mismatch")
        recommended = question.get("recommended_question_lambda_calibration", {})
        if int(recommended.get("eligible_sample_count", 0)) <= 0:
            errors.append("question-lambda calibration has no eligible samples")
    report = {
        "pass": not errors,
        "errors": errors,
        "expected_items": args.expected_items,
        "expected_layers": expected_layers,
        "expected_heads": expected_heads,
        "attention_score_mass": total_score,
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (summary_path, merge_path, manifest_path, signal_path, question_path)
            if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
