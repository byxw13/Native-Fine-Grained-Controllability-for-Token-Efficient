from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_raw(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(glob.glob(str(run_dir / "raw" / "predictions.shard*.jsonl"))):
        rows.extend(read_jsonl(Path(name)))
    return rows


def finite_numbers(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(
    rows: list[dict[str, Any]], judge_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    judged = [judge_by_id[str(row.get("id"))] for row in rows if str(row.get("id")) in judge_by_id]
    output = finite_numbers(rows, "output_tokens")
    thinking = finite_numbers(rows, "thinking_tokens")
    seconds = finite_numbers(rows, "generate_time_seconds")
    k_values = [
        float(value)
        for row in rows
        for value in ((row.get("dynamic_summary") or {}).get("k_sequence") or [])
    ]
    correct = sum(item.get("final_is_correct") is True for item in judged)
    return {
        "rows": len(rows),
        "judged_rows": len(judged),
        "correct": correct,
        "accuracy": correct / len(judged) if judged else None,
        "runtime_errors": sum(bool(row.get("error")) for row in rows),
        "parse_failures": sum(not bool(row.get("parseable")) for row in rows),
        "maxed_rows": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
        "maxed_ratio": sum(bool(row.get("hit_max_new_tokens")) for row in rows) / len(rows) if rows else None,
        "output_tokens": {
            "sum": sum(output),
            "mean": statistics.fmean(output) if output else None,
            "median": statistics.median(output) if output else None,
            "p90": quantile(output, 0.90),
        },
        "thinking_tokens": {
            "mean": statistics.fmean(thinking) if thinking else None,
            "median": statistics.median(thinking) if thinking else None,
            "p90": quantile(thinking, 0.90),
        },
        "generation": {
            "sum_item_allocated_seconds": sum(seconds),
            "mean_item_seconds": statistics.fmean(seconds) if seconds else None,
            "aggregate_output_tokens_per_allocated_gpu_second": sum(output) / sum(seconds)
            if output and sum(seconds) > 0
            else None,
        },
        "dynamic": {
            "checkpoint_count": sum(
                int((row.get("dynamic_summary") or {}).get("checkpoint_count") or 0) for row in rows
            ),
            "increase_count": sum(
                int((row.get("dynamic_summary") or {}).get("increase_count") or 0) for row in rows
            ),
            "decrease_count": sum(
                int((row.get("dynamic_summary") or {}).get("decrease_count") or 0) for row in rows
            ),
            "k_observations": len(k_values),
            "k_mean": statistics.fmean(k_values) if k_values else None,
            "k_max": max(k_values) if k_values else None,
            "nonzero_k_observations": sum(value > 0 for value in k_values),
        },
    }


def validate(
    run_dir: Path, rows: list[dict[str, Any]], judges: list[dict[str, Any]], expected: int
) -> list[str]:
    errors = []
    ids = [str(row.get("id")) for row in rows]
    indices = [int(row.get("global_index", -1)) for row in rows]
    judge_ids = [str(row.get("id")) for row in judges]
    if len(rows) != expected:
        errors.append(f"raw row count is {len(rows)}, expected {expected}")
    if len(set(ids)) != len(ids):
        errors.append("raw rows contain duplicate ids")
    if sorted(indices) != list(range(expected)):
        errors.append("raw rows are not exactly global indices 0..expected-1")
    if len(judges) != expected:
        errors.append(f"judge row count is {len(judges)}, expected {expected}")
    if len(set(judge_ids)) != len(judge_ids):
        errors.append("judge rows contain duplicate ids")
    if set(judge_ids) != set(ids):
        errors.append("raw and judge id sets differ")
    if any(row.get("error") for row in rows):
        errors.append("one or more raw rows contain runtime/audit errors")
    if any(row.get("method") != "aris_dynamic_vllm_final" for row in rows):
        errors.append("one or more rows do not identify the final dynamic method")
    if any(int(row.get("method_prefix_count", -1)) != 1 for row in rows):
        errors.append("one or more rows do not contain exactly one method prefix")
    if any((row.get("trajectory_audit") or {}).get("pass") is not True for row in rows):
        errors.append("one or more rows failed the trajectory persistence audit")
    if any(len(row.get("output_token_ids") or []) != int(row.get("output_tokens") or -1) for row in rows):
        errors.append("one or more output-token arrays have inconsistent lengths")
    if not any(
        float(value) > 0
        for row in rows
        for value in ((row.get("dynamic_summary") or {}).get("k_sequence") or [])
    ):
        errors.append("no nonzero dynamic K was observed in the formal run")
    expected_shards = {f"predictions.shard{rank:02d}of8.jsonl" for rank in range(8)}
    actual_shards = {path.name for path in (run_dir / "raw").glob("predictions.shard*.jsonl")}
    if actual_shards != expected_shards:
        errors.append(
            f"formal split files differ: missing={sorted(expected_shards - actual_shards)}, "
            f"unexpected={sorted(actual_shards - expected_shards)}"
        )
    expected_groups = {"a6000_half0": 450, "l20_half1": 450}
    group_counts = Counter(str(row.get("execution_group")) for row in rows)
    if dict(group_counts) != expected_groups:
        errors.append(f"execution-group counts differ: {dict(group_counts)}")
    if any(
        str(row.get("execution_group")) != ("a6000_half0" if int(row.get("global_index", -1)) < 450 else "l20_half1")
        for row in rows
    ):
        errors.append("one or more rows were produced by the wrong half-job")
    if any(int(row.get("rank", -1)) not in range(8) for row in rows):
        errors.append("one or more rows have a worker rank outside 0..7")
    if any(
        int(row.get("rank", -1))
        != int(row.get("global_index", -1)) % 4
        + (0 if int(row.get("global_index", -1)) < 450 else 4)
        for row in rows
    ):
        errors.append("one or more rows have a rank inconsistent with the frozen split")
    generation_expectations = {
        "max_new_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "sample_seed": 0,
        "repetition_penalty": 1.05,
    }
    for field, expected_value in generation_expectations.items():
        if any(row.get(field) != expected_value for row in rows):
            errors.append(f"one or more rows differ from formal {field}={expected_value!r}")
    return errors


def paired_comparison(
    method_rows: list[dict[str, Any]],
    method_judges: dict[str, dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    baseline_judges: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_by_id = {str(row.get("id")): row for row in method_rows}
    baseline_by_id = {str(row.get("id")): row for row in baseline_rows}
    ids = sorted(set(method_by_id) & set(baseline_by_id) & set(method_judges) & set(baseline_judges))
    if not ids:
        return {"paired_rows": 0}
    method_correct = sum(method_judges[key].get("final_is_correct") is True for key in ids)
    baseline_correct = sum(baseline_judges[key].get("final_is_correct") is True for key in ids)
    method_tokens = sum(float(method_by_id[key].get("output_tokens") or 0) for key in ids)
    baseline_tokens = sum(float(baseline_by_id[key].get("output_tokens") or 0) for key in ids)
    return {
        "paired_rows": len(ids),
        "method_accuracy": method_correct / len(ids),
        "baseline_accuracy": baseline_correct / len(ids),
        "accuracy_delta": (method_correct - baseline_correct) / len(ids),
        "method_output_tokens": method_tokens,
        "baseline_output_tokens": baseline_tokens,
        "output_token_ratio": method_tokens / baseline_tokens if baseline_tokens else None,
    }


def _fsync_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--expected-rows", type=int, default=900)
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = read_raw(args.run_dir)
    judges = read_jsonl(args.run_dir / "stream_judge" / "judge_results.jsonl")
    judge_by_id = {str(row.get("id")): row for row in judges}
    errors = validate(args.run_dir, rows, judges, args.expected_rows)
    rows.sort(key=lambda row: int(row.get("global_index", -1)))
    _fsync_jsonl(args.run_dir / "predictions.jsonl", rows)
    report: dict[str, Any] = {
        "pass": not errors,
        "errors": errors,
        "generation_parameters": {
            "model": "Qwen3-VL-4B-Thinking",
            "max_new_tokens": 32768,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "sample_seed": 0,
            "repetition_penalty": 1.05,
        },
        "execution_topology": {
            "a6000_half0": {"partition": "YOUR_A6000_PARTITION", "gpus": 4, "range": [0, 450]},
            "l20_half1": {"partition": "YOUR_L20_PARTITION", "gpus": 4, "range": [450, 900]},
        },
        "all_900": summarize(rows, judge_by_id),
        "calibration_100": summarize(
            [row for row in rows if int(row.get("global_index", -1)) < args.calibration_count], judge_by_id
        ),
        "holdout_800": summarize(
            [row for row in rows if int(row.get("global_index", -1)) >= args.calibration_count], judge_by_id
        ),
    }
    if args.baseline_run_dir:
        baseline_rows = read_raw(args.baseline_run_dir)
        baseline_judges_list = read_jsonl(args.baseline_run_dir / "stream_judge" / "judge_results.jsonl")
        baseline_judges = {str(row.get("id")): row for row in baseline_judges_list}
        report["baseline_available_rows"] = len(baseline_rows)
        report["paired_baseline_comparison"] = paired_comparison(
            rows, judge_by_id, baseline_rows, baseline_judges
        )

    _fsync_json(args.output_json, report)
    lines = [
        "# Qwen3-VL-4B ARIS 动态方法 MMMUVal 报告",
        "",
        f"完整性：{'通过' if report['pass'] else '失败'}。",
        "",
        "| 范围 | 题数 | Accuracy | Runtime errors | Maxed | 平均 output tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("全部", "all_900"), ("Calibration", "calibration_100"), ("Holdout", "holdout_800")):
        item = report[key]
        accuracy = item["accuracy"]
        lines.append(
            f"| {label} | {item['rows']} | {accuracy:.4%} | {item['runtime_errors']} | "
            f"{item['maxed_rows']} | {item['output_tokens']['mean']:.2f} |"
        )
    paired = report.get("paired_baseline_comparison")
    if paired and paired.get("paired_rows"):
        lines.extend(
            [
                "",
                f"与现有 baseline 的共同 {paired['paired_rows']} 题配对对比："
                f"accuracy Δ={paired['accuracy_delta']:+.4%}，"
                f"output-token ratio={paired['output_token_ratio']:.4f}。",
            ]
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "output": str(args.output_json)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
