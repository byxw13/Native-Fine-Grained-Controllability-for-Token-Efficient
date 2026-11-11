#!/usr/bin/env python3
"""Compare the currently judged ARIS subset with the same frozen baseline items.

The formal run is still appending shards.  This script takes a read-only,
line-consistent snapshot, pairs rows by sample id, and never rewrites inference
or judge artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def read_jsonl_snapshot(path: Path) -> list[dict[str, Any]]:
    """Read only newline-terminated JSONL records visible at open time."""
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        payload = payload.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in payload else b""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL record: {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def read_raw_shards(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = sorted(glob.glob(str(run_dir / "raw" / "predictions.shard*.jsonl")))
    if not names:
        fallback = run_dir / "predictions.jsonl"
        if fallback.is_file():
            names = [str(fallback)]
    for name in names:
        rows.extend(read_jsonl_snapshot(Path(name)))
    return rows


def unique_by_id(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    indices: dict[int, str] = {}
    for row in rows:
        sample_id = str(row.get("id") or "")
        if not sample_id:
            raise ValueError(f"{label} contains a row without id")
        if sample_id in result:
            raise ValueError(f"{label} contains duplicate id: {sample_id}")
        index = int(row.get("global_index", -1))
        if index in indices:
            raise ValueError(
                f"{label} contains duplicate global_index {index}: "
                f"{indices[index]} and {sample_id}"
            )
        result[sample_id] = row
        indices[index] = sample_id
    return result


def finite_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def number_summary(values: list[float]) -> dict[str, Any]:
    return {
        "available_rows": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": quantile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    probability = correct / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1 - probability) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def model_summary(
    ids: list[str],
    raw_by_id: dict[str, dict[str, Any]],
    judge_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [raw_by_id[sample_id] for sample_id in ids]
    judges = [judge_by_id[sample_id] for sample_id in ids]
    correct = sum(row.get("final_is_correct") is True for row in judges)
    output_tokens = finite_values(rows, "output_tokens")
    generation_seconds = finite_values(rows, "generate_time_seconds")
    return {
        "rows": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids) if ids else None,
        "accuracy_wilson_95": wilson_interval(correct, len(ids)),
        "question_types": {
            "multiple_choice": sum(row.get("question_type") == "multiple_choice" for row in judges),
            "open": sum(row.get("question_type") == "open" for row in judges),
        },
        "runtime_error_rows": sum(bool(row.get("error")) for row in rows),
        "parse_failure_rows": sum(not bool(row.get("parseable")) for row in rows),
        "max_output_rows": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
        "thinking_status_counts": dict(
            sorted(Counter(str(row.get("thinking_status") or "missing") for row in rows).items())
        ),
        "input_tokens": number_summary(finite_values(rows, "input_tokens")),
        "visual_tokens": number_summary(finite_values(rows, "visual_tokens")),
        "output_tokens": number_summary(output_tokens),
        "thinking_tokens": number_summary(finite_values(rows, "thinking_tokens")),
        "total_tokens": number_summary(finite_values(rows, "total_tokens")),
        "generate_time_seconds": number_summary(generation_seconds),
        "aggregate_output_tokens_per_gpu_second": (
            sum(output_tokens) / sum(generation_seconds)
            if output_tokens and generation_seconds and sum(generation_seconds) > 0
            else None
        ),
    }


def paired_outcome(method_correct: bool, baseline_correct: bool) -> str:
    if method_correct and baseline_correct:
        return "both_correct"
    if method_correct:
        return "method_only_correct"
    if baseline_correct:
        return "baseline_only_correct"
    return "both_wrong"


def compare_subset(
    ids: list[str],
    method_raw: dict[str, dict[str, Any]],
    method_judges: dict[str, dict[str, Any]],
    baseline_raw: dict[str, dict[str, Any]],
    baseline_judges: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method = model_summary(ids, method_raw, method_judges)
    baseline = model_summary(ids, baseline_raw, baseline_judges)
    outcomes = {
        "both_correct": 0,
        "method_only_correct": 0,
        "baseline_only_correct": 0,
        "both_wrong": 0,
    }
    for sample_id in ids:
        outcome = paired_outcome(
            method_judges[sample_id].get("final_is_correct") is True,
            baseline_judges[sample_id].get("final_is_correct") is True,
        )
        outcomes[outcome] += 1
    method_output = method["output_tokens"]["sum"]
    baseline_output = baseline["output_tokens"]["sum"]
    method_total = method["total_tokens"]["sum"]
    baseline_total = baseline["total_tokens"]["sum"]
    return {
        "rows": len(ids),
        "method": method,
        "baseline": baseline,
        "delta": {
            "accuracy": method["accuracy"] - baseline["accuracy"] if ids else None,
            "correct": method["correct"] - baseline["correct"],
            "mean_output_tokens": (
                method["output_tokens"]["mean"] - baseline["output_tokens"]["mean"]
                if ids
                else None
            ),
            "output_token_ratio": method_output / baseline_output if baseline_output else None,
            "mean_thinking_tokens": (
                method["thinking_tokens"]["mean"] - baseline["thinking_tokens"]["mean"]
                if method["thinking_tokens"]["mean"] is not None
                and baseline["thinking_tokens"]["mean"] is not None
                else None
            ),
            "mean_total_tokens": (
                method["total_tokens"]["mean"] - baseline["total_tokens"]["mean"]
                if ids
                else None
            ),
            "total_token_ratio": method_total / baseline_total if baseline_total else None,
        },
        "paired_outcomes": outcomes,
    }


def paired_rows(
    ids: list[str],
    method_raw: dict[str, dict[str, Any]],
    method_judges: dict[str, dict[str, Any]],
    baseline_raw: dict[str, dict[str, Any]],
    baseline_judges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        method_row = method_raw[sample_id]
        baseline_row = baseline_raw[sample_id]
        method_correct = method_judges[sample_id].get("final_is_correct") is True
        baseline_correct = baseline_judges[sample_id].get("final_is_correct") is True
        rows.append(
            {
                "id": sample_id,
                "global_index": int(method_row.get("global_index", -1)),
                "subject": method_row.get("subject"),
                "question_type": method_judges[sample_id].get("question_type"),
                "ground_truth": method_row.get("ground_truth"),
                "method_prediction": method_judges[sample_id].get("prediction"),
                "baseline_prediction": baseline_judges[sample_id].get("prediction"),
                "method_correct": method_correct,
                "baseline_correct": baseline_correct,
                "paired_outcome": paired_outcome(method_correct, baseline_correct),
                "method_output_tokens": method_row.get("output_tokens"),
                "baseline_output_tokens": baseline_row.get("output_tokens"),
                "output_token_delta": (
                    float(method_row.get("output_tokens") or 0)
                    - float(baseline_row.get("output_tokens") or 0)
                ),
                "method_thinking_tokens": method_row.get("thinking_tokens"),
                "baseline_thinking_tokens": baseline_row.get("thinking_tokens"),
                "method_total_tokens": method_row.get("total_tokens"),
                "baseline_total_tokens": baseline_row.get("total_tokens"),
                "method_hit_max": bool(method_row.get("hit_max_new_tokens")),
                "baseline_hit_max": bool(baseline_row.get("hit_max_new_tokens")),
            }
        )
    return rows


def fsync_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def percentage_points(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f} pp"


def render_markdown(report: dict[str, Any]) -> str:
    baseline_validation = report["baseline_validation"]
    baseline_label = (
        f"{baseline_validation['correct']}/"
        f"{baseline_validation['judged_rows']} frozen baseline"
    )
    lines = [
        f"# ARIS 当前完成子集 vs {baseline_label}",
        "",
        f"快照时间：`{report['snapshot']['created_at']}`",
        "",
        f"方法已落盘 {report['snapshot']['method_raw_rows']} 题，已 judge "
        f"{report['snapshot']['method_judged_rows']} 题；本报告使用可严格配对的 "
        f"{report['snapshot']['paired_rows']} 题。",
        "",
        "| 范围 | 配对题数 | 方法正确/准确率 | Baseline 正确/准确率 | Accuracy Δ | 方法平均输出 token | Baseline 平均输出 token | 输出 token 比率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("全部已完成", "all"), ("Head/Calibration 0–99", "calibration_100"), ("Holdout 100–899", "holdout_800")):
        item = report["comparisons"][key]
        method = item["method"]
        baseline = item["baseline"]
        lines.append(
            f"| {label} | {item['rows']} | {method['correct']}/{item['rows']} ({percent(method['accuracy'])}) | "
            f"{baseline['correct']}/{item['rows']} ({percent(baseline['accuracy'])}) | "
            f"{percentage_points(item['delta']['accuracy'])} | {number(method['output_tokens']['mean'])} | "
            f"{number(baseline['output_tokens']['mean'])} | {number(item['delta']['output_token_ratio'])} |"
        )
    all_rows = report["comparisons"]["all"]
    method = all_rows["method"]
    baseline = all_rows["baseline"]
    lines.extend(
        [
            "",
            "## Token 消耗（严格同题）",
            "",
            "| 指标 | 方法 | Baseline | 差值/比率 |",
            "|---|---:|---:|---:|",
            f"| 平均 input tokens | {number(method['input_tokens']['mean'])} | {number(baseline['input_tokens']['mean'])} | {number(method['input_tokens']['mean'] - baseline['input_tokens']['mean'])} |",
            f"| 平均 visual tokens | {number(method['visual_tokens']['mean'])} | {number(baseline['visual_tokens']['mean'])} | {number(method['visual_tokens']['mean'] - baseline['visual_tokens']['mean'])} |",
            f"| 平均 output tokens | {number(method['output_tokens']['mean'])} | {number(baseline['output_tokens']['mean'])} | {number(all_rows['delta']['mean_output_tokens'])} |",
            f"| 平均 thinking tokens（现有解析字段） | {number(method['thinking_tokens']['mean'])} | {number(baseline['thinking_tokens']['mean'])} | {number(all_rows['delta']['mean_thinking_tokens'])} |",
            f"| 平均 total tokens | {number(method['total_tokens']['mean'])} | {number(baseline['total_tokens']['mean'])} | {number(all_rows['delta']['mean_total_tokens'])} |",
            f"| output token 总量 | {number(method['output_tokens']['sum'])} | {number(baseline['output_tokens']['sum'])} | ratio={number(all_rows['delta']['output_token_ratio'])} |",
            f"| total token 总量 | {number(method['total_tokens']['sum'])} | {number(baseline['total_tokens']['sum'])} | ratio={number(all_rows['delta']['total_token_ratio'])} |",
            f"| 达到最大输出题数 | {method['max_output_rows']} | {baseline['max_output_rows']} | — |",
            "",
            "> 注意：动态方法在 closed-think 前缀之后继续生成，现有 `thinking_tokens` 解析器可能把未重新出现 `<think>` 标签的推理记为 0。实际 token 消耗请以 `output_tokens` 和 `total_tokens` 为主。",
            "",
            "## 配对胜负",
            "",
        ]
    )
    outcomes = all_rows["paired_outcomes"]
    lines.extend(
        [
            f"- 两边都对：{outcomes['both_correct']}",
            f"- 仅方法正确：{outcomes['method_only_correct']}",
            f"- 仅 baseline 正确：{outcomes['baseline_only_correct']}",
            f"- 两边都错：{outcomes['both_wrong']}",
            "",
            "## 口径检查",
            "",
            f"- Baseline 全量 judge：{report['baseline_validation']['correct']}/{report['baseline_validation']['judged_rows']} "
            f"({percent(report['baseline_validation']['accuracy'])})。",
            f"- 方法待 judge：{report['snapshot']['method_pending_judge_rows']} 题；这些题不进入当前准确率。",
            f"- 方法错误行：{method['runtime_error_rows']}；baseline 错误行：{baseline['runtime_error_rows']}。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-baseline-rows", type=int, default=900)
    parser.add_argument("--expected-baseline-correct", type=int, default=631)
    parser.add_argument("--calibration-count", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir or args.method_run_dir / "partial_same_subset_vs_baseline"
    method_rows = read_raw_shards(args.method_run_dir)
    method_judges_list = read_jsonl_snapshot(
        args.method_run_dir / "stream_judge" / "judge_results.jsonl"
    )
    baseline_rows = read_raw_shards(args.baseline_run_dir)
    baseline_judges_list = read_jsonl_snapshot(
        args.baseline_run_dir / "stream_judge" / "judge_results.jsonl"
    )
    method_raw = unique_by_id(method_rows, "method raw")
    method_judges = unique_by_id(method_judges_list, "method judge")
    baseline_raw = unique_by_id(baseline_rows, "baseline raw")
    baseline_judges = unique_by_id(baseline_judges_list, "baseline judge")

    baseline_correct = sum(
        row.get("final_is_correct") is True for row in baseline_judges.values()
    )
    if len(baseline_raw) != args.expected_baseline_rows:
        raise ValueError(
            f"baseline raw rows={len(baseline_raw)}, expected {args.expected_baseline_rows}"
        )
    if len(baseline_judges) != args.expected_baseline_rows:
        raise ValueError(
            f"baseline judge rows={len(baseline_judges)}, expected {args.expected_baseline_rows}"
        )
    if baseline_correct != args.expected_baseline_correct:
        raise ValueError(
            f"baseline correct={baseline_correct}, expected {args.expected_baseline_correct}; "
            "refusing to compare against a different baseline"
        )

    method_raw_ids = set(method_raw)
    method_judged_ids = method_raw_ids & set(method_judges)
    missing_baseline_raw = method_judged_ids - set(baseline_raw)
    missing_baseline_judge = method_judged_ids - set(baseline_judges)
    if missing_baseline_raw or missing_baseline_judge:
        raise ValueError(
            f"baseline is missing paired ids: raw={sorted(missing_baseline_raw)}, "
            f"judge={sorted(missing_baseline_judge)}"
        )
    ids = sorted(method_judged_ids, key=lambda key: int(method_raw[key]["global_index"]))
    if not ids:
        raise ValueError("no method rows have both a raw prediction and a judge result")
    for sample_id in ids:
        method_index = int(method_raw[sample_id].get("global_index", -1))
        baseline_index = int(baseline_raw[sample_id].get("global_index", -1))
        if method_index != baseline_index:
            raise ValueError(
                f"global_index mismatch for {sample_id}: method={method_index}, baseline={baseline_index}"
            )
        if str(method_raw[sample_id].get("ground_truth")) != str(
            baseline_raw[sample_id].get("ground_truth")
        ):
            raise ValueError(f"ground_truth mismatch for {sample_id}")

    calibration_ids = [
        sample_id
        for sample_id in ids
        if int(method_raw[sample_id]["global_index"]) < args.calibration_count
    ]
    holdout_ids = [
        sample_id
        for sample_id in ids
        if int(method_raw[sample_id]["global_index"]) >= args.calibration_count
    ]
    report = {
        "snapshot": {
            "created_at": datetime.now().astimezone().isoformat(),
            "method_run_dir": str(args.method_run_dir),
            "baseline_run_dir": str(args.baseline_run_dir),
            "method_raw_rows": len(method_raw),
            "method_judged_rows": len(method_judged_ids),
            "method_pending_judge_rows": len(method_raw_ids - method_judged_ids),
            "paired_rows": len(ids),
            "minimum_global_index": min(int(method_raw[key]["global_index"]) for key in ids),
            "maximum_global_index": max(int(method_raw[key]["global_index"]) for key in ids),
            "ids": ids,
        },
        "baseline_validation": {
            "expected_rows": args.expected_baseline_rows,
            "expected_correct": args.expected_baseline_correct,
            "raw_rows": len(baseline_raw),
            "judged_rows": len(baseline_judges),
            "correct": baseline_correct,
            "accuracy": baseline_correct / len(baseline_judges),
            "pass": True,
        },
        "comparisons": {
            "all": compare_subset(
                ids, method_raw, method_judges, baseline_raw, baseline_judges
            ),
            "calibration_100": compare_subset(
                calibration_ids, method_raw, method_judges, baseline_raw, baseline_judges
            ),
            "holdout_800": compare_subset(
                holdout_ids, method_raw, method_judges, baseline_raw, baseline_judges
            ),
        },
    }
    details = paired_rows(ids, method_raw, method_judges, baseline_raw, baseline_judges)
    json_path = output_dir / "partial_same_subset_latest.json"
    markdown_path = output_dir / "partial_same_subset_latest.md"
    paired_path = output_dir / "partial_same_subset_rows_latest.jsonl"
    fsync_json(json_path, report)
    fsync_jsonl(paired_path, details)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "paired_rows": len(ids),
                "method_accuracy": report["comparisons"]["all"]["method"]["accuracy"],
                "baseline_accuracy": report["comparisons"]["all"]["baseline"]["accuracy"],
                "output_json": str(json_path),
                "output_markdown": str(markdown_path),
                "output_paired_jsonl": str(paired_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
