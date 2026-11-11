from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "to_dynamic16": Path(
        "/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/"
        "mmmu900_dynamic_closedthink_to_t07_seed0_rp105_16384"
    ),
    "closed_dynamic16": Path(
        "/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/"
        "mmmu900_dynamic_closedthink_only_t07_seed0_rp105_16384"
    ),
    "closed_baseline16": Path(
        "/path/to/aris_data/VLM/experiments/"
        "mmmuval_qwen3vl4b_prefix_closedthink_full900_t07_seed0_rp105_16384"
    ),
    "pure_baseline32": Path(
        "/path/to/aris_data/VLM/experiments/"
        "mmmuval_qwen3vl4b_full900_vllm_l20_vlmevalprompt_p0_temp07_seed0_rp105_32768"
    ),
}


@dataclass
class LoadedRun:
    rows: dict[int, dict[str, Any]]
    judges: dict[int, dict[str, Any]]
    raw_lines: int
    malformed_lines: int
    judge_lines: int
    malformed_judge_lines: int
    judge_set: str


def read_run(run_dir: Path) -> LoadedRun:
    rows: dict[int, dict[str, Any]] = {}
    raw_lines = 0
    malformed = 0
    pattern = str(run_dir / "raw" / "predictions.shard*.jsonl")
    for name in sorted(glob.glob(pattern)):
        with Path(name).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_lines += 1
                try:
                    row = json.loads(line)
                    index = int(row["global_index"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    malformed += 1
                    continue
                rows[index] = row
    judges: dict[int, dict[str, Any]] = {}
    judge_lines = 0
    malformed_judges = 0
    corrected_dir = run_dir / "stream_judge_corrected"
    if (corrected_dir / "JUDGE_COMPLETE").is_file():
        judge_path = corrected_dir / "judge_results.jsonl"
        judge_set = "corrected"
    else:
        judge_path = run_dir / "stream_judge" / "judge_results.jsonl"
        judge_set = "official"
    if judge_path.is_file():
        with judge_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                judge_lines += 1
                try:
                    judge = json.loads(line)
                    index = int(judge["global_index"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    malformed_judges += 1
                    continue
                judges[index] = judge
    return LoadedRun(
        rows=rows,
        judges=judges,
        raw_lines=raw_lines,
        malformed_lines=malformed,
        judge_lines=judge_lines,
        malformed_judge_lines=malformed_judges,
        judge_set=judge_set,
    )


def is_valid_completion(row: dict[str, Any]) -> bool:
    # Old dynamic rows may carry a trajectory-audit error despite containing a
    # valid prediction.  Keep those rows; reject only empty placeholder errors.
    return row.get("prediction") is not None or bool(row.get("raw_output"))


def output_tokens(row: dict[str, Any]) -> int:
    for field in ("output_tokens", "num_output_tokens", "completion_tokens"):
        value = row.get(field)
        if isinstance(value, (int, float)):
            return int(value)
    return len(row.get("output_token_ids") or [])


def is_correct(row: dict[str, Any]) -> bool:
    return row.get("is_correct") is True


def correctness(run: LoadedRun, index: int, source: str) -> bool | None:
    if source == "judge":
        judge = run.judges.get(index)
        return None if judge is None else judge.get("final_is_correct") is True
    row = run.rows.get(index)
    return None if row is None or not is_valid_completion(row) else is_correct(row)


def eligible(run: LoadedRun, index: int, source: str) -> bool:
    row = run.rows.get(index)
    return bool(
        row is not None
        and is_valid_completion(row)
        and correctness(run, index, source) is not None
    )


def nearest_rank(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def metric(rows: list[dict[str, Any]], correct_values: list[bool]) -> dict[str, Any]:
    tokens = [output_tokens(row) for row in rows]
    correct = sum(correct_values)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "sum_tokens": sum(tokens),
        "mean_tokens": statistics.fmean(tokens) if tokens else 0.0,
        "median_tokens": statistics.median(tokens) if tokens else 0.0,
        "p90_tokens": nearest_rank(tokens, 0.90) if tokens else 0,
        "ge8k": sum(value >= 8_000 for value in tokens),
        "ge16k": sum(value >= 16_000 for value in tokens),
        "hit_cap": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
        "parse_fail": sum(row.get("parseable") is False for row in rows),
    }


def paired(
    latest: LoadedRun, other: LoadedRun, anchor: set[int], accuracy_source: str
) -> dict[str, Any]:
    indices = sorted(
        index
        for index in anchor
        if eligible(other, index, accuracy_source)
    )
    latest_rows = [latest.rows[index] for index in indices]
    other_rows = [other.rows[index] for index in indices]
    latest_correct = [bool(correctness(latest, index, accuracy_source)) for index in indices]
    other_correct = [bool(correctness(other, index, accuracy_source)) for index in indices]
    transitions = Counter(
        zip(latest_correct, other_correct)
    )
    latest_metric = metric(latest_rows, latest_correct)
    other_metric = metric(other_rows, other_correct)
    token_ratio = (
        latest_metric["sum_tokens"] / other_metric["sum_tokens"]
        if other_metric["sum_tokens"]
        else 0.0
    )
    return {
        "indices": indices,
        "latest": latest_metric,
        "other": other_metric,
        "latest_only_correct": transitions[(True, False)],
        "other_only_correct": transitions[(False, True)],
        "both_correct": transitions[(True, True)],
        "both_wrong": transitions[(False, False)],
        "token_ratio": token_ratio,
    }


def row_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * width for width in widths])]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def render(args: argparse.Namespace) -> str:
    run_dirs = {
        "to_dynamic16": args.latest_run,
        "closed_dynamic16": args.closed_dynamic_run,
        "closed_baseline16": args.closed_baseline_run,
        "pure_baseline32": args.pure_baseline_run,
    }
    loaded = {name: read_run(path) for name, path in run_dirs.items()}
    latest_run = loaded["to_dynamic16"]
    latest = latest_run.rows
    raw_anchor = {index for index, row in latest.items() if is_valid_completion(row)}
    anchor = {
        index for index in raw_anchor if eligible(latest_run, index, args.accuracy_source)
    }

    lines = [
        f"MMMUVal latest comparison @ {datetime.now().isoformat(timespec='seconds')}",
        f"Accuracy source: {args.accuracy_source}",
        f"Anchor: to_dynamic16 raw valid = {len(raw_anchor)}/900; accuracy-covered = {len(anchor)}/900",
    ]
    raw_rows = []
    for name, item in loaded.items():
        valid = sum(is_valid_completion(row) for row in item.rows.values())
        placeholders = len(item.rows) - valid
        raw_rows.append(
            [
                name,
                str(item.raw_lines),
                str(valid),
                str(item.judge_lines),
                item.judge_set,
                str(placeholders),
                f"{item.malformed_lines}/{item.malformed_judge_lines}",
            ]
        )
    lines += ["", "Data health", row_table(
        ["run", "raw", "valid", "judged", "judge_set", "placeholders", "bad(raw/judge)"], raw_rows
    )]

    comparisons: dict[str, dict[str, Any]] = {}
    summary_rows = []
    latest_rows = [latest[index] for index in sorted(anchor)]
    latest_metric = metric(
        latest_rows,
        [bool(correctness(latest_run, index, args.accuracy_source)) for index in sorted(anchor)],
    )
    summary_rows.append(
        [
            "to_dynamic16",
            str(latest_metric["n"]),
            f"{latest_metric['correct']}/{latest_metric['n']}",
            f"{100 * latest_metric['accuracy']:.2f}%",
            f"{latest_metric['mean_tokens']:.1f}",
            f"{latest_metric['median_tokens']:.1f}",
            str(latest_metric["p90_tokens"]),
            str(latest_metric["ge8k"]),
            str(latest_metric["ge16k"]),
            str(latest_metric["hit_cap"]),
        ]
    )
    for name in ("closed_dynamic16", "closed_baseline16", "pure_baseline32"):
        result = paired(latest_run, loaded[name], anchor, args.accuracy_source)
        comparisons[name] = result
        value = result["other"]
        summary_rows.append(
            [
                name,
                str(value["n"]),
                f"{value['correct']}/{value['n']}",
                f"{100 * value['accuracy']:.2f}%",
                f"{value['mean_tokens']:.1f}",
                f"{value['median_tokens']:.1f}",
                str(value["p90_tokens"]),
                str(value["ge8k"]),
                str(value["ge16k"]),
                str(value["hit_cap"]),
            ]
        )
    lines += ["", "Same-subset metrics", row_table(
        ["run", "n", "correct", "accuracy", "mean_tok", "median", "p90", ">=8k", ">=16k", "hit_cap"],
        summary_rows,
    )]

    paired_rows = []
    for name, result in comparisons.items():
        latest_value = result["latest"]
        other_value = result["other"]
        paired_rows.append(
            [
                f"to vs {name}",
                str(len(result["indices"])),
                f"{100 * (latest_value['accuracy'] - other_value['accuracy']):+.2f}pp",
                f"{100 * (result['token_ratio'] - 1):+.2f}%",
                str(result["latest_only_correct"]),
                str(result["other_only_correct"]),
                str(result["both_correct"]),
                str(result["both_wrong"]),
            ]
        )
    lines += ["", "Paired deltas (latest To dynamic minus comparison)", row_table(
        ["pair", "n", "acc_delta", "token_delta", "To_only", "other_only", "both_C", "both_W"],
        paired_rows,
    )]

    bins = [
        ("<1k", 0, 1_000),
        ("1k-8k", 1_000, 8_000),
        ("8k-16k", 8_000, 16_000),
        (">=16k", 16_000, 10**18),
    ]
    bin_rows = []
    for label, low, high in bins:
        indices = [
            index for index in sorted(anchor) if low <= output_tokens(latest[index]) < high
        ]
        if not indices:
            continue
        for name in run_dirs:
            source = loaded[name]
            selected_indices = [
                index for index in indices if eligible(source, index, args.accuracy_source)
            ]
            selected = [source.rows[index] for index in selected_indices]
            value = metric(
                selected,
                [
                    bool(correctness(source, index, args.accuracy_source))
                    for index in selected_indices
                ],
            )
            bin_rows.append(
                [label, name, str(value["n"]), f"{value['correct']}/{value['n']}", f"{value['mean_tokens']:.1f}"]
            )
    lines += ["", "Breakdown by latest To trajectory length", row_table(
        ["latest_bin", "run", "n", "correct", "mean_tok"], bin_rows
    )]

    if args.show_disagreements > 0:
        other_name = args.disagreement_against
        source = loaded[other_name]
        result = comparisons[other_name]
        disagreement_rows = []
        for index in result["indices"]:
            left, right = latest[index], source.rows[index]
            left_correct = bool(correctness(latest_run, index, args.accuracy_source))
            right_correct = bool(correctness(source, index, args.accuracy_source))
            if left_correct == right_correct:
                continue
            outcome = "To_gain" if left_correct else "To_loss"
            disagreement_rows.append(
                [
                    outcome,
                    str(index),
                    str(left.get("id")),
                    str(left.get("ground_truth")),
                    f"{left.get('prediction')}({output_tokens(left)})",
                    f"{right.get('prediction')}({output_tokens(right)})",
                ]
            )
        disagreement_rows = disagreement_rows[: args.show_disagreements]
        lines += ["", f"Disagreements against {other_name}", row_table(
            ["outcome", "index", "id", "GT", "To(pred/tok)", "other(pred/tok)"],
            disagreement_rows,
        )]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the latest closed-think+To dynamic run on exact paired subsets."
    )
    parser.add_argument("--latest-run", type=Path, default=DEFAULT_RUNS["to_dynamic16"])
    parser.add_argument("--closed-dynamic-run", type=Path, default=DEFAULT_RUNS["closed_dynamic16"])
    parser.add_argument("--closed-baseline-run", type=Path, default=DEFAULT_RUNS["closed_baseline16"])
    parser.add_argument("--pure-baseline-run", type=Path, default=DEFAULT_RUNS["pure_baseline32"])
    parser.add_argument("--show-disagreements", type=int, default=20)
    parser.add_argument(
        "--accuracy-source",
        choices=("judge", "raw"),
        default="judge",
        help="Use official stream-judge decisions by default; raw is_correct is diagnostic only.",
    )
    parser.add_argument(
        "--disagreement-against",
        choices=("closed_dynamic16", "closed_baseline16", "pure_baseline32"),
        default="closed_dynamic16",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        try:
            print(render(args), flush=True)
        except Exception as exc:
            print(f"comparison failed: {type(exc).__name__}: {exc}", flush=True)
            if not args.watch:
                raise
        if not args.watch:
            return
        print(f"\nRefreshing in {args.interval:g}s; press Ctrl-C to stop.\n", flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
