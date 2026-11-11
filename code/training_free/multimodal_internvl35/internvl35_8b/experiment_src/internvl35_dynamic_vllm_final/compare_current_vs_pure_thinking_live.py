#!/usr/bin/env python3
"""Live same-subset comparison for the InternVL3.5-2B dynamic run.

The script is read-only.  It snapshots newline-terminated raw/judge JSONL
records, intersects the two runs by sample ``id``, and refreshes the report
every 60 seconds while the dynamic job is still appending results.

Typical remote usage::

    python -u compare_current_vs_pure_thinking_live.py
    python -u compare_current_vs_pure_thinking_live.py --once

``output_tokens`` is the generation count recorded by each runner.  The
forced method prefix (``<think>...Okay``) is prompt-side and is intentionally
not added again here, matching the existing experiment comparison scripts.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CURRENT_RUN = Path(
    "/path/to/aris_data/VLM/ARIS_internvl35_2b_dynamic_vllm/results/"
    "mmmuval_internvl35_2b_aris_dynamic_closedthink_okay_t06_topp095_topk50_"
    "rp10_seed0_32k_v1"
)
DEFAULT_BASELINE_RUN = Path(
    "/path/to/aris_data/VLM/ARIS_internvl35_2b_baseline/results/"
    "mmmuval_internvl35_2b_official_thinking_t06_topp095_topk50_"
    "rp10_seed0_32k_v1_native_sampler"
)
DEFAULT_JOB_ID = ""
DEFAULT_INTERVAL = 60.0
MAX_NEW_TOKENS = 32768
TAIL_THRESHOLD = math.ceil(0.8 * MAX_NEW_TOKENS)
SQUEUE = Path("squeue")


@dataclass(frozen=True)
class RunData:
    name: str
    run_dir: Path
    raw: dict[str, dict[str, Any]]
    judges: dict[str, dict[str, Any]]
    raw_lines: int
    malformed_raw: int
    judge_lines: int
    malformed_judge: int
    judge_path: Path | None
    duplicate_raw: int
    duplicate_judge: int


def snapshot_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read only complete JSONL lines; a concurrently written final line is skipped."""
    if not path.is_file():
        return [], 0
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        newline = payload.rfind(b"\n")
        payload = payload[: newline + 1] if newline >= 0 else b""
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in payload.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            malformed += 1
    return rows, malformed


def raw_paths(run_dir: Path) -> list[Path]:
    paths = [Path(name) for name in sorted(glob.glob(str(run_dir / "raw" / "predictions*.jsonl")))]
    if not paths and (run_dir / "predictions.jsonl").is_file():
        paths = [run_dir / "predictions.jsonl"]
    return paths


def indexed_rows(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    result: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        sample_id = str(row.get("id") or "")
        if not sample_id:
            continue
        if sample_id in result:
            duplicates += 1
        result[sample_id] = row
    return result, duplicates


def choose_judge_path(run_dir: Path) -> Path | None:
    # Prefer the finalized v3 judge only after it has a non-empty result file;
    # otherwise use the streaming judge, which is updated during evaluation.
    candidates = [
        run_dir / "judge_v3" / "judge_results.jsonl",
        run_dir / "stream_judge" / "judge_results.jsonl",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            if path.parent.name == "judge_v3" and not (path.parent / "JUDGE_COMPLETE").exists():
                # A partially created v3 directory must not hide live stream results.
                continue
            return path
    return None


def load_run(name: str, run_dir: Path) -> RunData:
    raw_rows: list[dict[str, Any]] = []
    malformed_raw = 0
    for path in raw_paths(run_dir):
        rows, malformed = snapshot_jsonl(path)
        raw_rows.extend(rows)
        malformed_raw += malformed
    raw, duplicate_raw = indexed_rows(raw_rows)

    judge_path = choose_judge_path(run_dir)
    judge_rows, malformed_judge = snapshot_jsonl(judge_path) if judge_path else ([], 0)
    judges, duplicate_judge = indexed_rows(judge_rows)
    return RunData(
        name=name,
        run_dir=run_dir,
        raw=raw,
        judges=judges,
        raw_lines=len(raw_rows),
        malformed_raw=malformed_raw,
        judge_lines=len(judge_rows),
        malformed_judge=malformed_judge,
        judge_path=judge_path,
        duplicate_raw=duplicate_raw,
        duplicate_judge=duplicate_judge,
    )


def judge_correct(judge: dict[str, Any]) -> bool | None:
    for key in ("final_is_correct", "rule_is_correct", "is_correct", "api_is_correct"):
        if key in judge and judge[key] is not None:
            return bool(judge[key])
    return None


def output_tokens(row: dict[str, Any]) -> int:
    for key in ("output_tokens", "num_output_tokens", "completion_tokens"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return max(0, int(value))
    token_ids = row.get("output_token_ids")
    return len(token_ids) if isinstance(token_ids, list) else 0


def valid_pair_ids(current: RunData, baseline: RunData) -> list[str]:
    ids: list[str] = []
    for sample_id in set(current.raw) & set(current.judges) & set(baseline.raw) & set(baseline.judges):
        current_row = current.raw[sample_id]
        baseline_row = baseline.raw[sample_id]
        if current_row.get("error") or baseline_row.get("error"):
            continue
        if judge_correct(current.judges[sample_id]) is None or judge_correct(baseline.judges[sample_id]) is None:
            continue
        ids.append(sample_id)
    return sorted(ids, key=lambda sample_id: int(current.raw[sample_id].get("global_index", 10**9)))


def quantile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def metric(run: RunData, ids: list[str]) -> dict[str, Any]:
    rows = [run.raw[sample_id] for sample_id in ids]
    judges = [run.judges[sample_id] for sample_id in ids]
    tokens = [output_tokens(row) for row in rows]
    correct = sum(judge_correct(judge) is True for judge in judges)
    return {
        "n": len(ids),
        "correct": correct,
        "accuracy": correct / len(ids) if ids else None,
        "mean_tokens": statistics.fmean(tokens) if tokens else None,
        "median_tokens": statistics.median(tokens) if tokens else None,
        "p90_tokens": quantile(tokens, 0.90),
        "total_tokens": sum(tokens),
        "ge8k": sum(value >= 8192 for value in tokens),
        "ge16k": sum(value >= 16384 for value in tokens),
        "ge80": sum(value >= TAIL_THRESHOLD for value in tokens),
        "hit_max": sum(bool(row.get("hit_max_new_tokens")) for row in rows),
        "parse_failed": sum(row.get("parseable") is False for row in rows),
        "runtime_errors": sum(bool(row.get("error")) for row in rows),
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def num(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f}"


def status(job_id: str | None) -> str:
    if not job_id:
        return "未指定 job"
    try:
        result = subprocess.run(
            [str(SQUEUE), "-h", "-j", str(job_id), "-o", "%T | %M | %R"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "squeue 不可用"
    return result.stdout.strip() or "不在 squeue（可能已完成）"


def markers(run_dir: Path) -> str:
    names: list[str] = []
    for name in ("INFERENCE_COMPLETE", "EVALUATION_COMPLETE", "INFERENCE_FAILED", "EVALUATION_FAILED"):
        if (run_dir / name).exists():
            names.append(name)
    if (run_dir / "stream_judge" / "JUDGE_COMPLETE").exists():
        names.append("JUDGE_COMPLETE")
    if (run_dir / "judge_v3" / "JUDGE_COMPLETE").exists():
        names.append("JUDGE_V3_COMPLETE")
    return ", ".join(names) if names else "运行中/无完成标志"


def question_type(run: RunData, sample_id: str) -> str:
    judge = run.judges.get(sample_id, {})
    raw = run.raw.get(sample_id, {})
    return str(judge.get("question_type") or raw.get("question_type") or "unknown")


def print_metric_table(current: dict[str, Any], baseline: dict[str, Any]) -> None:
    print("| 版本 | 题数 | 正确/准确率 | 平均 token | 中位 token | P90 token | 总 token | ≥8K | ≥16K | ≥80%上限 | 卡32K | 解析失败 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, item in (("当前 dynamic + Okay", current), ("纯思考 baseline", baseline)):
        print(
            f"| {label} | {item['n']} | {item['correct']}/{item['n']} ({pct(item['accuracy'])}) "
            f"| {num(item['mean_tokens'])} | {num(item['median_tokens'])} | {num(item['p90_tokens'])} "
            f"| {item['total_tokens']:,} | {item['ge8k']} | {item['ge16k']} | {item['ge80']} "
            f"| {item['hit_max']} | {item['parse_failed']} |"
        )


def print_group_table(label: str, groups: list[tuple[str, list[str]]], current: RunData, baseline: RunData) -> None:
    print(f"\n{label}")
    print("| 分组 | 题数 | dynamic正确 | baseline正确 | 准确率差 | dynamic平均token | baseline平均token | token差 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for group, ids in groups:
        if not ids:
            print(f"| {group} | 0 | — | — | — | — | — | — |")
            continue
        cm = metric(current, ids)
        bm = metric(baseline, ids)
        delta_pp = (cm["correct"] - bm["correct"]) / len(ids) * 100
        token_delta = (cm["mean_tokens"] or 0) - (bm["mean_tokens"] or 0)
        print(
            f"| {group} | {len(ids)} | {cm['correct']}/{len(ids)} | {bm['correct']}/{len(ids)} "
            f"| {delta_pp:+.2f} pp | {num(cm['mean_tokens'])} | {num(bm['mean_tokens'])} | {token_delta:+,.1f} |"
        )


def render(current: RunData, baseline: RunData, job_id: str | None) -> None:
    ids = valid_pair_ids(current, baseline)
    cm = metric(current, ids)
    bm = metric(baseline, ids)
    current_correct = {sample_id: judge_correct(current.judges[sample_id]) is True for sample_id in ids}
    baseline_correct = {sample_id: judge_correct(baseline.judges[sample_id]) is True for sample_id in ids}
    both = sum(current_correct[sample_id] and baseline_correct[sample_id] for sample_id in ids)
    method_only = sum(current_correct[sample_id] and not baseline_correct[sample_id] for sample_id in ids)
    baseline_only = sum(not current_correct[sample_id] and baseline_correct[sample_id] for sample_id in ids)
    token_saved = bm["total_tokens"] - cm["total_tokens"]
    token_reduction = token_saved / bm["total_tokens"] if bm["total_tokens"] else None
    accuracy_delta = (cm["correct"] - bm["correct"]) / len(ids) if ids else None

    print(f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"当前任务 {job_id or '—'}：{status(job_id)}")
    print(f"当前完成标志：{markers(current.run_dir)}")
    print(f"当前 raw/judge：{len(current.raw)}/{len(current.judges)}（文件行 {current.raw_lines}/{current.judge_lines}）；")
    print(f"baseline raw/judge：{len(baseline.raw)}/{len(baseline.judges)}；严格公共且已 judge：{len(ids)} 题")
    print(f"当前 judge 文件：{current.judge_path or '尚未生成'}；baseline judge 文件：{baseline.judge_path or '尚未生成'}")

    print("\n## 相同题目子集总体对比")
    if not ids:
        print("当前还没有可严格配对且双方都完成 judge 的题目。")
        return
    print_metric_table(cm, bm)
    print(
        f"\n差异（dynamic - baseline）：准确率 {accuracy_delta * 100:+.2f} pp，"
        f"正确题数 {cm['correct'] - bm['correct']:+d}；平均 token "
        f"{(cm['mean_tokens'] or 0) - (bm['mean_tokens'] or 0):+,.1f}；"
        f"累计节省 {token_saved:,} token（{pct(token_reduction)}）。"
    )
    print(
        f"逐题结果：两者都正确 {both}，仅 dynamic 正确 {method_only}，"
        f"仅 baseline 正确 {baseline_only}，两者都错 {len(ids) - both - method_only - baseline_only}。"
    )

    bucket_names = ("极短 0–128", "短 129–1023", "中 1024–8191", "长 ≥8192")
    def bucket(tokens: int) -> str:
        if tokens <= 128:
            return bucket_names[0]
        if tokens <= 1023:
            return bucket_names[1]
        if tokens <= 8191:
            return bucket_names[2]
        return bucket_names[3]
    groups = [(name, [sample_id for sample_id in ids if bucket(output_tokens(current.raw[sample_id])) == name]) for name in bucket_names]
    print_group_table("按当前 dynamic 输出长度分组（每组与 baseline 同题比较）", groups, current, baseline)

    type_groups = []
    for kind, label in (("multiple_choice", "多选题"), ("open", "开放题")):
        type_groups.append((label, [sample_id for sample_id in ids if question_type(current, sample_id) == kind]))
    print_group_table("按题型分组", type_groups, current, baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--once", action="store_true", help="只输出一次，不循环刷新")
    parser.add_argument("--no-clear", action="store_true", help="刷新时不清屏，保留历史输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = True
    try:
        while True:
            current = load_run("current", args.current_run)
            baseline = load_run("baseline", args.baseline_run)
            if not args.no_clear and not first:
                sys.stdout.write("\033[2J\033[H")
            first = False
            print(f"当前目录：{args.current_run}\nbaseline目录：{args.baseline_run}")
            render(current, baseline, args.job_id)
            sys.stdout.flush()
            if args.once:
                return
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n已停止自动刷新。")


if __name__ == "__main__":
    main()
