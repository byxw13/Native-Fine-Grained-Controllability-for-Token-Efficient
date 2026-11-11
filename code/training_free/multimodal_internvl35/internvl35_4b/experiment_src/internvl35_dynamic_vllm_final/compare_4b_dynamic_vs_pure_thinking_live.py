#!/usr/bin/env python3
"""InternVL3.5-4B 的 dynamic 与纯思考 baseline 同题目子集实时对比。

脚本只读实验结果目录，不会修改推理或 judge 状态。每次快照只纳入：

* 两个 run 都有完整 raw 记录；
* 两个 run 都有对应的 judge 记录；
* 记录没有运行错误，且 judge 已给出布尔正确性。

因此在动态任务尚未跑完时，报告的是“当前双方都完成的严格公共子集”，不会把
baseline 已完成但 dynamic 尚未完成的题目混入分母。默认每 3600 秒刷新一次。

示例（远程环境）：

    python -u compare_4b_dynamic_vs_pure_thinking_live.py
    python -u compare_4b_dynamic_vs_pure_thinking_live.py --once --no-clear

judge 优先使用已完成的 ``judge_v3``，否则使用正在逐题更新的 ``stream_judge``。
``output_tokens`` 优先使用 runner 记录的生成长度，不把 prompt token 算入消耗。
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CURRENT_RUN = Path(
    "/path/to/aris_data/VLM/ARIS_internvl35_4b_dynamic_vllm/results/"
    "mmmuval_internvl35_4b_aris_dynamic_closedthink_okay_t06_topp095_topk50_"
    "rp10_seed0_32k_v2"
)
DEFAULT_BASELINE_RUN = Path(
    "/path/to/aris_data/VLM/ARIS_internvl35_4b_baseline/results/"
    "mmmuval_internvl35_4b_official_thinking_t06_topp095_topk50_"
    "rp10_seed0_32k_v1_native_sampler"
)
DEFAULT_JOB_ID = ""
DEFAULT_INTERVAL = 3600.0
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


def snapshot_jsonl(path: Path | None) -> tuple[list[dict[str, Any]], int]:
    """读取完整 JSONL 行，忽略并发写入中的半行。"""
    if path is None or not path.is_file():
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
    paths = [
        Path(name)
        for name in sorted(glob.glob(str(run_dir / "raw" / "predictions*.jsonl")))
    ]
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
    """v3 完成后优先 v3；动态运行阶段使用逐题 stream judge。"""
    candidates: list[Path] = []
    final_v3 = run_dir / "judge_v3" / "judge_results.jsonl"
    if final_v3.is_file() and (final_v3.parent / "JUDGE_COMPLETE").exists():
        candidates.append(final_v3)
    candidates.extend(
        Path(name)
        for name in sorted(
            glob.glob(str(run_dir / "judge_v3*" / "judge_results.jsonl")),
            reverse=True,
        )
        if Path(name) != final_v3
    )
    candidates.append(run_dir / "stream_judge" / "judge_results.jsonl")
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file() and path.stat().st_size > 0:
            # 未完成的 v3 目录不应覆盖仍在更新的 stream judge。
            if path.parent.name.startswith("judge_v3") and not (
                path.parent / "JUDGE_COMPLETE"
            ).exists():
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
    judge_rows, malformed_judge = snapshot_jsonl(judge_path)
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


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "correct", "pass"}:
            return True
        if normalized in {"false", "0", "no", "n", "incorrect", "wrong", "fail"}:
            return False
    return None


def judge_correct(judge: dict[str, Any]) -> bool | None:
    for key in ("final_is_correct", "rule_is_correct", "is_correct", "api_is_correct"):
        if key in judge and judge[key] is not None:
            parsed = as_bool(judge[key])
            if parsed is not None:
                return parsed
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
    common = set(current.raw) & set(current.judges) & set(baseline.raw) & set(baseline.judges)
    for sample_id in common:
        current_row = current.raw[sample_id]
        baseline_row = baseline.raw[sample_id]
        if current_row.get("error") or baseline_row.get("error"):
            continue
        if judge_correct(current.judges[sample_id]) is None:
            continue
        if judge_correct(baseline.judges[sample_id]) is None:
            continue
        ids.append(sample_id)
    return sorted(
        ids,
        key=lambda sample_id: int(current.raw[sample_id].get("global_index", 10**9)),
    )


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
        "hit_max": sum(as_bool(row.get("hit_max_new_tokens")) is True for row in rows),
        "parse_failed": sum(as_bool(row.get("parseable")) is False for row in rows),
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
    for name in (
        "INFERENCE_COMPLETE",
        "EVALUATION_COMPLETE",
        "INFERENCE_FAILED",
        "EVALUATION_FAILED",
    ):
        if (run_dir / name).exists():
            names.append(name)
    if (run_dir / "stream_judge" / "JUDGE_COMPLETE").exists():
        names.append("STREAM_JUDGE_COMPLETE")
    if (run_dir / "judge_v3" / "JUDGE_COMPLETE").exists():
        names.append("JUDGE_V3_COMPLETE")
    return ", ".join(names) if names else "运行中/无完成标志"


def render(current: RunData, baseline: RunData, job_id: str | None) -> None:
    ids = valid_pair_ids(current, baseline)
    print(f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"当前任务 {job_id or '—'}：{status(job_id)}")
    print(f"dynamic 标志：{markers(current.run_dir)}")
    print(f"dynamic raw/judge：{len(current.raw)}/{len(current.judges)}（文件行 {current.raw_lines}/{current.judge_lines}）")
    print(f"baseline raw/judge：{len(baseline.raw)}/{len(baseline.judges)}（文件行 {baseline.raw_lines}/{baseline.judge_lines}）")
    print(f"严格公共且双方已 judge：{len(ids)} 题")
    print(f"dynamic judge：{current.judge_path or '尚未生成'}")
    print(f"baseline judge：{baseline.judge_path or '尚未生成'}")

    if not ids:
        print("\n当前还没有可严格配对且双方都完成 judge 的题目。")
        return

    cm = metric(current, ids)
    bm = metric(baseline, ids)
    print("\n## 相同题目子集总体对比")
    print("| 版本 | 题数 | 正确题数 | 准确率 | 平均 token | 中位 token | P90 token | 总 token | ≥8K | ≥16K | ≥80%上限 | 卡 32K |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, item in (("当前 dynamic", cm), ("纯思考 baseline", bm)):
        print(
            f"| {label} | {item['n']} | {item['correct']}/{item['n']} | {pct(item['accuracy'])} "
            f"| {num(item['mean_tokens'])} | {num(item['median_tokens'])} | {num(item['p90_tokens'])} "
            f"| {item['total_tokens']:,} | {item['ge8k']} | {item['ge16k']} | {item['ge80']} | {item['hit_max']} |"
        )

    baseline_total = bm["total_tokens"]
    saved = baseline_total - cm["total_tokens"]
    reduction = saved / baseline_total if baseline_total else None
    accuracy_delta = cm["correct"] - bm["correct"]
    print(
        f"\n差异（dynamic - baseline）：正确题数 {accuracy_delta:+d}，"
        f"准确率 {(cm['accuracy'] - bm['accuracy']) * 100:+.2f} pp；"
        f"平均 token {(cm['mean_tokens'] or 0) - (bm['mean_tokens'] or 0):+,.1f}；"
        f"累计节省 {saved:,} token（{pct(reduction)}）。"
    )
    both = method_only = baseline_only = both_wrong = 0
    for sample_id in ids:
        mc = judge_correct(current.judges[sample_id]) is True
        bc = judge_correct(baseline.judges[sample_id]) is True
        if mc and bc:
            both += 1
        elif mc:
            method_only += 1
        elif bc:
            baseline_only += 1
        else:
            both_wrong += 1
    print(
        f"逐题结果：两者都正确 {both}；仅 dynamic 正确 {method_only}；"
        f"仅 baseline 正确 {baseline_only}；两者都错 {both_wrong}。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="刷新间隔（秒），默认 3600")
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
