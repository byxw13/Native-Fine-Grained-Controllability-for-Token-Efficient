#!/usr/bin/env python3
"""InternVL3.5-8B selector 动态方法与纯思考 baseline 的严格同题子集实时对比。

脚本只读结果目录。每次刷新只纳入两个 run 同时具备：

* 完整 JSONL raw 记录；
* 对应的 stream judge 或已完成的 judge_v3 记录；
* 无推理错误，且 judge 给出了布尔 ``final_is_correct``（兼容旧字段）。

因此动态任务尚未完成时，分母是“当前双方都完成且都已 judge 的严格公共题目
子集”，不会把尚未完成的题目当成错误。默认每 3600 秒刷新一次；用 ``--once``
可只输出一轮。

典型远程用法::

    cd /path/to/aris_data/VLM/ARIS_internvl35_8b_dynamic_vllm/code
    export PYTHONPATH=/path/to/aris_data/VLM/ARIS_internvl35_8b_dynamic_vllm/code
    python -u -m experiment_src.internvl35_dynamic_vllm_final.compare_8b_dynamic_vs_pure_thinking_live

输出 token 只统计生成 token（``output_tokens``），不把 prompt/image token 算入
消耗；两个版本的 prompt 长度差异不会混入本表的输出 token 指标。
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
    "/path/to/aris_data/VLM/ARIS_internvl35_8b_dynamic_vllm/results/"
    "mmmuval_internvl35_8b_aris_dynamic_selector_closedthink_so_okay_t06_"
    "topp095_topk50_rp10_seed0_32k_v5"
)
DEFAULT_BASELINE_RUN = Path(
    "/path/to/aris_data/VLM/ARIS_internvl35_8b_baseline/results/"
    "mmmuval_internvl35_8b_official_thinking_t06_topp095_topk50_"
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
    """只读取完整 JSONL 行，忽略并发写入中的最后半行。"""
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
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        sample_id = str(row.get("id") or "")
        if not sample_id:
            continue
        if sample_id in indexed:
            duplicates += 1
        indexed[sample_id] = row
    return indexed, duplicates


def choose_judge_path(run_dir: Path) -> Path | None:
    """完成的 v3 优先；运行中使用仍在追加的 stream judge。"""
    final_v3 = run_dir / "judge_v3" / "judge_results.jsonl"
    if final_v3.is_file() and (final_v3.parent / "JUDGE_COMPLETE").exists():
        if final_v3.stat().st_size > 0:
            return final_v3
    stream = run_dir / "stream_judge" / "judge_results.jsonl"
    if stream.is_file() and stream.stat().st_size > 0:
        return stream
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
        value = value.strip().lower()
        if value in {"true", "1", "yes", "correct", "pass"}:
            return True
        if value in {"false", "0", "no", "incorrect", "wrong", "fail"}:
            return False
    return None


def judge_correct(judge: dict[str, Any]) -> bool | None:
    # final_is_correct is the v3/stream judge's final decision.  The other
    # fields keep this utility compatible with older InternVL result folders.
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
    common = set(current.raw) & set(current.judges) & set(baseline.raw) & set(baseline.judges)
    ids: list[str] = []
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
    for name in ("INFERENCE_COMPLETE", "EVALUATION_COMPLETE", "INFERENCE_FAILED", "EVALUATION_FAILED"):
        if (run_dir / name).exists():
            names.append(name)
    if (run_dir / "stream_judge" / "JUDGE_COMPLETE").exists():
        names.append("STREAM_JUDGE_COMPLETE")
    if (run_dir / "judge_v3" / "JUDGE_COMPLETE").exists():
        names.append("JUDGE_V3_COMPLETE")
    return ", ".join(names) if names else "运行中/无完成标志"


def question_type(run: RunData, sample_id: str) -> str:
    judge = run.judges.get(sample_id, {})
    raw = run.raw.get(sample_id, {})
    return str(judge.get("question_type") or raw.get("question_type") or "unknown")


def selector_word(row: dict[str, Any]) -> str:
    """Return the cue actually inserted into the method prompt.

    The explicit row field is preferred; the nested audit fallback keeps the
    report useful for rows written by an older runner revision.
    """

    value = row.get("selected_method_cue") or row.get("method_cue")
    if value:
        return str(value)
    audit = row.get("audit")
    if isinstance(audit, dict):
        selector = audit.get("first_token_selector")
        if isinstance(selector, dict) and selector.get("selected_word"):
            return str(selector["selected_word"])
    selector = row.get("first_token_selector")
    if isinstance(selector, dict) and selector.get("selected_word"):
        return str(selector["selected_word"])
    return "unknown"


def token_bucket(value: int) -> str:
    if value <= 128:
        return "极短 0–128"
    if value <= 1023:
        return "短 129–1023"
    if value <= 8191:
        return "中 1024–8191"
    return "长 ≥8192"


def print_group_table(
    title: str,
    groups: list[tuple[str, list[str]]],
    current: RunData,
    baseline: RunData,
) -> None:
    print(f"\n## {title}")
    print("| 分组 | 题数 | dynamic 正确 | baseline 正确 | 正确题数差 | dynamic 平均 token | baseline 平均 token | token 差 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, group_ids in groups:
        if not group_ids:
            print(f"| {label} | 0 | — | — | — | — | — | — |")
            continue
        cm = metric(current, group_ids)
        bm = metric(baseline, group_ids)
        print(
            f"| {label} | {len(group_ids)} | {cm['correct']} | {bm['correct']} "
            f"| {cm['correct'] - bm['correct']:+d} | {num(cm['mean_tokens'])} "
            f"| {num(bm['mean_tokens'])} | {(cm['mean_tokens'] or 0) - (bm['mean_tokens'] or 0):+,.1f} |"
        )


def render(current: RunData, baseline: RunData, job_id: str | None) -> None:
    ids = valid_pair_ids(current, baseline)
    print(f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"正式任务 {job_id or '—'}：{status(job_id)}")
    print(f"dynamic 标志：{markers(current.run_dir)}")
    print(
        f"dynamic raw/judge：{len(current.raw)}/{len(current.judges)}（文件行 "
        f"{current.raw_lines}/{current.judge_lines}，重复 {current.duplicate_raw}/{current.duplicate_judge}）"
    )
    print(
        f"baseline raw/judge：{len(baseline.raw)}/{len(baseline.judges)}（文件行 "
        f"{baseline.raw_lines}/{baseline.judge_lines}，重复 {baseline.duplicate_raw}/{baseline.duplicate_judge}）"
    )
    print(f"严格公共且双方已 judge：{len(ids)} 题")
    print(f"dynamic judge：{current.judge_path or '尚未生成'}")
    print(f"baseline judge：{baseline.judge_path or '尚未生成'}")
    if current.malformed_raw or current.malformed_judge or baseline.malformed_raw or baseline.malformed_judge:
        print(
            f"并发快照中跳过半行/坏行：dynamic raw/judge={current.malformed_raw}/{current.malformed_judge}，"
            f"baseline raw/judge={baseline.malformed_raw}/{baseline.malformed_judge}"
        )

    print("\n## 相同题目子集总体对比")
    if not ids:
        print("当前还没有可严格配对且双方都完成 judge 的题目。")
        return
    cm = metric(current, ids)
    bm = metric(baseline, ids)
    print("| 版本 | 题数 | 正确题数 | 准确率 | 平均 token | 中位 token | P90 token | 总 token | ≥8K | ≥16K | ≥80%上限 | 卡32K |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, item in (("InternVL3.5-8B 我的方法", cm), ("InternVL3.5-8B 纯思考 baseline", bm)):
        print(
            f"| {label} | {item['n']} | {item['correct']} | {pct(item['accuracy'])} "
            f"| {num(item['mean_tokens'])} | {num(item['median_tokens'])} | {num(item['p90_tokens'])} "
            f"| {item['total_tokens']:,} | {item['ge8k']} | {item['ge16k']} | {item['ge80']} | {item['hit_max']} |"
        )
    saved = bm["total_tokens"] - cm["total_tokens"]
    reduction = saved / bm["total_tokens"] if bm["total_tokens"] else None
    print(
        f"\n差异（我的方法 − baseline）：正确题数 {cm['correct'] - bm['correct']:+d}，"
        f"准确率 {(cm['accuracy'] - bm['accuracy']) * 100:+.2f} pp；"
        f"平均 token {(cm['mean_tokens'] or 0) - (bm['mean_tokens'] or 0):+,.1f}；"
        f"累计节省 {saved:,} token（{pct(reduction)}）。"
    )
    both = method_only = baseline_only = both_wrong = 0
    for sample_id in ids:
        method_ok = judge_correct(current.judges[sample_id]) is True
        baseline_ok = judge_correct(baseline.judges[sample_id]) is True
        if method_ok and baseline_ok:
            both += 1
        elif method_ok:
            method_only += 1
        elif baseline_ok:
            baseline_only += 1
        else:
            both_wrong += 1
    print(
        f"逐题结果：两者都正确 {both}；仅我的方法正确 {method_only}；"
        f"仅 baseline 正确 {baseline_only}；两者都错 {both_wrong}。"
    )

    # The selector is part of the current experiment's causal condition.  A
    # cue-count table makes it immediately visible if a live run has silently
    # collapsed to one word (for example all ``Okay``) and also compares that
    # subset against the same baseline questions.
    selector_labels = sorted({selector_word(current.raw[sample_id]) for sample_id in ids})
    if selector_labels:
        print("\n## 按动态首词 selector 分组（公共子集）")
        print("| selector 首词 | 题数 | 我的方法正确 | baseline 正确 | 正确题数差 | 我的方法平均 token | baseline 平均 token | token 差 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for label in selector_labels:
            group_ids = [
                sample_id
                for sample_id in ids
                if selector_word(current.raw[sample_id]) == label
            ]
            sm = metric(current, group_ids)
            sb = metric(baseline, group_ids)
            current_mean = sm["mean_tokens"] or 0.0
            baseline_mean = sb["mean_tokens"] or 0.0
            print(
                f"| {label} | {len(group_ids)} | {sm['correct']} | {sb['correct']} "
                f"| {sm['correct'] - sb['correct']:+d} | {num(sm['mean_tokens'])} "
                f"| {num(sb['mean_tokens'])} | {current_mean - baseline_mean:+,.1f} |"
            )

    type_groups = [
        ("多选题", [sample_id for sample_id in ids if question_type(current, sample_id) == "multiple_choice"]),
        ("开放题", [sample_id for sample_id in ids if question_type(current, sample_id) == "open"]),
    ]
    print_group_table("按题型（同一公共子集）", type_groups, current, baseline)
    bucket_names = ("极短 0–128", "短 129–1023", "中 1024–8191", "长 ≥8192")
    length_groups = [
        (label, [sample_id for sample_id in ids if token_bucket(output_tokens(current.raw[sample_id])) == label])
        for label in bucket_names
    ]
    print_group_table("按我的方法输出长度（每组与 baseline 同题比较）", length_groups, current, baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_CURRENT_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID, help="Slurm job id；不需要时传空字符串")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="刷新间隔秒数，默认3600")
    parser.add_argument("--once", action="store_true", help="只输出一次")
    parser.add_argument("--no-clear", action="store_true", help="循环刷新时不清屏")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = True
    try:
        while True:
            current = load_run("dynamic", args.current_run)
            baseline = load_run("baseline", args.baseline_run)
            if not args.no_clear and not first:
                sys.stdout.write("\033[2J\033[H")
            first = False
            print(f"当前目录：{args.current_run}\nbaseline目录：{args.baseline_run}")
            render(current, baseline, args.job_id or None)
            sys.stdout.flush()
            if args.once:
                return
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n已停止自动刷新。")


if __name__ == "__main__":
    main()
