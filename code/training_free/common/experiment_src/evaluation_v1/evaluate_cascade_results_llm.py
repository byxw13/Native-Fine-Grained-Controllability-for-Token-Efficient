#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from experiment_src.evaluation_v1.evaluate_budget_results_attn_alright import (
    DEFAULT_REQUEST_TIMEOUT,
    INITIAL_CONTEXT_CHARS,
    REVIEW_CONTEXT_CHARS,
    call_llm_judge,
    extract_boxed_context,
    resolve_primary_api_settings,
    resolve_review_model,
)


DEFAULT_INPUT_PATHS = [
    Path("pulled-results/evaluation-v1/ds7b_math500_cascade_eval/results/final_results.jsonl"),
    Path("pulled-results/evaluation-v1/ds7b_gsm8k_cascade_eval/results/final_results.jsonl"),
]
DEFAULT_OUTPUT_DIR = Path("pulled-results/evaluation-v1/eval")
DEFAULT_SUMMARY_CSV = "cascade_eval_summary_llm.csv"
DEFAULT_SUMMARY_JSON = "cascade_eval_summary_llm.json"


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
    return records


def run_name_from_result_path(path: Path) -> str:
    return path.parent.parent.name


def make_item_signature(item: dict) -> str:
    payload = {
        "id": item.get("id"),
        "answer": item.get("answer"),
        "final_level_id": item.get("final_level_id"),
        "final_text": item.get("final_text"),
        "attempt_count": item.get("attempt_count"),
        "total_generated_tokens": item.get("total_generated_tokens"),
    }
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_checkpoint_complete(record: dict) -> bool:
    if record.get("id") is None or record.get("is_correct") is None:
        return False
    primary_is_correct = record.get("primary_is_correct")
    if primary_is_correct is False:
        return record.get("review_consensus") in {"same", "different"}
    if primary_is_correct is True:
        return True
    if "review_consensus" in record:
        return record.get("review_consensus") in {"same", "different", "not_needed"}
    return record.get("is_correct") is True


def load_latest_checkpoint_records(checkpoint_path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not checkpoint_path.exists():
        return latest
    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(record.get("id"))
            if key:
                latest[key] = record
    return latest


def append_checkpoint_records(checkpoint_path: Path, records: Iterable[dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def evaluate_one_item(
    primary_client: OpenAI,
    primary_model: str,
    review_client: Optional[OpenAI],
    review_model: str,
    run_name: str,
    item: dict,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> dict:
    item_id = str(item.get("id"))
    standard_answer = str(item.get("answer") or "")
    generated_answer = str(item.get("final_text") or "")
    primary_extracted_answer = extract_boxed_context(generated_answer, INITIAL_CONTEXT_CHARS)
    signature = make_item_signature(item)

    primary_is_correct, primary_error = call_llm_judge(
        client=primary_client,
        model=primary_model,
        generated_part=primary_extracted_answer,
        standard_part=standard_answer,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        request_timeout=request_timeout,
    )

    final_is_correct = primary_is_correct
    final_decision_source = "primary_qwen"
    review_needed = primary_is_correct is False
    review_checked = False
    review_extracted_answer = None
    review_is_correct = None
    review_error = None
    review_consensus = "not_needed"

    if review_needed:
        review_extracted_answer = extract_boxed_context(generated_answer, REVIEW_CONTEXT_CHARS)
        if review_client is None:
            review_consensus = "review_failed"
            review_error = "Review client is not configured"
        else:
            review_is_correct, review_error = call_llm_judge(
                client=review_client,
                model=review_model,
                generated_part=review_extracted_answer,
                standard_part=standard_answer,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                request_timeout=request_timeout,
            )
            if review_is_correct is None:
                review_consensus = "review_failed"
            else:
                review_checked = True
                if review_is_correct == primary_is_correct:
                    review_consensus = "same"
                else:
                    review_consensus = "different"
                    final_is_correct = review_is_correct
                    final_decision_source = "kimi_review"

    return {
        "run_name": run_name,
        "id": item_id,
        "signature": signature,
        "question": item.get("question"),
        "standard_answer": standard_answer,
        "extracted_answer": primary_extracted_answer,
        "total_generated_tokens": item.get("total_generated_tokens"),
        "attempt_count": item.get("attempt_count"),
        "final_level_id": item.get("final_level_id"),
        "is_correct": final_is_correct,
        "error": primary_error if primary_is_correct is None else None,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "primary_model": primary_model,
        "primary_is_correct": primary_is_correct,
        "primary_error": primary_error,
        "review_needed": review_needed,
        "review_checked": review_checked,
        "review_model": review_model,
        "review_extracted_answer": review_extracted_answer,
        "review_is_correct": review_is_correct,
        "review_error": review_error,
        "review_consensus": review_consensus,
        "final_decision_source": final_decision_source,
    }


def format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return ""
    return f"{(numerator / denominator) * 100:.2f}%"


def format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def build_summary_record(file_path: Path, checkpoint_records: Dict[str, dict]) -> dict:
    items = read_jsonl(file_path)
    run_name = run_name_from_result_path(file_path)
    token_values = [
        float(item["total_generated_tokens"])
        for item in items
        if isinstance(item.get("total_generated_tokens"), (int, float))
    ]
    attempt_values = [
        float(item["attempt_count"])
        for item in items
        if isinstance(item.get("attempt_count"), (int, float))
    ]
    avg_tokens = sum(token_values) / len(token_values) if token_values else None
    avg_attempts = sum(attempt_values) / len(attempt_values) if attempt_values else None

    evaluated_count = 0
    correct_count = 0
    pending_count = 0
    final_level_counts: Counter[str] = Counter()
    for item in items:
        item_id = str(item.get("id"))
        final_level = item.get("final_level_id")
        if final_level is not None:
            final_level_counts[str(final_level)] += 1
        signature = make_item_signature(item)
        checkpoint = checkpoint_records.get(item_id)
        is_reusable = (
            checkpoint is not None
            and checkpoint.get("signature") == signature
            and is_checkpoint_complete(checkpoint)
        )
        if is_reusable:
            evaluated_count += 1
            if checkpoint.get("is_correct") is True:
                correct_count += 1
        else:
            pending_count += 1

    accuracy = (correct_count / evaluated_count) if evaluated_count else None
    return {
        "run_name": run_name,
        "result_path": str(file_path),
        "accuracy": accuracy,
        "average_total_generated_tokens": avg_tokens,
        "average_attempt_count": avg_attempts,
        "correct_count": correct_count,
        "evaluated_count": evaluated_count,
        "pending_eval_count": pending_count,
        "total_items": len(items),
        "final_level_counts": dict(sorted(final_level_counts.items(), key=lambda kv: int(kv[0]))),
    }


def save_summary(output_dir: Path, summary_records: List[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_SUMMARY_CSV
    json_path = output_dir / DEFAULT_SUMMARY_JSON
    sorted_records = sorted(summary_records, key=lambda record: record["run_name"])

    csv_columns = [
        "run_name",
        "accuracy",
        "average_total_generated_tokens",
        "average_attempt_count",
        "correct_count",
        "evaluated_count",
        "pending_eval_count",
        "total_items",
        "final_level_counts",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    "run_name": record["run_name"],
                    "accuracy": format_percent(record["correct_count"], record["evaluated_count"]),
                    "average_total_generated_tokens": format_float(record["average_total_generated_tokens"]),
                    "average_attempt_count": format_float(record["average_attempt_count"]),
                    "correct_count": record["correct_count"],
                    "evaluated_count": record["evaluated_count"],
                    "pending_eval_count": record["pending_eval_count"],
                    "total_items": record["total_items"],
                    "final_level_counts": json.dumps(record["final_level_counts"], ensure_ascii=False),
                }
            )
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_records, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def evaluate_file(
    primary_client: OpenAI,
    primary_model: str,
    review_client: Optional[OpenAI],
    review_model: str,
    file_path: Path,
    output_dir: Path,
    max_workers: int,
    batch_size: int,
    pause_seconds: float,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
    summary_only: bool,
    limit_per_file: Optional[int],
) -> dict:
    items = read_jsonl(file_path)
    run_name = run_name_from_result_path(file_path)
    checkpoint_path = output_dir / "checkpoints" / f"{run_name}.eval.jsonl"
    checkpoint_records = load_latest_checkpoint_records(checkpoint_path)

    tasks: List[dict] = []
    for item in items:
        item_id = str(item.get("id"))
        signature = make_item_signature(item)
        checkpoint = checkpoint_records.get(item_id)
        is_reusable = (
            checkpoint is not None
            and checkpoint.get("signature") == signature
            and is_checkpoint_complete(checkpoint)
        )
        if not is_reusable and not summary_only:
            tasks.append(item)

    if limit_per_file is not None:
        tasks = tasks[:limit_per_file]

    if tasks:
        print(f"[{run_name}] pending new evaluations: {len(tasks)}")
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_one_item,
                    primary_client,
                    primary_model,
                    review_client,
                    review_model,
                    run_name=run_name,
                    item=item,
                    max_retries=max_retries,
                    retry_sleep=retry_sleep,
                    request_timeout=request_timeout,
                ): item
                for item in tasks
            }
            for future in as_completed(future_map):
                result = future.result()
                append_checkpoint_records(checkpoint_path, [result])
                completed += 1
                print(
                    f"[{run_name}] completed {completed}/{len(tasks)} "
                    f"(id={result.get('id')}, is_correct={result.get('is_correct')}, "
                    f"review_consensus={result.get('review_consensus')})"
                )
                if completed % batch_size == 0 and completed < len(tasks):
                    time.sleep(pause_seconds)

        checkpoint_records = load_latest_checkpoint_records(checkpoint_path)
    else:
        print(f"[{run_name}] no new evaluations needed")

    return build_summary_record(file_path=file_path, checkpoint_records=checkpoint_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cascade final_results.jsonl files with legacy LLM judging logic")
    parser.add_argument("--input-paths", nargs="+", type=Path, default=DEFAULT_INPUT_PATHS, help="Cascade result JSONL paths")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-workers", type=int, default=64, help="Concurrent worker count")
    parser.add_argument("--batch-size", type=int, default=200, help="Pause after every N completed items")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="Pause duration between batches")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per LLM call")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Sleep seconds between retries")
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT, help="Per-request timeout in seconds")
    parser.add_argument("--summary-only", action="store_true", help="Only build summary from existing checkpoints")
    parser.add_argument("--limit-per-file", type=int, default=None, help="Limit new evaluations per file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    missing_files = [str(path) for path in args.input_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing_files))

    primary_model = ""
    review_model = resolve_review_model()
    review_client: Optional[OpenAI] = None
    if not args.summary_only:
        primary_api_key, primary_base_url, primary_model = resolve_primary_api_settings()
        primary_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
        review_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
    else:
        primary_client = None  # type: ignore[assignment]

    summary_records: List[dict] = []
    for file_path in args.input_paths:
        summary_records.append(
            evaluate_file(
                primary_client=primary_client,  # type: ignore[arg-type]
                primary_model=primary_model,
                review_client=review_client,
                review_model=review_model,
                file_path=file_path,
                output_dir=args.output_dir,
                max_workers=args.max_workers,
                batch_size=args.batch_size,
                pause_seconds=args.pause_seconds,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                request_timeout=args.request_timeout,
                summary_only=args.summary_only,
                limit_per_file=args.limit_per_file,
            )
        )

    csv_path, json_path = save_summary(args.output_dir, summary_records)
    print("=" * 80)
    print("Summary finished")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
