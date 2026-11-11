#!/usr/bin/env python3
"""
Evaluate RL/budget_result JSONL files with resume support for Math500 Dsqwen_7B b0.

Flow per item:
1. Qwen2.5-14B first pass with boxed context +/-16 chars.
2. If first pass is False, use boxed context +/-32 chars and ask Kimi-K2.5.
3. If the two results differ, use the Kimi-K2.5 result as the final result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


TARGET_FILENAMES = [
    "Dsqwen_1.5B_math500_mixthink_2axis_supress_sup-attn_b0_enh-think_next_b0p2.jsonl",
    "Dsqwen_1.5B_math500_mixthink_2axis_supress_sup-attn_b0_enh-think_next_b0p7.jsonl",
    "Dsqwen_1.5B_math500_mixthink_2axis_supress_sup-attn_b0_enh-think_next_b1.jsonl",
]

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "budget_result"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "budget_result_eval"
DEFAULT_SUMMARY_CSV = "budget_summary_Dsqwen_1.5B_math500_2axis_supress_alright_attn.csv"
DEFAULT_SUMMARY_JSON = "budget_summary_Dsqwen_1.5B_2axis_supress_alright_attn.json"
DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"
INITIAL_CONTEXT_CHARS = 16
REVIEW_CONTEXT_CHARS = 32
DEFAULT_REVIEW_MODEL = "moonshotai/Kimi-K2-Thinking"
DEFAULT_REQUEST_TIMEOUT = 60.0


def extract_boxed_context(answer: str, context_chars: int = INITIAL_CONTEXT_CHARS) -> str:
    if not isinstance(answer, str):
        return ""

    pattern = r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = list(re.finditer(pattern, answer))
    if not matches:
        return answer[-500:]

    last_match = matches[-1]
    start_pos = max(0, last_match.start() - context_chars)
    end_pos = min(len(answer), last_match.end() + context_chars)
    return answer[start_pos:end_pos]


def resolve_primary_api_settings() -> Tuple[str, str, str]:
    api_key = os.environ.get("ARIS_PRIMARY_API_KEY") or DEFAULT_API_KEY
    base_url = os.environ.get("ARIS_PRIMARY_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("ARIS_PRIMARY_LLM_MODEL") or DEFAULT_LLM_MODEL
    return api_key, base_url, model


def resolve_review_model() -> str:
    return os.environ.get("ARIS_REVIEW_MODEL") or DEFAULT_REVIEW_MODEL


def parse_budget_from_filename(filename: str) -> float:
    match = re.search(r"_b(\d+)(?:p(\d+))?", filename)
    if not match:
        return 0.0
    
    integer_part = match.group(1)
    decimal_part = match.group(2) if match.group(2) else "0"
    return float(f"{integer_part}.{decimal_part}")


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


def make_item_signature(item: dict) -> str:
    payload = {
        "index": item.get("index"),
        "generated_answer": item.get("generated_answer"),
        "standard_answer": item.get("standard_answer"),
        "answer": item.get("answer"),
        "total_token_consumed": item.get("total_token_consumed"),
        "selected_heads_count": item.get("selected_heads_count"),
        "selected_heads": item.get("selected_heads"),
    }
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_checkpoint_complete(record: dict) -> bool:
    if record.get("index") is None or record.get("is_correct") is None:
        return False

    primary_is_correct = record.get("primary_is_correct")
    if primary_is_correct is False:
        return record.get("review_consensus") in {"same", "different"}
    if primary_is_correct is True:
        return True
    if "review_consensus" in record:
        return record.get("review_consensus") in {"same", "different", "not_needed"}
    return record.get("is_correct") is True


def load_latest_checkpoint_records(checkpoint_path: Path) -> Dict[int, dict]:
    latest: Dict[int, dict] = {}
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
            index = record.get("index")
            if isinstance(index, int):
                latest[index] = record
    return latest


def determine_select_num(file_path: Path, items: List[dict]) -> int:
    filename_match = re.search(r"_select(\d+)", file_path.name)
    if filename_match:
        return int(filename_match.group(1))

    selected_counts = {
        int(item["selected_heads_count"])
        for item in items
        if item.get("selected_heads_count") is not None
    }
    if len(selected_counts) == 1:
        return next(iter(selected_counts))
    
    observed_lengths = {
        len(item["selected_heads"])
        for item in items
        if isinstance(item.get("selected_heads"), list)
    }
    if len(observed_lengths) == 1:
        return next(iter(observed_lengths))

    if len(observed_lengths) > 1:
        return int(sum(observed_lengths) / len(observed_lengths))

    return 0


def call_llm_judge(
    client: OpenAI,
    model: str,
    generated_part: str,
    standard_part: str,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> Tuple[Optional[bool], Optional[str]]:
    if not generated_part or not standard_part:
        return False, None

    prompt = f"""
You are an evaluator. Determine whether the generated answer matches the standard answer.

Rules:
1. Ignore formatting differences such as \\boxed{{}} or spacing.
2. Judge whether the final mathematical answer is the same or logically equivalent.
3. Reply with exactly one sentence:
   - The answer is correct.
   - The answer is incorrect.

Generated Answer Segment:
{generated_part}

Standard Answer:
{standard_part}
"""

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.1,
                timeout=request_timeout,
            )
            content = (response.choices[0].message.content or "").strip().lower()
            is_correct = "the answer is correct" in content and "incorrect" not in content
            return is_correct, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(retry_sleep)

    return None, last_error


def evaluate_one_item(
    primary_client: OpenAI,
    primary_model: str,
    review_client: Optional[OpenAI],
    review_model: str,
    file_name: str,
    item: dict,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> dict:
    index = item.get("index")
    standard_answer = item.get("standard_answer") or item.get("answer") or ""
    generated_answer = item.get("generated_answer", "")
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
        "file_name": file_name,
        "index": index,
        "signature": signature,
        "question": item.get("question"),
        "standard_answer": standard_answer,
        "extracted_answer": primary_extracted_answer,
        "total_token_consumed": item.get("total_token_consumed"),
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


def build_summary_record(
    file_path: Path,
    items: List[dict],
    checkpoint_records: Dict[int, dict],
    expected_total: int,
) -> dict:
    budget = parse_budget_from_filename(file_path.name)
    select_num = determine_select_num(file_path, items)

    token_values = [
        float(item["total_token_consumed"])
        for item in items
        if isinstance(item.get("total_token_consumed"), (int, float))
    ]
    avg_tokens = sum(token_values) / len(token_values) if token_values else None

    evaluated_count = 0
    correct_count = 0
    pending_count = 0
    for item in items:
        index = item.get("index")
        if not isinstance(index, int):
            continue

        signature = make_item_signature(item)
        checkpoint = checkpoint_records.get(index)
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
        "file_name": file_path.name,
        "budget": budget,
        "select_num": select_num,
        "accuracy": accuracy,
        "average_token_consumed": avg_tokens,
        "correct_count": correct_count,
        "evaluated_count": evaluated_count,
        "current_item_count": len(items),
        "pending_eval_count": pending_count,
        "expected_total": expected_total,
        "coverage": (len(items) / expected_total) if expected_total else None,
    }


def save_summary(output_dir: Path, summary_records: List[dict]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / DEFAULT_SUMMARY_CSV
    json_path = output_dir / DEFAULT_SUMMARY_JSON

    sorted_records = sorted(summary_records, key=lambda record: (record["budget"], record["select_num"]))

    csv_columns = [
        "file_name",
        "budget",
        "select_num",
        "accuracy",
        "average_token_consumed",
        "correct_count",
        "evaluated_count",
        "current_item_count",
        "pending_eval_count",
        "expected_total",
        "coverage",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    "file_name": record["file_name"],
                    "budget": format_float(record["budget"]),
                    "select_num": record["select_num"],
                    "accuracy": format_percent(record["correct_count"], record["evaluated_count"]),
                    "average_token_consumed": format_float(record["average_token_consumed"]),
                    "correct_count": record["correct_count"],
                    "evaluated_count": record["evaluated_count"],
                    "current_item_count": record["current_item_count"],
                    "pending_eval_count": record["pending_eval_count"],
                    "expected_total": record["expected_total"],
                    "coverage": format_percent(record["current_item_count"], record["expected_total"]),
                }
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(sorted_records, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def append_checkpoint_records(checkpoint_path: Path, records: Iterable[dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def evaluate_file(
    primary_client: OpenAI,
    primary_model: str,
    review_client: Optional[OpenAI],
    review_model: str,
    file_path: Path,
    output_dir: Path,
    expected_total: int,
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
    checkpoint_path = output_dir / "checkpoints" / f"{file_path.stem}.eval.jsonl"
    checkpoint_records = load_latest_checkpoint_records(checkpoint_path)

    tasks: List[dict] = []
    for item in items:
        index = item.get("index")
        if not isinstance(index, int):
            continue

        signature = make_item_signature(item)
        checkpoint = checkpoint_records.get(index)
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
        print(f"[{file_path.name}] pending new evaluations: {len(tasks)}")
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_one_item,
                    primary_client,
                    primary_model,
                    review_client,
                    review_model,
                    file_name=file_path.name,
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
                    f"[{file_path.name}] completed {completed}/{len(tasks)} "
                    f"(index={result.get('index')}, is_correct={result.get('is_correct')}, "
                    f"review_consensus={result.get('review_consensus')})"
                )
                if completed % batch_size == 0 and completed < len(tasks):
                    time.sleep(pause_seconds)

        checkpoint_records = load_latest_checkpoint_records(checkpoint_path)
    else:
        print(f"[{file_path.name}] no new evaluations needed")

    return build_summary_record(
        file_path=file_path,
        items=items,
        checkpoint_records=checkpoint_records,
        expected_total=expected_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Math500 Dsqwen_1.5B budget_result files with resume support")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input JSONL directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--files", nargs="*", default=TARGET_FILENAMES, help="Target file names")
    parser.add_argument("--expected-total", type=int, default=500, help="Expected total problem count")
    parser.add_argument("--max-workers", type=int, default=10, help="Concurrent worker count")
    parser.add_argument("--batch-size", type=int, default=500, help="Pause after every N completed items")
    parser.add_argument("--pause-seconds", type=float, default=0.0, help="Pause duration between batches")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries per LLM call")
    parser.add_argument("--retry-sleep", type=float, default=3.0, help="Sleep seconds between retries")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Per-request timeout in seconds for each LLM API call",
    )
    parser.add_argument("--summary-only", action="store_true", help="Only build summary from existing checkpoints")
    parser.add_argument("--limit-per-file", type=int, default=None, help="Limit new evaluations per file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [args.input_dir / filename for filename in args.files]
    missing_files = [str(path) for path in input_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing_files))

    primary_api_key = ""
    primary_base_url = ""
    primary_model = ""
    review_client: Optional[OpenAI] = None
    review_model = DEFAULT_REVIEW_MODEL

    if not args.summary_only:
        primary_api_key, primary_base_url, primary_model = resolve_primary_api_settings()
        primary_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)

        review_model = resolve_review_model()
        review_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
    else:
        # Dummy values are unused in summary-only mode.
        primary_client = None  # type: ignore[assignment]

    summary_records: List[dict] = []
    for file_path in input_paths:
        summary_records.append(
            evaluate_file(
                primary_client=primary_client,  # type: ignore[arg-type]
                primary_model=primary_model,
                review_client=review_client,
                review_model=review_model,
                file_path=file_path,
                output_dir=args.output_dir,
                expected_total=args.expected_total,
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
