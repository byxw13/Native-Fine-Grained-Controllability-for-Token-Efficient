from __future__ import annotations

import argparse
import glob
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is invalid JSON: {exc}") from exc
    return rows


def read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A streaming producer may be writing the last line. It will be
                # retried on the next poll.
                continue
    return rows


def read_input_items(input_path: Path | None, input_glob: str | None) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    if input_path is not None:
        for row in read_jsonl_lenient(input_path):
            index = row.get("index")
            if isinstance(index, int):
                by_index[index] = row
    if input_glob:
        for raw_path in sorted(glob.glob(input_glob)):
            for row in read_jsonl_lenient(Path(raw_path)):
                index = row.get("index")
                if isinstance(index, int):
                    by_index[index] = row
    return [by_index[index] for index in sorted(by_index)]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def item_signature(item: dict[str, Any]) -> str:
    payload = {
        "index": item.get("index"),
        "question": item.get("question"),
        "standard_answer": item.get("standard_answer"),
        "generated_answer": item.get("generated_answer"),
        "total_token_consumed": item.get("total_token_consumed"),
        "attn_implementation": item.get("attn_implementation"),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = row.get("index")
            if isinstance(index, int):
                latest[index] = row
    return latest


def checkpoint_complete(row: dict[str, Any], signature: str) -> bool:
    if row.get("signature") != signature:
        return False
    if row.get("is_correct") is None:
        return False
    primary = row.get("primary_is_correct")
    if primary is True:
        return True
    if primary is False:
        return row.get("review_consensus") in {"same", "different"}
    return row.get("review_consensus") in {"same", "different", "not_needed"}


def evaluate_one(
    item: dict[str, Any],
    *,
    run_name: str,
    primary_client: OpenAI,
    primary_model: str,
    review_client: OpenAI,
    review_model: str,
    max_retries: int,
    retry_sleep: float,
    request_timeout: float,
) -> dict[str, Any]:
    standard_answer = str(item.get("standard_answer") or item.get("answer") or "")
    generated_answer = str(item.get("generated_answer") or item.get("final_text") or "")
    extracted_answer = extract_boxed_context(generated_answer, INITIAL_CONTEXT_CHARS)
    primary_is_correct, primary_error = call_llm_judge(
        client=primary_client,
        model=primary_model,
        generated_part=extracted_answer,
        standard_part=standard_answer,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        request_timeout=request_timeout,
    )

    final_is_correct = primary_is_correct
    review_needed = primary_is_correct is False
    review_checked = False
    review_extracted_answer = None
    review_is_correct = None
    review_error = None
    review_consensus = "not_needed"
    final_decision_source = "primary_qwen"

    if review_needed:
        review_extracted_answer = extract_boxed_context(generated_answer, REVIEW_CONTEXT_CHARS)
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
        "index": item.get("index"),
        "signature": item_signature(item),
        "question": item.get("question"),
        "standard_answer": standard_answer,
        "extracted_answer": extracted_answer,
        "total_token_consumed": item.get("total_token_consumed"),
        "attn_implementation": item.get("attn_implementation"),
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


def write_summary(
    path: Path,
    *,
    run_name: str,
    input_label: Path,
    checkpoint_path: Path,
    checkpoint_rows: dict[int, dict[str, Any]],
) -> None:
    evaluated = [row for row in checkpoint_rows.values() if row.get("is_correct") is not None]
    correct = [row for row in evaluated if row.get("is_correct") is True]
    token_values = [
        float(row["total_token_consumed"])
        for row in checkpoint_rows.values()
        if isinstance(row.get("total_token_consumed"), (int, float))
    ]
    review_counts = Counter(str(row.get("review_consensus")) for row in evaluated)
    payload = {
        "run_name": run_name,
        "input": str(input_label),
        "checkpoint": str(checkpoint_path),
        "total_items": len(checkpoint_rows),
        "evaluated_count": len(evaluated),
        "correct_count": len(correct),
        "accuracy": (len(correct) / len(evaluated)) if evaluated else None,
        "average_total_token_consumed": (sum(token_values) / len(token_values)) if token_values else None,
        "review_consensus_counts": dict(sorted(review_counts.items())),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote LLM judge for Qwen3 pure-thinking eager outputs.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--input-glob", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--idle-polls-after-complete", type=int, default=2)
    return parser.parse_args()


def evaluate_pending_items(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    checkpoint: dict[int, dict[str, Any]],
    checkpoint_path: Path,
    summary_path: Path,
    input_label: Path,
    primary_client: OpenAI,
    primary_model: str,
    review_client: OpenAI,
    review_model: str,
) -> int:
    pending: list[dict[str, Any]] = []
    for item in items:
        index = item.get("index")
        if not isinstance(index, int):
            continue
        signature = item_signature(item)
        existing = checkpoint.get(index)
        if existing is not None and checkpoint_complete(existing, signature):
            continue
        pending.append(item)
    print(json.dumps({"input": str(input_label), "total_seen": len(items), "pending": len(pending)}, ensure_ascii=False), flush=True)
    if not pending:
        write_summary(
            summary_path,
            run_name=args.run_name,
            input_label=input_label,
            checkpoint_path=checkpoint_path,
            checkpoint_rows=checkpoint,
        )
        return 0
    completed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                item,
                run_name=args.run_name,
                primary_client=primary_client,
                primary_model=primary_model,
                review_client=review_client,
                review_model=review_model,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                request_timeout=args.request_timeout,
            )
            for item in pending
        ]
        for future in as_completed(futures):
            record = future.result()
            append_jsonl(checkpoint_path, [record])
            index = record.get("index")
            if isinstance(index, int):
                checkpoint[index] = record
            completed += 1
            if completed % 25 == 0:
                write_summary(
                    summary_path,
                    run_name=args.run_name,
                    input_label=input_label,
                    checkpoint_path=checkpoint_path,
                    checkpoint_rows=checkpoint,
                )
                print(json.dumps({"completed": completed, "pending": len(pending), "elapsed": time.time() - started}, ensure_ascii=False), flush=True)
    write_summary(
        summary_path,
        run_name=args.run_name,
        input_label=input_label,
        checkpoint_path=checkpoint_path,
        checkpoint_rows=checkpoint,
    )
    return completed


def main() -> None:
    args = parse_args()
    if not args.input and not args.input_glob:
        raise ValueError("Either --input or --input-glob is required.")
    input_path = Path(args.input) if args.input else None
    input_label = input_path or Path(args.input_glob or "streaming_inputs")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = input_path.stem if input_path is not None else args.run_name
    checkpoint_path = output_dir / f"evaluation_checkpoint_{checkpoint_stem}.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint = load_checkpoint(checkpoint_path)

    primary_api_key, primary_base_url, primary_model = resolve_primary_api_settings()
    review_model = resolve_review_model()
    primary_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
    review_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)

    idle_polls = 0
    while True:
        items = read_input_items(input_path, args.input_glob)
        completed = evaluate_pending_items(
            args=args,
            items=items,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            input_label=input_label,
            primary_client=primary_client,
            primary_model=primary_model,
            review_client=review_client,
            review_model=review_model,
        )
        expected_reached = args.expected_count is not None and len(checkpoint) >= int(args.expected_count)
        if not args.follow or expected_reached:
            break
        if completed == 0:
            idle_polls += 1
        else:
            idle_polls = 0
        if idle_polls >= int(args.idle_polls_after_complete) and args.expected_count is None:
            break
        time.sleep(max(float(args.poll_seconds), 1.0))

    print(summary_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
