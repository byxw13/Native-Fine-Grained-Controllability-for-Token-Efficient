from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from experiment_src.sampling_v1.io_utils import iter_jsonl, write_json, write_jsonl


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def load_math500_questions(path: Path) -> set[str]:
    questions: set[str] = set()
    for row in iter_jsonl(path):
        question = str(row.get("question") or row.get("problem") or "").strip()
        if question:
            questions.add(normalize_question(question))
    return questions


def load_eval_outcomes(path: Path) -> tuple[dict[str, bool | None], int]:
    outcomes: dict[str, bool | None] = {}
    malformed_rows = 0
    if not path.exists():
        return outcomes, malformed_rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            sample_id = row.get("index")
            if sample_id is None:
                continue
            sample_key = str(sample_id)
            value = row.get("is_correct")
            if isinstance(value, bool):
                outcomes[sample_key] = value
                continue
            outcomes.setdefault(sample_key, None)
    return outcomes, malformed_rows


def load_hidden_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return rows
    for row in iter_jsonl(path):
        sample_id = str(row["id"])
        level_id = int(row["level_id"])
        rows[(sample_id, level_id)] = row
    return rows


def build_hidden_record(hidden_row: dict) -> dict[str, Any]:
    return {
        "id": hidden_row["id"],
        "level_id": int(hidden_row["level_id"]),
        "checkpoint_window_hidden_mean": hidden_row["checkpoint_window_hidden_mean"],
        "checkpoint_last_hidden": hidden_row["checkpoint_last_hidden"],
    }


def build_completed_run(raw_row: dict, correct: bool | None, has_hidden: bool) -> dict[str, Any]:
    return {
        "level_id": int(raw_row["level_id"]),
        "correct": correct,
        "status": "completed",
        "final_text": raw_row["final_text"],
        "n_tokens": int(raw_row["n_tokens"]),
        "entropy": raw_row["entropy"],
        "margin": raw_row["margin"],
        "logprob": raw_row["logprob"],
        "checkpoint_end_token_idx": raw_row["checkpoint_end_token_idx"],
        "hidden_ref": (
            {
                "id": raw_row["id"],
                "level_id": int(raw_row["level_id"]),
            }
            if has_hidden
            else None
        ),
    }


def build_pending_run(level_id: int) -> dict[str, Any]:
    return {
        "level_id": level_id,
        "correct": None,
        "status": "unfinished",
        "final_text": None,
        "n_tokens": None,
        "entropy": None,
        "margin": None,
        "logprob": None,
        "checkpoint_end_token_idx": None,
        "hidden_ref": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--completed-levels", required=True, help="Comma-separated completed level ids")
    parser.add_argument("--pending-levels", default="", help="Comma-separated pending level ids")
    parser.add_argument("--math500-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    completed_levels = [int(item) for item in args.completed_levels.split(",") if item.strip()]
    pending_levels = [int(item) for item in args.pending_levels.split(",") if item.strip()]
    math500_questions = load_math500_questions(Path(args.math500_input))

    records: dict[str, dict[str, Any]] = {}
    hidden_output_rows: list[dict[str, Any]] = []
    malformed_eval_rows = 0
    missing_hidden_runs = 0
    pending_by_level: dict[int, set[str]] = {}

    for level_id in completed_levels:
        raw_dir = stage_dir / "raw_outputs" / f"level_{level_id}"
        eval_dir = stage_dir / "eval_results" / f"level_{level_id}" / "checkpoints"
        for raw_path in sorted(raw_dir.glob("shard_*.jsonl")):
            if raw_path.name.endswith(".hidden.jsonl"):
                continue
            hidden_path = raw_dir / f"{raw_path.stem}.hidden.jsonl"
            hidden_rows = load_hidden_rows(hidden_path)
            eval_path = eval_dir / f"{raw_path.stem}.eval.jsonl"
            outcomes, malformed = load_eval_outcomes(eval_path)
            malformed_eval_rows += malformed
            for raw_row in iter_jsonl(raw_path):
                sample_id = str(raw_row["id"])
                hidden_key = (sample_id, int(raw_row["level_id"]))
                hidden_row = hidden_rows.get(hidden_key)
                record = records.setdefault(
                    sample_id,
                    {
                        "id": raw_row["id"],
                        "question": raw_row["question"],
                        "answer": raw_row["answer"],
                        "runs": [],
                        "hidden_runs": [],
                    },
                )
                correct = outcomes.get(sample_id)
                record["runs"].append(build_completed_run(raw_row, correct, hidden_row is not None))
                if hidden_row is None:
                    missing_hidden_runs += 1
                else:
                    record["hidden_runs"].append(build_hidden_record(hidden_row))

    for level_id in pending_levels:
        pending_ids: set[str] = set()
        pending_dir = stage_dir / "next_inputs" / f"level_{level_id}"
        for pending_path in sorted(pending_dir.glob("shard_*.jsonl")):
            for row in iter_jsonl(pending_path):
                pending_ids.add(str(row["id"]))
        pending_by_level[level_id] = pending_ids

    output_rows: list[dict[str, Any]] = []
    removed_math500 = 0
    pending_level_items = 0
    hard_unsolved_count = 0

    for sample_id, record in sorted(records.items(), key=lambda item: int(item[0])):
        record["runs"].sort(key=lambda row: int(row["level_id"]))
        success_level = None
        for run in record["runs"]:
            if run["correct"] is True:
                success_level = int(run["level_id"])
                break
        pending_for_item = [level_id for level_id, ids in pending_by_level.items() if sample_id in ids]
        if success_level is None:
            for level_id in sorted(pending_for_item):
                record["runs"].append(build_pending_run(level_id))
        if success_level is not None:
            hard_unsolved = False
        elif pending_for_item:
            hard_unsolved = None
            pending_level_items += 1
        else:
            hard_unsolved = True
            hard_unsolved_count += 1

        normalized = normalize_question(record["question"])
        if normalized in math500_questions:
            removed_math500 += 1
            continue

        hidden_output_rows.extend(record["hidden_runs"])
        output_rows.append(
            {
                "id": record["id"],
                "question": record["question"],
                "answer": record["answer"],
                "success_level": success_level,
                "hard_unsolved": hard_unsolved,
                "runs": record["runs"],
                "pending_levels": sorted(pending_for_item),
            }
        )

    write_jsonl(Path(args.output), output_rows)
    write_jsonl(Path(args.hidden_output), hidden_output_rows)
    write_json(
        Path(args.summary_json),
        {
            "split_name": args.split_name,
            "stage_dir": str(stage_dir),
            "completed_levels": completed_levels,
            "pending_levels": pending_levels,
            "total_records_before_math500_filter": len(records),
            "removed_for_math500_overlap": removed_math500,
            "final_records": len(output_rows),
            "hidden_records": len(hidden_output_rows),
            "runs_missing_hidden": missing_hidden_runs,
            "records_with_pending_levels": pending_level_items,
            "hard_unsolved_records": hard_unsolved_count,
            "malformed_eval_rows": malformed_eval_rows,
            "math500_question_count": len(math500_questions),
        },
    )


if __name__ == "__main__":
    main()
