from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_src.sampling_v1.io_utils import read_jsonl, write_json, write_jsonl


def load_eval_outcomes(path: Path) -> tuple[dict[str, bool | None], int]:
    outcomes: dict[str, bool | None] = {}
    malformed_rows = 0
    if not path.exists():
        return outcomes, malformed_rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            index = row.get("index")
            if index is None:
                continue
            sample_id = str(index)
            value = row.get("is_correct")
            if isinstance(value, bool):
                outcomes[sample_id] = value
                continue
            outcomes.setdefault(sample_id, None)
    return outcomes, malformed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--next-input", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    raw_rows = read_jsonl(Path(args.raw_input))
    outcomes, malformed_rows = load_eval_outcomes(Path(args.eval_output))

    incorrect_rows: list[dict] = []
    correct_count = 0
    missing_eval = 0
    incomplete_eval = 0
    for row in raw_rows:
        sample_id = str(row["id"])
        is_correct = outcomes.get(sample_id)
        if is_correct is None:
            if sample_id in outcomes:
                incomplete_eval += 1
            else:
                missing_eval += 1
            continue
        if is_correct:
            correct_count += 1
        else:
            incorrect_rows.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "source_level_id": row.get("level_id"),
                    "source_level_label": row.get("level_label"),
                    "source_hidden_state_ref": row.get("hidden_state_ref"),
                    "source_eval_correct": False,
                }
            )

    write_jsonl(Path(args.next_input), incorrect_rows)
    write_json(
        Path(args.summary_json),
        {
            "raw_input": args.raw_input,
            "eval_output": args.eval_output,
            "total_rows": len(raw_rows),
            "correct_rows": correct_count,
            "incorrect_rows": len(incorrect_rows),
            "missing_eval_rows": missing_eval,
            "incomplete_eval_rows": incomplete_eval,
            "malformed_eval_rows": malformed_rows,
        },
    )


if __name__ == "__main__":
    main()
