from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiment_src.sampling_v1.io_utils import iter_jsonl, read_jsonl, write_json, write_jsonl


def load_eval_outcomes(path: Path) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for row in iter_jsonl(path):
        index = row.get("index")
        if index is None:
            continue
        outcomes[str(index)] = bool(row.get("is_correct"))
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    merged: dict[str, dict] = {}
    entered_level_counts: Counter[int] = Counter()
    successful_level_counts: Counter[int] = Counter()
    missing_eval_records: list[dict] = []

    for level_dir in sorted((stage_dir / "eval_results").glob("level_*")):
        level_id = int(level_dir.name.split("_")[-1])
        for raw_path in sorted((stage_dir / "raw_outputs" / f"level_{level_id}").glob("shard_*.jsonl")):
            if raw_path.name.endswith(".hidden.jsonl"):
                continue
            shard_id = raw_path.stem.split("_")[-1]
            eval_path = level_dir / f"shard_{shard_id}.eval.jsonl"
            outcomes = load_eval_outcomes(eval_path)
            for row in read_jsonl(raw_path):
                sample_id = str(row["id"])
                item = merged.setdefault(
                    sample_id,
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": row["answer"],
                        "success_level": None,
                        "hard_unsolved": False,
                        "runs": [],
                    },
                )
                if any(existing["level_id"] == row["level_id"] for existing in item["runs"]):
                    continue
                correct = outcomes.get(sample_id)
                if correct is None:
                    missing_eval_records.append({"level_id": level_id, "shard_id": shard_id, "id": row["id"]})
                    correct = False
                item["runs"].append(
                    {
                        "level_id": row["level_id"],
                        "correct": bool(correct),
                        "final_text": row["final_text"],
                        "n_tokens": row["n_tokens"],
                        "entropy": row["entropy"],
                        "margin": row["margin"],
                        "logprob": row["logprob"],
                        "checkpoint_end_token_idx": row["checkpoint_end_token_idx"],
                        "hidden_state_ref": row.get("hidden_state_ref"),
                    }
                )
                entered_level_counts[level_id] += 1
                if item["success_level"] is None and correct:
                    item["success_level"] = level_id
                    successful_level_counts[level_id] += 1

    for item in merged.values():
        item["runs"].sort(key=lambda run: run["level_id"])
        if item["success_level"] is None:
            item["hard_unsolved"] = True

    write_jsonl(Path(args.output), [merged[key] for key in sorted(merged, key=lambda value: str(value))])
    write_json(
        Path(args.audit_json),
        {
            "total_questions": len(merged),
            "entered_level_counts": {str(key): value for key, value in sorted(entered_level_counts.items())},
            "successful_level_counts": {str(key): value for key, value in sorted(successful_level_counts.items())},
            "missing_eval_records": missing_eval_records,
        },
    )


if __name__ == "__main__":
    main()
