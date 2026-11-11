from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiment_src.sampling_v1.io_utils import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-jsonl", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage-name", required=True)
    args = parser.parse_args()

    records = read_jsonl(Path(args.merged_jsonl))
    audit = read_json(Path(args.audit_json))
    success_counter: Counter[str] = Counter()
    total_tokens = 0
    hard_unsolved = 0
    shard_counter: Counter[str] = Counter()

    for record in records:
        success_counter[str(record.get("success_level"))] += 1
        if record.get("hard_unsolved"):
            hard_unsolved += 1
        for run in record.get("runs", []):
            total_tokens += int(run.get("n_tokens", 0))

    for row in audit.get("missing_eval_records", []):
        shard_counter[str(row["shard_id"])] += 1

    write_json(
        Path(args.output),
        {
            "stage_name": args.stage_name,
            "total_questions": len(records),
            "entered_level_counts": audit.get("entered_level_counts", {}),
            "successful_level_counts": audit.get("successful_level_counts", {}),
            "hard_unsolved": hard_unsolved,
            "total_generated_tokens": total_tokens,
            "missing_eval_records": audit.get("missing_eval_records", []),
            "missing_eval_by_shard": dict(sorted(shard_counter.items())),
            "success_level_counts": dict(sorted(success_counter.items())),
        },
    )


if __name__ == "__main__":
    main()
