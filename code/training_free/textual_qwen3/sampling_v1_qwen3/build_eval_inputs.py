from __future__ import annotations

import argparse
from pathlib import Path

from experiment_src.sampling_v1.io_utils import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", required=True)
    parser.add_argument("--eval-input", required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for row in iter_jsonl(Path(args.raw_input)):
        rows.append(
            {
                "index": row["id"],
                "generated_answer": row["final_text"],
                "answer": row["answer"],
                "standard_answer": row["answer"],
                "question": row["question"],
                "source_level_id": row["level_id"],
            }
        )
    write_jsonl(Path(args.eval_input), rows)


if __name__ == "__main__":
    main()
