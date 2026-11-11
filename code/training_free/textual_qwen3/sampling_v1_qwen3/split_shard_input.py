from __future__ import annotations

import argparse
from pathlib import Path

from experiment_src.sampling_v1.io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parts", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    part_count = max(int(args.parts), 1)
    for part_idx in range(part_count):
        start = len(rows) * part_idx // part_count
        end = len(rows) * (part_idx + 1) // part_count
        write_jsonl(output_dir / f"part_{part_idx}.jsonl", rows[start:end])


if __name__ == "__main__":
    main()
