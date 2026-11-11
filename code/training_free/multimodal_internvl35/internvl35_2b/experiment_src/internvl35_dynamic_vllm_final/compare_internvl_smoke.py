#!/usr/bin/env python3
"""Compare vLLM and Transformers smoke trajectories at 32/128/512 tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_vllm(path: Path) -> list[list[int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return [[int(x) for x in row["token_ids"]] for row in report["outputs"][0]]


def read_transformers(path: Path) -> list[list[int]]:
    rows = []
    for item in sorted(path.glob("*.json")):
        row = json.loads(item.read_text(encoding="utf-8"))
        if row.get("error"):
            rows.append([])
        else:
            rows.append([int(x) for x in row.get("output_token_ids", [])])
    return rows


def compare(a: list[int], b: list[int]) -> dict[str, object]:
    n = min(len(a), len(b))
    common = 0
    while common < n and a[common] == b[common]:
        common += 1
    mismatches = sum(x != y for x, y in zip(a[:n], b[:n]))
    return {
        "vllm_tokens": len(a),
        "transformers_tokens": len(b),
        "common_prefix_tokens": common,
        "length_equal": len(a) == len(b),
        "exact_token_sequence": a == b,
        "mismatch_count_aligned": mismatches,
        "aligned_mismatch_rate": mismatches / n if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--transformers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    args = parser.parse_args()
    report: dict[str, object] = {"criteria": {"exact_sequence_required": True}, "by_max_tokens": {}}
    passed = True
    for value in args.max_tokens:
        left = read_vllm(args.vllm_dir / f"t{value}.json")
        right = read_transformers(args.transformers_dir / f"t{value}")
        if len(left) != len(right):
            passed = False
        pairs = [compare(a, b) for a, b in zip(left, right)]
        if any(not bool(pair["exact_token_sequence"]) for pair in pairs):
            passed = False
        report["by_max_tokens"][str(value)] = {"num_pairs": len(pairs), "pairs": pairs}
    report["pass"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
