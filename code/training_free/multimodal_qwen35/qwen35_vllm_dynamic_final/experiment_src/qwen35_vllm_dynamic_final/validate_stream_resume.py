from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path


def _rows(pattern: str) -> list[dict]:
    result = []
    for raw in sorted(glob.glob(pattern)):
        for line in Path(raw).read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
    return result


def _fsync_json(path: Path, value: dict) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = _rows(str(args.run_dir / "raw" / "predictions.shard*.jsonl"))
    judged = _rows(str(args.run_dir / "stream_judge" / "judge_results.jsonl"))
    phase1 = json.loads((args.run_dir / "phase1_snapshot.json").read_text(encoding="utf-8"))
    ids = [str(row.get("id")) for row in raw]
    judge_ids = [str(row.get("id")) for row in judged]
    errors = []
    if len(raw) != args.expected or len(set(ids)) != args.expected:
        errors.append(f"raw rows/unique={len(raw)}/{len(set(ids))}, expected {args.expected}")
    if len(judged) != args.expected or set(judge_ids) != set(ids):
        errors.append("judge rows are not a one-to-one match with durable raw rows")
    duplicates = {key: value for key, value in Counter(ids).items() if value != 1}
    if duplicates:
        errors.append(f"resumption recomputed durable items: {duplicates}")
    phase1_ids = [str(value) for value in phase1.get("ids") or []]
    if not phase1_ids or any(ids.count(value) != 1 for value in phase1_ids):
        errors.append("phase-1 durable rows were not preserved exactly once")
    bad_audits = [
        str(row.get("id"))
        for row in raw
        if row.get("error") or not (row.get("trajectory_audit") or {}).get("pass")
    ]
    if bad_audits:
        errors.append(f"trajectory audits failed: {bad_audits}")
    if not (args.run_dir / "INTERRUPTION_INJECTED").is_file():
        errors.append("missing interruption marker")
    if not (args.run_dir / "stream_judge" / "JUDGE_COMPLETE").is_file():
        errors.append("stream judge did not complete")
    report = {
        "pass": not errors,
        "raw_rows": len(raw),
        "unique_raw_ids": len(set(ids)),
        "judge_rows": len(judged),
        "phase1_durable_ids": phase1_ids,
        "errors": errors,
    }
    _fsync_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
