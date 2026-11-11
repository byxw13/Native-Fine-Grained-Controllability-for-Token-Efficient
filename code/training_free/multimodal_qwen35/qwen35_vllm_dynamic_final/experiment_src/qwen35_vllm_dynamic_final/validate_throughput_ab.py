from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _median_generate_tps(path: Path) -> float:
    report = _read(path)
    runs = report.get("runs") or []
    if len(runs) != 21:
        raise RuntimeError(f"{path}: expected 21 measured repeats, found {len(runs)}")
    return statistics.median(float(row["throughput_tokens_per_second"]) for row in runs)


def _fsync_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    args = parser.parse_args()
    lengths = []
    errors = []
    for tokens in args.max_tokens:
        # Phase A: GPU0 prefix-control, GPU1 dynamic. Phase B swaps the modes.
        p0 = _median_generate_tps(args.root / "phase_a_prefix" / f"t{tokens}.json")
        d1 = _median_generate_tps(args.root / "phase_a_dynamic" / f"t{tokens}.json")
        d0 = _median_generate_tps(args.root / "phase_b_dynamic" / f"t{tokens}.json")
        p1 = _median_generate_tps(args.root / "phase_b_prefix" / f"t{tokens}.json")
        ratios = [d0 / p0, d1 / p1]
        geometric = math.sqrt(ratios[0] * ratios[1])
        passed = min(ratios) >= args.threshold
        if not passed:
            errors.append(
                f"T{tokens}: per-GPU dynamic/control ratios {ratios} fall below {args.threshold}"
            )
        lengths.append(
            {
                "max_tokens": tokens,
                "gpu0_prefix_tps": p0,
                "gpu0_dynamic_tps": d0,
                "gpu1_prefix_tps": p1,
                "gpu1_dynamic_tps": d1,
                "per_gpu_ratios": ratios,
                "geometric_mean_ratio": geometric,
                "minimum_ratio": min(ratios),
                "pass": passed,
            }
        )
    result = {
        "pass": not errors,
        "control": "same closed-think+To prefix, ARIS patch off",
        "measurement": "pure llm.generate output-token throughput",
        "warmups_per_mode": 3,
        "measured_repeats_per_mode": 21,
        "threshold": args.threshold,
        "lengths": lengths,
        "errors": errors,
    }
    _fsync_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
