from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank_one(step: dict) -> str | None:
    ranked = [
        (int(row["rank"]), str(token_id))
        for token_id, row in step.items()
        if row.get("rank") is not None
    ]
    return min(ranked)[1] if ranked else None


def _compare_logprobs(
    left: list[dict],
    right: list[dict],
    *,
    min_top5_overlap: int,
    top1_logprob_atol: float,
    shared_logprob_atol: float,
) -> dict:
    """Compare decisions strictly while allowing documented BF16 build noise.

    Recompiling the upstream FlashAttention kernels can move low-ranked BF16
    candidates slightly even when suppression is disabled.  S1 therefore gates
    on the actual greedy decision at every position, and records (rather than
    hides) the residual top-5 numerical differences.
    """
    report = {
        "pass": True,
        "steps": len(left),
        "top1_equal_steps": 0,
        "minimum_top5_overlap": 5,
        "top5_exact_set_steps": 0,
        "lower_rank_swap_count": 0,
        "max_top1_logprob_abs_delta": 0.0,
        "max_shared_logprob_abs_delta": 0.0,
        "errors": [],
    }
    if len(left) != len(right):
        report["pass"] = False
        report["errors"].append(f"logprob step count differs: {len(left)} != {len(right)}")
        return report
    for step_index, (left_step, right_step) in enumerate(zip(left, right)):
        left_ids, right_ids = set(left_step), set(right_step)
        shared = left_ids & right_ids
        overlap = len(shared)
        report["minimum_top5_overlap"] = min(report["minimum_top5_overlap"], overlap)
        report["top5_exact_set_steps"] += int(left_ids == right_ids)
        if overlap < min_top5_overlap:
            report["errors"].append(
                f"step {step_index} top-5 overlap {overlap} < {min_top5_overlap}"
            )
        left_top1, right_top1 = _rank_one(left_step), _rank_one(right_step)
        if left_top1 != right_top1 or left_top1 is None:
            report["errors"].append(
                f"step {step_index} rank-1 token differs: {left_top1} != {right_top1}"
            )
        else:
            report["top1_equal_steps"] += 1
            delta = abs(
                float(left_step[left_top1]["logprob"])
                - float(right_step[right_top1]["logprob"])
            )
            report["max_top1_logprob_abs_delta"] = max(
                report["max_top1_logprob_abs_delta"], delta
            )
            if delta > top1_logprob_atol:
                report["errors"].append(
                    f"step {step_index} rank-1 logprob delta {delta} > {top1_logprob_atol}"
                )
        for token_id in shared:
            left_row, right_row = left_step[token_id], right_step[token_id]
            report["lower_rank_swap_count"] += int(
                token_id != left_top1 and left_row.get("rank") != right_row.get("rank")
            )
            delta = abs(float(left_row["logprob"]) - float(right_row["logprob"]))
            report["max_shared_logprob_abs_delta"] = max(
                report["max_shared_logprob_abs_delta"], delta
            )
            if delta > shared_logprob_atol:
                report["errors"].append(
                    f"step {step_index} token {token_id} logprob delta "
                    f"{delta} > {shared_logprob_atol}"
                )
    report["pass"] = not report["errors"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-top5-overlap", type=int, default=4)
    parser.add_argument("--top1-logprob-atol", type=float, default=0.1)
    parser.add_argument("--shared-logprob-atol", type=float, default=0.5)
    args = parser.parse_args()
    left, right = _load(args.left), _load(args.right)
    errors: list[str] = []
    for field in ("mode", "max_tokens", "indices", "ids", "audits"):
        if left.get(field) != right.get(field):
            errors.append(f"{field} differs")
    left_outputs = left.get("outputs") or []
    right_outputs = right.get("outputs") or []
    if len(left_outputs) != len(right_outputs):
        errors.append("repeat count differs")
    output_pairs = []
    for repeat, (left_repeat, right_repeat) in enumerate(zip(left_outputs, right_outputs)):
        if len(left_repeat) != len(right_repeat):
            errors.append(f"repeat {repeat} output count differs")
            continue
        for index, (left_item, right_item) in enumerate(zip(left_repeat, right_repeat)):
            same = left_item.get("token_ids") == right_item.get("token_ids")
            logprob_comparison = _compare_logprobs(
                left_item.get("top5_logprobs") or [],
                right_item.get("top5_logprobs") or [],
                min_top5_overlap=args.min_top5_overlap,
                top1_logprob_atol=args.top1_logprob_atol,
                shared_logprob_atol=args.shared_logprob_atol,
            )
            output_pairs.append(
                {
                    "repeat": repeat,
                    "index": index,
                    "token_ids_equal": same,
                    "decision_equivalent": logprob_comparison["pass"],
                    "logprob_comparison": logprob_comparison,
                }
            )
            if not same:
                errors.append(f"repeat {repeat} output {index} token ids differ")
            if not logprob_comparison["pass"]:
                errors.append(f"repeat {repeat} output {index} greedy decisions differ")
    report = {
        "pass": not errors,
        "errors": errors,
        "left": str(args.left),
        "right": str(args.right),
        "criteria": {
            "token_ids": "exact",
            "rank1_token_each_step": "exact",
            "minimum_top5_overlap": args.min_top5_overlap,
            "rank1_logprob_atol": args.top1_logprob_atol,
            "shared_top5_logprob_atol": args.shared_logprob_atol,
            "lower_rank_order": "reported_not_gated",
        },
        "output_pairs": output_pairs,
        "left_versions": left.get("versions"),
        "right_versions": right.get("versions"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
