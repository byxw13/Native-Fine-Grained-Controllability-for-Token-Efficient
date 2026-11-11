#!/usr/bin/env python3
"""Compare numerical/decision surfaces without requiring identical rollouts.

The report separates three questions:

* K=0 identity: dynamic plumbing must reduce to the closed-think prefix control.
* K>0 effect: the first-step distribution/decision should actually change for
  at least some requests (otherwise suppression is not functionally active).
* HF-vLLM agreement: compare top-k distributions and first decisions while
  allowing normal BF16/backend drift; exact full trajectories are not a gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top1(step: dict[str, Any]) -> str | None:
    ranked = [(int(v.get("rank", 10**9)), str(k)) for k, v in step.items() if v.get("rank") is not None]
    return min(ranked)[1] if ranked else None


def _topk_overlap(left: dict[str, Any], right: dict[str, Any]) -> int:
    return len(set(left) & set(right))


def _step_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    shared = set(left) & set(right)
    deltas = [abs(float(left[token]["logprob"]) - float(right[token]["logprob"])) for token in shared]
    return {
        "top1_left": _top1(left),
        "top1_right": _top1(right),
        "top1_equal": _top1(left) is not None and _top1(left) == _top1(right),
        "topk_overlap": len(shared),
        "max_shared_logprob_abs_delta": max(deltas) if deltas else None,
        "mean_shared_logprob_abs_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def _items(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get("outputs") or []
    # vLLM output is [repeat][request], HF output is written one JSON per item;
    # this helper is used only for vLLM reports.
    if rows and isinstance(rows[0], list):
        rows = rows[0]
    return {int(row.get("global_index", i)): row for i, row in enumerate(rows)}


def _hf_items(directory: Path, max_tokens: int) -> dict[int, dict[str, Any]]:
    result = {}
    for path in sorted((directory / f"t{max_tokens}").glob("*.json")):
        row = _load(path)
        result[int(row.get("global_index", -1))] = row
    return result


def compare_pair(left: dict[int, dict[str, Any]], right: dict[int, dict[str, Any]], label: str) -> dict[str, Any]:
    rows = []
    for index in sorted(set(left) & set(right)):
        a, b = left[index], right[index]
        a_lp = a.get("topk_logprobs") or []
        b_lp = b.get("topk_logprobs") or []
        steps = [_step_delta(x, y) for x, y in zip(a_lp, b_lp)]
        rows.append({
            "global_index": index,
            "tokens_left": len(a.get("token_ids", a.get("output_token_ids", []))),
            "tokens_right": len(b.get("token_ids", b.get("output_token_ids", []))),
            "exact_token_sequence": a.get("token_ids", a.get("output_token_ids", [])) == b.get("token_ids", b.get("output_token_ids", [])),
            "common_prefix": _common_prefix(a.get("token_ids", a.get("output_token_ids", [])), b.get("token_ids", b.get("output_token_ids", []))),
            "logprob_steps": steps,
            "first_step": steps[0] if steps else None,
        })
    first = [row["first_step"] for row in rows if row["first_step"]]
    return {
        "label": label,
        "num_items": len(rows),
        "exact_sequence_items": sum(bool(row["exact_token_sequence"]) for row in rows),
        "first_step_top1_equal": sum(bool(row["first_step"]["top1_equal"]) for row in rows if row["first_step"]),
        "first_step_topk_overlap_min": min((row["first_step"]["topk_overlap"] for row in rows if row["first_step"]), default=None),
        "first_step_max_logprob_delta": max((row["first_step"]["max_shared_logprob_abs_delta"] or 0.0 for row in rows if row["first_step"]), default=None),
        "rows": rows,
    }


def _common_prefix(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    for i in range(n):
        if int(left[i]) != int(right[i]):
            return i
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-prefix", type=Path, required=True)
    parser.add_argument("--vllm-k0", type=Path, required=True)
    parser.add_argument("--vllm-k1", type=Path, required=True)
    parser.add_argument("--hf-native", type=Path, required=True)
    parser.add_argument("--hf-k0", type=Path, required=True)
    parser.add_argument("--hf-k1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    v_prefix = _items(_load(args.vllm_prefix))
    v_k0 = _items(_load(args.vllm_k0))
    v_k1 = _items(_load(args.vllm_k1))
    hf_native = _hf_items(args.hf_native, args.max_tokens)
    hf_k0 = _hf_items(args.hf_k0, args.max_tokens)
    hf_k1 = _hf_items(args.hf_k1, args.max_tokens)
    report = {
        "criteria": {
            "exact_full_sequence": "diagnostic only; not a cross-backend gate",
            "k0_identity": "vLLM dynamic K=0 vs vLLM prefix; top-k/first-decision and sequence recorded",
            "suppression_effect": "vLLM dynamic K=0 vs K=0.1; at least one first-step decision/distribution change is expected",
            "cross_backend": "HF native vs vLLM prefix and HF dynamic vs vLLM dynamic; report top-k overlap/logprob drift",
        },
        "vllm": {
            "k0_identity": compare_pair(v_prefix, v_k0, "vllm_prefix_vs_dynamic_k0"),
            "suppression_effect": compare_pair(v_k0, v_k1, "vllm_dynamic_k0_vs_k01"),
        },
        "cross_backend": {
            "native_calibration": compare_pair(v_prefix, hf_native, "vllm_prefix_vs_hf_native"),
            "k0_calibration": compare_pair(v_k0, hf_k0, "vllm_k0_vs_hf_k0"),
            "dynamic_calibration": compare_pair(v_k1, hf_k1, "vllm_k01_vs_hf_k01"),
        },
    }
    k0 = report["vllm"]["k0_identity"]
    effect = report["vllm"]["suppression_effect"]
    report["pass"] = bool(
        k0["num_items"] > 0
        and k0["first_step_top1_equal"] == k0["num_items"]
        and effect["num_items"] > 0
        and (
            effect["first_step_top1_equal"] < effect["num_items"]
            or float(effect["first_step_max_logprob_delta"] or 0.0) > 1e-5
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "k0": k0, "effect": effect}, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
