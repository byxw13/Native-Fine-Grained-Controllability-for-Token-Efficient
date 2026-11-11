from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fsync_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def _checkpoints(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in events if row.get("event") == "checkpoint"]


def _forced_signature(events: list[dict[str, Any]]) -> list[tuple[str, int | None]]:
    result = []
    for row in events:
        event = str(row.get("event") or "")
        if event.startswith("forced_") or event in {"closure_enter", "tail_signal_armed"}:
            result.append((event, None if row.get("token_id") is None else int(row["token_id"])))
    return result


def _event_numeric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    fields = (
        "avg_entropy",
        "avg_margin",
        "positive_entropy_delta",
        "signal_score",
        "target_suppression",
    )
    return {
        field: abs(float(left[field]) - float(right[field]))
        for field in fields
        if isinstance(left.get(field), (int, float)) and isinstance(right.get(field), (int, float))
    }


def _resolve_trace_request_id(output_request_id: str, traces: dict[str, list[dict]]) -> str | None:
    """Resolve vLLM's public id to its internally uniquified engine id."""
    if output_request_id in traces:
        return output_request_id
    matches = [value for value in traces if value.startswith(output_request_id + "-")]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--entropy-tolerance", type=float, default=0.15)
    parser.add_argument("--margin-tolerance", type=float, default=0.02)
    parser.add_argument("--signal-tolerance", type=float, default=0.02)
    parser.add_argument("--k-tolerance", type=float, default=0.01)
    # Compatibility for an already-submitted Slurm copy of the S3 script.
    # The former single threshold cannot express the calibrated per-field
    # contract, so it is accepted for parsing but intentionally not applied.
    parser.add_argument("--metric-tolerance", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()

    reports = []
    errors: list[str] = []
    for max_tokens in args.max_tokens:
        vllm_report = _read_json(args.vllm_root / f"t{max_tokens}" / f"t{max_tokens}.json")
        outputs = (vllm_report.get("outputs") or [[]])[0]
        traces: dict[str, list[dict[str, Any]]] = {}
        for row in _read_jsonl(args.vllm_root / f"t{max_tokens}" / "dynamic_trace_tp0.jsonl"):
            traces.setdefault(str(row.get("request_id")), []).append(row)
        oracle_rows = [
            _read_json(path)
            for path in sorted((args.oracle_root / f"t{max_tokens}").glob("*.json"))
        ]
        oracle_by_index = {int(row["global_index"]): row for row in oracle_rows if not row.get("error")}
        output_indices = [int(value) for value in vllm_report.get("indices") or []]
        items = []
        for position, output in enumerate(outputs):
            if position >= len(output_indices):
                errors.append(f"T{max_tokens}: output {position} has no global index")
                continue
            global_index = output_indices[position]
            oracle = oracle_by_index.get(global_index)
            if oracle is None:
                errors.append(f"T{max_tokens}: missing oracle item {global_index}")
                continue
            request_id = str(output["request_id"])
            v_ids = [int(value) for value in output.get("token_ids") or []]
            h_ids = [int(value) for value in oracle.get("output_token_ids") or []]
            common = _prefix_len(v_ids, h_ids)
            trace_request_id = _resolve_trace_request_id(request_id, traces)
            events = [] if trace_request_id is None else traces[trace_request_id]
            if not events:
                errors.append(f"T{max_tokens}: missing vLLM trace for {request_id}")
            v_cp = _checkpoints(events)
            h_cp = _checkpoints(oracle.get("events") or [])
            comparable = []
            for v_event, h_event in zip(v_cp, h_cp):
                v_pos = int(v_event.get("cumulative_generated_tokens") or 0)
                h_pos = int(h_event.get("cumulative_generated_tokens") or 0)
                if max(v_pos, h_pos) > common:
                    break
                deltas = _event_numeric_delta(v_event, h_event)
                types_match = v_event.get("checkpoint_type") == h_event.get("checkpoint_type")
                k_match = deltas.get("target_suppression", float("inf")) <= args.k_tolerance
                metric_ok = (
                    deltas.get("avg_entropy", float("inf")) <= args.entropy_tolerance
                    and deltas.get("avg_margin", float("inf")) <= args.margin_tolerance
                    and deltas.get("positive_entropy_delta", 0.0) <= args.entropy_tolerance
                    and deltas.get("signal_score", float("inf")) <= args.signal_tolerance
                )
                comparable.append(
                    {
                        "checkpoint_type_vllm": v_event.get("checkpoint_type"),
                        "checkpoint_type_oracle": h_event.get("checkpoint_type"),
                        "vllm_position": v_pos,
                        "oracle_position": h_pos,
                        "numeric_abs_delta": deltas,
                        "types_match": types_match,
                        "k_match": k_match,
                        "metric_ok": metric_ok,
                    }
                )
                if not (types_match and k_match and metric_ok):
                    errors.append(
                        f"T{max_tokens} item {global_index}: checkpoint mismatch {comparable[-1]}"
                    )
            first_forward = [row for row in events if row.get("event") == "first_forward_budget_asserted"]
            init = [row for row in events if row.get("event") == "request_init"]
            finish = [row for row in events if row.get("event") == "request_finish"]
            lifecycle_ok = len(init) == 1 and len(first_forward) == 1 and len(finish) == 1
            if not lifecycle_ok:
                errors.append(
                    f"T{max_tokens} item {global_index}: lifecycle init/first/finish="
                    f"{len(init)}/{len(first_forward)}/{len(finish)}"
                )
            forced_v = _forced_signature(events)
            forced_h = _forced_signature(oracle.get("events") or [])
            forced_prefix_match = forced_v[: min(len(forced_v), len(forced_h))] == forced_h[: min(len(forced_v), len(forced_h))]
            if not forced_prefix_match:
                errors.append(f"T{max_tokens} item {global_index}: forced-event order differs")
            if not comparable or comparable[0].get("checkpoint_type_vllm") != "increase_check_checkpoint0":
                errors.append(
                    f"T{max_tokens} item {global_index}: no comparable diagnostic checkpoint0"
                )
            items.append(
                {
                    "global_index": global_index,
                    "request_id": request_id,
                    "trace_request_id": trace_request_id,
                    "vllm_tokens": len(v_ids),
                    "oracle_tokens": len(h_ids),
                    "exact_tokens": v_ids == h_ids,
                    "common_prefix_tokens": common,
                    "first_divergence": None if v_ids == h_ids else common,
                    "vllm_checkpoints": len(v_cp),
                    "oracle_checkpoints": len(h_cp),
                    "comparable_checkpoints": comparable,
                    "lifecycle_ok": lifecycle_ok,
                    "forced_event_prefix_match": forced_prefix_match,
                    "vllm_forced_signature": forced_v,
                    "oracle_forced_signature": forced_h,
                }
            )
        oracle_errors = [row.get("error") for row in oracle_rows if row.get("error")]
        if oracle_errors:
            errors.extend(f"T{max_tokens}: oracle error: {value}" for value in oracle_errors)
        if len(items) != len(outputs) or not items:
            errors.append(f"T{max_tokens}: compared {len(items)} of {len(outputs)} outputs")
        reports.append(
            {
                "max_tokens": max_tokens,
                "items": items,
                "vllm_trace_requests": len(traces),
                "oracle_items": len(oracle_rows),
            }
        )

    result = {
        "pass": not errors,
        "tolerances": {
            "avg_entropy": args.entropy_tolerance,
            "avg_margin": args.margin_tolerance,
            "positive_entropy_delta": args.entropy_tolerance,
            "signal_score": args.signal_tolerance,
            "target_suppression": args.k_tolerance,
        },
        "lengths": reports,
        "errors": errors,
    }
    _fsync_json(args.output, result)
    print(json.dumps({"pass": result["pass"], "errors": errors}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
