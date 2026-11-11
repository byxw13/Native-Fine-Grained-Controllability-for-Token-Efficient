from __future__ import annotations

import argparse
import json
from pathlib import Path


FULL_LAYERS = {"3", "7", "11", "15", "19", "23", "27", "31"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-stats", type=Path, required=True)
    parser.add_argument("--dynamic-stats", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--lifecycle-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = _load(args.native_stats)
    dynamic = _load(args.dynamic_stats)
    traces = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    checks = {
        "first_forward_assertions": int(dynamic.get("first_forward_budget_assertions", 0)),
        "first_forward_assertion_failures": int(
            dynamic.get("first_forward_budget_assertion_failures", 0)
        ),
        "states_created": int(dynamic.get("states_created", 0)),
        "states_finished": int(dynamic.get("states_finished", 0)),
        "shape_failures": int(dynamic.get("shape_failures", 0)),
        "max_live_states": int(dynamic.get("max_live_states", 0)),
        "native_fused_prefill_calls": int(native.get("native_fused_prefill_calls", 0)),
        "native_fused_decode_calls": int(native.get("native_fused_decode_calls", 0)),
        "fallback_calls": int(native.get("native_fused_decode_fallback_calls", 0)),
        "unparsed_layer_calls": int(native.get("unparsed_layer_calls", 0)),
        "hit_layers": sorted(native.get("layers", {}), key=int),
    }
    if checks["first_forward_assertions"] < args.expected_requests:
        errors.append("not every request passed the device-side first-forward K assertion")
    if checks["first_forward_assertion_failures"]:
        errors.append("first-forward K assertion failed")
    if checks["states_created"] != checks["states_finished"]:
        errors.append("controller state lifecycle is unbalanced")
    if checks["shape_failures"]:
        errors.append("dynamic controller saw a shape failure")
    if checks["max_live_states"] < args.expected_requests:
        errors.append("the real engine never held the expected concurrent request states")
    if checks["native_fused_prefill_calls"] <= 0:
        errors.append("initial K never reached the fused prefill/first-logit path")
    if checks["native_fused_decode_calls"] <= 0:
        errors.append("fused decode suppression never ran")
    if checks["fallback_calls"]:
        errors.append(f"native fused fallback occurred: {native.get('native_fused_decode_fallback_reasons')}")
    if checks["unparsed_layer_calls"]:
        errors.append(f"attention layer parsing failed: {native.get('unparsed_layer_names')}")
    if set(native.get("layers", {})) != FULL_LAYERS:
        errors.append(f"suppression hit layers differ from Qwen3.5 full layers: {checks['hit_layers']}")
    grouped: dict[str, set[str]] = {}
    grouped_rows: dict[str, list[dict]] = {}
    for event in traces:
        grouped.setdefault(str(event.get("request_id")), set()).add(str(event.get("event")))
        grouped_rows.setdefault(str(event.get("request_id")), []).append(event)
    complete = [
        request_id
        for request_id, events in grouped.items()
        if {"request_init", "first_forward_budget_asserted", "request_finish"} <= events
    ]
    checks["complete_trace_requests"] = len(complete)
    if len(complete) < args.expected_requests:
        errors.append("trace lacks init/first-forward/finish evidence for some requests")
    init_values = [
        float(row.get("current_k"))
        for row in traces
        if row.get("event") == "request_init" and isinstance(row.get("current_k"), (int, float))
    ]
    first_values = [
        float(row.get("expected_initial_k"))
        for row in traces
        if row.get("event") == "first_forward_budget_asserted"
        and isinstance(row.get("expected_initial_k"), (int, float))
    ]
    checks["request_init_k_values"] = sorted(set(init_values))
    checks["first_forward_k_values"] = sorted(set(first_values))
    if not init_values or any(abs(value - 0.1) > 1e-6 for value in init_values):
        errors.append("request metadata did not initialize every controller at K=0.1")
    if not first_values or any(abs(value - 0.1) > 1e-6 for value in first_values):
        errors.append("the first device attention forward did not consume K=0.1")
    checkpoint_k = [
        float(row["target_suppression"])
        for row in traces
        if row.get("event") == "checkpoint" and isinstance(row.get("target_suppression"), (int, float))
    ]
    checks["checkpoint_k_min"] = min(checkpoint_k) if checkpoint_k else None
    checks["checkpoint_k_max"] = max(checkpoint_k) if checkpoint_k else None
    if not checkpoint_k:
        errors.append("no runtime K checkpoint was observed")
    elif any(value < 0.0 or value > 0.8 + 1e-6 for value in checkpoint_k):
        errors.append("runtime K exceeded the effective [0, 0.8] bound")
    tail_armed = 0
    tail_inserted = 0
    duplicate_tail_requests = []
    for request_id, rows in grouped_rows.items():
        armed = sum(row.get("event") == "tail_signal_armed" for row in rows)
        inserted = sum(
            row.get("event") == "closure_enter"
            and str(row.get("reason", "")).startswith("tail_signal:")
            for row in rows
        )
        queued = sum(
            row.get("event") == "forced_tokens_queued" and row.get("reason") == "tail_signal"
            for row in rows
        )
        tail_armed += armed
        tail_inserted += inserted
        if armed > 1 or inserted > 1 or queued > 1:
            duplicate_tail_requests.append(request_id)
    checks["tail_signal_armed_requests"] = tail_armed
    checks["tail_signal_insertions"] = tail_inserted
    checks["duplicate_tail_request_ids"] = duplicate_tail_requests
    if tail_armed <= 0 or tail_inserted <= 0:
        errors.append("the short-run 80% tail mechanism was not armed and inserted")
    if duplicate_tail_requests:
        errors.append("the 80% tail mechanism triggered more than once for a request")
    if args.lifecycle_report is not None:
        lifecycle = _load(args.lifecycle_report)
        checks["lifecycle_abort"] = lifecycle
        if not lifecycle.get("abort_called"):
            errors.append("lifecycle smoke did not invoke abort")
        if not lifecycle.get("normal_finished"):
            errors.append("the non-aborted concurrent request did not finish")
        if not lifecycle.get("aborted_absent_from_finished_outputs"):
            errors.append("the aborted request leaked into finished outputs")
    report = {"pass": not errors, "errors": errors, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
