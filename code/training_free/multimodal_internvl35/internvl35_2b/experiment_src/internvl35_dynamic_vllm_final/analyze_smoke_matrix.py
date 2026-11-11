from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


TOKENS = (32, 128, 512)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def token_comparison(left: list[int], right: list[int]) -> dict[str, Any]:
    prefix = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        prefix += 1
    width = max(len(left), len(right), 1)
    positional = sum(int(a) == int(b) for a, b in zip(left, right))
    return {
        "left_tokens": len(left),
        "right_tokens": len(right),
        "exact_match": left == right,
        "common_prefix_tokens": prefix,
        "token_match_ratio": round(positional / width, 6),
        "first_divergence": first_divergence(left, right),
    }


def flatten_hf_tokens(row: dict[str, Any]) -> list[int]:
    if "output_token_ids" in row:
        return [int(item) for item in row["output_token_ids"]]
    return [
        int(token)
        for run in row.get("runs", [])
        for token in (run.get("generated_ids") or [])
    ]


def flatten_hf_checkpoints(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "events" in row:
        return [event for event in row["events"] if event.get("event") == "checkpoint"]
    return [
        event
        for run in row.get("runs", [])
        for event in (run.get("checkpoint_events") or [])
        if str(event.get("checkpoint_type", "")).endswith("check")
        or event.get("checkpoint_type") == "increase_check_checkpoint0"
    ]


def load_hf(root: Path, kind: str, max_tokens: int) -> list[dict[str, Any]]:
    if kind == "text":
        return read_jsonl(root / f"text_transformers_t{max_tokens}" / "rollouts" / "eval_rollout.jsonl")
    paths = sorted((root / f"t{max_tokens}").glob("*.json"))
    return [read_json(path) for path in paths]


def load_traces(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row.get("request_id"))].append(row)
    return grouped


def selected_heads_for_budget(summary: dict[str, Any] | None, budget: float) -> dict[str, int]:
    if summary is None or budget <= 0.0:
        return {"selected_heads": 0, "selected_layers": 0}
    flat = [
        (int(layer), int(head), max(float(score), 0.0))
        for layer, record in summary["summary_by_layer"].items()
        for head, score in enumerate(record["attn_mean_over_items"])
    ]
    total = sum(item[2] for item in flat)
    if total <= 0.0:
        return {"selected_heads": 0, "selected_layers": 0}
    flat.sort(key=lambda item: item[2], reverse=True)
    selected: list[tuple[int, int]] = []
    cumulative = 0.0
    for layer, head, score in flat:
        selected.append((layer, head))
        cumulative += score / total
        if cumulative >= budget:
            break
    return {
        "selected_heads": len(selected),
        "selected_layers": len({item[0] for item in selected}),
    }


def checkpoint_delta(
    vllm_events: list[dict[str, Any]], hf_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fields = ("avg_entropy", "avg_margin", "signal_score", "target_suppression")
    result = []
    for index, (left, right) in enumerate(zip(vllm_events, hf_events)):
        row: dict[str, Any] = {
            "index": index,
            "vllm_type": left.get("checkpoint_type"),
            "transformers_type": right.get("checkpoint_type"),
        }
        for field in fields:
            lv, rv = left.get(field), right.get(field)
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                row[f"{field}_abs_diff"] = round(abs(float(lv) - float(rv)), 8)
        result.append(row)
    return result


def stats_gate(stats_root: Path, max_tokens: int) -> dict[str, Any]:
    native_paths = sorted(stats_root.glob(f"native_dynamic_t{max_tokens}_r*_tp*.json"))
    if not native_paths:
        # Performance runs keep debug aggregation disabled.  A separate short
        # hook audit records all TP ranks once because the fused path is shared
        # by the three max-token settings.
        native_paths = sorted(stats_root.glob("hook_audit_tp*.json"))
    dynamic_paths = sorted(stats_root.glob(f"dynamic_t{max_tokens}_r*_tp*.json"))
    if not dynamic_paths:
        dynamic_paths = sorted(stats_root.glob("hook_audit_dynamic_tp*.json"))
    native = [read_json(path) for path in native_paths]
    dynamic = [read_json(path) for path in dynamic_paths]
    fallback_reasons = [
        row.get("native_fused_decode_fallback_reasons", {}) for row in native
    ]
    # TP workers receive the prepared suppression metadata by broadcast.  Only
    # the driver rank executes/records prepare_inputs, whereas every TP rank
    # must observe enabled decode calls.  A non_decode fallback is expected for
    # prefill; every other fallback reason is a failed native-fused-path gate.
    unexpected_fallbacks = [
        {key: int(value) for key, value in reasons.items() if key != "non_decode" and int(value) > 0}
        for reasons in fallback_reasons
    ]
    states_balanced = all(
        int(row.get("states_created", 0)) == int(row.get("states_finished", 0))
        for row in dynamic
    )
    processed_metric_steps = [int(row.get("processed_metric_steps", 0)) for row in dynamic]
    return {
        "native_rank_files": len(native),
        "dynamic_rank_files": len(dynamic),
        "fused_calls_by_rank": [int(row.get("native_fused_decode_calls", 0)) for row in native],
        "enabled_flash_calls_by_rank": [int(row.get("any_enabled_flash_calls", 0)) for row in native],
        "enabled_budget_kinds_by_rank": [len(row.get("prepare_enabled_budget_counts", {})) for row in native],
        "fallback_reasons_by_rank": fallback_reasons,
        "unexpected_fallbacks_by_rank": unexpected_fallbacks,
        "shape_failures": sum(int(row.get("shape_failures", 0)) for row in dynamic),
        "forced_tokens": sum(int(row.get("forced_tokens", 0)) for row in dynamic),
        "processed_metric_steps_by_rank": processed_metric_steps,
        "states_balanced": states_balanced,
        "pass": bool(native)
        and all(int(row.get("native_fused_decode_calls", 0)) > 0 for row in native)
        and all(int(row.get("any_enabled_flash_calls", 0)) > 0 for row in native)
        and any(bool(row.get("prepare_enabled_budget_counts")) for row in native)
        and bool(dynamic)
        and all(value > 0 for value in processed_metric_steps)
        and not any(unexpected_fallbacks)
        and not any(int(row.get("shape_failures", 0)) for row in dynamic)
        and states_balanced,
    }


def median(value: dict[str, Any]) -> float:
    return float(value["median_throughput_tokens_per_second"])


def _report_path(root: Path, mode: str, max_tokens: int) -> Path:
    candidates = (
        root / "results" / mode / "r0" / f"t{max_tokens}.json",
        root / mode / f"t{max_tokens}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _aggregate_throughput(
    root: Path,
    mode: str,
    max_tokens: int,
    fallback: dict[str, Any],
    throughput_report: dict[str, Any] | None,
) -> float:
    if throughput_report is not None:
        row = throughput_report.get("groups", {}).get(f"{mode}_t{max_tokens}")
        if row is not None:
            return float(row["aggregate_tokens_per_second"])
    aggregate = root / "aggregate_report.json"
    if aggregate.exists():
        row = read_json(aggregate).get("groups", {}).get(f"{mode}_t{max_tokens}")
        if row is not None:
            return float(row["aggregate_tokens_per_second"])
    return median(fallback)


def analyze_length(
    *,
    vllm_root: Path,
    hf_root: Path,
    kind: str,
    max_tokens: int,
    threshold: float,
    head_summary: dict[str, Any] | None,
    throughput_report: dict[str, Any] | None,
) -> dict[str, Any]:
    reports = {
        mode: read_json(_report_path(vllm_root, mode, max_tokens))
        for mode in ("native", "prefix", "dynamic")
    }
    throughputs = {
        mode: _aggregate_throughput(vllm_root, mode, max_tokens, report, throughput_report)
        for mode, report in reports.items()
    }
    speed_native = throughputs["dynamic"] / throughputs["native"]
    speed_prefix = throughputs["dynamic"] / throughputs["prefix"]
    prefix_counts_ok = (
        all(int(item) == 0 for item in reports["native"].get("prefix_counts", [a.get("method_prefix_count") for a in reports["native"].get("audits", [])]))
        and all(int(item) == 1 for item in reports["prefix"].get("prefix_counts", [a.get("method_prefix_count") for a in reports["prefix"].get("audits", [])]))
        and all(int(item) == 1 for item in reports["dynamic"].get("prefix_counts", [a.get("method_prefix_count") for a in reports["dynamic"].get("audits", [])]))
    )
    prefix_outputs = reports["prefix"]["outputs"][0]
    dynamic_outputs = reports["dynamic"]["outputs"][0]
    trace_path = vllm_root / "traces" / f"dynamic_t{max_tokens}_r0_tp0.jsonl"
    if not trace_path.exists():
        trace_path = vllm_root / "traces" / f"dynamic_t{max_tokens}_tp0.jsonl"
    traces = load_traces(trace_path)
    hf_all_rows = load_hf(hf_root, kind, max_tokens)
    hf_errors = [row.get("error") for row in hf_all_rows if row.get("error")]
    hf_rows = [row for row in hf_all_rows if not row.get("error")]
    transformers_gate = len(hf_rows) == len(dynamic_outputs) and not hf_errors
    prefix_dynamic = []
    dynamic_hf = []
    checkpoint_deltas = []
    for index, dynamic_output in enumerate(dynamic_outputs):
        dynamic_ids = [int(item) for item in dynamic_output["token_ids"]]
        if index < len(prefix_outputs):
            prefix_dynamic.append(token_comparison(prefix_outputs[index]["token_ids"], dynamic_ids))
        if index < len(hf_rows):
            dynamic_hf.append(token_comparison(dynamic_ids, flatten_hf_tokens(hf_rows[index])))
            request_events = [
                row for row in traces.get(str(dynamic_output["request_id"]), []) if row.get("event") == "checkpoint"
            ]
            hf_events = flatten_hf_checkpoints(hf_rows[index])
            checkpoint_deltas.append(
                {
                    "request_id": str(dynamic_output["request_id"]),
                    "vllm_checkpoint_count": len(request_events),
                    "transformers_checkpoint_count": len(hf_events),
                    "deltas": checkpoint_delta(request_events, hf_events),
                }
            )
    checkpoint0_gate = len(checkpoint_deltas) == len(dynamic_outputs) and all(
        row["vllm_checkpoint_count"] > 0
        and row["transformers_checkpoint_count"] > 0
        and bool(row["deltas"])
        and row["deltas"][0].get("vllm_type") == "increase_check_checkpoint0"
        and row["deltas"][0].get("transformers_type") == "increase_check_checkpoint0"
        and float(row["deltas"][0].get("avg_entropy_abs_diff", math.inf)) <= 5e-3
        and float(row["deltas"][0].get("avg_margin_abs_diff", math.inf)) <= 5e-3
        and float(row["deltas"][0].get("signal_score_abs_diff", math.inf)) <= 5e-3
        and float(row["deltas"][0].get("target_suppression_abs_diff", math.inf)) <= 1e-6
        for row in checkpoint_deltas
    )
    transformers_selected_ids_gate = all(
        "processor_selected_ids" not in row
        or flatten_hf_tokens(row) == [int(value) for value in row["processor_selected_ids"]]
        for row in hf_rows
    )
    hook = stats_gate(vllm_root / "stats", max_tokens)
    dynamic_checkpoints = [
        row
        for rows in traces.values()
        for row in rows
        if row.get("event") == "checkpoint"
    ]
    k_values = sorted(
        {
            round(float(row["target_suppression"]), 6)
            for row in dynamic_checkpoints
            if isinstance(row.get("target_suppression"), (int, float))
        }
    )
    event_counts: dict[str, int] = defaultdict(int)
    for rows in traces.values():
        for event in rows:
            event_counts[str(event.get("event"))] += 1
    selected_head_counts = [
        {"k": value, **selected_heads_for_budget(head_summary, value)} for value in k_values
    ]
    finish_reason_counts: dict[str, int] = defaultdict(int)
    for output in dynamic_outputs:
        finish_reason_counts[str(output.get("finish_reason"))] += 1
    incomplete_requests = [
        request_id
        for request_id, rows in traces.items()
        if sum(row.get("event") == "request_init" for row in rows) != 1
        or sum(row.get("event") == "request_finish" for row in rows) != 1
    ]
    trace_integrity = {
        "request_count": len(traces),
        "request_init_count": sum(
            row.get("event") == "request_init" for rows in traces.values() for row in rows
        ),
        "request_finish_count": sum(
            row.get("event") == "request_finish" for rows in traces.values() for row in rows
        ),
        "incomplete_request_ids": incomplete_requests,
        "pass": bool(traces) and not incomplete_requests,
    }
    return {
        "max_tokens": max_tokens,
        "throughput_tokens_per_second": {mode: round(value, 6) for mode, value in throughputs.items()},
        "dynamic_over_native": round(speed_native, 6),
        "dynamic_over_prefix": round(speed_prefix, 6),
        "throughput_gate": speed_native >= threshold,
        "prefix_counts_ok": prefix_counts_ok,
        "hook_gate": hook,
        "dynamic_k_values": k_values,
        "dynamic_k_changed": any(value > 0.0 for value in k_values),
        "controller_checkpoint_observed": bool(dynamic_checkpoints),
        "selected_head_counts_by_k": selected_head_counts,
        "dynamic_event_counts": dict(sorted(event_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "trace_integrity": trace_integrity,
        "prefix_vs_dynamic": prefix_dynamic,
        "dynamic_vs_transformers": dynamic_hf,
        "checkpoint_abs_deltas": checkpoint_deltas,
        "checkpoint0_semantic_gate": checkpoint0_gate,
        "transformers_items": len(hf_rows),
        "transformers_expected_items": len(dynamic_outputs),
        "transformers_errors": hf_errors,
        "transformers_gate": transformers_gate,
        "transformers_selected_ids_gate": transformers_selected_ids_gate,
        "pass": speed_native >= threshold
        and prefix_counts_ok
        and hook["pass"]
        and trace_integrity["pass"]
        and transformers_gate
        and transformers_selected_ids_gate
        and checkpoint0_gate
        and bool(dynamic_checkpoints),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['kind']} 动态 vLLM Smoke 报告",
        "",
        f"结论：**{'通过' if report['pass'] else '未通过'}**；吞吐硬门槛为动态/原生 ≥ {report['threshold']:.0%}。",
        "",
        "| T | Native tok/s | Prefix tok/s | Dynamic tok/s | Dynamic/Native | Hook | K 动态 | 结论 |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in report["lengths"]:
        tps = row["throughput_tokens_per_second"]
        lines.append(
            f"| {row['max_tokens']} | {tps['native']:.2f} | {tps['prefix']:.2f} | {tps['dynamic']:.2f} | "
            f"{row['dynamic_over_native']:.1%} | {'通过' if row['hook_gate']['pass'] else '失败'} | "
            f"{'是' if row['dynamic_k_changed'] else '否'} | {'通过' if row['pass'] else '失败'} |"
        )
    lines.extend(["", "逐样本 token 首分叉、match ratio 及 checkpoint 数值差保存在同目录 JSON 报告中。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("text", "vl"), required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--head-summary", type=Path)
    parser.add_argument(
        "--throughput-report",
        type=Path,
        help="optional aggregate report whose available mode/T groups override the main run",
    )
    args = parser.parse_args()
    head_summary = read_json(args.head_summary) if args.head_summary else None
    throughput_report = read_json(args.throughput_report) if args.throughput_report else None
    lengths = [
        analyze_length(
            vllm_root=args.vllm_root,
            hf_root=args.transformers_root,
            kind=args.kind,
            max_tokens=value,
            threshold=args.threshold,
            head_summary=head_summary,
            throughput_report=throughput_report,
        )
        for value in TOKENS
    ]
    lifecycle_path = args.vllm_root / "results" / "dynamic" / "r0" / "lifecycle_abort.json"
    if not lifecycle_path.exists():
        lifecycle_path = args.vllm_root / "dynamic" / "lifecycle_abort.json"
    lifecycle = read_json(lifecycle_path) if lifecycle_path.exists() else None
    lifecycle_pass = True if args.kind == "text" else bool(
        lifecycle
        and lifecycle.get("abort_called")
        and lifecycle.get("normal_finished")
        and lifecycle.get("aborted_absent_from_finished_outputs")
    )
    report = {
        "kind": args.kind,
        "threshold": args.threshold,
        "throughput_report": str(args.throughput_report) if args.throughput_report else None,
        "lengths": lengths,
        "lifecycle": lifecycle,
        "lifecycle_pass": lifecycle_pass,
        "dynamic_k_changed_in_suite": any(row["dynamic_k_changed"] for row in lengths),
        "pass": all(row["pass"] for row in lengths)
        and lifecycle_pass
        and any(row["dynamic_k_changed"] for row in lengths),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "output": str(args.output_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
