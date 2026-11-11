from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_src.sampling_v1.io_utils import read_jsonl, write_json


def compute_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile input is empty")
    if len(values) == 1:
        return round(float(values[0]), 6)
    ordered = sorted(float(value) for value in values)
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 6)


def summarize_scalar_list(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "q10": compute_quantile(values, 0.10),
        "q50": compute_quantile(values, 0.50),
        "q90": compute_quantile(values, 0.90),
        "max": round(max(values), 6),
    }


def collect_signal_checkpoint_events(rollout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rollout_rows:
        for run in row.get("runs", []):
            for checkpoint_event in run.get("checkpoint_events", []):
                if checkpoint_event.get("checkpoint_type") in {"increase_check_checkpoint0", "increase_check"}:
                    events.append(checkpoint_event)
    return events


def build_report(
    *,
    checkpoint_events: list[dict[str, Any]],
    run_name: str,
    dataset: str,
    selected_sample_count: int,
) -> dict[str, Any]:
    if not checkpoint_events:
        raise RuntimeError("no increase checkpoints found in rollout")
    avg_entropy_values = [float(event["avg_entropy"]) for event in checkpoint_events]
    avg_margin_values = [float(event["avg_margin"]) for event in checkpoint_events]
    positive_delta_values = [float(event["positive_entropy_delta"]) for event in checkpoint_events]
    entropy_summary = summarize_scalar_list(avg_entropy_values)
    margin_summary = summarize_scalar_list(avg_margin_values)
    delta_summary = summarize_scalar_list(positive_delta_values)
    recommended = {
        "entropy_lo": entropy_summary["q10"],
        "entropy_hi": entropy_summary["q90"],
        "delta_entropy_hi": delta_summary["q90"],
        "margin_lo": margin_summary["q10"],
        "margin_hi": margin_summary["q90"],
    }
    return {
        "method": "evaluation_v3_8_composite_signal_lambdaq_v1",
        "run_name": run_name,
        "dataset": dataset,
        "selected_sample_count": selected_sample_count,
        "checkpoint_event_types": ["increase_check_checkpoint0", "increase_check"],
        "checkpoint_count": len(checkpoint_events),
        "avg_entropy": entropy_summary,
        "avg_margin": margin_summary,
        "positive_entropy_delta": delta_summary,
        "recommended_signal_calibration": recommended,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild signal_calibration.json from a normal suppressed rollout.")
    parser.add_argument("--output-dir", required=True, help="Evaluation run root that contains rollouts/ and reports/.")
    parser.add_argument("--output-json", default=None, help="Optional explicit calibration json output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rollout_path = output_dir / "rollouts" / "eval_rollout.jsonl"
    summary_path = output_dir / "reports" / "eval_summary.json"
    if not rollout_path.exists():
        raise FileNotFoundError(f"missing rollout file: {rollout_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"missing eval summary: {summary_path}")

    rollout_rows = read_jsonl(rollout_path)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint_events = collect_signal_checkpoint_events(rollout_rows)
    report = build_report(
        checkpoint_events=checkpoint_events,
        run_name=str(summary_payload.get("run_name", output_dir.name)),
        dataset=str(summary_payload.get("dataset", "unknown")),
        selected_sample_count=int(summary_payload.get("selected_sample_count", len(rollout_rows))),
    )

    output_json = Path(args.output_json) if args.output_json else (output_dir / "reports" / "signal_calibration.json")
    metrics_json = output_dir / "metrics" / "signal_calibration.json"
    write_json(output_json, report)
    if output_json != metrics_json:
        write_json(metrics_json, report)
    print(json.dumps({"output_json": str(output_json), "metrics_json": str(metrics_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
