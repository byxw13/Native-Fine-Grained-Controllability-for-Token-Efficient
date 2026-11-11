from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_reports(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "mode" in value and "median_throughput_tokens_per_second" in value:
            rows.append((path, value))
    return rows


def _replica_exact(rows: list[dict[str, Any]]) -> bool | None:
    if len(rows) < 2 or not all(row.get("outputs") for row in rows):
        return None
    reference = [[item["token_ids"] for item in repeat] for repeat in rows[0]["outputs"]]
    return all(
        [[item["token_ids"] for item in repeat] for repeat in row["outputs"]] == reference
        for row in rows[1:]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-ratio", type=float, default=0.80)
    parser.add_argument("--expected-replicas", type=int, default=4)
    parser.add_argument(
        "--throughput-basis",
        choices=("reported", "generate-call"),
        default="reported",
    )
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, row in _read_reports(args.report_dir):
        grouped[(str(row["mode"]), int(row["max_tokens"]))].append((path, row))

    summary: dict[str, Any] = {}
    for (mode, max_tokens), items in sorted(grouped.items()):
        if args.throughput_basis == "generate-call":
            values = [
                statistics.median(
                    float(run["output_tokens"]) / float(run["generate_seconds"])
                    for run in row["runs"]
                )
                for _, row in items
            ]
        else:
            values = [float(row["median_throughput_tokens_per_second"]) for _, row in items]
        summary[f"{mode}_t{max_tokens}"] = {
            "replicas": len(items),
            "files": [str(path.relative_to(args.report_dir)) for path, _ in items],
            "per_replica_tokens_per_second": values,
            "aggregate_tokens_per_second": sum(values),
            "median_replica_tokens_per_second": statistics.median(values),
            "replica_outputs_exact": _replica_exact([row for _, row in items]),
        }

    token_lengths = sorted({token for _, token in grouped})
    ratios: dict[str, Any] = {}
    gates = []
    for max_tokens in token_lengths:
        native = summary.get(f"native_t{max_tokens}")
        dynamic = summary.get(f"dynamic_t{max_tokens}")
        if not native or not dynamic:
            continue
        ratio = dynamic["aggregate_tokens_per_second"] / native["aggregate_tokens_per_second"]
        replica_gate = (
            native["replicas"] == args.expected_replicas
            and dynamic["replicas"] == args.expected_replicas
        )
        passed = ratio >= args.minimum_ratio and replica_gate
        ratios[f"t{max_tokens}"] = {
            "dynamic_over_native": ratio,
            "minimum_required": args.minimum_ratio,
            "replica_count_gate": replica_gate,
            "passed": passed,
        }
        gates.append(passed)
        prefix = summary.get(f"prefix_t{max_tokens}")
        if prefix:
            ratios[f"t{max_tokens}"]["dynamic_over_prefix"] = (
                dynamic["aggregate_tokens_per_second"] / prefix["aggregate_tokens_per_second"]
            )

    report = {
        "deployment": "four independent TP=1 model replicas",
        "throughput_basis": args.throughput_basis,
        "throughput_definition": (
            "sum of per-replica median output tokens / llm.generate call wall time"
            if args.throughput_basis == "generate-call"
            else "sum of per-replica reported median output-token throughput"
        ),
        "groups": summary,
        "ratios": ratios,
        "passed": bool(gates) and all(gates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
