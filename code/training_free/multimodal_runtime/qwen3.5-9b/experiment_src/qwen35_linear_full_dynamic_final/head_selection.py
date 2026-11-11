from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LINEAR_SCORE_FIELDS = (
    "state_contribution_mean_over_items",
    "output_contribution_mean_over_items",
    "attn_mean_over_items",
)


@dataclass(frozen=True)
class RankedHead:
    layer: int
    head: int
    score: float
    cumulative: float


def _score_vector(record: dict[str, Any]) -> list[float]:
    for field in LINEAR_SCORE_FIELDS:
        values = record.get(field)
        if isinstance(values, list):
            return [max(float(value), 0.0) for value in values]
    raise KeyError(f"linear head record has none of {LINEAR_SCORE_FIELDS}")


def rank_linear_heads(summary: str | Path | dict[str, Any]) -> list[RankedHead]:
    payload = (
        json.loads(Path(summary).read_text(encoding="utf-8"))
        if isinstance(summary, (str, Path))
        else summary
    )
    by_layer = payload.get("linear_summary_by_layer", payload.get("summary_by_layer"))
    if not isinstance(by_layer, dict):
        raise KeyError("linear summary is missing linear_summary_by_layer/summary_by_layer")
    flat: list[tuple[int, int, float]] = []
    for layer_text, record in by_layer.items():
        for head, score in enumerate(_score_vector(record)):
            flat.append((int(layer_text), int(head), float(score)))
    flat.sort(key=lambda item: (-item[2], item[0], item[1]))
    total = sum(item[2] for item in flat)
    if total <= 0.0:
        raise ValueError("linear head summary has no positive score mass")
    cumulative = 0.0
    ranked: list[RankedHead] = []
    for layer, head, score in flat:
        cumulative += score / total
        ranked.append(RankedHead(layer, head, score, cumulative))
    return ranked


def select_linear_heads(
    summary: str | Path | dict[str, Any],
    budget: float,
    *,
    alpha: float = 1.0,
) -> dict[int, list[int]]:
    target = min(max(float(budget) * float(alpha), 0.0), 1.0)
    if target <= 0.0:
        return {}
    selected: dict[int, list[int]] = {}
    for item in rank_linear_heads(summary):
        selected.setdefault(item.layer, []).append(item.head)
        if item.cumulative >= target:
            break
    return {layer: sorted(heads) for layer, heads in sorted(selected.items())}


def cumulative_thresholds(summary: str | Path | dict[str, Any]) -> list[float]:
    return [item.cumulative for item in rank_linear_heads(summary)]


def selected_prefix_count(
    thresholds: list[float],
    budget: float,
    *,
    alpha: float = 1.0,
) -> int:
    target = min(max(float(budget) * float(alpha), 0.0), 1.0)
    return 0 if target <= 0.0 else min(bisect.bisect_left(thresholds, target) + 1, len(thresholds))
