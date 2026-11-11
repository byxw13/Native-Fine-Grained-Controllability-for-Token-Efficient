from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_WINDOWS = {
    "prev_end_next": (-1, 0, 1),
    "end_next": (0, 1),
    "prev_end": (-1, 0),
}


def _heads(raw: object, family: str) -> dict[int, tuple[int, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{family} must be a layer -> heads mapping")
    result: dict[int, tuple[int, ...]] = {}
    for layer_raw, heads_raw in raw.items():
        layer = int(layer_raw)
        if layer < 0:
            raise ValueError(f"{family} layer must be non-negative: {layer}")
        if not isinstance(heads_raw, Sequence) or isinstance(heads_raw, (str, bytes)):
            raise ValueError(f"{family}[{layer}] must be a head list")
        heads = tuple(sorted({int(item) for item in heads_raw}))
        if any(head < 0 for head in heads):
            raise ValueError(f"{family}[{layer}] contains a negative head")
        if heads:
            result[layer] = heads
    return result


@dataclass(frozen=True)
class SelectiveNoopConfig:
    window: str
    offsets: tuple[int, ...]
    full_heads: dict[int, tuple[int, ...]]
    linear_heads: dict[int, tuple[int, ...]]
    source_path: str
    sha256: str

    def positions(self, end_think_index: int) -> tuple[int, ...]:
        positions = tuple(int(end_think_index) + item for item in self.offsets)
        if not positions or positions[0] < 0:
            raise RuntimeError(
                f"invalid {self.window} boundary at </think> index {end_think_index}"
            )
        return positions


_CACHE: SelectiveNoopConfig | None = None


def load_config() -> SelectiveNoopConfig:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path_text = os.environ.get("ARIS_SELECTIVE_NOOP_CONFIG_JSON", "").strip()
    if not path_text:
        raise RuntimeError(
            "ARIS_SELECTIVE_NOOP_CONFIG_JSON is required when selective no-op is enabled"
        )
    path = Path(path_text).expanduser().resolve()
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if int(payload.get("version", 1)) != 1:
        raise ValueError("selective no-op config version must be 1")
    window = str(payload.get("token_window", "")).strip()
    if window not in _WINDOWS:
        raise ValueError(
            f"token_window must be one of {sorted(_WINDOWS)}, got {window!r}"
        )
    config = SelectiveNoopConfig(
        window=window,
        offsets=_WINDOWS[window],
        full_heads=_heads(payload.get("full_attention_heads"), "full_attention_heads"),
        linear_heads=_heads(
            payload.get("linear_attention_heads"), "linear_attention_heads"
        ),
        source_path=str(path),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if not config.full_heads and not config.linear_heads:
        raise ValueError("at least one Full or Linear head must be selected")
    _CACHE = config
    return config


def enabled() -> bool:
    return os.environ.get("ARIS_SELECTIVE_NOOP_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reset_config_cache_for_tests() -> None:
    global _CACHE
    _CACHE = None
