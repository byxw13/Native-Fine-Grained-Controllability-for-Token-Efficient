from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    parser.add_argument("--heads", type=int, default=16)
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if layers != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise ValueError(f"Qwen3.5 bootstrap must contain only full-attention layers: {layers}")
    values = [1.0 / args.heads] * args.heads
    payload = {
        "model": "Qwen3.5-4B",
        "bootstrap_only": True,
        "warning": "Uniform structural smoke summary; forbidden for final MMMU evaluation.",
        "num_total_items": 0,
        "num_valid_items_with_end_think": 0,
        "summary_by_layer": {
            str(layer): {"num_items": 0, "attn_mean_over_items": values, "contrib_mean_over_items": values}
            for layer in layers
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
