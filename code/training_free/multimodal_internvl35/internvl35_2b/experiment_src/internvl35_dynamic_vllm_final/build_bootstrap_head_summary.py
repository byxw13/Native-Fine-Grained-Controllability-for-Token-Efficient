from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--heads", type=int, default=32)
    args = parser.parse_args()
    values = [1.0 / args.heads] * args.heads
    payload = {
        "model": "Qwen3-VL-4B-Thinking",
        "bootstrap_only": True,
        "warning": "Uniform structural smoke summary; forbidden for final MMMU evaluation.",
        "num_total_items": 0,
        "num_valid_items_with_end_think": 0,
        "summary_by_layer": {
            str(layer): {"num_items": 0, "attn_mean_over_items": values, "contrib_mean_over_items": values}
            for layer in range(args.layers)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
