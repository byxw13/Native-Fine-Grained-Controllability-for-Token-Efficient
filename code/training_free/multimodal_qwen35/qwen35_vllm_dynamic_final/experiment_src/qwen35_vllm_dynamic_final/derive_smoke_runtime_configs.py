from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import ControllerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument(
        "--checkpoint-zero",
        action="store_true",
        help="Enable the diagnostic first-token checkpoint for oracle parity smokes only.",
    )
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for max_tokens in args.max_tokens:
        payload = dict(base)
        payload["max_new_tokens"] = int(max_tokens)
        if args.checkpoint_zero:
            payload["checkpoint_zero"] = True
        # Compress only the tail wait window so short T32/T128/T512 smokes
        # exercise the same 80%-arming/forced-token state machine before the
        # artificial generation cap. Formal runs retain the frozen 300 tokens.
        payload["tail_signal_wait_token_limit"] = max(2, int(max_tokens * 0.05))
        target = args.output_dir / f"runtime_init0p1_scale0p8_t{max_tokens}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ControllerConfig.from_json(target)
        written.append(str(target))
    print(json.dumps({"written": written}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
