#!/usr/bin/env python3
"""Create tokenizer-specific controller config and structural smoke summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def tokenize(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def newline_ids(tokenizer) -> list[int]:
    found: list[int] = []
    for token_id in range(len(tokenizer)):
        try:
            if "\n" in tokenizer.decode([token_id], skip_special_tokens=False):
                found.append(token_id)
        except Exception:
            continue
    if not found:
        raise RuntimeError("tokenizer contains no newline-bearing token")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--method-cue", default="So")
    parser.add_argument("--initial-k", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--max-k", type=float, default=0.8)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "max_new_tokens": 512,
        "initial_suppression": args.initial_k,
        "suppression_upper_bound": args.max_k,
        "cue_threshold": 0.1,
        "cue_gate_threshold": 0.1,
        "min_tokens_before_cue": 0,
        "enable_runtime_cue": False,
        "entropy_upper_bound": 1.0,
        "suppression_curve": "exp_saturating",
        "power_exponent": 0.65,
        "exp_beta": 2.0,
        "suppression_scale": args.scale,
        "raw_history_size": 10,
        "k_smoothing_initial_rho": 1.0,
        "k_smoothing_late_rho": 1.0,
        "k_smoothing_initial_checkpoints": 6,
        "upgrade_newline_span": 1,
        "downgrade_newline_span": 10,
        "downgrade_confirm_windows": 5,
        "post_downgrade_upgrade_freeze_newlines": 0,
        "tail_soft_ratio": 0.99999,
        "tail_full_ratio": 1.0,
        "tail_exp_beta": 3.0,
        "tail_threshold_max": 1.0,
        "tail_signal_trigger_ratio": 0.8,
        "tail_signal_wait_token_limit": 300,
        "tail_signal_next_checkpoint_only": False,
        "final_answer_prefix_trigger_ratio": 0.98,
        "rethink_cue_trigger_tokens": 0,
        "end_think_to_l1_lock": False,
        "end_think_mask_newlines": 0,
        "end_think_mask_tokens": 0,
        "end_think_mask_forever": False,
        "checkpoint_zero": False,
        "assistant_prefix_checkpoint": False,
        "calibration_only": False,
        "dynamic_question_lambda": False,
        "high_low_high_question_lambda": False,
        "linear_down_up_question_lambda": False,
        "end_think_token_ids": tokenize(tokenizer, "</think>"),
        "newline_token_ids": newline_ids(tokenizer),
        "cue_token_ids": tokenize(tokenizer, args.method_cue),
        "rethink_cue_token_ids": [],
        "tail_signal_token_ids": tokenize(tokenizer, "</think>\n"),
        "end_think_bridge_token_ids": [],
        "final_answer_prefix_token_ids": tokenize(tokenizer, "the answer is "),
        "signal_calibration": {
            "entropy_lo": 0.10,
            "entropy_hi": 0.90,
            "delta_entropy_hi": 0.50,
            "margin_lo": 0.01,
            "margin_hi": 0.50,
        },
        "forced_texts": {"method_prefix": f"<think>\\n\\n</think>\\n\\n{args.method_cue}"},
        "model": str(args.model),
        "model_language_architecture": "InternVL3.5-8B/InternVLChatModel/Qwen3ForCausalLM/full_attention",
        "smoke_only": True,
    }
    (args.output_dir / "controller_t512.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for value in (32, 128, 512):
        item = dict(config)
        item["max_new_tokens"] = value
        (args.output_dir / f"controller_t{value}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    uniform = [1.0] * args.heads
    summary = {
        "model": "InternVL3.5-8B/Qwen3-language-decoder",
        "bootstrap_only": True,
        "warning": "Uniform structural summary is for smoke only; collect an InternVL head summary before formal evaluation.",
        "num_total_items": 0,
        "num_valid_items_with_end_think": 0,
        "summary_by_layer": {
            str(layer): {
                "num_items": 0,
                "attn_mean_over_items": uniform,
                "contrib_mean_over_items": uniform,
            }
            for layer in range(args.layers)
        },
    }
    (args.output_dir / "internvl35_uniform_head_summary_smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"end_think_token_ids": config["end_think_token_ids"], "newline_count": len(config["newline_token_ids"]), "summary_heads": args.layers * args.heads}, ensure_ascii=False))


if __name__ == "__main__":
    main()
