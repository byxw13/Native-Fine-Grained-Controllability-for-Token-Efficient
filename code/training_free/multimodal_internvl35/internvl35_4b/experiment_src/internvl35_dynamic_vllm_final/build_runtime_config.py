from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .controller import ControllerConfig, QuestionLambdaCalibration, SignalCalibration


def _read_json(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def _tokenize(tokenizer, text: str) -> list[int]:
    return [int(item) for item in tokenizer.encode(text, add_special_tokens=False)] if text else []


def _newline_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for token_id in range(len(tokenizer)):
        try:
            if "\n" in tokenizer.decode([token_id], skip_special_tokens=False):
                ids.append(token_id)
        except Exception:
            continue
    if not ids:
        raise RuntimeError("tokenizer scan found no newline-bearing token")
    return ids


def build(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    signal = SignalCalibration.from_mapping(_read_json(args.signal_calibration))
    question_lambda = QuestionLambdaCalibration.from_mapping(_read_json(args.question_lambda_calibration))
    config = ControllerConfig(
        max_new_tokens=args.max_new_tokens,
        initial_suppression=args.initial_suppression,
        suppression_upper_bound=args.suppression_upper_bound,
        cue_threshold=args.cue_threshold,
        cue_gate_threshold=args.cue_gate_threshold,
        min_tokens_before_cue=args.min_tokens_before_cue,
        enable_runtime_cue=args.enable_runtime_cue,
        entropy_upper_bound=args.entropy_upper_bound,
        suppression_curve=args.suppression_curve,
        power_exponent=args.power_exponent,
        exp_beta=args.exp_beta,
        suppression_scale=args.suppression_scale,
        raw_history_size=args.raw_history_size,
        k_smoothing_initial_rho=args.k_smoothing_initial_rho,
        k_smoothing_late_rho=args.k_smoothing_late_rho,
        k_smoothing_initial_checkpoints=args.k_smoothing_initial_checkpoints,
        upgrade_newline_span=args.upgrade_newline_span,
        downgrade_newline_span=args.downgrade_newline_span,
        downgrade_confirm_windows=args.downgrade_confirm_windows,
        post_downgrade_upgrade_freeze_newlines=args.post_downgrade_upgrade_freeze_newlines,
        tail_soft_ratio=args.tail_soft_ratio,
        tail_full_ratio=args.tail_full_ratio,
        tail_exp_beta=args.tail_exp_beta,
        tail_threshold_max=args.tail_threshold_max,
        tail_signal_trigger_ratio=args.tail_signal_trigger_ratio,
        tail_signal_wait_token_limit=args.tail_signal_wait_token_limit,
        tail_signal_next_checkpoint_only=args.tail_signal_next_checkpoint_only,
        final_answer_prefix_trigger_ratio=args.final_answer_prefix_trigger_ratio,
        rethink_cue_trigger_tokens=args.rethink_cue_trigger_tokens,
        end_think_to_l1_lock=args.end_think_to_l1_lock,
        end_think_mask_newlines=args.end_think_mask_newlines,
        end_think_mask_tokens=args.end_think_mask_tokens,
        end_think_mask_forever=args.end_think_mask_forever,
        checkpoint_zero=args.checkpoint_zero,
        assistant_prefix_checkpoint=args.assistant_prefix_checkpoint,
        calibration_only=args.calibration_only,
        dynamic_question_lambda=args.dynamic_question_lambda,
        high_low_high_question_lambda=args.high_low_high_question_lambda,
        linear_down_up_question_lambda=args.linear_down_up_question_lambda,
        end_think_token_ids=_tokenize(tokenizer, "</think>"),
        newline_token_ids=_newline_token_ids(tokenizer),
        cue_token_ids=_tokenize(tokenizer, args.cue_text),
        rethink_cue_token_ids=_tokenize(tokenizer, args.rethink_cue_text),
        tail_signal_token_ids=_tokenize(tokenizer, args.tail_signal_text),
        end_think_bridge_token_ids=_tokenize(tokenizer, args.end_think_bridge_text),
        final_answer_prefix_token_ids=_tokenize(tokenizer, args.final_answer_prefix_text),
        signal_calibration=signal,
        question_lambda_calibration=question_lambda,
    )
    config.validate()
    payload = config.to_dict()
    payload["tokenizer"] = str(Path(args.tokenizer).resolve())
    payload["forced_texts"] = {
        "cue_text": args.cue_text,
        "rethink_cue_text": args.rethink_cue_text,
        "tail_signal_text": args.tail_signal_text,
        "end_think_bridge_text": args.end_think_bridge_text,
        "final_answer_prefix_text": args.final_answer_prefix_text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signal-calibration", type=Path)
    parser.add_argument("--question-lambda-calibration", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--initial-suppression", type=float, default=0.0)
    parser.add_argument("--suppression-upper-bound", type=float, default=1.0)
    parser.add_argument("--cue-threshold", type=float, default=0.1)
    parser.add_argument("--cue-gate-threshold", type=float, default=0.1)
    parser.add_argument("--min-tokens-before-cue", type=int, default=0)
    parser.add_argument("--enable-runtime-cue", action="store_true")
    parser.add_argument("--cue-text", default="")
    parser.add_argument("--entropy-upper-bound", type=float, default=1.0)
    parser.add_argument("--suppression-curve", choices=("power", "exp_saturating"), default="power")
    parser.add_argument("--power-exponent", type=float, default=0.65)
    parser.add_argument("--exp-beta", type=float, default=2.0)
    parser.add_argument("--suppression-scale", type=float, default=1.0)
    parser.add_argument("--raw-history-size", type=int, default=10)
    parser.add_argument("--k-smoothing-initial-rho", type=float, default=1.0)
    parser.add_argument("--k-smoothing-late-rho", type=float, default=0.75)
    parser.add_argument("--k-smoothing-initial-checkpoints", type=int, default=2)
    parser.add_argument("--upgrade-newline-span", type=int, default=1)
    parser.add_argument("--downgrade-newline-span", type=int, default=10)
    parser.add_argument("--downgrade-confirm-windows", type=int, default=3)
    parser.add_argument("--post-downgrade-upgrade-freeze-newlines", type=int, default=0)
    parser.add_argument("--tail-soft-ratio", type=float, default=0.75)
    parser.add_argument("--tail-full-ratio", type=float, default=0.90)
    parser.add_argument("--tail-exp-beta", type=float, default=3.0)
    parser.add_argument("--tail-threshold-max", type=float, default=0.6)
    parser.add_argument("--tail-signal-trigger-ratio", type=float, default=0.90)
    parser.add_argument("--tail-signal-wait-token-limit", type=int, default=300)
    parser.add_argument("--tail-signal-next-checkpoint-only", action="store_true")
    parser.add_argument("--tail-signal-text", default="</think>\n")
    parser.add_argument("--end-think-bridge-text", default="")
    parser.add_argument("--end-think-to-l1-lock", action="store_true")
    parser.add_argument("--end-think-mask-newlines", type=int, default=1)
    parser.add_argument("--end-think-mask-tokens", type=int, default=0)
    parser.add_argument("--end-think-mask-forever", action="store_true")
    parser.add_argument("--final-answer-prefix-text", default="the answer is ")
    parser.add_argument("--final-answer-prefix-trigger-ratio", type=float, default=0.98)
    parser.add_argument("--rethink-cue-text", default="")
    parser.add_argument("--rethink-cue-trigger-tokens", type=int, default=0)
    parser.add_argument("--checkpoint-zero", action="store_true")
    parser.add_argument("--assistant-prefix-checkpoint", action="store_true")
    parser.add_argument("--dynamic-question-lambda", action="store_true")
    parser.add_argument("--high-low-high-question-lambda", action="store_true")
    parser.add_argument("--linear-down-up-question-lambda", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    if args.signal_calibration is None and not args.calibration_only:
        parser.error("--signal-calibration is required unless --calibration-only is set")
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
