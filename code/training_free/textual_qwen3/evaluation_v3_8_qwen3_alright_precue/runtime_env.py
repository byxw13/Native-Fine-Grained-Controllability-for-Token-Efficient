from __future__ import annotations

import json
import math
import os
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_src.evaluation_v3_8_qwen3_alright_precue.config import SUPPORTED_SUPPRESSION_CURVES
from experiment_src.evaluation_v3_8_qwen3_alright_precue.judge import CascadeDualJudge
from experiment_src.evaluation_v3_8_qwen3_alright_precue.question_lambda import (
    QuestionLambdaCalibration,
    compute_question_lambda_q,
)
from experiment_src.sampling_v1_qwen3 import remote_sampling_runner as sampling_runtime


def _debug_env_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _debug_forced_insert_log(**payload: Any) -> None:
    if not _debug_env_flag("ARIS_QWEN3_DEBUG_FORCED_INSERT"):
        return
    print(json.dumps({"qwen3_debug": True, "source": "runtime_forced_insert", **payload}, ensure_ascii=False), flush=True)


@dataclass
class CascadeDecodeState:
    current_ids: torch.Tensor
    current_attention_mask: torch.Tensor
    past_key_values: Any | None
    generated_ids: list[int]


@dataclass(frozen=True)
class RuntimeResources:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    input_device: torch.device


@dataclass
class EpisodeRecord:
    sample_id: str
    question: str
    answer: str
    final_suppression_value: float
    peak_suppression_value: float
    question_lambda_q: float
    cue_inserted: bool
    tail_signal_inserted: bool
    attempt_count: int
    total_generated_tokens: int
    final_text: str
    runs: list[dict[str, Any]]
    judge_record: dict[str, Any]


@dataclass(frozen=True)
class SignalCalibration:
    entropy_lo: float
    entropy_hi: float
    delta_entropy_hi: float
    margin_lo: float
    margin_hi: float


def _runtime_suppression_label(suppression_value: float) -> str:
    return f"k_{float(suppression_value):.4f}"


def prepare_runtime_resources(
    model_path: str,
    tokenizer_path: str | None,
    llm_device: str,
    disable_custom_attention_patch: bool = False,
) -> RuntimeResources:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "eager",
    }
    if llm_device == "auto":
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        input_device = model.model.embed_tokens.weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(llm_device)
        input_device = torch.device(llm_device)
    if not disable_custom_attention_patch:
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(sampling_runtime.custom_qwen3_attention_forward, layer.self_attn)
    model.eval()
    return RuntimeResources(model=model, tokenizer=tokenizer, input_device=input_device)


def _extract_sample_id(row: dict[str, Any], fallback_index: int) -> str:
    for key in ("unique_id", "id", "index"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(fallback_index)


def _tokenize_plain_text(tokenizer: AutoTokenizer, text: str) -> list[int]:
    if not text:
        return []
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()


def _decode_token_ids(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _decode_piece_in_context(tokenizer: AutoTokenizer, running_ctx_ids: list[int], next_token_id: int) -> str:
    prior_text = _decode_token_ids(tokenizer, running_ctx_ids)
    new_text = _decode_token_ids(tokenizer, running_ctx_ids + [next_token_id])
    if new_text.startswith(prior_text):
        return new_text[len(prior_text) :]
    return _decode_token_ids(tokenizer, [next_token_id])


def _viz_text(text: str, maxlen: int = 400) -> str:
    text = text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    text = text.replace(" ", "·")
    if len(text) > maxlen:
        return text[:maxlen] + " …"
    return text


def _tail_ids(token_ids: list[int], n: int = 32) -> list[int]:
    if n <= 0:
        return []
    return list(token_ids[-n:])


def _longest_suffix_prefix_match(token_ids: list[int], pattern_ids: list[int]) -> int:
    if not token_ids or not pattern_ids:
        return 0
    max_len = min(len(token_ids), len(pattern_ids) - 1)
    for k in range(max_len, 0, -1):
        if token_ids[-k:] == pattern_ids[:k]:
            return k
    return 0


def _apply_end_think_mask(
    next_token_logits: torch.Tensor,
    generated_ids: list[int],
    end_think_tokens: list[int],
) -> list[int]:
    if not end_think_tokens:
        return []
    forbidden_ids = {end_think_tokens[0]}
    matched_prefix_len = _longest_suffix_prefix_match(generated_ids, end_think_tokens)
    if 0 < matched_prefix_len < len(end_think_tokens):
        forbidden_ids.add(end_think_tokens[matched_prefix_len])
    next_token_logits[:, list(forbidden_ids)] = float("-inf")
    return sorted(forbidden_ids)


def _infer_past_seq_len(past_key_values: Any | None) -> int | None:
    if past_key_values is None:
        return 0
    try:
        seq_len = getattr(past_key_values, "get_seq_length", None)
        if callable(seq_len):
            value = seq_len()
            if isinstance(value, int):
                return value
    except Exception:
        pass
    try:
        if isinstance(past_key_values, (list, tuple)) and past_key_values:
            first = past_key_values[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1 and hasattr(first[0], "shape"):
                return int(first[0].shape[-2])
    except Exception:
        pass
    return None


def _summarize_past_cache(past_key_values: Any | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": None if past_key_values is None else type(past_key_values).__name__,
        "seq_len": _infer_past_seq_len(past_key_values),
        "layers": None,
        "first_key_shape": None,
        "first_value_shape": None,
    }
    if past_key_values is None:
        return summary
    try:
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
        elif isinstance(past_key_values, (list, tuple)):
            legacy = past_key_values
        elif hasattr(past_key_values, "_past_key_values"):
            legacy = past_key_values._past_key_values
        else:
            legacy = None
        if isinstance(legacy, (list, tuple)):
            summary["layers"] = len(legacy)
            if legacy:
                first = legacy[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    if hasattr(first[0], "shape"):
                        summary["first_key_shape"] = list(first[0].shape)
                    if hasattr(first[1], "shape"):
                        summary["first_value_shape"] = list(first[1].shape)
    except Exception as exc:
        summary["summary_error"] = repr(exc)
    return summary


def _append_forced_token_ids(
    model: AutoModelForCausalLM,
    state: CascadeDecodeState,
    token_ids: list[int],
    forced_text: str | None = None,
) -> None:
    if not token_ids:
        return

    def _append_token(token_id: int) -> None:
        next_token = torch.tensor([[token_id]], dtype=state.current_ids.dtype, device=state.current_ids.device)
        state.generated_ids.append(int(token_id))
        state.current_ids = torch.cat([state.current_ids, next_token], dim=1)
        next_mask = torch.ones(
            (1, 1),
            dtype=state.current_attention_mask.dtype,
            device=state.current_attention_mask.device,
        )
        state.current_attention_mask = torch.cat([state.current_attention_mask, next_mask], dim=1)

    def _advance_last_token_into_cache() -> None:
        if state.current_ids.shape[1] <= 0:
            return
        if state.past_key_values is None:
            output = model(
                input_ids=state.current_ids,
                attention_mask=state.current_attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        else:
            output = model(
                input_ids=state.current_ids[:, -1:],
                attention_mask=state.current_attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        state.past_key_values = output.past_key_values

    def _fallback_full_rebuild() -> None:
        forced_tensor = torch.tensor([token_ids], dtype=state.current_ids.dtype, device=state.current_ids.device)
        state.generated_ids.extend(int(t) for t in token_ids)
        state.current_ids = torch.cat([state.current_ids, forced_tensor], dim=1)
        forced_mask = torch.ones(
            (1, len(token_ids)),
            dtype=state.current_attention_mask.dtype,
            device=state.current_attention_mask.device,
        )
        state.current_attention_mask = torch.cat([state.current_attention_mask, forced_mask], dim=1)

        if state.current_ids.shape[1] <= 1:
            state.past_key_values = None
        else:
            prefix_ids = state.current_ids[:, :-1]
            prefix_attention_mask = state.current_attention_mask[:, :-1]
            output = model(
                input_ids=prefix_ids,
                attention_mask=prefix_attention_mask,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            state.past_key_values = output.past_key_values

    with torch.no_grad():
        past_seq_len = _infer_past_seq_len(state.past_key_values)
        phase_gap = None if past_seq_len is None else int(state.current_ids.shape[1]) - int(past_seq_len)
        _debug_forced_insert_log(
            event="forced_insert_phase_check",
            phase_gap=phase_gap,
            past_seq_len=past_seq_len,
            current_ids_len=int(state.current_ids.shape[1]),
            forced_token_count=len(token_ids),
            forced_token_ids=[int(token_id) for token_id in token_ids],
            forced_text=forced_text,
            forced_text_visible=_viz_text(forced_text or ""),
        )

        if phase_gap == 1:
            _advance_last_token_into_cache()
            refreshed_past_len = _infer_past_seq_len(state.past_key_values)
            phase_gap = None if refreshed_past_len is None else int(state.current_ids.shape[1]) - int(refreshed_past_len)
            _debug_forced_insert_log(
                event="forced_insert_after_advance",
                phase_gap=phase_gap,
                past_seq_len=refreshed_past_len,
                current_ids_len=int(state.current_ids.shape[1]),
                forced_token_count=len(token_ids),
            )

        if phase_gap != 0:
            print(
                json.dumps(
                    {
                        "forced_insert_fallback": True,
                        "event": "forced_insert_full_rebuild_fallback",
                        "phase_gap": phase_gap,
                        "past_seq_len": past_seq_len,
                        "current_ids_len": int(state.current_ids.shape[1]),
                        "forced_token_count": len(token_ids),
                        "forced_token_ids": token_ids,
                        "forced_text": forced_text,
                        "forced_text_visible": _viz_text(forced_text or ""),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _fallback_full_rebuild()
            return

        _append_token(token_ids[0])
        for token_id in token_ids[1:]:
            _advance_last_token_into_cache()
            _append_token(token_id)
        _debug_forced_insert_log(
            event="forced_insert_applied",
            past_seq_len=_infer_past_seq_len(state.past_key_values),
            current_ids_len=int(state.current_ids.shape[1]),
            generated_ids_len=len(state.generated_ids),
            forced_token_count=len(token_ids),
        )


def _score_forced_suffix_token_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state: CascadeDecodeState,
    token_ids: list[int],
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Score forced suffix tokens with teacher forcing under their real context."""
    if not token_ids:
        return [], [], [], []
    suffix_len = len(token_ids)
    total_len = int(state.current_ids.shape[1])
    start_pos = total_len - suffix_len
    if start_pos <= 0:
        raise RuntimeError(
            f"无法为 forced suffix 评分: total_len={total_len}, suffix_len={suffix_len}"
        )
    output = model(
        input_ids=state.current_ids,
        attention_mask=state.current_attention_mask,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
    )
    entropy_values: list[float] = []
    margin_values: list[float] = []
    logprob_values: list[float] = []
    token_pieces: list[str] = []
    full_ids = state.current_ids[0].tolist()
    for offset, token_id in enumerate(token_ids):
        prediction_pos = start_pos + offset - 1
        token_logits = output.logits[:, prediction_pos, :]
        entropy_value, margin_value, logprob_value = sampling_runtime.compute_token_metric(
            token_logits[0],
            int(token_id),
        )
        entropy_values.append(entropy_value)
        margin_values.append(margin_value)
        logprob_values.append(logprob_value)
        token_pieces.append(_decode_piece_in_context(tokenizer, full_ids[: start_pos + offset], int(token_id)))
    return entropy_values, margin_values, logprob_values, token_pieces


class ContinuousEntropyRuntime:
    def __init__(
        self,
        run_name: str,
        resources: RuntimeResources,
        judge: CascadeDualJudge | None,
        summary_json: str,
        max_new_tokens: int,
        temperature: float,
        presence_penalty: float,
        assistant_prefix_text: str,
        cue_text: str,
        cue_threshold: float,
        cue_gate_threshold: float | None,
        min_tokens_before_cue: int,
        final_answer_prefix_text: str,
        final_answer_prefix_trigger_ratio: float,
        rethink_cue_text: str | None,
        rethink_cue_trigger_tokens: int,
        tail_soft_ratio: float,
        tail_full_ratio: float,
        tail_exp_beta: float,
        tail_threshold_max: float,
        tail_signal_trigger_ratio: float,
        tail_signal_text: str | None,
        tail_signal_wait_token_limit: int,
        tail_signal_next_checkpoint_only: bool,
        entropy_upper_bound: float,
        suppression_curve: str,
        power_exponent: float,
        exp_beta: float,
        suppression_scale: float,
        suppression_upper_bound: float,
        raw_history_size: int,
        k_smoothing_initial_rho: float,
        k_smoothing_late_rho: float,
        k_smoothing_initial_checkpoints: int,
        upgrade_newline_span: int,
        downgrade_newline_span: int,
        downgrade_confirm_windows: int,
        post_downgrade_upgrade_freeze_newlines: int,
        end_think_to_l1_lock: bool,
        end_think_mask_newlines: int,
        end_think_mask_tokens: int,
        end_think_mask_forever: bool,
        end_think_bridge_text: str | None,
        checkpoint_zero: bool,
        assistant_prefix_checkpoint: bool,
        initial_suppression: float,
        signal_calibration: SignalCalibration | None,
        question_lambda_calibration: QuestionLambdaCalibration | None,
        dynamic_question_lambda: bool = False,
        dynamic_lambda_probe_checkpoints: int = 4,
        dynamic_lambda_signal_lo: float = 0.18,
        dynamic_lambda_signal_hi: float = 0.55,
        dynamic_lambda_probe_scale_min: float = 0.30,
        dynamic_lambda_probe_scale_max: float = 0.45,
        dynamic_lambda_late_token_start: int = 5500,
        dynamic_lambda_late_token_end: int = 7000,
        dynamic_lambda_late_scale_start: float = 0.45,
        dynamic_lambda_late_scale_max: float = 0.80,
        high_low_high_question_lambda: bool = False,
        high_low_high_early_scale: float = 0.80,
        high_low_high_early_min_tokens: int = 800,
        high_low_high_early_min_checkpoints: int = 4,
        high_low_high_middle_scale: float = 0.40,
        high_low_high_late_token_start: int = 5500,
        high_low_high_late_token_end: int = 7000,
        high_low_high_late_scale_max: float = 0.80,
        high_low_high_adaptive_early: bool = False,
        high_low_high_adaptive_max_tokens: int = 2000,
        high_low_high_adaptive_window: int = 3,
        high_low_high_adaptive_avg_signal_threshold: float = 0.65,
        high_low_high_adaptive_max_signal_threshold: float = 0.90,
        high_low_high_adaptive_max_raw_threshold: float = 0.36,
        high_low_high_adaptive_raw_delta_threshold: float = 0.30,
        linear_down_up_question_lambda: bool = False,
        linear_down_up_high_scale: float = 0.80,
        linear_down_up_low_scale: float = 0.40,
        linear_down_up_hold_high_tokens: int = 800,
        linear_down_up_down_end_tokens: int = 1200,
        linear_down_up_up_start_tokens: int = 1200,
        linear_down_up_up_end_tokens: int = 7000,
        calibration_only: bool = False,
        debug_runtime_events: bool = False,
    ) -> None:
        self.run_name = run_name
        self.resources = resources
        self.judge = judge
        self.summary_json = summary_json
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.presence_penalty = float(presence_penalty)
        self.assistant_prefix_text = str(assistant_prefix_text or "")
        if self.presence_penalty < 0.0:
            raise ValueError(f"presence_penalty 必须非负: {self.presence_penalty}")
        self.cue_text = cue_text
        self.cue_threshold = max(float(cue_threshold), 0.0)
        self.cue_gate_threshold = (
            self.cue_threshold
            if cue_gate_threshold is None
            else max(float(cue_gate_threshold), 0.0)
        )
        self.min_tokens_before_cue = max(int(min_tokens_before_cue), 0)
        self.final_answer_prefix_text = str(final_answer_prefix_text or "")
        self.final_answer_prefix_trigger_ratio = float(final_answer_prefix_trigger_ratio)
        if not 0.0 <= self.final_answer_prefix_trigger_ratio <= 1.0:
            raise ValueError(
                f"final_answer_prefix_trigger_ratio 必须落在 [0, 1] 内: "
                f"{self.final_answer_prefix_trigger_ratio}"
            )
        self.rethink_cue_text = rethink_cue_text
        self.rethink_cue_trigger_tokens = max(int(rethink_cue_trigger_tokens), 0)
        self.tail_soft_ratio = float(tail_soft_ratio)
        self.tail_full_ratio = float(tail_full_ratio)
        self.tail_exp_beta = float(tail_exp_beta)
        self.tail_threshold_max = float(tail_threshold_max)
        self.tail_signal_trigger_ratio = float(tail_signal_trigger_ratio)
        self.tail_signal_text = tail_signal_text
        self.tail_signal_wait_token_limit = max(int(tail_signal_wait_token_limit), 0)
        self.tail_signal_next_checkpoint_only = bool(tail_signal_next_checkpoint_only)
        if not 0.0 <= self.tail_soft_ratio < self.tail_full_ratio <= 1.0:
            raise ValueError(
                f"tail ratios 必须满足 0 <= soft < full <= 1: "
                f"soft={self.tail_soft_ratio}, full={self.tail_full_ratio}"
            )
        if self.tail_exp_beta <= 0.0:
            raise ValueError(f"tail_exp_beta 必须为正数: {self.tail_exp_beta}")
        if not 0.0 <= self.tail_signal_trigger_ratio <= 1.0:
            raise ValueError(
                f"tail_signal_trigger_ratio 必须落在 [0, 1] 内: {self.tail_signal_trigger_ratio}"
            )
        if self.tail_threshold_max < self.cue_threshold:
            raise ValueError(
                f"tail_threshold_max 必须不小于 cue_threshold: "
                f"{self.tail_threshold_max} < {self.cue_threshold}"
            )
        self.entropy_upper_bound = float(entropy_upper_bound)
        if self.entropy_upper_bound <= self.cue_threshold:
            raise ValueError(
                f"entropy_upper_bound 必须大于 cue_threshold: "
                f"{self.entropy_upper_bound} <= {self.cue_threshold}"
            )
        if self.tail_threshold_max > self.entropy_upper_bound:
            raise ValueError(
                f"tail_threshold_max 必须不大于 entropy_upper_bound: "
                f"{self.tail_threshold_max} > {self.entropy_upper_bound}"
            )
        self.suppression_curve = str(suppression_curve).strip().lower()
        if self.suppression_curve not in SUPPORTED_SUPPRESSION_CURVES:
            raise ValueError(
                f"suppression_curve 必须是 {SUPPORTED_SUPPRESSION_CURVES} 之一: {self.suppression_curve!r}"
            )
        self.power_exponent = float(power_exponent)
        if self.power_exponent <= 0.0:
            raise ValueError(f"power_exponent 必须为正数: {self.power_exponent}")
        self.exp_beta = float(exp_beta)
        if self.exp_beta <= 0.0:
            raise ValueError(f"exp_beta 必须为正数: {self.exp_beta}")
        self.suppression_scale = float(suppression_scale)
        if self.suppression_scale < 0.0:
            raise ValueError(f"suppression_scale 必须非负: {self.suppression_scale}")
        self.suppression_upper_bound = max(float(suppression_upper_bound), 0.0)
        self.raw_history_size = max(int(raw_history_size), 1)
        self.k_smoothing_initial_rho = float(k_smoothing_initial_rho)
        self.k_smoothing_late_rho = float(k_smoothing_late_rho)
        for rho_name, rho_value in (
            ("k_smoothing_initial_rho", self.k_smoothing_initial_rho),
            ("k_smoothing_late_rho", self.k_smoothing_late_rho),
        ):
            if not 0.0 <= rho_value <= 1.0:
                raise ValueError(f"{rho_name} 必须落在 [0, 1] 内: {rho_value}")
        self.k_smoothing_initial_checkpoints = max(int(k_smoothing_initial_checkpoints), 0)
        self.upgrade_newline_span = max(int(upgrade_newline_span), 1)
        self.downgrade_newline_span = max(int(downgrade_newline_span), 1)
        self.downgrade_confirm_windows = max(int(downgrade_confirm_windows), 1)
        self.post_downgrade_upgrade_freeze_newlines = max(int(post_downgrade_upgrade_freeze_newlines), 0)
        self.end_think_to_l1_lock = bool(end_think_to_l1_lock)
        self.end_think_mask_newlines = max(int(end_think_mask_newlines), 0)
        self.end_think_mask_tokens = max(int(end_think_mask_tokens), 0)
        self.end_think_mask_forever = bool(end_think_mask_forever)
        self.end_think_bridge_text = end_think_bridge_text
        self.checkpoint_zero = bool(checkpoint_zero)
        self.assistant_prefix_checkpoint = bool(assistant_prefix_checkpoint)
        self.initial_suppression = self._clamp_suppression(initial_suppression)
        self.signal_calibration = signal_calibration
        self.question_lambda_calibration = question_lambda_calibration
        self.dynamic_question_lambda = bool(dynamic_question_lambda)
        self.dynamic_lambda_probe_checkpoints = max(int(dynamic_lambda_probe_checkpoints), 1)
        self.dynamic_lambda_signal_lo = float(dynamic_lambda_signal_lo)
        self.dynamic_lambda_signal_hi = float(dynamic_lambda_signal_hi)
        self.dynamic_lambda_probe_scale_min = float(dynamic_lambda_probe_scale_min)
        self.dynamic_lambda_probe_scale_max = float(dynamic_lambda_probe_scale_max)
        self.dynamic_lambda_late_token_start = max(int(dynamic_lambda_late_token_start), 0)
        self.dynamic_lambda_late_token_end = max(
            int(dynamic_lambda_late_token_end),
            self.dynamic_lambda_late_token_start + 1,
        )
        self.dynamic_lambda_late_scale_start = float(dynamic_lambda_late_scale_start)
        self.dynamic_lambda_late_scale_max = float(dynamic_lambda_late_scale_max)
        self.high_low_high_question_lambda = bool(high_low_high_question_lambda)
        self.high_low_high_early_scale = float(high_low_high_early_scale)
        self.high_low_high_early_min_tokens = max(int(high_low_high_early_min_tokens), 0)
        self.high_low_high_early_min_checkpoints = max(int(high_low_high_early_min_checkpoints), 0)
        self.high_low_high_middle_scale = float(high_low_high_middle_scale)
        self.high_low_high_late_token_start = max(int(high_low_high_late_token_start), 0)
        self.high_low_high_late_token_end = max(
            int(high_low_high_late_token_end),
            self.high_low_high_late_token_start + 1,
        )
        self.high_low_high_late_scale_max = float(high_low_high_late_scale_max)
        self.high_low_high_adaptive_early = bool(high_low_high_adaptive_early)
        self.high_low_high_adaptive_max_tokens = max(int(high_low_high_adaptive_max_tokens), 0)
        self.high_low_high_adaptive_window = max(int(high_low_high_adaptive_window), 1)
        self.high_low_high_adaptive_avg_signal_threshold = float(
            high_low_high_adaptive_avg_signal_threshold
        )
        self.high_low_high_adaptive_max_signal_threshold = float(
            high_low_high_adaptive_max_signal_threshold
        )
        self.high_low_high_adaptive_max_raw_threshold = float(high_low_high_adaptive_max_raw_threshold)
        self.high_low_high_adaptive_raw_delta_threshold = float(
            high_low_high_adaptive_raw_delta_threshold
        )
        self.linear_down_up_question_lambda = bool(linear_down_up_question_lambda)
        self.linear_down_up_high_scale = float(linear_down_up_high_scale)
        self.linear_down_up_low_scale = float(linear_down_up_low_scale)
        self.linear_down_up_hold_high_tokens = max(int(linear_down_up_hold_high_tokens), 0)
        self.linear_down_up_down_end_tokens = max(
            int(linear_down_up_down_end_tokens),
            self.linear_down_up_hold_high_tokens + 1,
        )
        self.linear_down_up_up_start_tokens = max(
            int(linear_down_up_up_start_tokens),
            self.linear_down_up_down_end_tokens,
        )
        self.linear_down_up_up_end_tokens = max(
            int(linear_down_up_up_end_tokens),
            self.linear_down_up_up_start_tokens + 1,
        )
        enabled_question_lambda_modes = sum(
            bool(value)
            for value in (
                self.dynamic_question_lambda,
                self.high_low_high_question_lambda,
                self.linear_down_up_question_lambda,
            )
        )
        if enabled_question_lambda_modes > 1:
            raise ValueError("动态 question_lambda 模式之间不能同时启用。")
        if enabled_question_lambda_modes:
            if self.suppression_scale <= 0.0:
                raise ValueError("动态 question_lambda 要求 suppression_scale > 0。")
            if self.question_lambda_calibration is not None:
                raise ValueError("动态 question_lambda 与旧版 question_lambda_calibration 不能同时启用。")
        if self.dynamic_question_lambda:
            if self.dynamic_lambda_signal_hi <= self.dynamic_lambda_signal_lo:
                raise ValueError("dynamic_lambda_signal_hi 必须大于 dynamic_lambda_signal_lo。")
            if self.dynamic_lambda_probe_scale_max < self.dynamic_lambda_probe_scale_min:
                raise ValueError("dynamic_lambda_probe_scale_max 必须不小于 dynamic_lambda_probe_scale_min。")
            if self.dynamic_lambda_late_scale_max < self.dynamic_lambda_late_scale_start:
                raise ValueError("dynamic_lambda_late_scale_max 必须不小于 dynamic_lambda_late_scale_start。")
        if self.high_low_high_question_lambda:
            if self.high_low_high_late_scale_max < self.high_low_high_middle_scale:
                raise ValueError("high_low_high_late_scale_max 必须不小于 high_low_high_middle_scale。")
        if self.linear_down_up_question_lambda:
            if self.linear_down_up_high_scale < self.linear_down_up_low_scale:
                raise ValueError("linear_down_up_high_scale 必须不小于 linear_down_up_low_scale。")
        self.calibration_only = bool(calibration_only)
        if self.signal_calibration is None and not self.calibration_only:
            raise ValueError("非 calibration_only 模式必须提供 signal_calibration。")
        self.debug_runtime_events = bool(debug_runtime_events)

    def _debug_event(self, **payload: Any) -> None:
        if not self.debug_runtime_events:
            return
        print(json.dumps({"debug_runtime_event": True, **payload}, ensure_ascii=False), flush=True)

    def _clamp_suppression(self, suppression_value: float) -> float:
        return round(min(max(float(suppression_value), 0.0), self.suppression_upper_bound), 6)

    def _tail_soft_token_index(self) -> float:
        return float(self.max_new_tokens) * self.tail_soft_ratio

    def _tail_full_token_index(self) -> float:
        return float(self.max_new_tokens) * self.tail_full_ratio

    def _tail_signal_trigger_token_index(self) -> float:
        return float(self.max_new_tokens) * self.tail_signal_trigger_ratio

    def _tail_progress(self, cumulative_generated_tokens: int) -> float:
        token_count = max(float(cumulative_generated_tokens), 0.0)
        soft = self._tail_soft_token_index()
        full = self._tail_full_token_index()
        if token_count <= soft:
            return 0.0
        if token_count >= full:
            return 1.0
        progress = (token_count - soft) / (full - soft)
        return round(min(max(progress, 0.0), 1.0), 6)

    def _tail_activation_threshold(self, cumulative_generated_tokens: int) -> float:
        progress = self._tail_progress(cumulative_generated_tokens)
        if progress <= 0.0 or self.tail_threshold_max <= self.cue_threshold:
            return round(self.cue_threshold, 6)
        denominator = 1.0 - math.exp(-self.tail_exp_beta)
        numerator = 1.0 - math.exp(-self.tail_exp_beta * progress)
        smooth = numerator / denominator if denominator > 0.0 else progress
        threshold = self.cue_threshold + (self.tail_threshold_max - self.cue_threshold) * smooth
        return round(min(max(threshold, self.cue_threshold), self.tail_threshold_max), 6)

    def _normalize_control_score(
        self,
        control_score: float,
        cumulative_generated_tokens: int,
    ) -> tuple[float, float, float]:
        activation_threshold = self._tail_activation_threshold(cumulative_generated_tokens)
        if float(control_score) < activation_threshold:
            return 0.0, self._tail_progress(cumulative_generated_tokens), activation_threshold
        denominator = self.entropy_upper_bound - activation_threshold
        if denominator <= 0.0:
            normalized = 1.0 if float(control_score) >= self.entropy_upper_bound else 0.0
        else:
            normalized = (float(control_score) - activation_threshold) / denominator
        return (
            round(min(max(normalized, 0.0), 1.0), 6),
            self._tail_progress(cumulative_generated_tokens),
            activation_threshold,
        )

    def _mapped_suppression_from_normalized_signal(self, normalized_signal: float) -> float:
        normalized_signal = min(max(float(normalized_signal), 0.0), 1.0)
        if self.suppression_curve == "power":
            mapped = normalized_signal ** self.power_exponent
        else:
            denominator = 1.0 - math.exp(-self.exp_beta)
            numerator = 1.0 - math.exp(-self.exp_beta * normalized_signal)
            mapped = numerator / denominator if denominator > 0.0 else normalized_signal
        return self._clamp_suppression(self.suppression_upper_bound * self.suppression_scale * mapped)

    def _set_suppression_runtime(self, suppression_value: float) -> float:
        suppression_value = self._clamp_suppression(suppression_value)
        budget = None if suppression_value <= 0.0 else suppression_value
        sampling_runtime.TARGET_SUPPRESS_HEADS_BY_LAYER = sampling_runtime.load_target_heads(
            Path(self.summary_json),
            budget,
        )
        self._debug_event(
            event="set_suppression_runtime",
            requested_suppression=round(float(suppression_value), 6),
            selected_head_count=int(sum(len(heads) for heads in sampling_runtime.TARGET_SUPPRESS_HEADS_BY_LAYER.values())),
            selected_layer_count=int(len(sampling_runtime.TARGET_SUPPRESS_HEADS_BY_LAYER)),
        )
        return suppression_value

    def _mapped_suppression_from_signal_score(
        self,
        signal_score: float,
        cumulative_generated_tokens: int,
    ) -> tuple[float, float, float, float]:
        normalized_signal, tail_progress, activation_threshold = self._normalize_control_score(
            signal_score,
            cumulative_generated_tokens,
        )
        return (
            normalized_signal,
            self._mapped_suppression_from_normalized_signal(normalized_signal),
            tail_progress,
            activation_threshold,
        )

    def _should_trigger_cue(self, signal_score: float | None, cumulative_generated_tokens: int) -> bool:
        del signal_score, cumulative_generated_tokens
        # This ablation uses only the one-token assistant prefix before generation.
        # Runtime cue insertion is disabled so any effect is attributable to the prefix.
        return False

    def _should_arm_tail_signal(self, cumulative_generated_tokens: int) -> bool:
        if not self.tail_signal_text:
            return False
        return float(cumulative_generated_tokens) >= self._tail_signal_trigger_token_index()

    def _apply_question_lambda_to_suppression(
        self,
        base_suppression: float,
        question_lambda_q: float,
    ) -> float:
        return self._clamp_suppression(float(base_suppression) * float(question_lambda_q))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(float(value), 0.0), 1.0)
        return value * value * (3.0 - (2.0 * value))

    @staticmethod
    def _quantile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)

    def _resolve_dynamic_question_lambda(
        self,
        *,
        signal_score: float | None,
        cumulative_generated_tokens: int,
        probe_signal_scores: list[float],
    ) -> tuple[float, dict[str, Any]]:
        if signal_score is not None and len(probe_signal_scores) < self.dynamic_lambda_probe_checkpoints:
            probe_signal_scores.append(min(max(float(signal_score), 0.0), 1.0))
        probe_u_q75 = self._quantile(probe_signal_scores, 0.75)
        probe_progress = self._smoothstep(
            (probe_u_q75 - self.dynamic_lambda_signal_lo)
            / (self.dynamic_lambda_signal_hi - self.dynamic_lambda_signal_lo)
        )
        probe_scale = self.dynamic_lambda_probe_scale_min + (
            (self.dynamic_lambda_probe_scale_max - self.dynamic_lambda_probe_scale_min) * probe_progress
        )
        late_progress = self._smoothstep(
            (float(cumulative_generated_tokens) - float(self.dynamic_lambda_late_token_start))
            / float(self.dynamic_lambda_late_token_end - self.dynamic_lambda_late_token_start)
        )
        late_scale = self.dynamic_lambda_late_scale_start + (
            (self.dynamic_lambda_late_scale_max - self.dynamic_lambda_late_scale_start) * late_progress
        )
        effective_scale = max(probe_scale, late_scale if cumulative_generated_tokens > self.dynamic_lambda_late_token_start else probe_scale)
        effective_scale = min(
            max(effective_scale, self.dynamic_lambda_probe_scale_min),
            self.dynamic_lambda_late_scale_max,
        )
        lambda_q = effective_scale / self.suppression_scale
        payload = {
            "question_lambda_mode": "dynamic_probe_late",
            "probe_signal_scores": [round(float(value), 6) for value in probe_signal_scores],
            "probe_u_q75": round(float(probe_u_q75), 6),
            "probe_scale": round(float(probe_scale), 6),
            "late_progress": round(float(late_progress), 6),
            "late_scale": round(float(late_scale), 6),
            "effective_scale": round(float(effective_scale), 6),
            "base_scale": round(float(self.suppression_scale), 6),
            "question_lambda_q": round(float(lambda_q), 6),
            "late_token_start": int(self.dynamic_lambda_late_token_start),
            "late_token_end": int(self.dynamic_lambda_late_token_end),
        }
        return lambda_q, payload

    def _resolve_high_low_high_question_lambda(
        self,
        *,
        increase_checkpoint_count: int,
        cumulative_generated_tokens: int,
        signal_score: float | None,
        raw_suppression: float | None,
        high_low_high_signal_scores: list[float],
        high_low_high_raw_suppressions: list[float],
    ) -> tuple[float, dict[str, Any]]:
        if signal_score is not None:
            high_low_high_signal_scores.append(min(max(float(signal_score), 0.0), 1.0))
        if raw_suppression is not None:
            high_low_high_raw_suppressions.append(min(max(float(raw_suppression), 0.0), 1.0))
        recent_signals = high_low_high_signal_scores[-self.high_low_high_adaptive_window :]
        recent_raws = high_low_high_raw_suppressions[-self.high_low_high_adaptive_window :]
        recent_avg_signal = sum(recent_signals) / len(recent_signals) if recent_signals else 0.0
        recent_max_signal = max(recent_signals) if recent_signals else 0.0
        recent_max_raw = max(recent_raws) if recent_raws else 0.0
        recent_raw_delta = (recent_raws[-1] - recent_raws[0]) if len(recent_raws) >= 2 else 0.0
        adaptive_keep_high = (
            self.high_low_high_adaptive_early
            and int(cumulative_generated_tokens) < self.high_low_high_adaptive_max_tokens
            and (
                recent_avg_signal >= self.high_low_high_adaptive_avg_signal_threshold
                or recent_max_signal >= self.high_low_high_adaptive_max_signal_threshold
                or recent_max_raw >= self.high_low_high_adaptive_max_raw_threshold
                or recent_raw_delta >= self.high_low_high_adaptive_raw_delta_threshold
            )
        )
        if (
            int(increase_checkpoint_count) <= self.high_low_high_early_min_checkpoints
            or int(cumulative_generated_tokens) < self.high_low_high_early_min_tokens
            or adaptive_keep_high
        ):
            phase = "early_high_adaptive" if adaptive_keep_high else "early_high"
            effective_scale = self.high_low_high_early_scale
            late_progress = 0.0
        elif int(cumulative_generated_tokens) <= self.high_low_high_late_token_start:
            phase = "middle_low"
            effective_scale = self.high_low_high_middle_scale
            late_progress = 0.0
        else:
            phase = "late_ramp"
            late_progress = self._smoothstep(
                (float(cumulative_generated_tokens) - float(self.high_low_high_late_token_start))
                / float(self.high_low_high_late_token_end - self.high_low_high_late_token_start)
            )
            effective_scale = self.high_low_high_middle_scale + (
                (self.high_low_high_late_scale_max - self.high_low_high_middle_scale) * late_progress
            )
        effective_scale = min(
            max(float(effective_scale), 0.0),
            float(self.high_low_high_late_scale_max),
        )
        lambda_q = effective_scale / self.suppression_scale
        payload = {
            "question_lambda_mode": "high_low_high",
            "phase": phase,
            "increase_checkpoint_count": int(increase_checkpoint_count),
            "cumulative_generated_tokens": int(cumulative_generated_tokens),
            "early_scale": round(float(self.high_low_high_early_scale), 6),
            "early_min_tokens": int(self.high_low_high_early_min_tokens),
            "early_min_checkpoints": int(self.high_low_high_early_min_checkpoints),
            "middle_scale": round(float(self.high_low_high_middle_scale), 6),
            "late_token_start": int(self.high_low_high_late_token_start),
            "late_token_end": int(self.high_low_high_late_token_end),
            "late_scale_max": round(float(self.high_low_high_late_scale_max), 6),
            "adaptive_early": bool(self.high_low_high_adaptive_early),
            "adaptive_max_tokens": int(self.high_low_high_adaptive_max_tokens),
            "adaptive_window": int(self.high_low_high_adaptive_window),
            "recent_signal_scores": [round(float(value), 6) for value in recent_signals],
            "recent_raw_suppressions": [round(float(value), 6) for value in recent_raws],
            "recent_avg_signal": round(float(recent_avg_signal), 6),
            "recent_max_signal": round(float(recent_max_signal), 6),
            "recent_max_raw_suppression": round(float(recent_max_raw), 6),
            "recent_raw_delta": round(float(recent_raw_delta), 6),
            "adaptive_avg_signal_threshold": round(
                float(self.high_low_high_adaptive_avg_signal_threshold),
                6,
            ),
            "adaptive_max_signal_threshold": round(
                float(self.high_low_high_adaptive_max_signal_threshold),
                6,
            ),
            "adaptive_max_raw_threshold": round(float(self.high_low_high_adaptive_max_raw_threshold), 6),
            "adaptive_raw_delta_threshold": round(
                float(self.high_low_high_adaptive_raw_delta_threshold),
                6,
            ),
            "adaptive_keep_high": bool(adaptive_keep_high),
            "late_progress": round(float(late_progress), 6),
            "effective_scale": round(float(effective_scale), 6),
            "base_scale": round(float(self.suppression_scale), 6),
            "question_lambda_q": round(float(lambda_q), 6),
        }
        return lambda_q, payload

    def _resolve_linear_down_up_question_lambda(
        self,
        *,
        cumulative_generated_tokens: int,
    ) -> tuple[float, dict[str, Any]]:
        token_count = max(float(cumulative_generated_tokens), 0.0)
        high_scale = float(self.linear_down_up_high_scale)
        low_scale = float(self.linear_down_up_low_scale)
        if token_count <= float(self.linear_down_up_hold_high_tokens):
            phase = "hold_high"
            effective_scale = high_scale
            down_progress = 0.0
            up_progress = 0.0
        elif token_count <= float(self.linear_down_up_down_end_tokens):
            phase = "linear_down"
            down_progress = (
                (token_count - float(self.linear_down_up_hold_high_tokens))
                / float(self.linear_down_up_down_end_tokens - self.linear_down_up_hold_high_tokens)
            )
            up_progress = 0.0
            effective_scale = high_scale + ((low_scale - high_scale) * down_progress)
        elif token_count <= float(self.linear_down_up_up_start_tokens):
            phase = "hold_low"
            down_progress = 1.0
            up_progress = 0.0
            effective_scale = low_scale
        elif token_count <= float(self.linear_down_up_up_end_tokens):
            phase = "linear_up"
            down_progress = 1.0
            up_progress = (
                (token_count - float(self.linear_down_up_up_start_tokens))
                / float(self.linear_down_up_up_end_tokens - self.linear_down_up_up_start_tokens)
            )
            effective_scale = low_scale + ((high_scale - low_scale) * up_progress)
        else:
            phase = "hold_high_late"
            down_progress = 1.0
            up_progress = 1.0
            effective_scale = high_scale
        effective_scale = min(max(float(effective_scale), 0.0), float(high_scale))
        lambda_q = effective_scale / self.suppression_scale
        payload = {
            "question_lambda_mode": "linear_down_up",
            "phase": phase,
            "cumulative_generated_tokens": int(cumulative_generated_tokens),
            "high_scale": round(high_scale, 6),
            "low_scale": round(low_scale, 6),
            "hold_high_tokens": int(self.linear_down_up_hold_high_tokens),
            "down_end_tokens": int(self.linear_down_up_down_end_tokens),
            "up_start_tokens": int(self.linear_down_up_up_start_tokens),
            "up_end_tokens": int(self.linear_down_up_up_end_tokens),
            "down_progress": round(float(down_progress), 6),
            "up_progress": round(float(up_progress), 6),
            "effective_scale": round(float(effective_scale), 6),
            "base_scale": round(float(self.suppression_scale), 6),
            "question_lambda_q": round(float(lambda_q), 6),
        }
        return lambda_q, payload

    def _resolve_effective_suppression(
        self,
        base_suppression: float,
        increase_checkpoint_count: int,
        early_base_suppressions: list[float],
        probe_signal_scores: list[float],
        high_low_high_signal_scores: list[float],
        high_low_high_raw_suppressions: list[float],
        signal_score: float | None,
        raw_suppression: float | None,
        cumulative_generated_tokens: int,
        question_lambda_q: float,
        question_lambda_frozen: bool,
    ) -> tuple[float, float, bool, dict[str, Any] | None]:
        if self.dynamic_question_lambda and not self.calibration_only:
            base_suppression = self._clamp_suppression(base_suppression)
            next_lambda_q, lambda_payload = self._resolve_dynamic_question_lambda(
                signal_score=signal_score,
                cumulative_generated_tokens=cumulative_generated_tokens,
                probe_signal_scores=probe_signal_scores,
            )
            return (
                self._apply_question_lambda_to_suppression(base_suppression, next_lambda_q),
                next_lambda_q,
                False,
                lambda_payload,
            )
        if self.high_low_high_question_lambda and not self.calibration_only:
            base_suppression = self._clamp_suppression(base_suppression)
            next_lambda_q, lambda_payload = self._resolve_high_low_high_question_lambda(
                increase_checkpoint_count=increase_checkpoint_count,
                cumulative_generated_tokens=cumulative_generated_tokens,
                signal_score=signal_score,
                raw_suppression=raw_suppression,
                high_low_high_signal_scores=high_low_high_signal_scores,
                high_low_high_raw_suppressions=high_low_high_raw_suppressions,
            )
            return (
                self._apply_question_lambda_to_suppression(base_suppression, next_lambda_q),
                next_lambda_q,
                False,
                lambda_payload,
            )
        if self.linear_down_up_question_lambda and not self.calibration_only:
            base_suppression = self._clamp_suppression(base_suppression)
            next_lambda_q, lambda_payload = self._resolve_linear_down_up_question_lambda(
                cumulative_generated_tokens=cumulative_generated_tokens,
            )
            return (
                self._apply_question_lambda_to_suppression(base_suppression, next_lambda_q),
                next_lambda_q,
                False,
                lambda_payload,
            )
        if self.question_lambda_calibration is None or self.calibration_only:
            return self._clamp_suppression(base_suppression), question_lambda_q, question_lambda_frozen, None

        base_suppression = self._clamp_suppression(base_suppression)
        lambda_payload: dict[str, Any] | None = None
        next_lambda_q = float(question_lambda_q)
        next_lambda_frozen = bool(question_lambda_frozen)
        if increase_checkpoint_count <= 2:
            early_base_suppressions.append(base_suppression)
        if (not next_lambda_frozen) and increase_checkpoint_count >= 2 and len(early_base_suppressions) >= 2:
            d_q, normalized_d_q, resolved_lambda_q = compute_question_lambda_q(
                early_base_suppressions[0],
                early_base_suppressions[1],
                self.question_lambda_calibration,
            )
            next_lambda_q = resolved_lambda_q
            next_lambda_frozen = True
            lambda_payload = {
                "k1_base": round(float(early_base_suppressions[0]), 6),
                "k2_base": round(float(early_base_suppressions[1]), 6),
                "d_q": round(float(d_q), 6),
                "normalized_d_q": round(float(normalized_d_q), 6),
                "question_lambda_q": round(float(resolved_lambda_q), 6),
                "lambda_applied_from_checkpoint": int(increase_checkpoint_count),
                "lambda_min": round(float(self.question_lambda_calibration.lambda_min), 6),
                "lambda_max": round(float(self.question_lambda_calibration.lambda_max), 6),
                "d_lo": round(float(self.question_lambda_calibration.d_lo), 6),
                "d_hi": round(float(self.question_lambda_calibration.d_hi), 6),
            }
            self._debug_event(
                event="question_lambda_freeze",
                increase_checkpoint_count=int(increase_checkpoint_count),
                base_suppression=round(float(base_suppression), 6),
                early_base_suppressions=[round(float(value), 6) for value in early_base_suppressions[:2]],
                **lambda_payload,
            )
        return (
            self._apply_question_lambda_to_suppression(base_suppression, next_lambda_q),
            next_lambda_q,
            next_lambda_frozen,
            lambda_payload,
        )

    def _resolve_k_smoothing_rho(self, checkpoint_index: int) -> float:
        if checkpoint_index <= self.k_smoothing_initial_checkpoints:
            return self.k_smoothing_initial_rho
        return self.k_smoothing_late_rho

    def _smooth_target_suppression(
        self,
        current_suppression: float,
        raw_target_suppression: float,
        checkpoint_index: int,
    ) -> tuple[float, float]:
        rho_t = self._resolve_k_smoothing_rho(checkpoint_index)
        smoothed_target = ((1.0 - rho_t) * float(current_suppression)) + (rho_t * float(raw_target_suppression))
        return round(rho_t, 6), self._clamp_suppression(smoothed_target)

    def _normalize_component(self, value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if float(value) > upper else 0.0
        normalized = (float(value) - lower) / (upper - lower)
        return round(min(max(normalized, 0.0), 1.0), 6)

    def _build_signal_features(
        self,
        avg_entropy: float,
        avg_margin: float,
        previous_checkpoint_entropy: float | None,
        cumulative_generated_tokens: int,
    ) -> dict[str, Any]:
        avg_entropy = round(float(avg_entropy), 6)
        avg_margin = round(float(avg_margin), 6)
        positive_entropy_delta = round(
            max(avg_entropy - float(previous_checkpoint_entropy), 0.0) if previous_checkpoint_entropy is not None else 0.0,
            6,
        )
        payload: dict[str, Any] = {
            "avg_entropy": avg_entropy,
            "avg_margin": avg_margin,
            "positive_entropy_delta": positive_entropy_delta,
            "previous_checkpoint_avg_entropy": None
            if previous_checkpoint_entropy is None
            else round(float(previous_checkpoint_entropy), 6),
            "entropy_component": None,
            "delta_component": None,
            "margin_uncertainty_component": None,
            "signal_score": None,
            "normalized_signal": None,
            "raw_suppression_preview": None,
            "cumulative_generated_tokens": int(cumulative_generated_tokens),
            "tail_progress": None,
            "tail_activation_threshold": None,
            "tail_soft_token_index": round(self._tail_soft_token_index(), 3),
            "tail_full_token_index": round(self._tail_full_token_index(), 3),
        }
        if self.signal_calibration is None:
            return payload
        z_entropy = self._normalize_component(
            avg_entropy,
            self.signal_calibration.entropy_lo,
            self.signal_calibration.entropy_hi,
        )
        z_delta = self._normalize_component(
            positive_entropy_delta,
            0.0,
            self.signal_calibration.delta_entropy_hi,
        )
        normalized_margin = self._normalize_component(
            avg_margin,
            self.signal_calibration.margin_lo,
            self.signal_calibration.margin_hi,
        )
        z_margin = round(1.0 - normalized_margin, 6)
        signal_score = round(0.5 * z_entropy + 0.3 * z_delta + 0.2 * z_margin, 6)
        normalized_signal, raw_suppression_preview, tail_progress, tail_activation_threshold = self._mapped_suppression_from_signal_score(
            signal_score,
            cumulative_generated_tokens,
        )
        payload.update(
            {
                "entropy_component": z_entropy,
                "delta_component": z_delta,
                "margin_uncertainty_component": z_margin,
                "signal_score": signal_score,
                "normalized_signal": normalized_signal,
                "raw_suppression_preview": raw_suppression_preview,
                "tail_progress": tail_progress,
                "tail_activation_threshold": tail_activation_threshold,
            }
        )
        return payload

    def _apply_forced_entry_text(
        self,
        state: CascadeDecodeState,
        runtime_suppression: float,
        text: str,
    ) -> list[int]:
        token_ids = _tokenize_plain_text(self.resources.tokenizer, text)
        if token_ids:
            self._set_suppression_runtime(runtime_suppression)
            _append_forced_token_ids(
                self.resources.model,
                state,
                token_ids,
                forced_text=text,
            )
        return token_ids

    def _start_segment(
        self,
        state: CascadeDecodeState,
        suppression_value: float,
        cue_inserted: bool,
        cue_pending: bool = False,
        forced_entry_text: str | None = None,
        forced_entry_runtime_suppression: float | None = None,
        forced_entry_reason: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        suppression_value = self._clamp_suppression(suppression_value)
        entry_text = ""
        entry_reason: str | None = None
        entry_token_ids: list[int] = []
        if forced_entry_text:
            entry_text = forced_entry_text
            entry_reason = forced_entry_reason or "forced"
            entry_token_ids = self._apply_forced_entry_text(
                state=state,
                runtime_suppression=(
                    self._clamp_suppression(forced_entry_runtime_suppression)
                    if forced_entry_runtime_suppression is not None
                    else suppression_value
                ),
                text=entry_text,
            )
        elif cue_pending and not cue_inserted and self.cue_text:
            # cue 只负责一次性的语言强化提示；真正的 suppression 强度仍然由
            # 当前 checkpoint 算出的连续值决定。
            entry_text = self.cue_text
            entry_reason = "cue"
            entry_token_ids = self._apply_forced_entry_text(
                state=state,
                runtime_suppression=0.0,
                text=entry_text,
            )
            cue_inserted = True
        active_suppression = self._set_suppression_runtime(suppression_value)
        segment_generated_ids = list(entry_token_ids)
        return {
            "suppression_budget": active_suppression,
            "suppression_label": _runtime_suppression_label(active_suppression),
            "entry_text": entry_text,
            "entry_reason": entry_reason,
            "entry_text_applied_before_run": bool(entry_token_ids),
            "entry_token_count": len(entry_token_ids),
            "entry_token_positions": list(range(len(entry_token_ids))),
            "generated_ids": segment_generated_ids,
            "checkpoint_events": [],
            "transition_triggered": False,
            "transition_target_suppression": None,
            "transition_type": None,
            "finish_reason": None,
        }, cue_inserted

    def _finalize_segment(
        self,
        segment: dict[str, Any],
        finish_reason: str,
    ) -> dict[str, Any]:
        segment["finish_reason"] = finish_reason
        segment["final_text"] = _decode_token_ids(self.resources.tokenizer, segment["generated_ids"])
        segment["n_tokens"] = len(segment["generated_ids"])
        segment["checkpoint_count"] = len(segment["checkpoint_events"])
        return segment

    def _build_judge_record(self, final_row: dict[str, Any], answer: str) -> dict[str, Any]:
        if self.judge is None:
            return {
                "standard_answer": answer,
                "is_correct": None,
                "judge_failed": False,
                "judge_skipped": True,
                "review_consensus": "skipped",
                "final_decision_source": "skipped",
            }
        judge_record = self.judge.evaluate_episode(self.run_name, final_row)
        judge_record["judge_skipped"] = False
        return judge_record

    def rollout_episode(self, row: dict[str, Any], fallback_index: int, deterministic: bool = False) -> EpisodeRecord:
        del deterministic
        sample_id = _extract_sample_id(row, fallback_index)
        question = sampling_runtime.extract_question(row)
        answer = sampling_runtime.extract_answer(row)
        prompt_ids = sampling_runtime.build_prompt(
            self.resources.tokenizer,
            question,
            assistant_suffix="",
        ).to(self.resources.input_device)
        end_think_tokens = self.resources.tokenizer.encode("</think>", add_special_tokens=False)
        if not end_think_tokens:
            raise RuntimeError("Tokenizer 无法编码 </think>。")
        sampling_runtime.initialize_end_think_state(prompt_ids, end_think_tokens[0])
        if self.debug_runtime_events:
            prompt_token_ids = prompt_ids[0].tolist()
            prompt_text = _decode_token_ids(self.resources.tokenizer, prompt_token_ids)
            self._debug_event(
                sample_id=sample_id,
                event="prompt_init",
                prompt_len=int(prompt_ids.shape[1]),
                assistant_prefix_text=self.assistant_prefix_text,
                end_think_token_ids=end_think_tokens,
                detected_end_think_index=int(sampling_runtime.GLOBAL_END_THINK_INDEX),
                suppression_active_after_init=bool(sampling_runtime.GLOBAL_SUPPRESSION_ACTIVE),
                prompt_end_think_literal_count=int(prompt_text.count("</think>")),
                prompt_tail_ids=_tail_ids(prompt_token_ids, n=48),
                prompt_tail_text_visible=_viz_text(prompt_text[-400:]),
            )
        state = CascadeDecodeState(
            current_ids=prompt_ids,
            current_attention_mask=torch.ones_like(prompt_ids, device=self.resources.input_device),
            past_key_values=None,
            generated_ids=[],
        )
        runs: list[dict[str, Any]] = []
        current_base_suppression = self.initial_suppression
        current_effective_suppression = self.initial_suppression
        peak_effective_suppression = current_effective_suppression
        question_lambda_q = 1.0
        question_lambda_frozen = (
            (self.question_lambda_calibration is None and not self.dynamic_question_lambda)
            or self.calibration_only
        )
        early_base_suppressions: list[float] = []
        probe_signal_scores: list[float] = []
        high_low_high_signal_scores: list[float] = []
        high_low_high_raw_suppressions: list[float] = []
        cue_inserted = False
        tail_signal_inserted = False
        tail_signal_armed = False
        no_more_increases = False
        no_more_downgrades = False
        upgrade_freeze_remaining_newlines = 0
        end_think_block_remaining_newlines = 0
        end_think_block_remaining_tokens = 0
        end_think_mask_forever_active = False
        closure_mode_active = False
        closure_debug_emitted_tokens = 0
        closure_mode_start_generated_len = 0
        end_think_intercept_consumed = False
        consecutive_downgrade_hits = 0
        pending_downgrade_target_suppression: float | None = None
        recent_normalized_signal_history: list[float] = []
        previous_checkpoint_entropy: float | None = None
        increase_checkpoint_count = 0
        closure_token_budget = 10000
        tail_signal_armed_generated_len: int | None = None
        tail_signal_wait_token_limit = self.tail_signal_wait_token_limit
        final_answer_prefix_inserted = False
        rethink_cue_inserted = False
        segment, cue_inserted = self._start_segment(
            state=state,
            suppression_value=current_effective_suppression,
            cue_inserted=cue_inserted,
            forced_entry_text=self.assistant_prefix_text if self.assistant_prefix_text else None,
            forced_entry_runtime_suppression=0.0 if self.assistant_prefix_text else None,
            forced_entry_reason="assistant_prefix" if self.assistant_prefix_text else None,
        )
        up_token_ids: list[int] = []
        up_entropy: list[float] = []
        up_margin: list[float] = []
        up_logprob: list[float] = []
        up_newlines = 0
        down_token_ids: list[int] = []
        down_entropy: list[float] = []
        down_margin: list[float] = []
        down_logprob: list[float] = []
        down_newlines = 0
        running_ctx_ids = state.current_ids[0].tolist()
        finish_reason = "max_new_tokens"
        # Prefix checkpoint is configurable: older Alright-only runs use the
        # prefix as context only, while newer ablations score it immediately.
        assistant_prefix_checkpoint_pending = bool(self.assistant_prefix_text) and self.assistant_prefix_checkpoint
        checkpoint_zero_pending = self.checkpoint_zero or assistant_prefix_checkpoint_pending

        def _append_recent_normalized_signal(normalized_signal: float | None) -> None:
            if normalized_signal is None:
                return
            recent_normalized_signal_history.append(min(max(float(normalized_signal), 0.0), 1.0))
            if len(recent_normalized_signal_history) > self.raw_history_size:
                del recent_normalized_signal_history[0]

        def _maybe_arm_tail_signal() -> None:
            nonlocal tail_signal_armed, tail_signal_armed_generated_len
            if tail_signal_inserted or tail_signal_armed:
                return False
            if self._should_arm_tail_signal(len(state.generated_ids)):
                tail_signal_armed = True
                tail_signal_armed_generated_len = len(state.generated_ids)
                return True
            return False

        def _prepare_tail_signal_forced_entry() -> tuple[str | None, float | None, float, float, bool]:
            nonlocal current_base_suppression, current_effective_suppression
            nonlocal no_more_increases, no_more_downgrades
            nonlocal upgrade_freeze_remaining_newlines
            nonlocal end_think_block_remaining_newlines, end_think_block_remaining_tokens
            nonlocal end_think_mask_forever_active, closure_mode_active, closure_debug_emitted_tokens
            nonlocal end_think_intercept_consumed
            nonlocal consecutive_downgrade_hits, pending_downgrade_target_suppression
            nonlocal recent_normalized_signal_history

            next_base_suppression = current_base_suppression
            next_effective_suppression = current_effective_suppression
            forced_entry_text = self.tail_signal_text
            forced_entry_runtime_suppression = current_effective_suppression
            closure_mode_applied = False
            if self.end_think_to_l1_lock:
                next_base_suppression = 0.0
                next_effective_suppression = 0.0
                current_base_suppression = 0.0
                current_effective_suppression = 0.0
                no_more_increases = True
                no_more_downgrades = True
                end_think_intercept_consumed = True
                end_think_block_remaining_newlines = self.end_think_mask_newlines
                end_think_block_remaining_tokens = self.end_think_mask_tokens
                end_think_mask_forever_active = self.end_think_mask_forever
                closure_mode_active = True
                closure_debug_emitted_tokens = 0
                upgrade_freeze_remaining_newlines = 0
                consecutive_downgrade_hits = 0
                pending_downgrade_target_suppression = None
                recent_normalized_signal_history = []
                forced_entry_runtime_suppression = 0.0
                closure_mode_applied = True
                if self.end_think_bridge_text:
                    forced_entry_text = f"{self.tail_signal_text}{self.end_think_bridge_text}"
            return (
                forced_entry_text,
                forced_entry_runtime_suppression,
                next_base_suppression,
                next_effective_suppression,
                closure_mode_applied,
            )

        def _insert_tail_signal_immediately(trigger_source: str) -> None:
            nonlocal segment, cue_inserted, tail_signal_inserted, tail_signal_armed
            nonlocal tail_signal_armed_generated_len
            nonlocal up_token_ids, up_entropy, up_margin, up_logprob, up_newlines
            nonlocal down_token_ids, down_entropy, down_margin, down_logprob, down_newlines
            nonlocal running_ctx_ids
            nonlocal closure_mode_start_generated_len

            if self.debug_runtime_events:
                self._debug_event(
                    sample_id=sample_id,
                    event="tail_signal_immediate",
                    trigger_source=trigger_source,
                    current_suppression=round(float(current_effective_suppression), 6),
                    current_base_suppression=round(float(current_base_suppression), 6),
                    cumulative_generated_tokens=len(state.generated_ids),
                    tail_signal_trigger_token_index=round(float(self._tail_signal_trigger_token_index()), 6),
                    current_ids_len_before=int(state.current_ids.shape[1]),
                    past_seq_len_before=_infer_past_seq_len(state.past_key_values),
                    generated_tail_text_visible=_viz_text(
                        _decode_token_ids(self.resources.tokenizer, state.generated_ids[-120:])
                    ),
                    tail_signal_text=self.tail_signal_text,
                    tail_signal_text_visible=_viz_text(self.tail_signal_text or ""),
                )

            (
                forced_entry_text,
                forced_entry_runtime_suppression,
                next_base_suppression,
                next_effective_suppression,
                closure_mode_applied,
            ) = _prepare_tail_signal_forced_entry()

            segment["transition_triggered"] = True
            segment["transition_target_suppression"] = next_effective_suppression
            segment["transition_type"] = "tail_signal_insert"
            segment["checkpoint_events"].append(
                {
                    "checkpoint_type": "tail_signal_immediate",
                    "current_suppression": current_effective_suppression,
                    "current_base_suppression": current_base_suppression,
                    "tail_signal_pending": True,
                    "tail_signal_triggered": True,
                    "tail_signal_trigger_source": trigger_source,
                    "cumulative_generated_tokens": len(state.generated_ids),
                    "tail_signal_trigger_token_index": self._tail_signal_trigger_token_index(),
                    "transition_applied": True,
                    "transition_type": "tail_signal_insert",
                    "target_suppression": next_effective_suppression,
                    "closure_mode_applied": closure_mode_applied,
                    "forced_entry_text": forced_entry_text,
                    "bridge_text": self.end_think_bridge_text if closure_mode_applied else None,
                }
            )
            runs.append(self._finalize_segment(segment, finish_reason="transition_tail_signal_insert_immediate"))
            segment, cue_inserted = self._start_segment(
                state=state,
                suppression_value=next_effective_suppression,
                cue_inserted=cue_inserted,
                forced_entry_text=forced_entry_text,
                forced_entry_runtime_suppression=forced_entry_runtime_suppression,
                forced_entry_reason="tail_signal",
            )
            if self.debug_runtime_events:
                self._debug_event(
                    sample_id=sample_id,
                    event="tail_signal_immediate_applied",
                    closure_mode_applied=closure_mode_applied,
                    next_base_suppression=round(float(next_base_suppression), 6),
                    next_effective_suppression=round(float(next_effective_suppression), 6),
                    forced_entry_text=forced_entry_text,
                    forced_entry_text_visible=_viz_text(forced_entry_text or ""),
                    current_ids_len_after=int(state.current_ids.shape[1]),
                    past_seq_len_after=_infer_past_seq_len(state.past_key_values),
                    generated_ids_len_after=len(state.generated_ids),
                    generated_tail_text_visible=_viz_text(
                        _decode_token_ids(self.resources.tokenizer, state.generated_ids[-120:])
                    ),
                )
            tail_signal_inserted = True
            tail_signal_armed = False
            tail_signal_armed_generated_len = None
            if closure_mode_applied:
                closure_mode_start_generated_len = len(state.generated_ids)
            up_token_ids = []
            up_entropy = []
            up_margin = []
            up_logprob = []
            up_newlines = 0
            down_token_ids = []
            down_entropy = []
            down_margin = []
            down_logprob = []
            down_newlines = 0
            running_ctx_ids = state.current_ids[0].tolist()

        def _insert_forced_phrase_immediately(text: str, reason: str) -> None:
            nonlocal segment, cue_inserted, running_ctx_ids, final_answer_prefix_inserted
            nonlocal up_token_ids, up_entropy, up_margin, up_logprob, up_newlines
            nonlocal down_token_ids, down_entropy, down_margin, down_logprob, down_newlines
            segment["transition_triggered"] = True
            segment["transition_target_suppression"] = current_effective_suppression
            segment["transition_type"] = reason
            segment["checkpoint_events"].append(
                {
                    "checkpoint_type": reason,
                    "current_suppression": current_effective_suppression,
                    "current_base_suppression": current_base_suppression,
                    "transition_applied": True,
                    "transition_type": reason,
                    "target_suppression": current_effective_suppression,
                    "forced_entry_text": text,
                    "cumulative_generated_tokens": len(state.generated_ids),
                }
            )
            runs.append(self._finalize_segment(segment, finish_reason=f"transition_{reason}"))
            segment, cue_inserted = self._start_segment(
                state=state,
                suppression_value=current_effective_suppression,
                cue_inserted=cue_inserted,
                forced_entry_text=text,
                forced_entry_reason=reason,
            )
            final_answer_prefix_inserted = True
            up_token_ids = []
            up_entropy = []
            up_margin = []
            up_logprob = []
            up_newlines = 0
            down_token_ids = []
            down_entropy = []
            down_margin = []
            down_logprob = []
            down_newlines = 0
            running_ctx_ids = state.current_ids[0].tolist()

        def _should_insert_rethink_cue_at_checkpoint() -> bool:
            if rethink_cue_inserted:
                return False
            if not self.rethink_cue_text:
                return False
            return len(state.generated_ids) >= self.rethink_cue_trigger_tokens

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                if checkpoint_zero_pending:
                    checkpoint_zero_pending = False
                    if not no_more_increases:
                        checkpoint_token_ids = list(segment["generated_ids"]) if assistant_prefix_checkpoint_pending else []
                        checkpoint_text = _decode_token_ids(self.resources.tokenizer, checkpoint_token_ids)
                        metric_source = "teacher_forced_prefix_tokens" if checkpoint_token_ids else "next_token_probe"
                        probe_token_id: int | None = None
                        probe_piece: str | None = None
                        scored_token_pieces: list[str] = []
                        if checkpoint_token_ids:
                            entropy_values, margin_values, logprob_values, scored_token_pieces = (
                                _score_forced_suffix_token_metrics(
                                    self.resources.model,
                                    self.resources.tokenizer,
                                    state,
                                    checkpoint_token_ids,
                                )
                            )
                            entropy_value = sum(entropy_values) / len(entropy_values)
                            margin_value = sum(margin_values) / len(margin_values)
                        else:
                            probe_input_ids = state.current_ids if state.past_key_values is None else state.current_ids[:, -1:]
                            probe_output = self.resources.model(
                                input_ids=probe_input_ids,
                                attention_mask=state.current_attention_mask,
                                past_key_values=state.past_key_values,
                                use_cache=False,
                                return_dict=True,
                            )
                            probe_logits = probe_output.logits[:, -1, :]
                            probe_token = torch.argmax(probe_logits, dim=-1, keepdim=True)
                            probe_token_id = int(probe_token.item())
                            probe_piece = _decode_piece_in_context(
                                self.resources.tokenizer,
                                running_ctx_ids,
                                probe_token_id,
                            )
                            entropy_value, margin_value, logprob_value = sampling_runtime.compute_token_metric(
                                probe_logits[0],
                                probe_token_id,
                            )
                            entropy_values = [entropy_value]
                            margin_values = [margin_value]
                            logprob_values = [logprob_value]
                        signal_features = self._build_signal_features(
                            avg_entropy=entropy_value,
                            avg_margin=margin_value,
                            previous_checkpoint_entropy=previous_checkpoint_entropy,
                            cumulative_generated_tokens=len(state.generated_ids),
                        )
                        normalized_signal = signal_features["normalized_signal"]
                        raw_suppression_preview = signal_features["raw_suppression_preview"]
                        increase_checkpoint_count += 1
                        checkpoint_rho_t = None
                        smoothed_suppression_preview = None
                        if not self.calibration_only and raw_suppression_preview is not None:
                            checkpoint_rho_t, smoothed_suppression_preview = self._smooth_target_suppression(
                                current_suppression=current_base_suppression,
                                raw_target_suppression=raw_suppression_preview,
                                checkpoint_index=increase_checkpoint_count,
                            )
                        next_base_suppression = (
                            current_base_suppression
                            if self.calibration_only or smoothed_suppression_preview is None
                            else smoothed_suppression_preview
                        )
                        base_output_suppression = (
                            current_base_suppression
                            if self.calibration_only or next_base_suppression <= current_base_suppression
                            else next_base_suppression
                        )
                        (
                            next_effective_suppression,
                            question_lambda_q,
                            question_lambda_frozen,
                            lambda_payload,
                        ) = self._resolve_effective_suppression(
                            base_suppression=base_output_suppression,
                            increase_checkpoint_count=increase_checkpoint_count,
                            early_base_suppressions=early_base_suppressions,
                            probe_signal_scores=probe_signal_scores,
                            high_low_high_signal_scores=high_low_high_signal_scores,
                            high_low_high_raw_suppressions=high_low_high_raw_suppressions,
                            signal_score=signal_features["signal_score"],
                            raw_suppression=raw_suppression_preview,
                            cumulative_generated_tokens=len(state.generated_ids),
                            question_lambda_q=question_lambda_q,
                            question_lambda_frozen=question_lambda_frozen,
                        )
                        cue_pending = (
                            (not self.calibration_only)
                            and (not cue_inserted)
                            and self._should_trigger_cue(signal_features["signal_score"], len(state.generated_ids))
                        )
                        _append_recent_normalized_signal(normalized_signal)
                        checkpoint_event = {
                            "checkpoint_type": "increase_check_checkpoint0",
                            "increase_checkpoint_index": increase_checkpoint_count,
                            "current_suppression": current_effective_suppression,
                            "current_base_suppression": current_base_suppression,
                            "normalized_signal": normalized_signal,
                            "raw_suppression": raw_suppression_preview,
                            "k_smoothing_rho": checkpoint_rho_t,
                            "smoothed_suppression": smoothed_suppression_preview,
                            "base_target_suppression": base_output_suppression,
                            "target_suppression": next_effective_suppression,
                            "cue_pending": cue_pending,
                            "token_count": len(checkpoint_token_ids),
                            "newline_count": checkpoint_text.count("\n"),
                            "checkpoint_text": checkpoint_text,
                            "entropy": list(entropy_values),
                            "margin": list(margin_values),
                            "logprob": list(logprob_values),
                            "metric_source": metric_source,
                            "scored_token_ids": [int(token_id) for token_id in checkpoint_token_ids],
                            "scored_token_pieces": scored_token_pieces,
                            "probe_token_id": probe_token_id,
                            "probe_token_text": probe_piece,
                            "transition_applied": False,
                        }
                        checkpoint_event.update(signal_features)
                        checkpoint_event["question_lambda_q"] = round(float(question_lambda_q), 6)
                        checkpoint_event["question_lambda_frozen"] = bool(question_lambda_frozen)
                        if lambda_payload is not None:
                            checkpoint_event["question_lambda_payload"] = lambda_payload
                        previous_checkpoint_entropy = signal_features["avg_entropy"]
                        current_base_suppression = base_output_suppression
                        if next_effective_suppression != current_effective_suppression:
                            checkpoint_event["transition_applied"] = True
                            checkpoint_event["transition_type"] = (
                                "increase"
                                if next_effective_suppression > current_effective_suppression
                                else "lambda_adjust"
                            )
                            segment["checkpoint_events"].append(checkpoint_event)
                            segment["transition_triggered"] = True
                            segment["transition_target_suppression"] = next_effective_suppression
                            segment["transition_type"] = checkpoint_event["transition_type"]
                            runs.append(
                                self._finalize_segment(
                                    segment,
                                    finish_reason=f"transition_{checkpoint_event['transition_type']}_checkpoint0",
                                )
                            )
                            current_effective_suppression = next_effective_suppression
                            peak_effective_suppression = max(
                                peak_effective_suppression,
                                current_effective_suppression,
                            )
                            segment, cue_inserted = self._start_segment(
                                state=state,
                                suppression_value=current_effective_suppression,
                                cue_inserted=cue_inserted,
                                cue_pending=cue_pending,
                            )
                            consecutive_downgrade_hits = 0
                            pending_downgrade_target_suppression = None
                            up_token_ids = []
                            up_entropy = []
                            up_margin = []
                            up_logprob = []
                            up_newlines = 0
                            down_token_ids = []
                            down_entropy = []
                            down_margin = []
                            down_logprob = []
                            down_newlines = 0
                            running_ctx_ids = state.current_ids[0].tolist()
                            assistant_prefix_checkpoint_pending = False
                            continue
                        if assistant_prefix_checkpoint_pending and cue_pending:
                            checkpoint_event["transition_applied"] = True
                            checkpoint_event["transition_type"] = "cue_only"
                            segment["checkpoint_events"].append(checkpoint_event)
                            segment["transition_triggered"] = True
                            segment["transition_target_suppression"] = current_effective_suppression
                            segment["transition_type"] = "cue_only"
                            runs.append(
                                self._finalize_segment(
                                    segment,
                                    finish_reason="transition_cue_only_checkpoint0",
                                )
                            )
                            segment, cue_inserted = self._start_segment(
                                state=state,
                                suppression_value=current_effective_suppression,
                                cue_inserted=cue_inserted,
                                cue_pending=True,
                            )
                            consecutive_downgrade_hits = 0
                            pending_downgrade_target_suppression = None
                            up_token_ids = []
                            up_entropy = []
                            up_margin = []
                            up_logprob = []
                            up_newlines = 0
                            down_token_ids = []
                            down_entropy = []
                            down_margin = []
                            down_logprob = []
                            down_newlines = 0
                            running_ctx_ids = state.current_ids[0].tolist()
                            assistant_prefix_checkpoint_pending = False
                            continue
                        segment["checkpoint_events"].append(checkpoint_event)
                        assistant_prefix_checkpoint_pending = False

                previous_past_key_values = state.past_key_values
                previous_current_ids = state.current_ids
                model_input_ids = state.current_ids if state.past_key_values is None else state.current_ids[:, -1:]
                output = self.resources.model(
                    input_ids=model_input_ids,
                    attention_mask=state.current_attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                state.past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
                if (
                    end_think_mask_forever_active
                    or end_think_block_remaining_newlines > 0
                    or end_think_block_remaining_tokens > 0
                ):
                    _apply_end_think_mask(
                        next_token_logits=next_token_logits,
                        generated_ids=state.generated_ids,
                        end_think_tokens=end_think_tokens,
                    )
                if self.temperature <= 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token = sampling_runtime.sample_next_token(
                        next_token_logits,
                        temperature=self.temperature,
                        top_p=0.95,
                        generated_ids=state.generated_ids,
                        presence_penalty=self.presence_penalty,
                    )
                token_id = int(next_token.item())
                if (
                    self.end_think_to_l1_lock
                    and not end_think_intercept_consumed
                    and token_id == end_think_tokens[0]
                ):
                    pre_intercept_tail_text = _decode_token_ids(self.resources.tokenizer, state.generated_ids[-80:])
                    bridge_token_ids = (
                        _tokenize_plain_text(self.resources.tokenizer, self.end_think_bridge_text)
                        if self.end_think_bridge_text
                        else []
                    )
                    state.past_key_values = previous_past_key_values
                    segment["transition_triggered"] = True
                    segment["transition_target_suppression"] = 0.0
                    segment["transition_type"] = "end_think_to_zero"
                    segment["checkpoint_events"].append(
                        {
                            "checkpoint_type": "end_think_intercept",
                            "current_suppression": current_effective_suppression,
                            "current_base_suppression": current_base_suppression,
                            "target_suppression": 0.0,
                            "transition_applied": True,
                            "transition_type": "end_think_to_zero",
                            "mask_newlines": self.end_think_mask_newlines,
                            "mask_tokens": self.end_think_mask_tokens,
                            "mask_forever": self.end_think_mask_forever,
                            "bridge_text": self.end_think_bridge_text,
                        }
                    )
                    runs.append(self._finalize_segment(segment, finish_reason="transition_end_think_to_zero"))
                    current_base_suppression = 0.0
                    current_effective_suppression = 0.0
                    end_think_intercept_consumed = True
                    no_more_increases = True
                    no_more_downgrades = True
                    end_think_block_remaining_newlines = self.end_think_mask_newlines
                    end_think_block_remaining_tokens = self.end_think_mask_tokens
                    end_think_mask_forever_active = self.end_think_mask_forever
                    closure_mode_active = True
                    closure_debug_emitted_tokens = 0
                    closure_mode_start_generated_len = len(state.generated_ids)
                    upgrade_freeze_remaining_newlines = 0
                    consecutive_downgrade_hits = 0
                    pending_downgrade_target_suppression = None
                    segment, cue_inserted = self._start_segment(
                        state=state,
                        suppression_value=current_effective_suppression,
                        cue_inserted=cue_inserted,
                        forced_entry_text=self.end_think_bridge_text,
                        forced_entry_runtime_suppression=0.0,
                    )
                    if self.debug_runtime_events:
                        self._debug_event(
                            sample_id=sample_id,
                            event="end_think_intercept",
                            suppression_before=runs[-1]["suppression_budget"],
                            intercepted_token_id=token_id,
                            end_think_token_ids=end_think_tokens,
                            previous_current_tail_ids=_tail_ids(previous_current_ids[0].tolist(), n=32),
                            previous_generated_tail_ids=_tail_ids(runs[-1]["generated_ids"], n=32),
                            current_ids_len_before=int(previous_current_ids.shape[1]),
                            generated_ids_len_before=len(state.generated_ids) - segment["entry_token_count"],
                            previous_past_seq_len=_infer_past_seq_len(previous_past_key_values),
                            advanced_past_seq_len=_infer_past_seq_len(output.past_key_values),
                            previous_past_summary=_summarize_past_cache(previous_past_key_values),
                            advanced_past_summary=_summarize_past_cache(output.past_key_values),
                            phase_gap_before=None
                            if _infer_past_seq_len(previous_past_key_values) is None
                            else int(previous_current_ids.shape[1]) - int(_infer_past_seq_len(previous_past_key_values)),
                            generated_tokens_before_bridge=len(state.generated_ids) - segment["entry_token_count"],
                            bridge_text=self.end_think_bridge_text,
                            bridge_text_visible=_viz_text(self.end_think_bridge_text or ""),
                            bridge_token_ids=bridge_token_ids,
                            bridge_token_count=segment["entry_token_count"],
                            bridge_decoded_text=_decode_token_ids(self.resources.tokenizer, bridge_token_ids),
                            bridge_decoded_text_visible=_viz_text(_decode_token_ids(self.resources.tokenizer, bridge_token_ids)),
                            current_ids_len_after_bridge=int(state.current_ids.shape[1]),
                            generated_ids_len_after_bridge=len(state.generated_ids),
                            post_bridge_past_seq_len=_infer_past_seq_len(state.past_key_values),
                            post_bridge_past_summary=_summarize_past_cache(state.past_key_values),
                            phase_gap_after=None
                            if _infer_past_seq_len(state.past_key_values) is None
                            else int(state.current_ids.shape[1]) - int(_infer_past_seq_len(state.past_key_values)),
                            current_ids_tail_after_bridge=_tail_ids(state.current_ids[0].tolist(), n=32),
                            generated_ids_tail_after_bridge=_tail_ids(state.generated_ids, n=32),
                            pre_intercept_tail_text=pre_intercept_tail_text,
                            pre_intercept_tail_text_visible=_viz_text(pre_intercept_tail_text),
                            post_bridge_tail_text=_decode_token_ids(self.resources.tokenizer, state.generated_ids[-80:]),
                            post_bridge_tail_text_visible=_viz_text(
                                _decode_token_ids(self.resources.tokenizer, state.generated_ids[-80:])
                            ),
                            end_think_mask_forever=self.end_think_mask_forever,
                            end_think_mask_newlines=self.end_think_mask_newlines,
                            end_think_mask_tokens=self.end_think_mask_tokens,
                        )
                    up_token_ids = []
                    up_entropy = []
                    up_margin = []
                    up_logprob = []
                    up_newlines = 0
                    down_token_ids = []
                    down_entropy = []
                    down_margin = []
                    down_logprob = []
                    down_newlines = 0
                    recent_normalized_signal_history = []
                    running_ctx_ids = state.current_ids[0].tolist()
                    continue
                running_ctx_ids = state.current_ids[0].tolist()
                end_think_token_mask_active = end_think_block_remaining_tokens > 0
                entropy_value, margin_value, logprob_value = sampling_runtime.compute_token_metric(next_token_logits[0], token_id)
                state.generated_ids.append(token_id)
                segment["generated_ids"].append(token_id)
                up_token_ids.append(token_id)
                up_entropy.append(entropy_value)
                up_margin.append(margin_value)
                up_logprob.append(logprob_value)
                if current_effective_suppression > 0.0 or cue_inserted or recent_normalized_signal_history:
                    down_token_ids.append(token_id)
                    down_entropy.append(entropy_value)
                    down_margin.append(margin_value)
                    down_logprob.append(logprob_value)
                if self.debug_runtime_events and closure_mode_active and closure_debug_emitted_tokens < 40:
                    token_piece = _decode_piece_in_context(self.resources.tokenizer, running_ctx_ids, token_id)
                    self._debug_event(
                        sample_id=sample_id,
                        event="closure_token",
                        closure_token_index=closure_debug_emitted_tokens,
                        token_id=token_id,
                        token_piece=token_piece,
                        token_piece_visible=_viz_text(token_piece),
                        current_suppression=current_effective_suppression,
                        current_ids_len=int(state.current_ids.shape[1]),
                        generated_ids_len=len(state.generated_ids),
                        past_seq_len=_infer_past_seq_len(state.past_key_values),
                        past_summary=_summarize_past_cache(state.past_key_values),
                        running_ctx_tail_ids=_tail_ids(running_ctx_ids, n=24),
                        generated_tail_text_visible=_viz_text(
                            _decode_token_ids(self.resources.tokenizer, state.generated_ids[-40:] + [token_id])
                        ),
                    )
                    closure_debug_emitted_tokens += 1
                state.current_ids = torch.cat([state.current_ids, next_token.to(state.current_ids.device)], dim=1)
                state.current_attention_mask = torch.cat(
                    [
                        state.current_attention_mask,
                        torch.ones((1, 1), dtype=state.current_attention_mask.dtype, device=state.current_attention_mask.device),
                    ],
                    dim=1,
                )
                piece = _decode_piece_in_context(self.resources.tokenizer, running_ctx_ids, token_id)
                running_ctx_ids.append(token_id)
                _maybe_arm_tail_signal()
                # Count at most one newline hit per decoded token piece so multi-newline pieces
                # do not collapse multiple checkpoint windows into one generation step.
                newline_delta = 1 if "\n" in piece else 0
                up_newlines += newline_delta
                if down_token_ids:
                    down_newlines += newline_delta
                if upgrade_freeze_remaining_newlines > 0 and newline_delta > 0:
                    upgrade_freeze_remaining_newlines = max(upgrade_freeze_remaining_newlines - newline_delta, 0)
                if end_think_block_remaining_newlines > 0 and newline_delta > 0:
                    end_think_block_remaining_newlines = max(end_think_block_remaining_newlines - newline_delta, 0)
                if end_think_token_mask_active:
                    # mask_tokens 期间仍要正常推进解码状态，否则 cache 和文本会错位。
                    end_think_block_remaining_tokens = max(end_think_block_remaining_tokens - 1, 0)
                if (
                    tail_signal_armed
                    and (not tail_signal_inserted)
                    and (not self.tail_signal_next_checkpoint_only)
                    and newline_delta > 0
                ):
                    _insert_tail_signal_immediately(trigger_source="tail_signal_next_newline")
                    continue
                if (
                    tail_signal_armed
                    and (not tail_signal_inserted)
                    and tail_signal_armed_generated_len is not None
                    and (len(state.generated_ids) - tail_signal_armed_generated_len) >= tail_signal_wait_token_limit
                ):
                    _insert_tail_signal_immediately(
                        trigger_source=f"tail_signal_wait_timeout_{tail_signal_wait_token_limit}"
                    )
                    continue
                if (
                    (not final_answer_prefix_inserted)
                    and self.final_answer_prefix_text
                    and float(len(state.generated_ids))
                    >= (float(self.max_new_tokens) * self.final_answer_prefix_trigger_ratio)
                ):
                    _insert_forced_phrase_immediately(
                        text=self.final_answer_prefix_text,
                        reason="final_answer_prefix",
                    )
                    continue

                if (
                    closure_mode_active
                    and closure_token_budget is not None
                    and (len(state.generated_ids) - closure_mode_start_generated_len) >= closure_token_budget
                ):
                    finish_reason = "closure_token_budget"
                    break

                if (
                    not no_more_increases
                    and upgrade_freeze_remaining_newlines <= 0
                    and up_newlines >= self.upgrade_newline_span
                    and up_entropy
                ):
                    up_avg_entropy = sum(up_entropy) / len(up_entropy)
                    up_avg_margin = sum(up_margin) / len(up_margin)
                    signal_features = self._build_signal_features(
                        avg_entropy=up_avg_entropy,
                        avg_margin=up_avg_margin,
                        previous_checkpoint_entropy=previous_checkpoint_entropy,
                        cumulative_generated_tokens=len(state.generated_ids),
                    )
                    normalized_signal = signal_features["normalized_signal"]
                    raw_suppression_preview = signal_features["raw_suppression_preview"]
                    increase_checkpoint_count += 1
                    checkpoint_rho_t = None
                    smoothed_suppression_preview = None
                    if not self.calibration_only and raw_suppression_preview is not None:
                        checkpoint_rho_t, smoothed_suppression_preview = self._smooth_target_suppression(
                            current_suppression=current_base_suppression,
                            raw_target_suppression=raw_suppression_preview,
                            checkpoint_index=increase_checkpoint_count,
                        )
                    next_base_suppression = (
                        current_base_suppression
                        if self.calibration_only or smoothed_suppression_preview is None
                        else smoothed_suppression_preview
                    )
                    base_output_suppression = (
                        current_base_suppression
                        if self.calibration_only or next_base_suppression <= current_base_suppression
                        else next_base_suppression
                    )
                    (
                        next_effective_suppression,
                        question_lambda_q,
                        question_lambda_frozen,
                        lambda_payload,
                    ) = self._resolve_effective_suppression(
                        base_suppression=base_output_suppression,
                        increase_checkpoint_count=increase_checkpoint_count,
                        early_base_suppressions=early_base_suppressions,
                        probe_signal_scores=probe_signal_scores,
                        high_low_high_signal_scores=high_low_high_signal_scores,
                        high_low_high_raw_suppressions=high_low_high_raw_suppressions,
                        signal_score=signal_features["signal_score"],
                        raw_suppression=raw_suppression_preview,
                        cumulative_generated_tokens=len(state.generated_ids),
                        question_lambda_q=question_lambda_q,
                        question_lambda_frozen=question_lambda_frozen,
                    )
                    cue_pending = (
                        (not self.calibration_only)
                        and (not cue_inserted)
                        and self._should_trigger_cue(signal_features["signal_score"], len(state.generated_ids))
                    )
                    tail_signal_pending = (
                        tail_signal_armed
                        and (not tail_signal_inserted)
                        and bool(self.tail_signal_text)
                    )
                    rethink_cue_pending = (not tail_signal_pending) and _should_insert_rethink_cue_at_checkpoint()
                    _append_recent_normalized_signal(normalized_signal)
                    checkpoint_event = {
                        "checkpoint_type": "increase_check",
                        "increase_checkpoint_index": increase_checkpoint_count,
                        "current_suppression": current_effective_suppression,
                        "current_base_suppression": current_base_suppression,
                        "normalized_signal": normalized_signal,
                        "raw_suppression": raw_suppression_preview,
                        "k_smoothing_rho": checkpoint_rho_t,
                        "smoothed_suppression": smoothed_suppression_preview,
                        "base_target_suppression": base_output_suppression,
                        "target_suppression": next_effective_suppression,
                        "cue_pending": cue_pending,
                        "tail_signal_pending": tail_signal_pending,
                        "tail_signal_triggered": False,
                        "rethink_cue_pending": rethink_cue_pending,
                        "rethink_cue_triggered": False,
                        "token_count": len(up_token_ids),
                        "newline_count": up_newlines,
                        "checkpoint_text": _decode_token_ids(self.resources.tokenizer, up_token_ids),
                        "entropy": list(up_entropy),
                        "margin": list(up_margin),
                        "logprob": list(up_logprob),
                        "transition_applied": False,
                    }
                    checkpoint_event.update(signal_features)
                    checkpoint_event["question_lambda_q"] = round(float(question_lambda_q), 6)
                    checkpoint_event["question_lambda_frozen"] = bool(question_lambda_frozen)
                    if lambda_payload is not None:
                        checkpoint_event["question_lambda_payload"] = lambda_payload
                    previous_checkpoint_entropy = signal_features["avg_entropy"]
                    current_base_suppression = base_output_suppression
                    if next_effective_suppression != current_effective_suppression:
                        checkpoint_event["transition_applied"] = True
                        checkpoint_event["transition_type"] = (
                            "increase"
                            if next_effective_suppression > current_effective_suppression
                            else "lambda_adjust"
                        )
                        segment["checkpoint_events"].append(checkpoint_event)
                        segment["transition_triggered"] = True
                        segment["transition_target_suppression"] = next_effective_suppression
                        segment["transition_type"] = checkpoint_event["transition_type"]
                        runs.append(
                            self._finalize_segment(
                                segment,
                                finish_reason=f"transition_{checkpoint_event['transition_type']}",
                            )
                        )
                        current_effective_suppression = next_effective_suppression
                        peak_effective_suppression = max(
                            peak_effective_suppression,
                            current_effective_suppression,
                        )
                        if tail_signal_pending:
                            checkpoint_event["tail_signal_triggered"] = True
                            checkpoint_event["tail_signal_trigger_source"] = "tail_signal_checkpoint"
                            (
                                forced_entry_text,
                                forced_entry_runtime_suppression,
                                _tail_next_base,
                                _tail_next_effective,
                                closure_mode_applied,
                            ) = _prepare_tail_signal_forced_entry()
                            checkpoint_event["closure_mode_applied"] = closure_mode_applied
                            checkpoint_event["forced_entry_text"] = forced_entry_text
                        elif rethink_cue_pending:
                            checkpoint_event["rethink_cue_triggered"] = True
                            checkpoint_event["rethink_cue_trigger_source"] = "rethink_cue_checkpoint"
                            checkpoint_event["forced_entry_text"] = self.rethink_cue_text
                            forced_entry_text = self.rethink_cue_text
                            forced_entry_runtime_suppression = 0.0
                        else:
                            forced_entry_text = None
                            forced_entry_runtime_suppression = None
                        segment, cue_inserted = self._start_segment(
                            state=state,
                            suppression_value=current_effective_suppression,
                            cue_inserted=cue_inserted,
                            cue_pending=cue_pending,
                            forced_entry_text=forced_entry_text,
                            forced_entry_runtime_suppression=forced_entry_runtime_suppression,
                            forced_entry_reason=(
                                "tail_signal"
                                if tail_signal_pending
                                else ("rethink_cue" if rethink_cue_pending else None)
                            ),
                        )
                        if tail_signal_pending:
                            tail_signal_inserted = True
                            tail_signal_armed = False
                            tail_signal_armed_generated_len = None
                            if closure_mode_applied:
                                closure_mode_start_generated_len = len(state.generated_ids)
                        if rethink_cue_pending:
                            rethink_cue_inserted = True
                        consecutive_downgrade_hits = 0
                        pending_downgrade_target_suppression = None
                        up_token_ids = []
                        up_entropy = []
                        up_margin = []
                        up_logprob = []
                        up_newlines = 0
                        down_token_ids = []
                        down_entropy = []
                        down_margin = []
                        down_logprob = []
                        down_newlines = 0
                        running_ctx_ids = state.current_ids[0].tolist()
                        continue
                    segment["checkpoint_events"].append(checkpoint_event)
                    if tail_signal_pending or rethink_cue_pending:
                        checkpoint_event["tail_signal_triggered"] = True
                        if tail_signal_pending:
                            checkpoint_event["tail_signal_trigger_source"] = "tail_signal_checkpoint"
                            (
                                forced_entry_text,
                                forced_entry_runtime_suppression,
                                _tail_next_base,
                                _tail_next_effective,
                                closure_mode_applied,
                            ) = _prepare_tail_signal_forced_entry()
                            checkpoint_event["closure_mode_applied"] = closure_mode_applied
                            checkpoint_event["forced_entry_text"] = forced_entry_text
                            forced_entry_reason = "tail_signal"
                            transition_type_forced = "tail_signal_insert"
                        else:
                            checkpoint_event["tail_signal_triggered"] = False
                            checkpoint_event["rethink_cue_triggered"] = True
                            checkpoint_event["rethink_cue_trigger_source"] = "rethink_cue_checkpoint"
                            checkpoint_event["forced_entry_text"] = self.rethink_cue_text
                            forced_entry_text = self.rethink_cue_text
                            forced_entry_runtime_suppression = 0.0
                            closure_mode_applied = False
                            forced_entry_reason = "rethink_cue"
                            transition_type_forced = "rethink_cue_insert"
                        segment["transition_triggered"] = True
                        segment["transition_target_suppression"] = current_effective_suppression
                        segment["transition_type"] = transition_type_forced
                        runs.append(self._finalize_segment(segment, finish_reason=f"transition_{transition_type_forced}"))
                        segment, cue_inserted = self._start_segment(
                            state=state,
                            suppression_value=current_effective_suppression,
                            cue_inserted=cue_inserted,
                            forced_entry_text=forced_entry_text,
                            forced_entry_runtime_suppression=forced_entry_runtime_suppression,
                            forced_entry_reason=forced_entry_reason,
                        )
                        if tail_signal_pending:
                            tail_signal_inserted = True
                            tail_signal_armed = False
                            tail_signal_armed_generated_len = None
                        if tail_signal_pending and closure_mode_applied:
                            closure_mode_start_generated_len = len(state.generated_ids)
                        if rethink_cue_pending:
                            rethink_cue_inserted = True
                        up_token_ids = []
                        up_entropy = []
                        up_margin = []
                        up_logprob = []
                        up_newlines = 0
                        down_token_ids = []
                        down_entropy = []
                        down_margin = []
                        down_logprob = []
                        down_newlines = 0
                        running_ctx_ids = state.current_ids[0].tolist()
                        continue
                    up_token_ids = []
                    up_entropy = []
                    up_margin = []
                    up_logprob = []
                    up_newlines = 0

                if (
                    not no_more_downgrades
                    and down_newlines >= self.downgrade_newline_span
                    and down_entropy
                    and recent_normalized_signal_history
                ):
                    down_avg_entropy = sum(down_entropy) / len(down_entropy)
                    down_avg_margin = sum(down_margin) / len(down_margin)
                    down_signal_features = self._build_signal_features(
                        avg_entropy=down_avg_entropy,
                        avg_margin=down_avg_margin,
                        previous_checkpoint_entropy=previous_checkpoint_entropy,
                        cumulative_generated_tokens=len(state.generated_ids),
                    )
                    window_normalized_signal = down_signal_features["normalized_signal"]
                    history_avg_normalized_signal = round(
                        sum(recent_normalized_signal_history) / len(recent_normalized_signal_history),
                        6,
                    )
                    history_avg_raw_suppression = self._mapped_suppression_from_normalized_signal(
                        history_avg_normalized_signal
                    )
                    window_raw_suppression = (
                        None
                        if window_normalized_signal is None
                        else self._mapped_suppression_from_normalized_signal(window_normalized_signal)
                    )
                    transition_applied = False
                    transition_type: str | None = None
                    if not self.calibration_only and history_avg_raw_suppression < current_base_suppression:
                        if (
                            pending_downgrade_target_suppression is None
                            or history_avg_raw_suppression <= pending_downgrade_target_suppression
                        ):
                            consecutive_downgrade_hits += 1
                            pending_downgrade_target_suppression = history_avg_raw_suppression
                        else:
                            consecutive_downgrade_hits = 1
                            pending_downgrade_target_suppression = history_avg_raw_suppression
                        if consecutive_downgrade_hits >= self.downgrade_confirm_windows:
                            transition_applied = True
                            transition_type = "decrease"
                    else:
                        consecutive_downgrade_hits = 0
                        pending_downgrade_target_suppression = None
                    tail_signal_pending = (
                        tail_signal_armed
                        and (not tail_signal_inserted)
                        and bool(self.tail_signal_text)
                    )
                    rethink_cue_pending = (not tail_signal_pending) and _should_insert_rethink_cue_at_checkpoint()
                    checkpoint_event = {
                        "checkpoint_type": "decrease_check",
                        "current_suppression": current_effective_suppression,
                        "current_base_suppression": current_base_suppression,
                        "window_normalized_signal": window_normalized_signal,
                        "window_raw_suppression": window_raw_suppression,
                        "history_avg_normalized_signal": history_avg_normalized_signal,
                        "history_avg_raw_suppression": history_avg_raw_suppression,
                        "history_size": len(recent_normalized_signal_history),
                        "token_count": len(down_token_ids),
                        "newline_count": down_newlines,
                        "checkpoint_text": _decode_token_ids(self.resources.tokenizer, down_token_ids),
                        "entropy": list(down_entropy),
                        "margin": list(down_margin),
                        "logprob": list(down_logprob),
                        "transition_applied": transition_applied,
                        "downgrade_confirm_windows": self.downgrade_confirm_windows,
                        "downgrade_confirm_hits": consecutive_downgrade_hits,
                        "pending_downgrade_target_suppression": pending_downgrade_target_suppression,
                        "tail_signal_pending": tail_signal_pending,
                        "tail_signal_triggered": False,
                        "rethink_cue_pending": rethink_cue_pending,
                        "rethink_cue_triggered": False,
                    }
                    checkpoint_event.update(down_signal_features)
                    checkpoint_event["question_lambda_q"] = round(float(question_lambda_q), 6)
                    checkpoint_event["question_lambda_frozen"] = bool(question_lambda_frozen)
                    if transition_applied:
                        checkpoint_event["transition_type"] = transition_type
                        checkpoint_event["base_target_suppression"] = history_avg_raw_suppression
                        checkpoint_event["target_suppression"] = self._apply_question_lambda_to_suppression(
                            history_avg_raw_suppression,
                            question_lambda_q,
                        )
                        segment["checkpoint_events"].append(checkpoint_event)
                        segment["transition_triggered"] = True
                        segment["transition_target_suppression"] = checkpoint_event["target_suppression"]
                        segment["transition_type"] = transition_type
                        runs.append(self._finalize_segment(segment, finish_reason="transition_decrease"))
                        current_base_suppression = history_avg_raw_suppression
                        current_effective_suppression = checkpoint_event["target_suppression"]
                        upgrade_freeze_remaining_newlines = self.post_downgrade_upgrade_freeze_newlines
                        consecutive_downgrade_hits = 0
                        pending_downgrade_target_suppression = None
                        if tail_signal_pending:
                            checkpoint_event["tail_signal_triggered"] = True
                            checkpoint_event["tail_signal_trigger_source"] = "tail_signal_checkpoint"
                            (
                                forced_entry_text,
                                forced_entry_runtime_suppression,
                                _tail_next_base,
                                _tail_next_effective,
                                closure_mode_applied,
                            ) = _prepare_tail_signal_forced_entry()
                            checkpoint_event["closure_mode_applied"] = closure_mode_applied
                            checkpoint_event["forced_entry_text"] = forced_entry_text
                        elif rethink_cue_pending:
                            checkpoint_event["rethink_cue_triggered"] = True
                            checkpoint_event["rethink_cue_trigger_source"] = "rethink_cue_checkpoint"
                            checkpoint_event["forced_entry_text"] = self.rethink_cue_text
                            forced_entry_text = self.rethink_cue_text
                            forced_entry_runtime_suppression = 0.0
                        else:
                            forced_entry_text = None
                            forced_entry_runtime_suppression = None
                        segment, cue_inserted = self._start_segment(
                            state=state,
                            suppression_value=current_effective_suppression,
                            cue_inserted=cue_inserted,
                            forced_entry_text=forced_entry_text,
                            forced_entry_runtime_suppression=forced_entry_runtime_suppression,
                            forced_entry_reason=(
                                "tail_signal"
                                if tail_signal_pending
                                else ("rethink_cue" if rethink_cue_pending else None)
                            ),
                        )
                        if tail_signal_pending:
                            tail_signal_inserted = True
                            tail_signal_armed = False
                            tail_signal_armed_generated_len = None
                            if closure_mode_applied:
                                closure_mode_start_generated_len = len(state.generated_ids)
                        if rethink_cue_pending:
                            rethink_cue_inserted = True
                        up_token_ids = []
                        up_entropy = []
                        up_margin = []
                        up_logprob = []
                        up_newlines = 0
                        down_token_ids = []
                        down_entropy = []
                        down_margin = []
                        down_logprob = []
                        down_newlines = 0
                        running_ctx_ids = state.current_ids[0].tolist()
                        continue
                    segment["checkpoint_events"].append(checkpoint_event)
                    if tail_signal_pending or rethink_cue_pending:
                        if tail_signal_pending:
                            checkpoint_event["tail_signal_triggered"] = True
                            checkpoint_event["tail_signal_trigger_source"] = "tail_signal_checkpoint"
                            (
                                forced_entry_text,
                                forced_entry_runtime_suppression,
                                _tail_next_base,
                                _tail_next_effective,
                                closure_mode_applied,
                            ) = _prepare_tail_signal_forced_entry()
                            checkpoint_event["closure_mode_applied"] = closure_mode_applied
                            checkpoint_event["forced_entry_text"] = forced_entry_text
                            forced_entry_reason = "tail_signal"
                            transition_type_forced = "tail_signal_insert"
                        else:
                            checkpoint_event["rethink_cue_triggered"] = True
                            checkpoint_event["rethink_cue_trigger_source"] = "rethink_cue_checkpoint"
                            checkpoint_event["forced_entry_text"] = self.rethink_cue_text
                            forced_entry_text = self.rethink_cue_text
                            forced_entry_runtime_suppression = 0.0
                            closure_mode_applied = False
                            forced_entry_reason = "rethink_cue"
                            transition_type_forced = "rethink_cue_insert"
                        segment["transition_triggered"] = True
                        segment["transition_target_suppression"] = current_effective_suppression
                        segment["transition_type"] = transition_type_forced
                        runs.append(self._finalize_segment(segment, finish_reason=f"transition_{transition_type_forced}"))
                        segment, cue_inserted = self._start_segment(
                            state=state,
                            suppression_value=current_effective_suppression,
                            cue_inserted=cue_inserted,
                            forced_entry_text=forced_entry_text,
                            forced_entry_runtime_suppression=forced_entry_runtime_suppression,
                            forced_entry_reason=forced_entry_reason,
                        )
                        if tail_signal_pending:
                            tail_signal_inserted = True
                            tail_signal_armed = False
                            tail_signal_armed_generated_len = None
                        if tail_signal_pending and closure_mode_applied:
                            closure_mode_start_generated_len = len(state.generated_ids)
                        if rethink_cue_pending:
                            rethink_cue_inserted = True
                        up_token_ids = []
                        up_entropy = []
                        up_margin = []
                        up_logprob = []
                        up_newlines = 0
                        down_token_ids = []
                        down_entropy = []
                        down_margin = []
                        down_logprob = []
                        down_newlines = 0
                        running_ctx_ids = state.current_ids[0].tolist()
                        continue
                    down_token_ids = []
                    down_entropy = []
                    down_margin = []
                    down_logprob = []
                    down_newlines = 0

                if token_id == self.resources.tokenizer.eos_token_id:
                    finish_reason = "eos"
                    break

        runs.append(self._finalize_segment(segment, finish_reason=finish_reason))
        final_row = {
            "id": sample_id,
            "question": question,
            "answer": answer,
            "final_suppression_value": current_effective_suppression,
            "peak_suppression_value": peak_effective_suppression,
            "initial_suppression_value": self.initial_suppression,
            "assistant_prefix_checkpoint": self.assistant_prefix_checkpoint,
            "question_lambda_q": round(float(question_lambda_q), 6),
            "cue_inserted": cue_inserted,
            "tail_signal_inserted": tail_signal_inserted,
            "attempt_count": len(runs),
            "total_generated_tokens": len(state.generated_ids),
            "final_text": _decode_token_ids(self.resources.tokenizer, state.generated_ids),
            "runs": runs,
        }
        judge_record = self._build_judge_record(final_row, answer)
        return EpisodeRecord(
            sample_id=sample_id,
            question=question,
            answer=answer,
            final_suppression_value=current_effective_suppression,
            peak_suppression_value=peak_effective_suppression,
            question_lambda_q=round(float(question_lambda_q), 6),
            cue_inserted=cue_inserted,
            tail_signal_inserted=tail_signal_inserted,
            attempt_count=len(runs),
            total_generated_tokens=len(state.generated_ids),
            final_text=str(final_row["final_text"]),
            runs=runs,
            judge_record=judge_record,
        )


def summarize_episodes(episodes: list[EpisodeRecord]) -> dict[str, Any]:
    review_consensus_counts: Counter[str] = Counter()
    decision_source_counts: Counter[str] = Counter()
    transition_type_counts: Counter[str] = Counter()
    total_tokens = 0
    correct = 0
    judge_failed_count = 0
    judge_skipped_count = 0
    cue_inserted_count = 0
    tail_signal_inserted_count = 0
    total_transition_count = 0
    total_increase_count = 0
    total_decrease_count = 0
    total_final_suppression = 0.0
    total_peak_suppression = 0.0
    max_peak_suppression = 0.0
    total_question_lambda_q = 0.0
    max_question_lambda_q = 0.0
    for episode in episodes:
        total_tokens += episode.total_generated_tokens
        total_final_suppression += episode.final_suppression_value
        total_peak_suppression += episode.peak_suppression_value
        max_peak_suppression = max(max_peak_suppression, episode.peak_suppression_value)
        total_question_lambda_q += episode.question_lambda_q
        max_question_lambda_q = max(max_question_lambda_q, episode.question_lambda_q)
        if episode.cue_inserted:
            cue_inserted_count += 1
        if episode.tail_signal_inserted:
            tail_signal_inserted_count += 1
        review_consensus = str(episode.judge_record.get("review_consensus") or "unknown")
        final_decision_source = str(episode.judge_record.get("final_decision_source") or "unknown")
        review_consensus_counts[review_consensus] += 1
        decision_source_counts[final_decision_source] += 1
        if episode.judge_record.get("judge_failed") is True:
            judge_failed_count += 1
        if episode.judge_record.get("judge_skipped") is True:
            judge_skipped_count += 1
        if episode.judge_record.get("is_correct") is True:
            correct += 1
        for run in episode.runs:
            transition_type = run.get("transition_type")
            if transition_type:
                transition_type_counts[str(transition_type)] += 1
                total_transition_count += 1
                if transition_type == "increase":
                    total_increase_count += 1
                elif transition_type == "decrease":
                    total_decrease_count += 1
    total = len(episodes)
    return {
        "episode_count": total,
        "accuracy": (correct / total) if total else 0.0,
        "avg_total_generated_tokens": (total_tokens / total) if total else 0.0,
        "avg_transition_count": (total_transition_count / total) if total else 0.0,
        "avg_increase_count": (total_increase_count / total) if total else 0.0,
        "avg_decrease_count": (total_decrease_count / total) if total else 0.0,
        "avg_final_suppression": (total_final_suppression / total) if total else 0.0,
        "avg_peak_suppression": (total_peak_suppression / total) if total else 0.0,
        "max_peak_suppression": max_peak_suppression,
        "avg_question_lambda_q": (total_question_lambda_q / total) if total else 1.0,
        "max_question_lambda_q": max_question_lambda_q if total else 1.0,
        "cue_inserted_count": cue_inserted_count,
        "tail_signal_inserted_count": tail_signal_inserted_count,
        "judge_failed_count": judge_failed_count,
        "judge_skipped_count": judge_skipped_count,
        "review_consensus_counts": {key: count for key, count in sorted(review_consensus_counts.items())},
        "decision_source_counts": {key: count for key, count in sorted(decision_source_counts.items())},
        "transition_type_counts": {key: count for key, count in sorted(transition_type_counts.items())},
    }
