from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(min(max(float(value), float(lo)), float(hi)), 6)


@dataclass(frozen=True)
class SignalCalibration:
    entropy_lo: float
    entropy_hi: float
    delta_entropy_hi: float
    margin_lo: float
    margin_hi: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> SignalCalibration | None:
        if not value:
            return None
        for key in ("signal_calibration", "recommended_signal_calibration"):
            if key in value:
                value = value[key]
                break
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class QuestionLambdaCalibration:
    d_lo: float
    d_hi: float
    lambda_min: float
    lambda_max: float
    lambda_exp_beta: float
    d_mean_weight: float
    d_min_weight: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> QuestionLambdaCalibration | None:
        if not value:
            return None
        for key in ("question_lambda_calibration", "recommended_question_lambda_calibration"):
            if key in value:
                value = value[key]
                break
        return cls(**{name: float(value[name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class TokenMetric:
    entropy: float
    margin: float
    logprob: float


@dataclass
class ControllerConfig:
    max_new_tokens: int = 16384
    initial_suppression: float = 0.0
    suppression_upper_bound: float = 1.0
    cue_threshold: float = 0.1
    cue_gate_threshold: float = 0.1
    min_tokens_before_cue: int = 0
    enable_runtime_cue: bool = False
    entropy_upper_bound: float = 1.0
    suppression_curve: str = "power"
    power_exponent: float = 0.65
    exp_beta: float = 2.0
    suppression_scale: float = 1.0
    raw_history_size: int = 10
    k_smoothing_initial_rho: float = 1.0
    k_smoothing_late_rho: float = 0.75
    k_smoothing_initial_checkpoints: int = 2
    upgrade_newline_span: int = 1
    downgrade_newline_span: int = 10
    downgrade_confirm_windows: int = 3
    post_downgrade_upgrade_freeze_newlines: int = 0
    tail_soft_ratio: float = 0.75
    tail_full_ratio: float = 0.90
    tail_exp_beta: float = 3.0
    tail_threshold_max: float = 0.6
    tail_signal_trigger_ratio: float = 0.90
    tail_signal_wait_token_limit: int = 300
    tail_signal_next_checkpoint_only: bool = False
    final_answer_prefix_trigger_ratio: float = 0.98
    rethink_cue_trigger_tokens: int = 0
    end_think_to_l1_lock: bool = False
    end_think_mask_newlines: int = 1
    end_think_mask_tokens: int = 0
    end_think_mask_forever: bool = False
    checkpoint_zero: bool = True
    assistant_prefix_checkpoint: bool = False
    calibration_only: bool = False
    dynamic_question_lambda: bool = False
    dynamic_lambda_probe_checkpoints: int = 4
    dynamic_lambda_signal_lo: float = 0.18
    dynamic_lambda_signal_hi: float = 0.55
    dynamic_lambda_probe_scale_min: float = 0.30
    dynamic_lambda_probe_scale_max: float = 0.45
    dynamic_lambda_late_token_start: int = 5500
    dynamic_lambda_late_token_end: int = 7000
    dynamic_lambda_late_scale_start: float = 0.45
    dynamic_lambda_late_scale_max: float = 0.80
    high_low_high_question_lambda: bool = False
    high_low_high_early_scale: float = 0.80
    high_low_high_early_min_tokens: int = 800
    high_low_high_early_min_checkpoints: int = 4
    high_low_high_middle_scale: float = 0.40
    high_low_high_late_token_start: int = 5500
    high_low_high_late_token_end: int = 7000
    high_low_high_late_scale_max: float = 0.80
    high_low_high_adaptive_early: bool = False
    high_low_high_adaptive_max_tokens: int = 2000
    high_low_high_adaptive_window: int = 3
    high_low_high_adaptive_avg_signal_threshold: float = 0.65
    high_low_high_adaptive_max_signal_threshold: float = 0.90
    high_low_high_adaptive_max_raw_threshold: float = 0.36
    high_low_high_adaptive_raw_delta_threshold: float = 0.30
    linear_down_up_question_lambda: bool = False
    linear_down_up_high_scale: float = 0.80
    linear_down_up_low_scale: float = 0.40
    linear_down_up_hold_high_tokens: int = 800
    linear_down_up_down_end_tokens: int = 1200
    linear_down_up_up_start_tokens: int = 1200
    linear_down_up_up_end_tokens: int = 7000
    end_think_token_ids: list[int] = field(default_factory=list)
    newline_token_ids: list[int] = field(default_factory=list)
    cue_token_ids: list[int] = field(default_factory=list)
    rethink_cue_token_ids: list[int] = field(default_factory=list)
    tail_signal_token_ids: list[int] = field(default_factory=list)
    end_think_bridge_token_ids: list[int] = field(default_factory=list)
    final_answer_prefix_token_ids: list[int] = field(default_factory=list)
    signal_calibration: SignalCalibration | None = None
    question_lambda_calibration: QuestionLambdaCalibration | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ControllerConfig:
        known = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in known}
        kwargs["signal_calibration"] = SignalCalibration.from_mapping(payload.get("signal_calibration"))
        kwargs["question_lambda_calibration"] = QuestionLambdaCalibration.from_mapping(
            payload.get("question_lambda_calibration")
        )
        for key in (
            "end_think_token_ids",
            "newline_token_ids",
            "cue_token_ids",
            "rethink_cue_token_ids",
            "tail_signal_token_ids",
            "end_think_bridge_token_ids",
            "final_answer_prefix_token_ids",
        ):
            kwargs[key] = [int(item) for item in kwargs.get(key, [])]
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> ControllerConfig:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.suppression_curve not in {"power", "exp_saturating"}:
            raise ValueError(f"unsupported suppression_curve={self.suppression_curve!r}")
        if not 0.0 <= self.tail_soft_ratio < self.tail_full_ratio <= 1.0:
            raise ValueError("tail ratios must satisfy 0 <= soft < full <= 1")
        if not self.cue_threshold < self.entropy_upper_bound:
            raise ValueError("cue_threshold must be below entropy_upper_bound")
        if not self.cue_threshold <= self.tail_threshold_max <= self.entropy_upper_bound:
            raise ValueError("tail_threshold_max is outside the valid range")
        if sum(
            int(item)
            for item in (
                self.dynamic_question_lambda,
                self.high_low_high_question_lambda,
                self.linear_down_up_question_lambda,
            )
        ) > 1:
            raise ValueError("question-lambda modes are mutually exclusive")
        if self.signal_calibration is None and not self.calibration_only:
            raise ValueError("signal_calibration is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForcedToken:
    token_id: int
    reason: str


@dataclass
class RequestState:
    request_id: str
    current_base_suppression: float
    current_effective_suppression: float
    peak_effective_suppression: float
    question_lambda_q: float = 1.0
    question_lambda_frozen: bool = False
    generated_ids: list[int] = field(default_factory=list)
    up_token_ids: list[int] = field(default_factory=list)
    up_entropy: list[float] = field(default_factory=list)
    up_margin: list[float] = field(default_factory=list)
    up_logprob: list[float] = field(default_factory=list)
    up_newlines: int = 0
    down_token_ids: list[int] = field(default_factory=list)
    down_entropy: list[float] = field(default_factory=list)
    down_margin: list[float] = field(default_factory=list)
    down_logprob: list[float] = field(default_factory=list)
    down_newlines: int = 0
    recent_normalized_signal_history: list[float] = field(default_factory=list)
    previous_checkpoint_entropy: float | None = None
    increase_checkpoint_count: int = 0
    early_base_suppressions: list[float] = field(default_factory=list)
    probe_signal_scores: list[float] = field(default_factory=list)
    high_low_high_signal_scores: list[float] = field(default_factory=list)
    high_low_high_raw_suppressions: list[float] = field(default_factory=list)
    cue_inserted: bool = False
    rethink_cue_inserted: bool = False
    tail_signal_armed: bool = False
    tail_signal_inserted: bool = False
    tail_signal_armed_generated_len: int | None = None
    final_answer_prefix_inserted: bool = False
    no_more_increases: bool = False
    no_more_downgrades: bool = False
    upgrade_freeze_remaining_newlines: int = 0
    end_think_block_remaining_newlines: int = 0
    end_think_block_remaining_tokens: int = 0
    end_think_mask_forever_active: bool = False
    end_think_intercept_consumed: bool = False
    closure_mode_active: bool = False
    closure_mode_start_generated_len: int = 0
    consecutive_downgrade_hits: int = 0
    pending_downgrade_target_suppression: float | None = None
    checkpoint_zero_pending: bool = False
    forced_queue: list[ForcedToken] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events


class DynamicSuppressionController:
    """Pure per-request controller shared by text and Qwen3-VL runners."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.newline_ids = set(config.newline_token_ids)

    def new_state(self, request_id: str) -> RequestState:
        value = self._clamp(self.config.initial_suppression)
        state = RequestState(
            request_id=str(request_id),
            current_base_suppression=value,
            current_effective_suppression=value,
            peak_effective_suppression=value,
            question_lambda_frozen=self.config.question_lambda_calibration is None,
            checkpoint_zero_pending=bool(self.config.checkpoint_zero),
        )
        state.events.append(
            {
                "event": "request_init",
                "request_id": state.request_id,
                "current_k": value,
                "checkpoint_zero_pending": state.checkpoint_zero_pending,
            }
        )
        return state

    def _clamp(self, value: float) -> float:
        return _clamp(value, 0.0, self.config.suppression_upper_bound)

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(float(value), 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(value) for value in values)
        if len(ordered) == 1:
            return ordered[0]
        pos = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    def _tail_progress(self, token_count: int) -> float:
        soft = self.config.max_new_tokens * self.config.tail_soft_ratio
        full = self.config.max_new_tokens * self.config.tail_full_ratio
        if token_count <= soft:
            return 0.0
        if token_count >= full:
            return 1.0
        return _clamp((token_count - soft) / (full - soft))

    def _activation_threshold(self, token_count: int) -> float:
        progress = self._tail_progress(token_count)
        if progress <= 0.0:
            return round(self.config.cue_threshold, 6)
        denominator = 1.0 - math.exp(-self.config.tail_exp_beta)
        smooth = (1.0 - math.exp(-self.config.tail_exp_beta * progress)) / denominator
        return round(
            self.config.cue_threshold
            + (self.config.tail_threshold_max - self.config.cue_threshold) * smooth,
            6,
        )

    def _mapped_suppression(self, normalized_signal: float) -> float:
        u = min(max(float(normalized_signal), 0.0), 1.0)
        if self.config.suppression_curve == "power":
            mapped = u**self.config.power_exponent
        else:
            denominator = 1.0 - math.exp(-self.config.exp_beta)
            mapped = (1.0 - math.exp(-self.config.exp_beta * u)) / denominator
        return self._clamp(self.config.suppression_upper_bound * self.config.suppression_scale * mapped)

    @staticmethod
    def _normalize(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if value > upper else 0.0
        return _clamp((value - lower) / (upper - lower))

    def _signal_features(
        self,
        avg_entropy: float,
        avg_margin: float,
        previous_entropy: float | None,
        token_count: int,
    ) -> dict[str, Any]:
        avg_entropy = round(float(avg_entropy), 6)
        avg_margin = round(float(avg_margin), 6)
        delta = round(max(avg_entropy - float(previous_entropy), 0.0), 6) if previous_entropy is not None else 0.0
        payload: dict[str, Any] = {
            "avg_entropy": avg_entropy,
            "avg_margin": avg_margin,
            "positive_entropy_delta": delta,
            "previous_checkpoint_avg_entropy": previous_entropy,
            "cumulative_generated_tokens": int(token_count),
            "entropy_component": None,
            "delta_component": None,
            "margin_uncertainty_component": None,
            "signal_score": None,
            "normalized_signal": None,
            "raw_suppression_preview": None,
            "tail_progress": self._tail_progress(token_count),
            "tail_activation_threshold": self._activation_threshold(token_count),
        }
        calibration = self.config.signal_calibration
        if calibration is None:
            return payload
        z_entropy = self._normalize(avg_entropy, calibration.entropy_lo, calibration.entropy_hi)
        z_delta = self._normalize(delta, 0.0, calibration.delta_entropy_hi)
        z_margin = round(1.0 - self._normalize(avg_margin, calibration.margin_lo, calibration.margin_hi), 6)
        signal = round(0.5 * z_entropy + 0.3 * z_delta + 0.2 * z_margin, 6)
        threshold = self._activation_threshold(token_count)
        if signal < threshold:
            normalized = 0.0
        else:
            denominator = self.config.entropy_upper_bound - threshold
            normalized = 1.0 if denominator <= 0.0 else _clamp((signal - threshold) / denominator)
        payload.update(
            entropy_component=z_entropy,
            delta_component=z_delta,
            margin_uncertainty_component=z_margin,
            signal_score=signal,
            normalized_signal=normalized,
            raw_suppression_preview=self._mapped_suppression(normalized),
        )
        return payload

    def _smooth_target(self, current: float, raw: float, checkpoint_index: int) -> tuple[float, float]:
        rho = (
            self.config.k_smoothing_initial_rho
            if checkpoint_index <= self.config.k_smoothing_initial_checkpoints
            else self.config.k_smoothing_late_rho
        )
        return round(rho, 6), self._clamp((1.0 - rho) * current + rho * raw)

    def _legacy_lambda(self, state: RequestState, base: float) -> tuple[float, dict[str, Any] | None]:
        calibration = self.config.question_lambda_calibration
        if calibration is None or self.config.calibration_only:
            return state.question_lambda_q, None
        if state.increase_checkpoint_count <= 2:
            state.early_base_suppressions.append(base)
        if state.question_lambda_frozen or len(state.early_base_suppressions) < 2:
            return state.question_lambda_q, None
        k1, k2 = state.early_base_suppressions[:2]
        d_q = calibration.d_mean_weight * ((k1 + k2) / 2.0) + calibration.d_min_weight * min(k1, k2)
        u_q = self._normalize(d_q, calibration.d_lo, calibration.d_hi)
        denominator = 1.0 - math.exp(-calibration.lambda_exp_beta)
        mapped = (1.0 - math.exp(-calibration.lambda_exp_beta * u_q)) / denominator
        value = calibration.lambda_min + (calibration.lambda_max - calibration.lambda_min) * mapped
        state.question_lambda_q = round(value, 6)
        state.question_lambda_frozen = True
        return state.question_lambda_q, {
            "question_lambda_mode": "calibrated_k1_k2",
            "k1_base": k1,
            "k2_base": k2,
            "d_q": round(d_q, 6),
            "normalized_d_q": u_q,
            "question_lambda_q": state.question_lambda_q,
        }

    def _dynamic_lambda(self, state: RequestState, signal: float | None, token_count: int) -> tuple[float, dict[str, Any]]:
        if signal is not None and len(state.probe_signal_scores) < self.config.dynamic_lambda_probe_checkpoints:
            state.probe_signal_scores.append(_clamp(signal))
        q75 = self._quantile(state.probe_signal_scores, 0.75)
        probe_progress = self._smoothstep(
            (q75 - self.config.dynamic_lambda_signal_lo)
            / (self.config.dynamic_lambda_signal_hi - self.config.dynamic_lambda_signal_lo)
        )
        probe_scale = self.config.dynamic_lambda_probe_scale_min + (
            self.config.dynamic_lambda_probe_scale_max - self.config.dynamic_lambda_probe_scale_min
        ) * probe_progress
        late_progress = self._smoothstep(
            (token_count - self.config.dynamic_lambda_late_token_start)
            / (self.config.dynamic_lambda_late_token_end - self.config.dynamic_lambda_late_token_start)
        )
        late_scale = self.config.dynamic_lambda_late_scale_start + (
            self.config.dynamic_lambda_late_scale_max - self.config.dynamic_lambda_late_scale_start
        ) * late_progress
        scale = max(probe_scale, late_scale) if token_count > self.config.dynamic_lambda_late_token_start else probe_scale
        scale = min(max(scale, self.config.dynamic_lambda_probe_scale_min), self.config.dynamic_lambda_late_scale_max)
        value = scale / self.config.suppression_scale
        return value, {
            "question_lambda_mode": "dynamic_probe_late",
            "probe_signal_scores": list(state.probe_signal_scores),
            "probe_u_q75": round(q75, 6),
            "effective_scale": round(scale, 6),
            "question_lambda_q": round(value, 6),
        }

    def _high_low_high_lambda(
        self, state: RequestState, signal: float | None, raw: float | None, token_count: int
    ) -> tuple[float, dict[str, Any]]:
        if signal is not None:
            state.high_low_high_signal_scores.append(_clamp(signal))
        if raw is not None:
            state.high_low_high_raw_suppressions.append(_clamp(raw))
        window = self.config.high_low_high_adaptive_window
        recent_s = state.high_low_high_signal_scores[-window:]
        recent_r = state.high_low_high_raw_suppressions[-window:]
        avg_s = sum(recent_s) / len(recent_s) if recent_s else 0.0
        max_s = max(recent_s) if recent_s else 0.0
        max_r = max(recent_r) if recent_r else 0.0
        delta_r = recent_r[-1] - recent_r[0] if len(recent_r) >= 2 else 0.0
        adaptive = self.config.high_low_high_adaptive_early and token_count < self.config.high_low_high_adaptive_max_tokens and (
            avg_s >= self.config.high_low_high_adaptive_avg_signal_threshold
            or max_s >= self.config.high_low_high_adaptive_max_signal_threshold
            or max_r >= self.config.high_low_high_adaptive_max_raw_threshold
            or delta_r >= self.config.high_low_high_adaptive_raw_delta_threshold
        )
        if (
            state.increase_checkpoint_count <= self.config.high_low_high_early_min_checkpoints
            or token_count < self.config.high_low_high_early_min_tokens
            or adaptive
        ):
            phase, scale = "early_high_adaptive" if adaptive else "early_high", self.config.high_low_high_early_scale
        elif token_count <= self.config.high_low_high_late_token_start:
            phase, scale = "middle_low", self.config.high_low_high_middle_scale
        else:
            phase = "late_ramp"
            progress = self._smoothstep(
                (token_count - self.config.high_low_high_late_token_start)
                / (self.config.high_low_high_late_token_end - self.config.high_low_high_late_token_start)
            )
            scale = self.config.high_low_high_middle_scale + (
                self.config.high_low_high_late_scale_max - self.config.high_low_high_middle_scale
            ) * progress
        value = scale / self.config.suppression_scale
        return value, {
            "question_lambda_mode": "high_low_high",
            "phase": phase,
            "effective_scale": round(scale, 6),
            "question_lambda_q": round(value, 6),
            "adaptive_keep_high": adaptive,
        }

    def _linear_lambda(self, token_count: int) -> tuple[float, dict[str, Any]]:
        high, low = self.config.linear_down_up_high_scale, self.config.linear_down_up_low_scale
        if token_count <= self.config.linear_down_up_hold_high_tokens:
            phase, scale = "hold_high", high
        elif token_count <= self.config.linear_down_up_down_end_tokens:
            phase = "linear_down"
            p = (token_count - self.config.linear_down_up_hold_high_tokens) / (
                self.config.linear_down_up_down_end_tokens - self.config.linear_down_up_hold_high_tokens
            )
            scale = high + (low - high) * p
        elif token_count <= self.config.linear_down_up_up_start_tokens:
            phase, scale = "hold_low", low
        elif token_count <= self.config.linear_down_up_up_end_tokens:
            phase = "linear_up"
            p = (token_count - self.config.linear_down_up_up_start_tokens) / (
                self.config.linear_down_up_up_end_tokens - self.config.linear_down_up_up_start_tokens
            )
            scale = low + (high - low) * p
        else:
            phase, scale = "hold_high_late", high
        value = scale / self.config.suppression_scale
        return value, {
            "question_lambda_mode": "linear_down_up",
            "phase": phase,
            "effective_scale": round(scale, 6),
            "question_lambda_q": round(value, 6),
        }

    def _resolve_effective(
        self, state: RequestState, base: float, signal: float | None, raw: float | None
    ) -> tuple[float, dict[str, Any] | None]:
        token_count = len(state.generated_ids)
        payload = None
        if self.config.dynamic_question_lambda and not self.config.calibration_only:
            state.question_lambda_q, payload = self._dynamic_lambda(state, signal, token_count)
        elif self.config.high_low_high_question_lambda and not self.config.calibration_only:
            state.question_lambda_q, payload = self._high_low_high_lambda(state, signal, raw, token_count)
        elif self.config.linear_down_up_question_lambda and not self.config.calibration_only:
            state.question_lambda_q, payload = self._linear_lambda(token_count)
        else:
            state.question_lambda_q, payload = self._legacy_lambda(state, base)
        return self._clamp(base * state.question_lambda_q), payload

    def token_ids_to_mask(self, state: RequestState) -> list[int]:
        active = (
            state.end_think_mask_forever_active
            or state.end_think_block_remaining_newlines > 0
            or state.end_think_block_remaining_tokens > 0
        )
        if not active or not self.config.end_think_token_ids:
            return []
        forbidden = {self.config.end_think_token_ids[0]}
        pattern = self.config.end_think_token_ids
        max_width = min(len(state.generated_ids), len(pattern) - 1)
        for width in range(max_width, 0, -1):
            if state.generated_ids[-width:] == pattern[:width]:
                forbidden.add(pattern[width])
                break
        return sorted(forbidden)

    def apply_checkpoint_zero(self, state: RequestState, metric: TokenMetric, probe_token_id: int) -> None:
        if not state.checkpoint_zero_pending:
            return
        state.checkpoint_zero_pending = False
        event = self._increase_checkpoint(
            state,
            token_ids=[],
            metrics=[metric],
            newline_count=0,
            checkpoint_type="increase_check_checkpoint0",
        )
        event["metric_source"] = "next_token_probe"
        event["probe_token_id"] = int(probe_token_id)

    def _append_history(self, state: RequestState, normalized: float | None) -> None:
        if normalized is None:
            return
        state.recent_normalized_signal_history.append(_clamp(normalized))
        del state.recent_normalized_signal_history[: -self.config.raw_history_size]

    def _increase_checkpoint(
        self,
        state: RequestState,
        token_ids: list[int],
        metrics: list[TokenMetric],
        newline_count: int,
        checkpoint_type: str = "increase_check",
    ) -> dict[str, Any]:
        avg_entropy = sum(item.entropy for item in metrics) / len(metrics)
        avg_margin = sum(item.margin for item in metrics) / len(metrics)
        features = self._signal_features(
            avg_entropy,
            avg_margin,
            state.previous_checkpoint_entropy,
            len(state.generated_ids),
        )
        state.increase_checkpoint_count += 1
        raw = features["raw_suppression_preview"]
        rho = smoothed = None
        if not self.config.calibration_only and raw is not None:
            rho, smoothed = self._smooth_target(
                state.current_base_suppression, raw, state.increase_checkpoint_count
            )
        next_base = state.current_base_suppression if smoothed is None else smoothed
        base_output = max(state.current_base_suppression, next_base)
        next_effective, lambda_payload = self._resolve_effective(
            state, base_output, features["signal_score"], raw
        )
        before_base = state.current_base_suppression
        before_effective = state.current_effective_suppression
        state.current_base_suppression = base_output
        state.current_effective_suppression = next_effective
        state.peak_effective_suppression = max(state.peak_effective_suppression, next_effective)
        state.previous_checkpoint_entropy = features["avg_entropy"]
        self._append_history(state, features["normalized_signal"])
        transition = "none"
        if next_effective > before_effective:
            transition = "increase"
        elif next_effective < before_effective:
            transition = "lambda_adjust"
        event: dict[str, Any] = {
            "event": "checkpoint",
            "checkpoint_type": checkpoint_type,
            "increase_checkpoint_index": state.increase_checkpoint_count,
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
            "newline_count": int(newline_count),
            "entropy": [round(item.entropy, 6) for item in metrics],
            "margin": [round(item.margin, 6) for item in metrics],
            "logprob": [round(item.logprob, 6) for item in metrics],
            "current_base_suppression_before": before_base,
            "current_suppression_before": before_effective,
            "k_smoothing_rho": rho,
            "smoothed_suppression": smoothed,
            "base_target_suppression": base_output,
            "target_suppression": next_effective,
            "transition_applied": transition != "none",
            "transition_type": transition,
            "question_lambda_q": round(state.question_lambda_q, 6),
            **features,
        }
        if lambda_payload is not None:
            event["question_lambda_payload"] = lambda_payload
        state.events.append(event)
        return event

    def _reset_windows(self, state: RequestState) -> None:
        state.up_token_ids.clear()
        state.up_entropy.clear()
        state.up_margin.clear()
        state.up_logprob.clear()
        state.up_newlines = 0
        state.down_token_ids.clear()
        state.down_entropy.clear()
        state.down_margin.clear()
        state.down_logprob.clear()
        state.down_newlines = 0

    def _queue(self, state: RequestState, token_ids: Iterable[int], reason: str) -> None:
        ids = [int(item) for item in token_ids]
        if not ids:
            return
        state.forced_queue.extend(ForcedToken(token_id=item, reason=reason) for item in ids)
        state.events.append(
            {
                "event": "forced_tokens_queued",
                "reason": reason,
                "token_ids": ids,
                "generated_index_before": len(state.generated_ids),
                "current_k": state.current_effective_suppression,
            }
        )
        self._reset_windows(state)

    def _enter_closure(self, state: RequestState, reason: str, include_tail: bool) -> None:
        state.current_base_suppression = 0.0
        state.current_effective_suppression = 0.0
        state.no_more_increases = True
        state.no_more_downgrades = True
        state.end_think_intercept_consumed = True
        state.end_think_block_remaining_newlines = self.config.end_think_mask_newlines
        state.end_think_block_remaining_tokens = self.config.end_think_mask_tokens
        state.end_think_mask_forever_active = self.config.end_think_mask_forever
        state.closure_mode_active = True
        state.closure_mode_start_generated_len = len(state.generated_ids)
        state.upgrade_freeze_remaining_newlines = 0
        state.consecutive_downgrade_hits = 0
        state.pending_downgrade_target_suppression = None
        state.recent_normalized_signal_history.clear()
        forced: list[int] = []
        if include_tail:
            forced.extend(self.config.tail_signal_token_ids)
        forced.extend(self.config.end_think_bridge_token_ids)
        self._queue(state, forced, reason)
        state.events.append(
            {
                "event": "closure_enter",
                "reason": reason,
                "target_suppression": 0.0,
                "mask_newlines": self.config.end_think_mask_newlines,
                "mask_tokens": self.config.end_think_mask_tokens,
                "mask_forever": self.config.end_think_mask_forever,
            }
        )

    def intercept_end_think(self, state: RequestState, candidate_token_id: int) -> bool:
        return bool(
            self.config.end_think_to_l1_lock
            and not state.end_think_intercept_consumed
            and self.config.end_think_token_ids
            and int(candidate_token_id) == self.config.end_think_token_ids[0]
        )

    def on_end_think_intercept(self, state: RequestState) -> None:
        # Preserve the sampled boundary: the sampler replaces the current
        # candidate with the first queued token, so queue </think> first and
        # append the native newline plus Answer bridge after it.
        self._enter_closure(state, "end_think_intercept", include_tail=True)
        # The natural boundary already committed the same sequence used by
        # the 80-percent tail path; do not insert a second boundary later.
        state.tail_signal_inserted = True
        state.tail_signal_armed = False
        state.tail_signal_armed_generated_len = None

    def pop_forced(self, state: RequestState) -> ForcedToken | None:
        return state.forced_queue.pop(0) if state.forced_queue else None

    def _arm_tail_if_needed(self, state: RequestState) -> None:
        if state.tail_signal_inserted or state.tail_signal_armed or not self.config.tail_signal_token_ids:
            return
        if len(state.generated_ids) >= self.config.max_new_tokens * self.config.tail_signal_trigger_ratio:
            state.tail_signal_armed = True
            state.tail_signal_armed_generated_len = len(state.generated_ids)
            state.events.append({"event": "tail_signal_armed", "token_index": len(state.generated_ids)})

    def _insert_tail(self, state: RequestState, source: str) -> None:
        if state.tail_signal_inserted:
            return
        if self.config.end_think_to_l1_lock:
            self._enter_closure(state, f"tail_signal:{source}", include_tail=True)
        else:
            self._queue(state, self.config.tail_signal_token_ids, "tail_signal")
        state.tail_signal_inserted = True
        state.tail_signal_armed = False
        state.tail_signal_armed_generated_len = None

    def on_token(self, state: RequestState, token_id: int, metric: TokenMetric, forced_reason: str | None = None) -> None:
        state.step_index += 1
        state.generated_ids.append(int(token_id))
        newline = 1 if int(token_id) in self.newline_ids else 0
        if state.end_think_block_remaining_tokens > 0:
            state.end_think_block_remaining_tokens -= 1
        if newline:
            state.end_think_block_remaining_newlines = max(state.end_think_block_remaining_newlines - 1, 0)
            state.upgrade_freeze_remaining_newlines = max(state.upgrade_freeze_remaining_newlines - 1, 0)

        if forced_reason is not None:
            state.events.append(
                {
                    "event": "forced_token_emitted",
                    "reason": forced_reason,
                    "token_id": int(token_id),
                    "generated_index": len(state.generated_ids) - 1,
                    "entropy": round(metric.entropy, 6),
                    "margin": round(metric.margin, 6),
                    "logprob": round(metric.logprob, 6),
                }
            )
            if not state.forced_queue:
                self._reset_windows(state)
            return

        state.up_token_ids.append(int(token_id))
        state.up_entropy.append(metric.entropy)
        state.up_margin.append(metric.margin)
        state.up_logprob.append(metric.logprob)
        state.up_newlines += newline
        if state.current_effective_suppression > 0.0 or state.cue_inserted or state.recent_normalized_signal_history:
            state.down_token_ids.append(int(token_id))
            state.down_entropy.append(metric.entropy)
            state.down_margin.append(metric.margin)
            state.down_logprob.append(metric.logprob)
            state.down_newlines += newline

        self._arm_tail_if_needed(state)
        if state.tail_signal_armed and not self.config.tail_signal_next_checkpoint_only and newline:
            self._insert_tail(state, "next_newline")
            return
        if (
            state.tail_signal_armed
            and state.tail_signal_armed_generated_len is not None
            and len(state.generated_ids) - state.tail_signal_armed_generated_len >= self.config.tail_signal_wait_token_limit
        ):
            self._insert_tail(state, "wait_timeout")
            return
        if (
            not state.final_answer_prefix_inserted
            and self.config.final_answer_prefix_token_ids
            and len(state.generated_ids) >= self.config.max_new_tokens * self.config.final_answer_prefix_trigger_ratio
        ):
            self._queue(state, self.config.final_answer_prefix_token_ids, "final_answer_prefix")
            state.final_answer_prefix_inserted = True
            return

        if (
            not state.no_more_increases
            and state.upgrade_freeze_remaining_newlines <= 0
            and state.up_newlines >= self.config.upgrade_newline_span
            and state.up_entropy
        ):
            metrics = [
                TokenMetric(e, m, lp)
                for e, m, lp in zip(state.up_entropy, state.up_margin, state.up_logprob)
            ]
            before = state.current_effective_suppression
            event = self._increase_checkpoint(
                state, list(state.up_token_ids), metrics, state.up_newlines
            )
            tail_pending = state.tail_signal_armed and not state.tail_signal_inserted
            cue_pending = (
                self.config.enable_runtime_cue
                and not state.cue_inserted
                and self.config.cue_token_ids
                and len(state.generated_ids) >= self.config.min_tokens_before_cue
                and float(event.get("signal_score") or 0.0) > self.config.cue_gate_threshold
            )
            rethink_pending = (
                not tail_pending
                and not state.rethink_cue_inserted
                and self.config.rethink_cue_token_ids
                and len(state.generated_ids) >= self.config.rethink_cue_trigger_tokens
            )
            state.up_token_ids.clear()
            state.up_entropy.clear()
            state.up_margin.clear()
            state.up_logprob.clear()
            state.up_newlines = 0
            if state.current_effective_suppression != before:
                state.consecutive_downgrade_hits = 0
                state.pending_downgrade_target_suppression = None
                state.down_token_ids.clear()
                state.down_entropy.clear()
                state.down_margin.clear()
                state.down_logprob.clear()
                state.down_newlines = 0
            if tail_pending:
                self._insert_tail(state, "checkpoint")
                return
            if cue_pending:
                self._queue(state, self.config.cue_token_ids, "cue")
                state.cue_inserted = True
                return
            if rethink_pending:
                self._queue(state, self.config.rethink_cue_token_ids, "rethink_cue")
                state.rethink_cue_inserted = True
                return

        if (
            not state.no_more_downgrades
            and state.down_newlines >= self.config.downgrade_newline_span
            and state.down_entropy
            and state.recent_normalized_signal_history
        ):
            features = self._signal_features(
                sum(state.down_entropy) / len(state.down_entropy),
                sum(state.down_margin) / len(state.down_margin),
                state.previous_checkpoint_entropy,
                len(state.generated_ids),
            )
            history_normalized = round(
                sum(state.recent_normalized_signal_history) / len(state.recent_normalized_signal_history), 6
            )
            target_base = self._mapped_suppression(history_normalized)
            transition = False
            if not self.config.calibration_only and target_base < state.current_base_suppression:
                if (
                    state.pending_downgrade_target_suppression is None
                    or target_base <= state.pending_downgrade_target_suppression
                ):
                    state.consecutive_downgrade_hits += 1
                else:
                    state.consecutive_downgrade_hits = 1
                state.pending_downgrade_target_suppression = target_base
                transition = state.consecutive_downgrade_hits >= self.config.downgrade_confirm_windows
            else:
                state.consecutive_downgrade_hits = 0
                state.pending_downgrade_target_suppression = None
            event = {
                "event": "checkpoint",
                "checkpoint_type": "decrease_check",
                "token_ids": list(state.down_token_ids),
                "entropy": [round(value, 6) for value in state.down_entropy],
                "margin": [round(value, 6) for value in state.down_margin],
                "logprob": [round(value, 6) for value in state.down_logprob],
                "newline_count": state.down_newlines,
                "history_avg_normalized_signal": history_normalized,
                "history_avg_raw_suppression": target_base,
                "transition_applied": transition,
                "downgrade_confirm_hits": state.consecutive_downgrade_hits,
                **features,
            }
            if transition:
                state.current_base_suppression = target_base
                state.current_effective_suppression = self._clamp(target_base * state.question_lambda_q)
                state.upgrade_freeze_remaining_newlines = self.config.post_downgrade_upgrade_freeze_newlines
                state.consecutive_downgrade_hits = 0
                state.pending_downgrade_target_suppression = None
                event["transition_type"] = "decrease"
                event["target_suppression"] = state.current_effective_suppression
            state.events.append(event)
            state.down_token_ids.clear()
            state.down_entropy.clear()
            state.down_margin.clear()
            state.down_logprob.clear()
            state.down_newlines = 0

    def finish_event(self, state: RequestState, reason: str) -> dict[str, Any]:
        return {
            "event": "request_finish",
            "request_id": state.request_id,
            "reason": reason,
            "generated_tokens": len(state.generated_ids),
            "final_suppression_value": state.current_effective_suppression,
            "peak_suppression_value": state.peak_effective_suppression,
            "question_lambda_q": state.question_lambda_q,
            "tail_signal_inserted": state.tail_signal_inserted,
            "final_answer_prefix_inserted": state.final_answer_prefix_inserted,
            "end_think_intercept_consumed": state.end_think_intercept_consumed,
        }
