from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionLambdaCalibration:
    d_lo: float
    d_hi: float
    lambda_min: float
    lambda_max: float
    lambda_exp_beta: float
    d_mean_weight: float
    d_min_weight: float
    source_q20: float | None = None
    source_q90: float | None = None
    u_at_lambda_one: float | None = None
    eligible_sample_count: int | None = None


def compute_question_difficulty_score(
    k1_base: float,
    k2_base: float,
    *,
    mean_weight: float,
    min_weight: float,
) -> float:
    mean_part = (float(k1_base) + float(k2_base)) / 2.0
    min_part = min(float(k1_base), float(k2_base))
    return round((float(mean_weight) * mean_part) + (float(min_weight) * min_part), 6)


def normalize_question_difficulty(d_q: float, calibration: QuestionLambdaCalibration) -> float:
    d_lo = float(calibration.d_lo)
    d_hi = float(calibration.d_hi)
    if d_hi <= d_lo:
        return 1.0 if float(d_q) > d_hi else 0.0
    normalized = (float(d_q) - d_lo) / (d_hi - d_lo)
    return round(min(max(normalized, 0.0), 1.0), 6)


def map_normalized_difficulty_to_lambda(u_q: float, calibration: QuestionLambdaCalibration) -> float:
    u_q = min(max(float(u_q), 0.0), 1.0)
    exp_beta = float(calibration.lambda_exp_beta)
    denominator = 1.0 - math.exp(-exp_beta)
    numerator = 1.0 - math.exp(-exp_beta * u_q)
    mapped = numerator / denominator if denominator > 0.0 else u_q
    value = float(calibration.lambda_min) + (
        (float(calibration.lambda_max) - float(calibration.lambda_min)) * mapped
    )
    return round(value, 6)


def compute_question_lambda_q(
    k1_base: float,
    k2_base: float,
    calibration: QuestionLambdaCalibration,
) -> tuple[float, float, float]:
    d_q = compute_question_difficulty_score(
        k1_base,
        k2_base,
        mean_weight=calibration.d_mean_weight,
        min_weight=calibration.d_min_weight,
    )
    u_q = normalize_question_difficulty(d_q, calibration)
    lambda_q = map_normalized_difficulty_to_lambda(u_q, calibration)
    return d_q, u_q, lambda_q
