from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

import torch

from experiment_src.evaluation_v3_8_ds.config import (
    DATASETS,
    DEFAULT_ASSISTANT_PREFIX_TEXT,
    DEFAULT_CUE_TEXT,
    DEFAULT_CUE_THRESHOLD,
    DEFAULT_CUE_GATE_THRESHOLD,
    DEFAULT_DOWNGRADE_CONFIRM_WINDOWS,
    DEFAULT_DOWNGRADE_NEWLINE_SPAN,
    DEFAULT_ENTROPY_UPPER_BOUND,
    DEFAULT_EXP_BETA,
    DEFAULT_FINAL_ANSWER_PREFIX_TEXT,
    DEFAULT_FINAL_ANSWER_PREFIX_TRIGGER_RATIO,
    DEFAULT_INITIAL_SUPPRESSION,
    DEFAULT_K_SMOOTHING_INITIAL_CHECKPOINTS,
    DEFAULT_K_SMOOTHING_INITIAL_RHO,
    DEFAULT_K_SMOOTHING_LATE_RHO,
    DEFAULT_MIN_TOKENS_BEFORE_CUE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_POST_DOWNGRADE_UPGRADE_FREEZE_NEWLINES,
    DEFAULT_POWER_EXPONENT,
    DEFAULT_QUESTION_LAMBDA_D_MEAN_WEIGHT,
    DEFAULT_QUESTION_LAMBDA_D_MIN_WEIGHT,
    DEFAULT_QUESTION_LAMBDA_EXP_BETA,
    DEFAULT_QUESTION_LAMBDA_MAX,
    DEFAULT_QUESTION_LAMBDA_MIN,
    DEFAULT_RAW_HISTORY_SIZE,
    DEFAULT_SEED,
    DEFAULT_SUPPRESSION_CURVE,
    DEFAULT_SUPPRESSION_SCALE,
    DEFAULT_SUPPRESSION_UPPER_BOUND,
    SUPPORTED_SUPPRESSION_CURVES,
    DEFAULT_TAIL_EXP_BETA,
    DEFAULT_TAIL_FULL_RATIO,
    DEFAULT_TAIL_SIGNAL_TEXT,
    DEFAULT_TAIL_SIGNAL_NEXT_CHECKPOINT_ONLY,
    DEFAULT_TAIL_SIGNAL_TRIGGER_RATIO,
    DEFAULT_TAIL_SIGNAL_WAIT_TOKEN_LIMIT,
    DEFAULT_TAIL_SOFT_RATIO,
    DEFAULT_TAIL_THRESHOLD_MAX,
    DEFAULT_DISABLE_CUSTOM_ATTENTION_PATCH,
    DEFAULT_TEMPERATURE,
    DEFAULT_UPGRADE_NEWLINE_SPAN,
    REMOTE,
)
from experiment_src.evaluation_v3_8_ds.judge import CascadeDualJudge, JudgeConfig
from experiment_src.evaluation_v3_8_ds.question_lambda import QuestionLambdaCalibration
from experiment_src.evaluation_v3_8_ds.runtime_env import (
    ContinuousEntropyRuntime,
    EpisodeRecord,
    SignalCalibration,
    prepare_runtime_resources,
    summarize_episodes,
)
from experiment_src.sampling_v1.io_utils import append_jsonl, ensure_parent, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 evaluation-v3-8 复合信号 + 题目级 λ_q + 预算感知尾部升阈控制器")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--model", default=REMOTE.model_path)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--summary-json", default=REMOTE.summary_json_path)
    parser.add_argument("--llm-device", default="cuda:0")
    parser.add_argument("--gpu-ids", default=None, help="手动指定 GPU 列表，例如 0,1,2。")
    parser.add_argument("--reset-seed-per-sample", action="store_true")
    parser.add_argument("--checkpoint-zero", action="store_true")
    parser.add_argument(
        "--disable-assistant-prefix-checkpoint",
        action="store_true",
        help="只把 assistant prefix 当作上下文，不额外触发 prefix checkpoint。",
    )
    parser.add_argument(
        "--initial-suppression",
        type=float,
        default=DEFAULT_INITIAL_SUPPRESSION,
        help="插入 assistant prefix 后开始自由生成时使用的初始 suppression。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--assistant-prefix-text", default=DEFAULT_ASSISTANT_PREFIX_TEXT)
    parser.add_argument("--cue-text", default=DEFAULT_CUE_TEXT)
    parser.add_argument("--cue-threshold", type=float, default=DEFAULT_CUE_THRESHOLD)
    parser.add_argument("--cue-gate-threshold", type=float, default=DEFAULT_CUE_GATE_THRESHOLD)
    parser.add_argument("--min-tokens-before-cue", type=int, default=DEFAULT_MIN_TOKENS_BEFORE_CUE)
    parser.add_argument("--final-answer-prefix-text", default=DEFAULT_FINAL_ANSWER_PREFIX_TEXT)
    parser.add_argument(
        "--final-answer-prefix-trigger-ratio",
        type=float,
        default=DEFAULT_FINAL_ANSWER_PREFIX_TRIGGER_RATIO,
    )
    parser.add_argument("--rethink-cue-text", default=None)
    parser.add_argument("--rethink-cue-trigger-tokens", type=int, default=0)
    parser.add_argument("--tail-soft-ratio", type=float, default=DEFAULT_TAIL_SOFT_RATIO)
    parser.add_argument("--tail-full-ratio", type=float, default=DEFAULT_TAIL_FULL_RATIO)
    parser.add_argument("--tail-exp-beta", type=float, default=DEFAULT_TAIL_EXP_BETA)
    parser.add_argument("--tail-threshold-max", type=float, default=DEFAULT_TAIL_THRESHOLD_MAX)
    parser.add_argument("--tail-signal-trigger-ratio", type=float, default=DEFAULT_TAIL_SIGNAL_TRIGGER_RATIO)
    parser.add_argument("--tail-signal-text", default=DEFAULT_TAIL_SIGNAL_TEXT)
    parser.add_argument("--tail-signal-wait-token-limit", type=int, default=DEFAULT_TAIL_SIGNAL_WAIT_TOKEN_LIMIT)
    parser.add_argument(
        "--tail-signal-next-checkpoint-only",
        action="store_true",
        default=DEFAULT_TAIL_SIGNAL_NEXT_CHECKPOINT_ONLY,
        help="tail signal 达到 token 阈值后只在下一个 checkpoint 或等待 token 上限处插入，不在普通换行处立刻插入。",
    )
    parser.add_argument("--entropy-upper-bound", type=float, default=DEFAULT_ENTROPY_UPPER_BOUND)
    parser.add_argument(
        "--suppression-curve",
        choices=SUPPORTED_SUPPRESSION_CURVES,
        default=DEFAULT_SUPPRESSION_CURVE,
    )
    parser.add_argument("--power-exponent", type=float, default=DEFAULT_POWER_EXPONENT)
    parser.add_argument("--exp-beta", type=float, default=DEFAULT_EXP_BETA)
    parser.add_argument("--suppression-scale", type=float, default=DEFAULT_SUPPRESSION_SCALE)
    parser.add_argument("--suppression-upper-bound", type=float, default=DEFAULT_SUPPRESSION_UPPER_BOUND)
    parser.add_argument("--raw-history-size", type=int, default=DEFAULT_RAW_HISTORY_SIZE)
    parser.add_argument("--k-smoothing-initial-rho", type=float, default=DEFAULT_K_SMOOTHING_INITIAL_RHO)
    parser.add_argument("--k-smoothing-late-rho", type=float, default=DEFAULT_K_SMOOTHING_LATE_RHO)
    parser.add_argument(
        "--k-smoothing-initial-checkpoints",
        type=int,
        default=DEFAULT_K_SMOOTHING_INITIAL_CHECKPOINTS,
    )
    parser.add_argument("--upgrade-newline-span", type=int, default=DEFAULT_UPGRADE_NEWLINE_SPAN)
    parser.add_argument("--downgrade-newline-span", type=int, default=DEFAULT_DOWNGRADE_NEWLINE_SPAN)
    parser.add_argument("--downgrade-confirm-windows", type=int, default=DEFAULT_DOWNGRADE_CONFIRM_WINDOWS)
    parser.add_argument(
        "--post-downgrade-upgrade-freeze-newlines",
        type=int,
        default=DEFAULT_POST_DOWNGRADE_UPGRADE_FREEZE_NEWLINES,
    )
    parser.add_argument("--end-think-to-l1-lock", action="store_true")
    parser.add_argument("--end-think-mask-newlines", type=int, default=1)
    parser.add_argument("--end-think-mask-tokens", type=int, default=0)
    parser.add_argument("--end-think-mask-forever", action="store_true")
    parser.add_argument("--end-think-bridge-text", default=None)
    parser.add_argument("--debug-runtime-events", action="store_true")
    parser.add_argument("--skip-judge", action="store_true", help="smoke 测试时跳过外部 judge 调用。")
    parser.add_argument("--subset-ids-path", default=None, help="仅评测该 id 列表中的样本。")
    parser.add_argument("--signal-calibration-json", default=None, help="复合信号归一化统计文件。")
    parser.add_argument("--question-lambda-calibration-json", default=None, help="题目级 λ_q 校准文件。")
    parser.add_argument(
        "--dynamic-question-lambda",
        action="store_true",
        help="启用动态 λ_q：前期按 U_probe 保守控制，后期按 token 单向升到高档。",
    )
    parser.add_argument("--dynamic-lambda-probe-checkpoints", type=int, default=4)
    parser.add_argument("--dynamic-lambda-signal-lo", type=float, default=0.18)
    parser.add_argument("--dynamic-lambda-signal-hi", type=float, default=0.55)
    parser.add_argument("--dynamic-lambda-probe-scale-min", type=float, default=0.30)
    parser.add_argument("--dynamic-lambda-probe-scale-max", type=float, default=0.45)
    parser.add_argument("--dynamic-lambda-late-token-start", type=int, default=5500)
    parser.add_argument("--dynamic-lambda-late-token-end", type=int, default=7000)
    parser.add_argument("--dynamic-lambda-late-scale-start", type=float, default=0.45)
    parser.add_argument("--dynamic-lambda-late-scale-max", type=float, default=0.80)
    parser.add_argument(
        "--high-low-high-question-lambda",
        action="store_true",
        help="启用先高、后低、长输出再升高的题目级 λ_q 调度。",
    )
    parser.add_argument("--high-low-high-early-scale", type=float, default=0.80)
    parser.add_argument("--high-low-high-early-min-tokens", type=int, default=800)
    parser.add_argument("--high-low-high-early-min-checkpoints", type=int, default=4)
    parser.add_argument("--high-low-high-middle-scale", type=float, default=0.40)
    parser.add_argument("--high-low-high-late-token-start", type=int, default=5500)
    parser.add_argument("--high-low-high-late-token-end", type=int, default=7000)
    parser.add_argument("--high-low-high-late-scale-max", type=float, default=0.80)
    parser.add_argument("--high-low-high-adaptive-early", action="store_true")
    parser.add_argument("--high-low-high-adaptive-max-tokens", type=int, default=2000)
    parser.add_argument("--high-low-high-adaptive-window", type=int, default=3)
    parser.add_argument("--high-low-high-adaptive-avg-signal-threshold", type=float, default=0.65)
    parser.add_argument("--high-low-high-adaptive-max-signal-threshold", type=float, default=0.90)
    parser.add_argument("--high-low-high-adaptive-max-raw-threshold", type=float, default=0.36)
    parser.add_argument("--high-low-high-adaptive-raw-delta-threshold", type=float, default=0.30)
    parser.add_argument(
        "--linear-down-up-question-lambda",
        action="store_true",
        help="启用 token 线性调度：高档保持、线性降档、再线性升档。",
    )
    parser.add_argument("--linear-down-up-high-scale", type=float, default=0.80)
    parser.add_argument("--linear-down-up-low-scale", type=float, default=0.40)
    parser.add_argument("--linear-down-up-hold-high-tokens", type=int, default=800)
    parser.add_argument("--linear-down-up-down-end-tokens", type=int, default=1200)
    parser.add_argument("--linear-down-up-up-start-tokens", type=int, default=1200)
    parser.add_argument("--linear-down-up-up-end-tokens", type=int, default=7000)
    parser.add_argument("--calibration-only", action="store_true", help="仅做 K=0 校准统计，不启用 cue/suppression。")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--disable-custom-attention-patch", action="store_true", default=DEFAULT_DISABLE_CUSTOM_ATTENTION_PATCH)
    return parser.parse_args()


def parse_gpu_ids(raw_gpu_ids: str | None) -> list[int]:
    if raw_gpu_ids is None:
        return []
    raw_gpu_ids = raw_gpu_ids.strip()
    if not raw_gpu_ids:
        return []
    gpu_ids: list[int] = []
    seen: set[int] = set()
    for chunk in raw_gpu_ids.split(","):
        item = chunk.strip()
        if not item:
            raise ValueError(f"非法 --gpu-ids: {raw_gpu_ids!r}")
        gpu_id = int(item)
        if gpu_id < 0:
            raise ValueError(f"GPU id 必须非负: {gpu_id}")
        if gpu_id in seen:
            raise ValueError(f"GPU id 重复: {gpu_id}")
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def set_seed(seed: int, include_cuda: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_manual_cuda_device(gpu_id: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，无法使用手动 GPU 分配。")
    torch.cuda.set_device(gpu_id)


def log_event(message: str, **payload: Any) -> None:
    record = {"message": message, **payload}
    print(json.dumps(record, ensure_ascii=False), flush=True)


def materialize_eval_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "results": output_dir / "results",
        "rollouts": output_dir / "rollouts",
        "reports": output_dir / "reports",
        "metrics": output_dir / "metrics",
    }
    for path in paths.values():
        ensure_parent(path / ".keep")
        path.mkdir(parents=True, exist_ok=True)
    return paths


def slice_rows(
    rows: list[dict[str, Any]],
    start_index: int,
    end_index: int | None,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    sliced = rows[start_index:end_index]
    if max_samples is not None:
        return sliced[:max_samples]
    return sliced


def load_subset_ids(subset_ids_path: str | None) -> set[str] | None:
    if subset_ids_path is None:
        return None
    path = Path(subset_ids_path)
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ids:
        raise RuntimeError(f"subset_ids_path 为空: {path}")
    return ids


def filter_rows_by_subset_ids(rows: list[dict[str, Any]], subset_ids: set[str] | None) -> list[dict[str, Any]]:
    if subset_ids is None:
        return rows
    filtered: list[dict[str, Any]] = []
    for row in rows:
        sample_id = None
        for key in ("unique_id", "id", "index"):
            value = row.get(key)
            if value is not None:
                sample_id = str(value)
                break
        if sample_id in subset_ids:
            filtered.append(row)
    return filtered


def load_signal_calibration(signal_calibration_json: str | None) -> SignalCalibration | None:
    if signal_calibration_json is None:
        return None
    payload = json.loads(Path(signal_calibration_json).read_text(encoding="utf-8"))
    recommended = payload.get("recommended_signal_calibration", payload)
    return SignalCalibration(
        entropy_lo=float(recommended["entropy_lo"]),
        entropy_hi=float(recommended["entropy_hi"]),
        delta_entropy_hi=float(recommended["delta_entropy_hi"]),
        margin_lo=float(recommended["margin_lo"]),
        margin_hi=float(recommended["margin_hi"]),
    )


def load_question_lambda_calibration(
    question_lambda_calibration_json: str | None,
) -> QuestionLambdaCalibration | None:
    if question_lambda_calibration_json is None:
        return None
    payload = json.loads(Path(question_lambda_calibration_json).read_text(encoding="utf-8"))
    recommended = payload.get("recommended_question_lambda_calibration", payload)
    return QuestionLambdaCalibration(
        d_lo=float(recommended["d_lo"]),
        d_hi=float(recommended["d_hi"]),
        lambda_min=float(recommended.get("lambda_min", DEFAULT_QUESTION_LAMBDA_MIN)),
        lambda_max=float(recommended.get("lambda_max", DEFAULT_QUESTION_LAMBDA_MAX)),
        lambda_exp_beta=float(recommended.get("lambda_exp_beta", DEFAULT_QUESTION_LAMBDA_EXP_BETA)),
        d_mean_weight=float(recommended.get("d_mean_weight", DEFAULT_QUESTION_LAMBDA_D_MEAN_WEIGHT)),
        d_min_weight=float(recommended.get("d_min_weight", DEFAULT_QUESTION_LAMBDA_D_MIN_WEIGHT)),
        source_q20=None if recommended.get("source_q20") is None else float(recommended["source_q20"]),
        source_q90=None if recommended.get("source_q90") is None else float(recommended["source_q90"]),
        u_at_lambda_one=(
            None if recommended.get("u_at_lambda_one") is None else float(recommended["u_at_lambda_one"])
        ),
        eligible_sample_count=(
            None if recommended.get("eligible_sample_count") is None else int(recommended["eligible_sample_count"])
        ),
    )


def compute_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile 输入为空。")
    if len(values) == 1:
        return round(float(values[0]), 6)
    ordered = sorted(float(value) for value in values)
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 6)


def summarize_scalar_list(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "q10": compute_quantile(values, 0.10),
        "q50": compute_quantile(values, 0.50),
        "q90": compute_quantile(values, 0.90),
        "max": round(max(values), 6),
    }


def iter_signal_checkpoint_events(episodes: list[EpisodeRecord]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for episode in episodes:
        for run in episode.runs:
            for checkpoint_event in run.get("checkpoint_events", []):
                checkpoint_type = checkpoint_event.get("checkpoint_type")
                if checkpoint_type in {"increase_check_checkpoint0", "increase_check"}:
                    events.append(checkpoint_event)
    return events


def build_signal_calibration_report(
    episodes: list[EpisodeRecord],
    *,
    run_name: str,
    dataset: str,
    selected_sample_count: int,
) -> dict[str, Any]:
    checkpoint_events = iter_signal_checkpoint_events(episodes)
    if not checkpoint_events:
        raise RuntimeError("calibration_only 运行未收集到任何 increase checkpoint。")
    avg_entropy_values = [float(event["avg_entropy"]) for event in checkpoint_events]
    avg_margin_values = [float(event["avg_margin"]) for event in checkpoint_events]
    positive_delta_values = [float(event["positive_entropy_delta"]) for event in checkpoint_events]
    entropy_summary = summarize_scalar_list(avg_entropy_values)
    margin_summary = summarize_scalar_list(avg_margin_values)
    delta_summary = summarize_scalar_list(positive_delta_values)
    recommended = {
        "entropy_lo": entropy_summary["q10"],
        "entropy_hi": entropy_summary["q90"],
        "delta_entropy_hi": delta_summary["q90"],
        "margin_lo": margin_summary["q10"],
        "margin_hi": margin_summary["q90"],
    }
    return {
        "method": "evaluation_v3_8_composite_signal_lambdaq_v1",
        "run_name": run_name,
        "dataset": dataset,
        "selected_sample_count": selected_sample_count,
        "checkpoint_event_types": ["increase_check_checkpoint0", "increase_check"],
        "checkpoint_count": len(checkpoint_events),
        "avg_entropy": entropy_summary,
        "avg_margin": margin_summary,
        "positive_entropy_delta": delta_summary,
        "recommended_signal_calibration": recommended,
    }


def build_result_row(
    run_name: str,
    dataset_name: str,
    episode: EpisodeRecord,
    episode_started_at: str,
    episode_finished_at: str,
    episode_elapsed_seconds: float,
) -> dict[str, Any]:
    judge_record = dict(episode.judge_record)
    result = {
        "run_name": run_name,
        "dataset": dataset_name,
        "id": episode.sample_id,
        "question": episode.question,
        "answer": episode.answer,
        "standard_answer": judge_record.get("standard_answer", episode.answer),
        "final_suppression_value": episode.final_suppression_value,
        "peak_suppression_value": episode.peak_suppression_value,
        "question_lambda_q": episode.question_lambda_q,
        "cue_inserted": episode.cue_inserted,
        "tail_signal_inserted": episode.tail_signal_inserted,
        "attempt_count": episode.attempt_count,
        "total_generated_tokens": episode.total_generated_tokens,
        "final_text": episode.final_text,
        "runs": episode.runs,
        "episode_started_at": episode_started_at,
        "episode_finished_at": episode_finished_at,
        "episode_elapsed_seconds": round(float(episode_elapsed_seconds), 6),
    }
    result.update(judge_record)
    return result


def build_runtime(
    args: argparse.Namespace,
    llm_device: str,
    signal_calibration: SignalCalibration | None,
    question_lambda_calibration: QuestionLambdaCalibration | None,
) -> ContinuousEntropyRuntime:
    resources = prepare_runtime_resources(
        model_path=args.model,
        tokenizer_path=args.tokenizer_path,
        llm_device=llm_device,
        disable_custom_attention_patch=args.disable_custom_attention_patch,
    )
    judge = None
    if not args.skip_judge:
        judge = CascadeDualJudge(
            JudgeConfig(
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                request_timeout=args.request_timeout,
            )
        )
    return ContinuousEntropyRuntime(
        run_name=args.run_name,
        resources=resources,
        judge=judge,
        summary_json=args.summary_json,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        assistant_prefix_text=args.assistant_prefix_text,
        cue_text=args.cue_text,
        cue_threshold=args.cue_threshold,
        cue_gate_threshold=args.cue_gate_threshold,
        min_tokens_before_cue=args.min_tokens_before_cue,
        final_answer_prefix_text=args.final_answer_prefix_text,
        final_answer_prefix_trigger_ratio=args.final_answer_prefix_trigger_ratio,
        rethink_cue_text=args.rethink_cue_text,
        rethink_cue_trigger_tokens=args.rethink_cue_trigger_tokens,
        tail_soft_ratio=args.tail_soft_ratio,
        tail_full_ratio=args.tail_full_ratio,
        tail_exp_beta=args.tail_exp_beta,
        tail_threshold_max=args.tail_threshold_max,
        tail_signal_trigger_ratio=args.tail_signal_trigger_ratio,
        tail_signal_text=args.tail_signal_text,
        tail_signal_wait_token_limit=args.tail_signal_wait_token_limit,
        tail_signal_next_checkpoint_only=args.tail_signal_next_checkpoint_only,
        entropy_upper_bound=args.entropy_upper_bound,
        suppression_curve=args.suppression_curve,
        power_exponent=args.power_exponent,
        exp_beta=args.exp_beta,
        suppression_scale=args.suppression_scale,
        suppression_upper_bound=args.suppression_upper_bound,
        raw_history_size=args.raw_history_size,
        k_smoothing_initial_rho=args.k_smoothing_initial_rho,
        k_smoothing_late_rho=args.k_smoothing_late_rho,
        k_smoothing_initial_checkpoints=args.k_smoothing_initial_checkpoints,
        upgrade_newline_span=args.upgrade_newline_span,
        downgrade_newline_span=args.downgrade_newline_span,
        downgrade_confirm_windows=args.downgrade_confirm_windows,
        post_downgrade_upgrade_freeze_newlines=args.post_downgrade_upgrade_freeze_newlines,
        end_think_to_l1_lock=args.end_think_to_l1_lock,
        end_think_mask_newlines=args.end_think_mask_newlines,
        end_think_mask_tokens=args.end_think_mask_tokens,
        end_think_mask_forever=args.end_think_mask_forever,
        end_think_bridge_text=args.end_think_bridge_text,
        checkpoint_zero=args.checkpoint_zero,
        assistant_prefix_checkpoint=not args.disable_assistant_prefix_checkpoint,
        initial_suppression=args.initial_suppression,
        signal_calibration=signal_calibration,
        question_lambda_calibration=question_lambda_calibration,
        dynamic_question_lambda=args.dynamic_question_lambda,
        dynamic_lambda_probe_checkpoints=args.dynamic_lambda_probe_checkpoints,
        dynamic_lambda_signal_lo=args.dynamic_lambda_signal_lo,
        dynamic_lambda_signal_hi=args.dynamic_lambda_signal_hi,
        dynamic_lambda_probe_scale_min=args.dynamic_lambda_probe_scale_min,
        dynamic_lambda_probe_scale_max=args.dynamic_lambda_probe_scale_max,
        dynamic_lambda_late_token_start=args.dynamic_lambda_late_token_start,
        dynamic_lambda_late_token_end=args.dynamic_lambda_late_token_end,
        dynamic_lambda_late_scale_start=args.dynamic_lambda_late_scale_start,
        dynamic_lambda_late_scale_max=args.dynamic_lambda_late_scale_max,
        high_low_high_question_lambda=args.high_low_high_question_lambda,
        high_low_high_early_scale=args.high_low_high_early_scale,
        high_low_high_early_min_tokens=args.high_low_high_early_min_tokens,
        high_low_high_early_min_checkpoints=args.high_low_high_early_min_checkpoints,
        high_low_high_middle_scale=args.high_low_high_middle_scale,
        high_low_high_late_token_start=args.high_low_high_late_token_start,
        high_low_high_late_token_end=args.high_low_high_late_token_end,
        high_low_high_late_scale_max=args.high_low_high_late_scale_max,
        high_low_high_adaptive_early=args.high_low_high_adaptive_early,
        high_low_high_adaptive_max_tokens=args.high_low_high_adaptive_max_tokens,
        high_low_high_adaptive_window=args.high_low_high_adaptive_window,
        high_low_high_adaptive_avg_signal_threshold=args.high_low_high_adaptive_avg_signal_threshold,
        high_low_high_adaptive_max_signal_threshold=args.high_low_high_adaptive_max_signal_threshold,
        high_low_high_adaptive_max_raw_threshold=args.high_low_high_adaptive_max_raw_threshold,
        high_low_high_adaptive_raw_delta_threshold=args.high_low_high_adaptive_raw_delta_threshold,
        linear_down_up_question_lambda=args.linear_down_up_question_lambda,
        linear_down_up_high_scale=args.linear_down_up_high_scale,
        linear_down_up_low_scale=args.linear_down_up_low_scale,
        linear_down_up_hold_high_tokens=args.linear_down_up_hold_high_tokens,
        linear_down_up_down_end_tokens=args.linear_down_up_down_end_tokens,
        linear_down_up_up_start_tokens=args.linear_down_up_up_start_tokens,
        linear_down_up_up_end_tokens=args.linear_down_up_up_end_tokens,
        calibration_only=args.calibration_only,
        debug_runtime_events=args.debug_runtime_events,
    )


def build_rollout_row(
    episode: EpisodeRecord,
    episode_started_at: str,
    episode_finished_at: str,
    episode_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "id": episode.sample_id,
        "final_suppression_value": episode.final_suppression_value,
        "peak_suppression_value": episode.peak_suppression_value,
        "question_lambda_q": episode.question_lambda_q,
        "cue_inserted": episode.cue_inserted,
        "tail_signal_inserted": episode.tail_signal_inserted,
        "attempt_count": episode.attempt_count,
        "total_generated_tokens": episode.total_generated_tokens,
        "episode_started_at": episode_started_at,
        "episode_finished_at": episode_finished_at,
        "episode_elapsed_seconds": round(float(episode_elapsed_seconds), 6),
        "judge_record": episode.judge_record,
        "runs": episode.runs,
    }


def write_episode_outputs(
    result_path: Path,
    rollout_path: Path,
    run_name: str,
    dataset_name: str,
    episode: EpisodeRecord,
    episode_started_at: str,
    episode_finished_at: str,
    episode_elapsed_seconds: float,
) -> None:
    append_jsonl(
        result_path,
        build_result_row(
            run_name=run_name,
            dataset_name=dataset_name,
            episode=episode,
            episode_started_at=episode_started_at,
            episode_finished_at=episode_finished_at,
            episode_elapsed_seconds=episode_elapsed_seconds,
        ),
    )
    append_jsonl(
        rollout_path,
        build_rollout_row(
            episode=episode,
            episode_started_at=episode_started_at,
            episode_finished_at=episode_finished_at,
            episode_elapsed_seconds=episode_elapsed_seconds,
        ),
    )


def materialize_multi_gpu_staging_dirs(paths: dict[str, Path]) -> dict[str, Path]:
    root = paths["metrics"] / "multi_gpu_staging"
    shutil.rmtree(root, ignore_errors=True)
    staging_paths = {
        "root": root,
        "results_parts": root / "results_parts",
        "rollout_parts": root / "rollout_parts",
    }
    for path in staging_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return staging_paths


def stage_episode_outputs(
    staging_paths: dict[str, Path],
    order_index: int,
    run_name: str,
    dataset_name: str,
    episode: EpisodeRecord,
    episode_started_at: str,
    episode_finished_at: str,
    episode_elapsed_seconds: float,
) -> None:
    part_name = f"{int(order_index):06d}.json"
    write_json(
        staging_paths["results_parts"] / part_name,
        build_result_row(
            run_name=run_name,
            dataset_name=dataset_name,
            episode=episode,
            episode_started_at=episode_started_at,
            episode_finished_at=episode_finished_at,
            episode_elapsed_seconds=episode_elapsed_seconds,
        ),
    )
    write_json(
        staging_paths["rollout_parts"] / part_name,
        build_rollout_row(
            episode=episode,
            episode_started_at=episode_started_at,
            episode_finished_at=episode_finished_at,
            episode_elapsed_seconds=episode_elapsed_seconds,
        ),
    )


def merge_staged_episode_outputs(
    staging_paths: dict[str, Path],
    result_path: Path,
    rollout_path: Path,
    expected_count: int,
) -> None:
    result_path.unlink(missing_ok=True)
    rollout_path.unlink(missing_ok=True)
    with result_path.open("w", encoding="utf-8") as result_handle, rollout_path.open(
        "w", encoding="utf-8"
    ) as rollout_handle:
        for order_index in range(expected_count):
            part_name = f"{int(order_index):06d}.json"
            result_part = staging_paths["results_parts"] / part_name
            rollout_part = staging_paths["rollout_parts"] / part_name
            if not result_part.exists() or not rollout_part.exists():
                raise RuntimeError(
                    f"multi-gpu staged output 缺失: order_index={order_index} "
                    f"result_exists={result_part.exists()} rollout_exists={rollout_part.exists()}"
                )
            result_handle.write(json.dumps(json.loads(result_part.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
            rollout_handle.write(
                json.dumps(json.loads(rollout_part.read_text(encoding="utf-8")), ensure_ascii=False) + "\n"
            )


def maybe_log_episode(
    index: int,
    total: int,
    episode: EpisodeRecord,
    episode_elapsed_seconds: float,
    log_every: int,
) -> None:
    if index == 1 or index == total or (log_every > 0 and index % log_every == 0):
        log_event(
            "evaluation_v3_8 episode finished",
            index=index,
            total=total,
            sample_id=episode.sample_id,
            correct=episode.judge_record.get("is_correct"),
            judge_skipped=episode.judge_record.get("judge_skipped"),
            final_suppression_value=episode.final_suppression_value,
            peak_suppression_value=episode.peak_suppression_value,
            cue_inserted=episode.cue_inserted,
            attempt_count=episode.attempt_count,
            total_generated_tokens=episode.total_generated_tokens,
            episode_elapsed_seconds=round(float(episode_elapsed_seconds), 3),
        )


def worker_payload(
    order_index: int,
    episode: EpisodeRecord,
    episode_started_at: str,
    episode_finished_at: str,
    episode_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "type": "episode",
        "order_index": order_index,
        "episode": episode,
        "episode_started_at": episode_started_at,
        "episode_finished_at": episode_finished_at,
        "episode_elapsed_seconds": episode_elapsed_seconds,
    }


def parallel_worker_main(
    worker_index: int,
    gpu_id: int,
    task_queue: Any,
    args_payload: dict[str, Any],
    output_queue: Any,
) -> None:
    try:
        args = argparse.Namespace(**args_payload)
        set_manual_cuda_device(gpu_id)
        set_seed(args.seed, include_cuda=True)
        runtime = build_runtime(
            args=args,
            llm_device=f"cuda:{gpu_id}",
            signal_calibration=load_signal_calibration(args.signal_calibration_json),
            question_lambda_calibration=load_question_lambda_calibration(args.question_lambda_calibration_json),
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            order_index, row = task
            if args.reset_seed_per_sample:
                set_manual_cuda_device(gpu_id)
                set_seed(args.seed, include_cuda=True)
            episode_started_at = datetime.now().isoformat(timespec="seconds")
            episode_start_time = time.perf_counter()
            episode = runtime.rollout_episode(row=row, fallback_index=args.start_index + order_index)
            episode_elapsed_seconds = time.perf_counter() - episode_start_time
            episode_finished_at = datetime.now().isoformat(timespec="seconds")
            output_queue.put(
                worker_payload(
                    order_index=order_index,
                    episode=episode,
                    episode_started_at=episode_started_at,
                    episode_finished_at=episode_finished_at,
                    episode_elapsed_seconds=episode_elapsed_seconds,
                )
            )
        output_queue.put({"type": "done", "worker_index": worker_index})
    except Exception as exc:
        output_queue.put(
            {
                "type": "error",
                "worker_index": worker_index,
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_single_device(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    paths: dict[str, Path],
    llm_device: str,
) -> list[EpisodeRecord]:
    include_cuda = llm_device.startswith("cuda")
    if include_cuda:
        device_index = torch.device(llm_device).index
        if device_index is None:
            raise RuntimeError(f"非法 CUDA 设备: {llm_device}")
        set_manual_cuda_device(device_index)
    set_seed(args.seed, include_cuda=include_cuda)
    runtime = build_runtime(
        args=args,
        llm_device=llm_device,
        signal_calibration=load_signal_calibration(args.signal_calibration_json),
        question_lambda_calibration=load_question_lambda_calibration(args.question_lambda_calibration_json),
    )
    result_path = paths["results"] / "final_results.jsonl"
    rollout_path = paths["rollouts"] / "eval_rollout.jsonl"
    episodes: list[EpisodeRecord] = []
    for order_index, row in enumerate(rows):
        if args.reset_seed_per_sample:
            if include_cuda:
                device_index = torch.device(llm_device).index
                if device_index is None:
                    raise RuntimeError(f"非法 CUDA 设备: {llm_device}")
                set_manual_cuda_device(device_index)
            set_seed(args.seed, include_cuda=include_cuda)
        episode_started_at = datetime.now().isoformat(timespec="seconds")
        episode_start_time = time.perf_counter()
        episode = runtime.rollout_episode(row=row, fallback_index=args.start_index + order_index)
        episode_elapsed_seconds = time.perf_counter() - episode_start_time
        episode_finished_at = datetime.now().isoformat(timespec="seconds")
        episodes.append(episode)
        write_episode_outputs(
            result_path=result_path,
            rollout_path=rollout_path,
            run_name=args.run_name,
            dataset_name=args.dataset,
            episode=episode,
            episode_started_at=episode_started_at,
            episode_finished_at=episode_finished_at,
            episode_elapsed_seconds=episode_elapsed_seconds,
        )
        maybe_log_episode(
            index=order_index + 1,
            total=len(rows),
            episode=episode,
            episode_elapsed_seconds=episode_elapsed_seconds,
            log_every=args.log_every,
        )
    return episodes


def run_multi_gpu(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    paths: dict[str, Path],
    gpu_ids: list[int],
) -> list[EpisodeRecord]:
    if not gpu_ids:
        raise RuntimeError("run_multi_gpu 至少需要一个 GPU id。")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    output_queue = ctx.Queue()
    for order_index, row in enumerate(rows):
        task_queue.put((order_index, row))
    for _ in gpu_ids:
        task_queue.put(None)

    args_payload = vars(args).copy()
    workers: list[Any] = []
    for worker_index, gpu_id in enumerate(gpu_ids):
        process = ctx.Process(
            target=parallel_worker_main,
            args=(
                worker_index,
                gpu_id,
                task_queue,
                args_payload,
                output_queue,
            ),
        )
        process.start()
        workers.append(process)

    result_path = paths["results"] / "final_results.jsonl"
    rollout_path = paths["rollouts"] / "eval_rollout.jsonl"
    staging_paths = materialize_multi_gpu_staging_dirs(paths)
    episodes_by_order: dict[int, EpisodeRecord] = {}
    completed_episode_count = 0
    done_workers = 0
    try:
        while done_workers < len(workers):
            try:
                payload = output_queue.get(timeout=30.0)
            except Empty:
                dead_workers = [process for process in workers if process.exitcode not in (None, 0)]
                if dead_workers:
                    raise RuntimeError(
                        "multi-gpu worker exited unexpectedly: "
                        + ", ".join(str(process.exitcode) for process in dead_workers)
                    )
                continue
            payload_type = payload.get("type")
            if payload_type == "episode":
                order_index = int(payload["order_index"])
                if order_index in episodes_by_order:
                    raise RuntimeError(f"multi-gpu 重复返回同一条样本: order_index={order_index}")
                stage_episode_outputs(
                    staging_paths=staging_paths,
                    order_index=order_index,
                    run_name=args.run_name,
                    dataset_name=args.dataset,
                    episode=payload["episode"],
                    episode_started_at=payload["episode_started_at"],
                    episode_finished_at=payload["episode_finished_at"],
                    episode_elapsed_seconds=payload["episode_elapsed_seconds"],
                )
                episodes_by_order[order_index] = payload["episode"]
                completed_episode_count += 1
                maybe_log_episode(
                    index=completed_episode_count,
                    total=len(rows),
                    episode=payload["episode"],
                    episode_elapsed_seconds=payload["episode_elapsed_seconds"],
                    log_every=args.log_every,
                )
                continue
            if payload_type == "done":
                done_workers += 1
                continue
            if payload_type == "error":
                raise RuntimeError(
                    f"multi-gpu worker failed on gpu={payload.get('gpu_id')} "
                    f"worker_index={payload.get('worker_index')}: {payload.get('error')}\n"
                    f"{payload.get('traceback', '')}"
                )
            raise RuntimeError(f"未知 worker payload: {payload}")
    except Exception:
        for process in workers:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in workers:
            process.join()

    if len(episodes_by_order) != len(rows):
        raise RuntimeError(
            f"多 GPU 汇总结果不完整: expected={len(rows)} received={len(episodes_by_order)}"
        )
    merge_staged_episode_outputs(
        staging_paths=staging_paths,
        result_path=result_path,
        rollout_path=rollout_path,
        expected_count=len(rows),
    )
    for process in workers:
        if process.exitcode not in (0, None):
            raise RuntimeError(f"worker 退出异常，exitcode={process.exitcode}")
    return [episodes_by_order[order_index] for order_index in range(len(rows))]


def resolve_run_devices(args: argparse.Namespace) -> tuple[list[int], str]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids and args.llm_device == "auto":
        raise RuntimeError("--gpu-ids 模式下不允许使用 --llm-device auto。")
    if gpu_ids:
        return gpu_ids, f"cuda:{gpu_ids[0]}"
    return [], args.llm_device


def write_markdown_report(report_path: Path, payload: dict[str, Any]) -> None:
    summary = payload["eval_summary"]
    lines = [
        "# evaluation-v3-8 composite signal + lambda_q + budget-aware tail threshold escalation evaluation",
        "",
        f"- run_name: {payload['run_name']}",
        f"- dataset: {payload['dataset']}",
        f"- input_jsonl: {payload['input_jsonl']}",
        f"- subset_ids_path: {payload['subset_ids_path']}",
        f"- subset_id_count: {payload['subset_id_count']}",
        f"- total_input_count: {payload['total_input_count']}",
        f"- filtered_input_count: {payload['filtered_input_count']}",
        f"- selected_sample_count: {payload['selected_sample_count']}",
        f"- llm_device: {payload['llm_device']}",
        f"- gpu_ids: {payload['gpu_ids']}",
        f"- reset_seed_per_sample: {payload['reset_seed_per_sample']}",
        f"- checkpoint_zero: {payload['checkpoint_zero']}",
        f"- skip_judge: {payload['skip_judge']}",
        f"- calibration_only: {payload['calibration_only']}",
        f"- signal_calibration_json: {payload['signal_calibration_json']}",
        f"- question_lambda_calibration_json: {payload['question_lambda_calibration_json']}",
        f"- linear_down_up_question_lambda: {payload['linear_down_up_question_lambda']}",
        f"- linear_down_up_high_scale: {payload['linear_down_up_high_scale']}",
        f"- linear_down_up_low_scale: {payload['linear_down_up_low_scale']}",
        f"- linear_down_up_hold_high_tokens: {payload['linear_down_up_hold_high_tokens']}",
        f"- linear_down_up_down_end_tokens: {payload['linear_down_up_down_end_tokens']}",
        f"- linear_down_up_up_start_tokens: {payload['linear_down_up_up_start_tokens']}",
        f"- linear_down_up_up_end_tokens: {payload['linear_down_up_up_end_tokens']}",
        f"- cue_threshold: {payload['cue_threshold']}",
        f"- cue_gate_threshold: {payload['cue_gate_threshold']}",
        f"- min_tokens_before_cue: {payload['min_tokens_before_cue']}",
        f"- rethink_cue_text: {payload['rethink_cue_text']!r}",
        f"- rethink_cue_trigger_tokens: {payload['rethink_cue_trigger_tokens']}",
        f"- tail_soft_ratio: {payload['tail_soft_ratio']}",
        f"- tail_full_ratio: {payload['tail_full_ratio']}",
        f"- tail_exp_beta: {payload['tail_exp_beta']}",
        f"- tail_threshold_max: {payload['tail_threshold_max']}",
        f"- disable_custom_attention_patch: {payload['disable_custom_attention_patch']}",
        f"- entropy_upper_bound: {payload['entropy_upper_bound']}",
        f"- suppression_curve: {payload['suppression_curve']}",
        f"- power_exponent: {payload['power_exponent']}",
        f"- exp_beta: {payload['exp_beta']}",
        f"- suppression_scale: {payload['suppression_scale']}",
        f"- suppression_upper_bound: {payload['suppression_upper_bound']}",
        f"- presence_penalty: {payload['presence_penalty']}",
        f"- raw_history_size: {payload['raw_history_size']}",
        f"- k_smoothing_initial_rho: {payload['k_smoothing_initial_rho']}",
        f"- k_smoothing_late_rho: {payload['k_smoothing_late_rho']}",
        f"- k_smoothing_initial_checkpoints: {payload['k_smoothing_initial_checkpoints']}",
        f"- assistant_prefix_text: {payload['assistant_prefix_text']!r}",
        f"- cue_text: {payload['cue_text']!r}",
        f"- upgrade_newline_span: {payload['upgrade_newline_span']}",
        f"- downgrade_newline_span: {payload['downgrade_newline_span']}",
        f"- downgrade_confirm_windows: {payload['downgrade_confirm_windows']}",
        f"- post_downgrade_upgrade_freeze_newlines: {payload['post_downgrade_upgrade_freeze_newlines']}",
        f"- end_think_to_l1_lock: {payload['end_think_to_l1_lock']}",
        f"- end_think_mask_newlines: {payload['end_think_mask_newlines']}",
        f"- end_think_mask_tokens: {payload['end_think_mask_tokens']}",
        f"- end_think_mask_forever: {payload['end_think_mask_forever']}",
        f"- end_think_bridge_text: {payload['end_think_bridge_text']!r}",
        f"- started_at: {payload['timing']['started_at']}",
        f"- finished_at: {payload['timing']['finished_at']}",
        f"- elapsed_seconds: {payload['timing']['elapsed_seconds']:.2f}",
        "",
        "## Summary",
        f"- episode_count: {summary['episode_count']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- avg_total_generated_tokens: {summary['avg_total_generated_tokens']:.2f}",
        f"- avg_transition_count: {summary['avg_transition_count']:.2f}",
        f"- avg_increase_count: {summary['avg_increase_count']:.2f}",
        f"- avg_decrease_count: {summary['avg_decrease_count']:.2f}",
        f"- avg_final_suppression: {summary['avg_final_suppression']:.4f}",
        f"- avg_peak_suppression: {summary['avg_peak_suppression']:.4f}",
        f"- max_peak_suppression: {summary['max_peak_suppression']:.4f}",
        f"- avg_question_lambda_q: {summary['avg_question_lambda_q']:.4f}",
        f"- max_question_lambda_q: {summary['max_question_lambda_q']:.4f}",
        f"- cue_inserted_count: {summary['cue_inserted_count']}",
        f"- tail_signal_inserted_count: {summary['tail_signal_inserted_count']}",
        f"- judge_failed_count: {summary['judge_failed_count']}",
        f"- judge_skipped_count: {summary['judge_skipped_count']}",
        f"- transition_type_counts: {json.dumps(summary['transition_type_counts'], ensure_ascii=False)}",
        f"- review_consensus_counts: {json.dumps(summary['review_consensus_counts'], ensure_ascii=False)}",
        f"- decision_source_counts: {json.dumps(summary['decision_source_counts'], ensure_ascii=False)}",
    ]
    signal_calibration = payload.get("signal_calibration")
    if signal_calibration is not None:
        lines.extend(
            [
                "",
                "## Signal Calibration",
                f"- checkpoint_count: {signal_calibration['checkpoint_count']}",
                f"- entropy_q10_q90: ({signal_calibration['avg_entropy']['q10']}, {signal_calibration['avg_entropy']['q90']})",
                f"- delta_entropy_q90: {signal_calibration['positive_entropy_delta']['q90']}",
                f"- margin_q10_q90: ({signal_calibration['avg_margin']['q10']}, {signal_calibration['avg_margin']['q90']})",
                f"- recommended_signal_calibration: {json.dumps(signal_calibration['recommended_signal_calibration'], ensure_ascii=False)}",
            ]
        )
    ensure_parent(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    gpu_ids, resolved_llm_device = resolve_run_devices(args)
    dataset = DATASETS.get(args.dataset)
    input_jsonl = args.input_jsonl or (dataset.remote_input_path if dataset is not None else None)
    if input_jsonl is None:
        raise RuntimeError(f"dataset={args.dataset} 未内置 remote_input_path，必须显式传 --input-jsonl。")
    signal_calibration = load_signal_calibration(args.signal_calibration_json)
    question_lambda_calibration = load_question_lambda_calibration(args.question_lambda_calibration_json)
    if signal_calibration is None and not args.calibration_only:
        raise RuntimeError("evaluation-v3-8 非校准运行必须提供 --signal-calibration-json。")

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()
    set_seed(args.seed, include_cuda=False)

    output_dir = Path(args.output_dir)
    paths = materialize_eval_output_dirs(output_dir)
    all_rows = read_jsonl(Path(input_jsonl))
    subset_ids = load_subset_ids(args.subset_ids_path)
    base_rows = filter_rows_by_subset_ids(all_rows, subset_ids)
    rows = slice_rows(
        base_rows,
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
    )
    if not rows:
        raise RuntimeError("评测输入为空。")

    result_path = paths["results"] / "final_results.jsonl"
    rollout_path = paths["rollouts"] / "eval_rollout.jsonl"
    result_path.unlink(missing_ok=True)
    rollout_path.unlink(missing_ok=True)
    log_event(
        "evaluation_v3_8 started",
        run_name=args.run_name,
        dataset=args.dataset,
        input_jsonl=input_jsonl,
        subset_ids_path=args.subset_ids_path,
        subset_id_count=None if subset_ids is None else len(subset_ids),
        total_input_count=len(all_rows),
        filtered_input_count=len(base_rows),
        sample_count=len(rows),
        seed=args.seed,
        reset_seed_per_sample=args.reset_seed_per_sample,
        checkpoint_zero=args.checkpoint_zero,
        skip_judge=args.skip_judge,
        calibration_only=args.calibration_only,
        signal_calibration_json=args.signal_calibration_json,
        question_lambda_calibration_json=args.question_lambda_calibration_json,
        dynamic_question_lambda=args.dynamic_question_lambda,
        high_low_high_question_lambda=args.high_low_high_question_lambda,
        llm_device=resolved_llm_device,
        gpu_ids=gpu_ids,
        cue_threshold=args.cue_threshold,
        cue_gate_threshold=args.cue_gate_threshold,
        tail_soft_ratio=args.tail_soft_ratio,
        tail_full_ratio=args.tail_full_ratio,
        tail_exp_beta=args.tail_exp_beta,
        tail_threshold_max=args.tail_threshold_max,
        entropy_upper_bound=args.entropy_upper_bound,
        suppression_curve=args.suppression_curve,
        power_exponent=args.power_exponent,
        exp_beta=args.exp_beta,
        suppression_scale=args.suppression_scale,
        suppression_upper_bound=args.suppression_upper_bound,
        raw_history_size=args.raw_history_size,
        k_smoothing_initial_rho=args.k_smoothing_initial_rho,
        k_smoothing_late_rho=args.k_smoothing_late_rho,
        k_smoothing_initial_checkpoints=args.k_smoothing_initial_checkpoints,
        upgrade_newline_span=args.upgrade_newline_span,
        downgrade_newline_span=args.downgrade_newline_span,
    )
    if len(gpu_ids) > 1:
        episodes = run_multi_gpu(
            args=args,
            rows=rows,
            paths=paths,
            gpu_ids=gpu_ids,
        )
    else:
        episodes = run_single_device(
            args=args,
            rows=rows,
            paths=paths,
            llm_device=resolved_llm_device,
        )

    summary = summarize_episodes(episodes)
    signal_calibration_report = (
        build_signal_calibration_report(
            episodes,
            run_name=args.run_name,
            dataset=args.dataset,
            selected_sample_count=len(rows),
        )
        if args.calibration_only
        else None
    )
    finished_at = datetime.now().isoformat(timespec="seconds")
    elapsed_seconds = time.perf_counter() - start_time
    report_payload = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "input_jsonl": input_jsonl,
        "subset_ids_path": args.subset_ids_path,
        "subset_id_count": None if subset_ids is None else len(subset_ids),
        "total_input_count": len(all_rows),
        "filtered_input_count": len(base_rows),
        "selected_sample_count": len(rows),
        "llm_device": resolved_llm_device,
        "gpu_ids": gpu_ids,
        "reset_seed_per_sample": args.reset_seed_per_sample,
        "checkpoint_zero": args.checkpoint_zero,
        "skip_judge": args.skip_judge,
        "calibration_only": args.calibration_only,
        "signal_calibration_json": args.signal_calibration_json,
        "question_lambda_calibration_json": args.question_lambda_calibration_json,
        "dynamic_question_lambda": args.dynamic_question_lambda,
        "dynamic_lambda_probe_checkpoints": args.dynamic_lambda_probe_checkpoints,
        "dynamic_lambda_signal_lo": args.dynamic_lambda_signal_lo,
        "dynamic_lambda_signal_hi": args.dynamic_lambda_signal_hi,
        "dynamic_lambda_probe_scale_min": args.dynamic_lambda_probe_scale_min,
        "dynamic_lambda_probe_scale_max": args.dynamic_lambda_probe_scale_max,
        "dynamic_lambda_late_token_start": args.dynamic_lambda_late_token_start,
        "dynamic_lambda_late_token_end": args.dynamic_lambda_late_token_end,
        "dynamic_lambda_late_scale_start": args.dynamic_lambda_late_scale_start,
        "dynamic_lambda_late_scale_max": args.dynamic_lambda_late_scale_max,
        "high_low_high_question_lambda": args.high_low_high_question_lambda,
        "high_low_high_early_scale": args.high_low_high_early_scale,
        "high_low_high_early_min_tokens": args.high_low_high_early_min_tokens,
        "high_low_high_early_min_checkpoints": args.high_low_high_early_min_checkpoints,
        "high_low_high_middle_scale": args.high_low_high_middle_scale,
        "high_low_high_late_token_start": args.high_low_high_late_token_start,
        "high_low_high_late_token_end": args.high_low_high_late_token_end,
        "high_low_high_late_scale_max": args.high_low_high_late_scale_max,
        "high_low_high_adaptive_early": args.high_low_high_adaptive_early,
        "high_low_high_adaptive_max_tokens": args.high_low_high_adaptive_max_tokens,
        "high_low_high_adaptive_window": args.high_low_high_adaptive_window,
        "high_low_high_adaptive_avg_signal_threshold": args.high_low_high_adaptive_avg_signal_threshold,
        "high_low_high_adaptive_max_signal_threshold": args.high_low_high_adaptive_max_signal_threshold,
        "high_low_high_adaptive_max_raw_threshold": args.high_low_high_adaptive_max_raw_threshold,
        "high_low_high_adaptive_raw_delta_threshold": args.high_low_high_adaptive_raw_delta_threshold,
        "linear_down_up_question_lambda": args.linear_down_up_question_lambda,
        "linear_down_up_high_scale": args.linear_down_up_high_scale,
        "linear_down_up_low_scale": args.linear_down_up_low_scale,
        "linear_down_up_hold_high_tokens": args.linear_down_up_hold_high_tokens,
        "linear_down_up_down_end_tokens": args.linear_down_up_down_end_tokens,
        "linear_down_up_up_start_tokens": args.linear_down_up_up_start_tokens,
        "linear_down_up_up_end_tokens": args.linear_down_up_up_end_tokens,
        "cue_text": args.cue_text,
        "cue_threshold": args.cue_threshold,
        "cue_gate_threshold": args.cue_gate_threshold,
        "min_tokens_before_cue": args.min_tokens_before_cue,
        "rethink_cue_text": args.rethink_cue_text,
        "rethink_cue_trigger_tokens": args.rethink_cue_trigger_tokens,
        "tail_soft_ratio": args.tail_soft_ratio,
        "tail_full_ratio": args.tail_full_ratio,
        "tail_exp_beta": args.tail_exp_beta,
        "tail_threshold_max": args.tail_threshold_max,
        "disable_custom_attention_patch": args.disable_custom_attention_patch,
        "tail_signal_trigger_ratio": args.tail_signal_trigger_ratio,
        "tail_signal_text": args.tail_signal_text,
        "tail_signal_wait_token_limit": args.tail_signal_wait_token_limit,
        "tail_signal_next_checkpoint_only": args.tail_signal_next_checkpoint_only,
        "entropy_upper_bound": args.entropy_upper_bound,
        "suppression_curve": args.suppression_curve,
        "power_exponent": args.power_exponent,
        "exp_beta": args.exp_beta,
        "suppression_scale": args.suppression_scale,
        "suppression_upper_bound": args.suppression_upper_bound,
        "presence_penalty": args.presence_penalty,
        "assistant_prefix_text": args.assistant_prefix_text,
        "assistant_prefix_checkpoint": not args.disable_assistant_prefix_checkpoint,
        "disable_assistant_prefix_checkpoint": args.disable_assistant_prefix_checkpoint,
        "initial_suppression": args.initial_suppression,
        "raw_history_size": args.raw_history_size,
        "k_smoothing_initial_rho": args.k_smoothing_initial_rho,
        "k_smoothing_late_rho": args.k_smoothing_late_rho,
        "k_smoothing_initial_checkpoints": args.k_smoothing_initial_checkpoints,
        "upgrade_newline_span": args.upgrade_newline_span,
        "downgrade_newline_span": args.downgrade_newline_span,
        "downgrade_confirm_windows": args.downgrade_confirm_windows,
        "post_downgrade_upgrade_freeze_newlines": args.post_downgrade_upgrade_freeze_newlines,
        "end_think_to_l1_lock": args.end_think_to_l1_lock,
        "end_think_mask_newlines": args.end_think_mask_newlines,
        "end_think_mask_tokens": args.end_think_mask_tokens,
        "end_think_mask_forever": args.end_think_mask_forever,
        "end_think_bridge_text": args.end_think_bridge_text,
        "debug_runtime_events": args.debug_runtime_events,
        "signal_calibration": signal_calibration_report,
        "eval_summary": summary,
        "config": vars(args),
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
        },
    }
    write_json(paths["reports"] / "eval_summary.json", report_payload)
    write_markdown_report(paths["reports"] / "eval_summary.md", report_payload)
    if signal_calibration_report is not None:
        write_json(paths["reports"] / "signal_calibration.json", signal_calibration_report)
        write_json(paths["metrics"] / "signal_calibration.json", signal_calibration_report)
    write_json(
        paths["metrics"] / "run_manifest.json",
        {
            "run_name": args.run_name,
            "dataset": args.dataset,
            "input_jsonl": input_jsonl,
            "subset_ids_path": args.subset_ids_path,
            "subset_id_count": None if subset_ids is None else len(subset_ids),
            "total_input_count": len(all_rows),
            "filtered_input_count": len(base_rows),
            "selected_sample_count": len(rows),
            "model": args.model,
            "tokenizer_path": args.tokenizer_path,
            "summary_json": args.summary_json,
            "llm_device": resolved_llm_device,
            "gpu_ids": gpu_ids,
            "reset_seed_per_sample": args.reset_seed_per_sample,
            "checkpoint_zero": args.checkpoint_zero,
            "assistant_prefix_checkpoint": not args.disable_assistant_prefix_checkpoint,
            "disable_assistant_prefix_checkpoint": args.disable_assistant_prefix_checkpoint,
            "initial_suppression": args.initial_suppression,
            "skip_judge": args.skip_judge,
            "calibration_only": args.calibration_only,
            "signal_calibration_json": args.signal_calibration_json,
            "question_lambda_calibration_json": args.question_lambda_calibration_json,
            "dynamic_question_lambda": args.dynamic_question_lambda,
            "dynamic_lambda_probe_checkpoints": args.dynamic_lambda_probe_checkpoints,
            "dynamic_lambda_signal_lo": args.dynamic_lambda_signal_lo,
            "dynamic_lambda_signal_hi": args.dynamic_lambda_signal_hi,
            "dynamic_lambda_probe_scale_min": args.dynamic_lambda_probe_scale_min,
            "dynamic_lambda_probe_scale_max": args.dynamic_lambda_probe_scale_max,
            "dynamic_lambda_late_token_start": args.dynamic_lambda_late_token_start,
            "dynamic_lambda_late_token_end": args.dynamic_lambda_late_token_end,
            "dynamic_lambda_late_scale_start": args.dynamic_lambda_late_scale_start,
            "dynamic_lambda_late_scale_max": args.dynamic_lambda_late_scale_max,
            "high_low_high_question_lambda": args.high_low_high_question_lambda,
            "high_low_high_early_scale": args.high_low_high_early_scale,
            "high_low_high_early_min_tokens": args.high_low_high_early_min_tokens,
            "high_low_high_early_min_checkpoints": args.high_low_high_early_min_checkpoints,
            "high_low_high_middle_scale": args.high_low_high_middle_scale,
            "high_low_high_late_token_start": args.high_low_high_late_token_start,
            "high_low_high_late_token_end": args.high_low_high_late_token_end,
            "high_low_high_late_scale_max": args.high_low_high_late_scale_max,
            "high_low_high_adaptive_early": args.high_low_high_adaptive_early,
            "high_low_high_adaptive_max_tokens": args.high_low_high_adaptive_max_tokens,
            "high_low_high_adaptive_window": args.high_low_high_adaptive_window,
            "high_low_high_adaptive_avg_signal_threshold": args.high_low_high_adaptive_avg_signal_threshold,
            "high_low_high_adaptive_max_signal_threshold": args.high_low_high_adaptive_max_signal_threshold,
            "high_low_high_adaptive_max_raw_threshold": args.high_low_high_adaptive_max_raw_threshold,
            "high_low_high_adaptive_raw_delta_threshold": args.high_low_high_adaptive_raw_delta_threshold,
            "linear_down_up_question_lambda": args.linear_down_up_question_lambda,
            "linear_down_up_high_scale": args.linear_down_up_high_scale,
            "linear_down_up_low_scale": args.linear_down_up_low_scale,
            "linear_down_up_hold_high_tokens": args.linear_down_up_hold_high_tokens,
            "linear_down_up_down_end_tokens": args.linear_down_up_down_end_tokens,
            "linear_down_up_up_start_tokens": args.linear_down_up_up_start_tokens,
            "linear_down_up_up_end_tokens": args.linear_down_up_up_end_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "assistant_prefix_text": args.assistant_prefix_text,
            "assistant_prefix_checkpoint": not args.disable_assistant_prefix_checkpoint,
            "disable_assistant_prefix_checkpoint": args.disable_assistant_prefix_checkpoint,
            "initial_suppression": args.initial_suppression,
            "cue_text": args.cue_text,
            "cue_threshold": args.cue_threshold,
            "cue_gate_threshold": args.cue_gate_threshold,
            "min_tokens_before_cue": args.min_tokens_before_cue,
            "rethink_cue_text": args.rethink_cue_text,
            "rethink_cue_trigger_tokens": args.rethink_cue_trigger_tokens,
            "tail_soft_ratio": args.tail_soft_ratio,
            "tail_full_ratio": args.tail_full_ratio,
            "tail_exp_beta": args.tail_exp_beta,
            "tail_threshold_max": args.tail_threshold_max,
        "disable_custom_attention_patch": args.disable_custom_attention_patch,
            "tail_signal_trigger_ratio": args.tail_signal_trigger_ratio,
            "tail_signal_text": args.tail_signal_text,
            "tail_signal_wait_token_limit": args.tail_signal_wait_token_limit,
            "tail_signal_next_checkpoint_only": args.tail_signal_next_checkpoint_only,
            "entropy_upper_bound": args.entropy_upper_bound,
            "suppression_curve": args.suppression_curve,
            "power_exponent": args.power_exponent,
            "exp_beta": args.exp_beta,
            "suppression_scale": args.suppression_scale,
            "suppression_upper_bound": args.suppression_upper_bound,
            "raw_history_size": args.raw_history_size,
            "k_smoothing_initial_rho": args.k_smoothing_initial_rho,
            "k_smoothing_late_rho": args.k_smoothing_late_rho,
            "k_smoothing_initial_checkpoints": args.k_smoothing_initial_checkpoints,
            "upgrade_newline_span": args.upgrade_newline_span,
            "downgrade_newline_span": args.downgrade_newline_span,
            "downgrade_confirm_windows": args.downgrade_confirm_windows,
            "post_downgrade_upgrade_freeze_newlines": args.post_downgrade_upgrade_freeze_newlines,
            "end_think_to_l1_lock": args.end_think_to_l1_lock,
            "end_think_mask_newlines": args.end_think_mask_newlines,
            "end_think_mask_tokens": args.end_think_mask_tokens,
            "end_think_mask_forever": args.end_think_mask_forever,
            "end_think_bridge_text": args.end_think_bridge_text,
            "debug_runtime_events": args.debug_runtime_events,
            "signal_calibration": signal_calibration_report,
            "timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": elapsed_seconds,
            },
        },
    )
    log_event(
        "evaluation_v3_8 finished",
        run_name=args.run_name,
        episode_count=summary["episode_count"],
        accuracy=summary["accuracy"],
        avg_total_generated_tokens=summary["avg_total_generated_tokens"],
        avg_final_suppression=summary["avg_final_suppression"],
        avg_peak_suppression=summary["avg_peak_suppression"],
        cue_inserted_count=summary["cue_inserted_count"],
        tail_signal_inserted_count=summary["tail_signal_inserted_count"],
        judge_skipped_count=summary["judge_skipped_count"],
        elapsed_seconds=round(float(elapsed_seconds), 3),
    )


if __name__ == "__main__":
    main()
