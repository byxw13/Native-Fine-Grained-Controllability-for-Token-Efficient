from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from remote_scripts.mmmu_qwen3vl.scripts import (
    run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline,
)


DEFAULT_METHOD_PREFIX = "<think>\n\n</think>\n\nAlright"


def _fsync_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_fsync_trace(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    for row in baseline.read_jsonl(path):
        sample_id = row.get("id")
        if sample_id is not None:
            result.add(str(sample_id))
    return result


def _read_trace_delta(path: Path | None, offset: int) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if path is None or not path.exists():
        return offset, grouped
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[str(row.get("request_id"))].append(row)
        return handle.tell(), grouped


def _trace_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints = [row for row in events if row.get("event") == "checkpoint"]
    forced = [row for row in events if str(row.get("event", "")).startswith("forced_")]
    k_values = [
        float(row["target_suppression"])
        for row in checkpoints
        if row.get("target_suppression") is not None
    ]
    return {
        "checkpoint_count": len(checkpoints),
        "forced_event_count": len(forced),
        "k_sequence": k_values,
        "increase_count": sum(row.get("transition_type") == "increase" for row in checkpoints),
        "decrease_count": sum(row.get("transition_type") == "decrease" for row in checkpoints),
        "end_think_intercept_count": sum(row.get("event") == "closure_enter" for row in events),
    }


def _error_row(sample: dict[str, Any], args, error: str) -> dict[str, Any]:
    rank = args.rank_offset + args.shard_index
    return {
        **baseline.error_row(sample, args, error, rank, args.seed),
        "method": "aris_dynamic_vllm_final" if args.mode == "dynamic" else args.mode,
        "tokenizer": args.tokenizer,
        "sample_seed": args.seed,
        "top_k": args.top_k,
        "execution_group": args.execution_group,
    }


def run(args: argparse.Namespace) -> None:
    method_prefix = args.method_prefix if args.mode != "native" else None
    if args.mode != "native" and not method_prefix:
        raise ValueError("--method-prefix must be non-empty outside native mode")
    finalize_dynamic = None
    initial_request_metadata = None
    if args.mode == "dynamic":
        from .engine_patch import (
            finalize_llm_dynamic_states,
            initial_request_suppression_metadata,
            install_dynamic_aris_patch,
        )

        install_dynamic_aris_patch(require_native_fused=True)
        finalize_dynamic = finalize_llm_dynamic_states
        initial_request_metadata = initial_request_suppression_metadata

    processor_source = args.model_path if args.mode == "native" else args.tokenizer
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    tokenizer = processor.tokenizer
    image_token_id = baseline.get_image_token_id(processor)

    if args.global_end_index is not None and args.global_end_index <= args.global_start_index:
        raise ValueError("--global-end-index must be greater than --global-start-index")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")

    candidate_rows = [
        row
        for row in baseline.read_jsonl(args.manifest)
        if int(row.get("global_index", -1)) >= args.global_start_index
        and (
            args.global_end_index is None
            or int(row.get("global_index", -1)) < args.global_end_index
        )
    ]
    rows = [
        row
        for row in candidate_rows
        if int(row.get("global_index", 0)) % args.shard_count == args.shard_index
    ]
    if args.indices:
        selected = {int(value) for value in args.indices.split(",") if value.strip()}
        rows = [row for row in rows if int(row.get("global_index", -1)) in selected]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = _completed_ids(args.output) if args.resume else set()
    rows = [row for row in rows if str(row.get("id")) not in completed]

    engine_kwargs = {}
    if args.mode == "dynamic":
        engine_kwargs["worker_extension_cls"] = (
            "experiment_src.qwen3vl_vllm_dynamic_final.worker_extension.DynamicArisWorkerExtension"
        )
    llm = LLM(
        model=args.model_path,
        tokenizer=processor_source,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        skip_special_tokens=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args) | {"manifest": str(args.manifest), "output": str(args.output)},
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "vllm": vllm.__version__,
                },
                "prompt_style": baseline.PROMPT_STYLE,
                "method_prefix": method_prefix,
                "dynamic_config": os.environ.get("ARIS_DYNAMIC_CONFIG_JSON"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    trace_path = Path(args.trace_path) if args.trace_path else None
    # A resumed engine restarts vLLM request ids at zero.  Ignore trace bytes
    # from the previous process so old request-id events cannot be attached to
    # newly generated items.
    trace_offset = trace_path.stat().st_size if trace_path is not None and trace_path.exists() else 0
    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            requests: list[dict[str, Any]] = []
            metas: list[dict[str, Any]] = []
            prepared: list[dict[str, Any]] = []
            for sample in batch:
                try:
                    request, meta = baseline.build_vllm_request(processor, sample, image_token_id)
                    prefix_count = str(request["prompt"]).count(method_prefix)
                    if args.mode != "native" and prefix_count != 1:
                        raise RuntimeError(f"method prefix count is {prefix_count}, expected exactly one")
                    if args.mode == "dynamic":
                        request["aris_suppression"] = initial_request_metadata()
                    requests.append(request)
                    metas.append(meta | {"method_prefix_count": prefix_count})
                    prepared.append(sample)
                except Exception as exc:
                    _fsync_jsonl(handle, _error_row(sample, args, f"{type(exc).__name__}: {exc}"))
            if not requests:
                continue

            started = time.perf_counter()
            finalize_reports: list[dict[str, Any]] = []
            try:
                outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
                if finalize_dynamic is not None:
                    finalize_reports = finalize_dynamic(llm, reason="evaluation_batch_return")
            except Exception as exc:
                for sample in prepared:
                    _fsync_jsonl(handle, _error_row(sample, args, f"{type(exc).__name__}: {exc}"))
                continue
            elapsed = time.perf_counter() - started
            trace_offset, trace_by_req = _read_trace_delta(trace_path, trace_offset)
            # Persist the complete trajectory before exposing the per-item
            # prediction row to the streaming judge.
            _fsync_path(trace_path)

            total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            for sample, meta, output in zip(prepared, metas, outputs):
                completion = output.outputs[0]
                token_ids = [int(value) for value in completion.token_ids]
                raw_output = completion.text
                prediction, prediction_source, parseable = baseline.extract_prediction(
                    raw_output,
                    sample.get("options") or [],
                    str(sample.get("ground_truth") or ""),
                )
                correct = baseline.is_correct_prediction(
                    prediction,
                    str(sample.get("ground_truth") or ""),
                    sample.get("options") or [],
                )
                thinking_tokens, thinking_status = baseline.count_reasoning_tokens(tokenizer, raw_output)
                events = trace_by_req.get(str(output.request_id), [])
                event_names = {str(event.get("event")) for event in events}
                trajectory_errors = []
                if args.mode == "dynamic":
                    for required_event in ("request_init", "checkpoint", "request_finish"):
                        if required_event not in event_names:
                            trajectory_errors.append(f"missing {required_event}")
                finish_event = {
                    "time": time.time(),
                    "event": "evaluation_item_finish",
                    "request_id": str(output.request_id),
                    "sample_id": str(sample.get("id")),
                    "global_index": int(sample.get("global_index")),
                    "output_tokens": len(token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
                _append_fsync_trace(trace_path, finish_event)
                events.append(finish_event)
                allocated = elapsed * (len(token_ids) / max(total_output_tokens, 1))
                row = {
                    "id": sample.get("id"),
                    "trajectory_id": baseline.trajectory_id(sample.get("id"), args.seed),
                    "global_index": sample.get("global_index"),
                    "subject": sample.get("subject"),
                    "question": sample.get("question"),
                    "options": sample.get("options") or [],
                    "ground_truth": sample.get("ground_truth"),
                    "prediction": prediction,
                    "prediction_source": prediction_source,
                    "parseable": parseable,
                    "is_correct": correct,
                    "raw_output": raw_output,
                    "full_assistant_text": (method_prefix + raw_output) if method_prefix is not None else raw_output,
                    "output_token_ids": token_ids,
                    "output_tokens": len(token_ids),
                    "total_tokens": int(meta.get("input_tokens") or 0) + len(token_ids),
                    "thinking_tokens": thinking_tokens,
                    "thinking_status": thinking_status,
                    "input_tokens": int(meta.get("input_tokens") or 0),
                    "visual_tokens": int(meta.get("visual_tokens") or 0),
                    "processed_images": meta.get("processed_images") or [],
                    "image_count": int(meta.get("image_count") or 0),
                    "image_occurrence_count": int(meta.get("image_occurrence_count") or 0),
                    "image_occurrence_order": meta.get("image_occurrence_order") or [],
                    "referenced_image_indices": meta.get("referenced_image_indices") or [],
                    "unused_image_indices": meta.get("unused_image_indices") or [],
                    "rendered_prompt_template": meta.get("rendered_prompt_template"),
                    "prompt_style": meta.get("prompt_style"),
                    "method_prefix_count": meta.get("method_prefix_count"),
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "sample_seed": args.seed,
                    "repetition_penalty": args.repetition_penalty,
                    "generate_time_seconds": allocated,
                    "time_seconds": allocated,
                    "batch_generate_wall_seconds": elapsed,
                    "tokens_per_second": len(token_ids) / allocated if allocated > 0 else None,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "hit_max_new_tokens": bool(
                        completion.finish_reason == "length" or len(token_ids) >= args.max_new_tokens
                    ),
                    "request_id": output.request_id,
                    "rank": args.rank_offset + args.shard_index,
                    "execution_group": args.execution_group,
                    "engine": "vllm",
                    "method": "aris_dynamic_vllm_final" if args.mode == "dynamic" else args.mode,
                    "dynamic_events": events,
                    "dynamic_summary": _trace_summary(events),
                    "trajectory_audit": {
                        "pass": not trajectory_errors,
                        "errors": trajectory_errors,
                        "event_count": len(events),
                        "persistence_order": "worker_trace_fsync -> evaluation_item_finish_fsync -> prediction_row_fsync",
                    },
                    "suppression_hook_evidence": {
                        "native_stats_path": os.environ.get("ARIS_VLLM_V1_STATS_PATH"),
                        "dynamic_stats_path": os.environ.get("ARIS_DYNAMIC_STATS_PATH"),
                        "trace_path": str(trace_path) if trace_path else None,
                        "worker_finalize_reports": finalize_reports,
                    },
                    "error": None
                    if not trajectory_errors
                    else "trajectory_audit_failed: " + "; ".join(trajectory_errors),
                }
                _fsync_jsonl(handle, row)
                print(
                    json.dumps(
                        {
                            "persisted_id": row["id"],
                            "global_index": row["global_index"],
                            "output_tokens": row["output_tokens"],
                            "checkpoint_count": row["dynamic_summary"]["checkpoint_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if trace_path is not None and trace_path.exists():
                trace_offset = trace_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), required=True)
    parser.add_argument("--method-prefix", default=DEFAULT_METHOD_PREFIX)
    parser.add_argument("--model-path", default="/path/to/aris_data/Qwen3-VL-4B-Thinking")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-path")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--rank-offset", type=int, default=0)
    parser.add_argument("--execution-group", default="default")
    parser.add_argument("--global-start-index", type=int, default=0)
    parser.add_argument("--global-end-index", type=int)
    parser.add_argument("--indices", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
