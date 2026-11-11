"""Continuous-batching MMMU evaluation with per-item durable persistence.

This frontend keeps multiple independent requests live in one vLLM engine.
Each completed request is finalized through a request-scoped worker RPC, then
its complete trace and prediction row are fsynced before the streaming judge
can observe that row.  Other live controller states are never finalized.
"""

from __future__ import annotations

import asyncio
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
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind

from experiment_src.qwen35_vllm_dynamic_final import mmmu_dynamic_vllm as base
from experiment_src.qwen35_vllm_dynamic_final import (
    vendor as _vendor_package,  # noqa: F401
)
from experiment_src.qwen35_vllm_dynamic_final.baseline_snapshot import (
    run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline,
)
from experiment_src.qwen35_vllm_dynamic_final.prompting import (
    build_qwen35_request,
)
from experiment_src.qwen35_vllm_dynamic_final.vendor import (
    aris_vllm_v1_suppression as native,
)

from . import engine_patch


def _tagged_request_id(base_id: str, metadata: dict[str, Any]) -> str:
    tag = native._format_request_id_tag(
        native.RequestSuppressionMetadata(
            end_think_index=int(metadata["end_think_index"]),
            enabled=bool(metadata["enabled"]),
            budget=(
                None
                if metadata.get("budget") is None
                else float(metadata["budget"])
            ),
        )
    )
    return str(base_id) + tag


def _assert_runtime_end_think_token_ids(tokenizer) -> list[int]:
    actual = [
        int(value)
        for value in tokenizer.encode("</think>", add_special_tokens=False)
    ]
    if not actual:
        raise RuntimeError("tokenizer produced no token ids for </think>")
    raw = os.environ.get("ARIS_VLLM_V1_END_THINK_TOKEN_IDS", "")
    configured = [int(value) for value in raw.split(",") if value.strip()]
    if configured != actual:
        raise RuntimeError(
            "ARIS_VLLM_V1_END_THINK_TOKEN_IDS does not match the active tokenizer: "
            f"configured={configured}, tokenizer={actual}"
        )
    return actual


def _linear_first_forward_evidenced(
    finalize_reports: list[dict[str, Any]],
) -> bool:
    for report in finalize_reports:
        linear = report.get("linear_stats") or {}
        if (
            int(linear.get("first_forward_budget_assertions", 0)) >= 1
            and int(linear.get("first_forward_budget_assertion_failures", -1)) == 0
            and int(linear.get("decode_integrated_calls", 0)) > 0
            and not bool(report.get("request_remaining", True))
        ):
            return True
    return False


def _result_row(
    *,
    sample: dict[str, Any],
    meta: dict[str, Any],
    output,
    tokenizer,
    args,
    elapsed: float,
    events: list[dict[str, Any]],
    trace_request_id: str | None,
    finalize_reports: list[dict[str, Any]],
    trajectory_errors: list[str],
    aggregate_tokens: int,
    aggregate_wall: float,
) -> dict[str, Any]:
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
    thinking_tokens, thinking_status = baseline.count_reasoning_tokens(
        tokenizer,
        raw_output,
    )
    return {
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
        "full_assistant_text": str(meta["assistant_generation_prefix"]) + raw_output,
        "assistant_generation_prefix": meta["assistant_generation_prefix"],
        "template_enable_thinking": meta["template_enable_thinking"],
        "assistant_cue": meta["assistant_cue"],
        "end_think_token_ids": meta.get("end_think_token_ids") or [],
        "end_think_token_count": meta.get("end_think_token_count"),
        "end_think_token_index": meta.get("end_think_token_index"),
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
        "image_order": "full_rendered_prompt_placeholder_order",
        "method_prefix_count": meta.get("method_prefix_count"),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "sample_seed": args.seed,
        "repetition_penalty": args.repetition_penalty,
        "generate_time_seconds": elapsed,
        "time_seconds": elapsed,
        "request_wall_seconds": elapsed,
        "tokens_per_second": len(token_ids) / elapsed if elapsed > 0 else None,
        "rolling_engine_tokens": aggregate_tokens,
        "rolling_engine_wall_seconds": aggregate_wall,
        "rolling_engine_tokens_per_second": (
            aggregate_tokens / aggregate_wall if aggregate_wall > 0 else None
        ),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
        "hit_max_new_tokens": bool(
            completion.finish_reason == "length"
            or len(token_ids) >= args.max_new_tokens
        ),
        "request_id": output.request_id,
        "trace_request_id": trace_request_id,
        "rank": args.rank_offset + args.shard_index,
        "execution_group": args.execution_group,
        "engine": "vllm_async_continuous_batching",
        "method": "aris_dynamic_vllm_final",
        "dynamic_events": events,
        "dynamic_summary": base._trace_summary(events),
        "trajectory_audit": {
            "pass": not trajectory_errors,
            "errors": trajectory_errors,
            "event_count": len(events),
            "persistence_order": (
                "request_worker_trace_fsync -> evaluation_item_finish_fsync "
                "-> prediction_row_fsync"
            ),
        },
        "suppression_hook_evidence": {
            "native_stats_path": os.environ.get("ARIS_VLLM_V1_STATS_PATH"),
            "dynamic_stats_path": os.environ.get("ARIS_DYNAMIC_STATS_PATH"),
            "trace_path": args.trace_path,
            "worker_finalize_reports": finalize_reports,
        },
        "error": (
            None
            if not trajectory_errors
            else "trajectory_audit_failed: " + "; ".join(trajectory_errors)
        ),
    }


async def run_async(args) -> None:
    if args.mode != "dynamic":
        raise ValueError("the continuous-batching frontend currently requires --mode dynamic")
    if not args.method_cue:
        raise ValueError("--method-cue must be non-empty")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be positive")
    if args.max_num_seqs < args.max_concurrency:
        raise ValueError("--max-num-seqs must be >= --max-concurrency")

    # Parent verification catches a wrong FA2 ABI before model loading.  Every
    # worker independently installs the same patch via worker_extension_cls.
    engine_patch.install_dynamic_aris_patch(require_native_fused=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    runtime_end_think_token_ids = _assert_runtime_end_think_token_ids(tokenizer)
    image_token_id = baseline.get_image_token_id(processor)

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
        rows = [
            row
            for row in rows
            if int(row.get("global_index", -1)) in selected
        ]
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = base._completed_ids(args.output) if args.resume else set()
    rows = [row for row in rows if str(row.get("id")) not in completed]

    # A resume run with no pending rows is a true idempotent no-op.  Do not
    # start an AsyncLLM engine in that case: vLLM would initialize workers and
    # then wait for an RPC despite there being no requests, which can leave a
    # smoke/evaluation process alive indefinitely.  The caller compares the
    # durable output snapshot to prove that this path added no rows.
    if not rows:
        print(
            json.dumps(
                {
                    "resume_noop": True,
                    "completed_rows": len(completed),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Qwen3.5-2B otherwise may select the unified attention backend for
        # decode, bypassing the audited FlashAttention-2 Full suppression hook.
        # The 9B regression gate uses the same explicit backend contract.
        attention_backend="FLASH_ATTN",
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        max_num_seqs=args.max_num_seqs,
        worker_extension_cls=(
            "experiment_src.qwen35_linear_full_dynamic_final.worker_extension."
            "DynamicArisLinearWorkerExtension"
        ),
        # Fixed compacted-stream convolution replacement must see each
        # request's complete prompt once.  Continuous batching remains active
        # across requests, but vLLM must not split one request's prefill into
        # multiple recurrent transitions.
        enable_chunked_prefill=False,
        enforce_eager=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        skip_special_tokens=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config_path.write_text(
        json.dumps(
            {
                "args": vars(args)
                | {"manifest": str(args.manifest), "output": str(args.output)},
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "vllm": vllm.__version__,
                },
                "prompt_style": baseline.PROMPT_STYLE,
                "template_enable_thinking": False,
                "method_cue": args.method_cue,
                "dynamic_config": os.environ.get("ARIS_DYNAMIC_CONFIG_JSON"),
                "persistence": "request-scoped immediate fsync",
                "scheduler": "AsyncLLM continuous batching",
                "chunked_prefill": False,
                "chunked_prefill_reason": (
                    "fixed rank-3 compacted-stream conv replacement requires "
                    "one complete prefill transition per request"
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    trace_path = Path(args.trace_path) if args.trace_path else None
    trace_offset = (
        trace_path.stat().st_size
        if trace_path is not None and trace_path.exists()
        else 0
    )
    trace_pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    persist_lock = asyncio.Lock()
    run_started = time.perf_counter()
    aggregate_tokens = 0

    async def persist_error(sample: dict[str, Any], error: str) -> None:
        async with persist_lock:
            with args.output.open("a", encoding="utf-8") as handle:
                base._fsync_jsonl(handle, base._error_row(sample, args, error))

    async def generate_one(
        sample: dict[str, Any],
        request: dict[str, Any],
        meta: dict[str, Any],
        request_id: str,
    ) -> None:
        nonlocal trace_offset, aggregate_tokens
        started = time.perf_counter()
        final_output = None
        try:
            async for output in engine.generate(
                request,
                sampling,
                request_id=request_id,
            ):
                final_output = output
            if final_output is None or not bool(final_output.finished):
                raise RuntimeError("AsyncLLM returned no finished output")
            elapsed = time.perf_counter() - started
            # Serialize finalization and trace reads.  The worker RPC closes
            # only this request; other in-flight states continue decoding.
            async with persist_lock:
                finalize_reports = await engine_patch.finalize_async_dynamic_request(
                    engine,
                    request_id=request_id,
                    reason="async_evaluation_request_return",
                )
                new_offset, grouped = base._read_trace_delta(
                    trace_path,
                    trace_offset,
                )
                trace_offset = new_offset
                for trace_id, new_events in grouped.items():
                    trace_pending[trace_id].extend(new_events)
                base._fsync_path(trace_path)

                resolved_worker_ids = {
                    str(report["resolved_request_id"])
                    for report in finalize_reports
                    if report.get("resolved_request_id")
                }
                if len(resolved_worker_ids) > 1:
                    raise RuntimeError(
                        "workers resolved different request ids: "
                        f"{sorted(resolved_worker_ids)!r}"
                    )
                trace_request_id = (
                    next(iter(resolved_worker_ids))
                    if len(resolved_worker_ids) == 1
                    else None
                )
                if trace_request_id not in trace_pending:
                    trace_request_id = base._resolve_trace_request_id(
                        str(final_output.request_id),
                        trace_pending,
                    )
                events = (
                    []
                    if trace_request_id is None
                    else trace_pending.pop(trace_request_id, [])
                )
                event_names = {str(event.get("event")) for event in events}
                trajectory_errors = [
                    f"missing {event}"
                    for event in ("request_init", "request_finish")
                    if event not in event_names
                ]
                if base._env_enabled("ARIS_FULL_ENABLE", True):
                    if "first_forward_budget_asserted" not in event_names:
                        trajectory_errors.append(
                            "missing first_forward_budget_asserted"
                        )
                if base._env_enabled("ARIS_LINEAR_ENABLE", False):
                    if not _linear_first_forward_evidenced(finalize_reports):
                        trajectory_errors.append(
                            "missing linear_first_forward_budget_asserted"
                        )

                completion = final_output.outputs[0]
                token_ids = [int(value) for value in completion.token_ids]
                finish_event = {
                    "time": time.time(),
                    "event": "evaluation_item_finish",
                    "request_id": trace_request_id or request_id,
                    "public_request_id": str(final_output.request_id),
                    "sample_id": str(sample.get("id")),
                    "global_index": int(sample.get("global_index")),
                    "output_tokens": len(token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
                base._append_fsync_trace(trace_path, finish_event)
                events.append(finish_event)
                aggregate_tokens += len(token_ids)
                aggregate_wall = time.perf_counter() - run_started
                row = _result_row(
                    sample=sample,
                    meta=meta,
                    output=final_output,
                    tokenizer=tokenizer,
                    args=args,
                    elapsed=elapsed,
                    events=events,
                    trace_request_id=trace_request_id,
                    finalize_reports=finalize_reports,
                    trajectory_errors=trajectory_errors,
                    aggregate_tokens=aggregate_tokens,
                    aggregate_wall=aggregate_wall,
                )
                with args.output.open("a", encoding="utf-8") as handle:
                    base._fsync_jsonl(handle, row)
                print(
                    json.dumps(
                        {
                            "persisted_id": row["id"],
                            "global_index": row["global_index"],
                            "output_tokens": row["output_tokens"],
                            "checkpoint_count": row["dynamic_summary"][
                                "checkpoint_count"
                            ],
                            "live_concurrency": args.max_concurrency,
                            "rolling_engine_tokens_per_second": row[
                                "rolling_engine_tokens_per_second"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        except Exception as exc:
            await persist_error(
                sample,
                f"{type(exc).__name__}: {exc}",
            )

    pending: set[asyncio.Task[None]] = set()
    row_iter = iter(rows)
    sequence = 0

    def prepare_next() -> asyncio.Task[None] | None:
        nonlocal sequence
        try:
            sample = next(row_iter)
        except StopIteration:
            return None
        try:
            request, meta = build_qwen35_request(
                processor,
                sample,
                image_token_id,
                enable_thinking=False,
                assistant_cue=args.method_cue,
            )
            rendered_prefix = str(meta["assistant_generation_prefix"])
            prefix_count = str(request["prompt"]).count(rendered_prefix)
            if prefix_count != 1:
                raise RuntimeError(
                    f"assistant generation prefix count is {prefix_count}, "
                    "expected exactly one"
                )
            if meta.get("end_think_token_ids") != runtime_end_think_token_ids:
                raise RuntimeError("per-request </think> token ids changed after startup")
            if int(meta.get("end_think_token_count", -1)) != 1:
                raise RuntimeError("per-request prompt does not contain exactly one </think>")
            metadata = engine_patch.initial_request_suppression_metadata()
            request_id = _tagged_request_id(
                (
                    f"aris-async-r{args.rank_offset + args.shard_index}"
                    f"-g{int(sample.get('global_index'))}-n{sequence}"
                ),
                metadata,
            )
            sequence += 1
            return asyncio.create_task(
                generate_one(
                    sample,
                    request,
                    meta | {"method_prefix_count": prefix_count},
                    request_id,
                )
            )
        except Exception as exc:
            sequence += 1
            return asyncio.create_task(
                persist_error(sample, f"{type(exc).__name__}: {exc}")
            )

    try:
        while len(pending) < args.max_concurrency:
            task = prepare_next()
            if task is None:
                break
            pending.add(task)
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Retrieve exceptions before immediately refilling the engine.
            for task in done:
                await task
            while len(pending) < args.max_concurrency:
                task = prepare_next()
                if task is None:
                    break
                pending.add(task)
    finally:
        # This is an end-of-process invariant check, not per-item lifecycle.
        reports = await engine.collective_rpc(
            "aris_finalize_dynamic_states",
            kwargs={"reason": "async_evaluation_shutdown"},
        )
        if reports and any(int(report.get("remaining", -1)) != 0 for report in reports):
            raise RuntimeError(f"ARIS shutdown left live states: {reports!r}")
        engine.shutdown()


def main() -> None:
    parser = base.build_parser()
    parser.description = (
        "Qwen3.5 ARIS MMMU evaluation with AsyncLLM continuous batching and "
        "per-item durable persistence"
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
