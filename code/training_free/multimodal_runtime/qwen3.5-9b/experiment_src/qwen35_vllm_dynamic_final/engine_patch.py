from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from .controller import ControllerConfig, DynamicSuppressionController, RequestState, TokenMetric
from .vendor import aris_vllm_v1_suppression as native


_INSTALLED = False
_TRACE_LOCK = threading.Lock()
_TRACE_FD: int | None = None
_TRACE_FD_PATH: Path | None = None
_TRACE_BUFFER_BY_REQUEST: dict[str, bytearray] = {}
_TRACE_COMPLETED_BUFFER = bytearray()
_TRACE_COMPLETED_PATH: Path | None = None
_DYNAMIC_STATS: dict[str, Any] = {
    "sample_calls": 0,
    "processed_metric_steps": 0,
    "bookkeeping_calls": 0,
    "processed_tokens": 0,
    "forced_tokens": 0,
    "end_think_intercepts": 0,
    "states_created": 0,
    "states_finished": 0,
    "batch_reorders": 0,
    "max_live_states": 0,
    "metric_seconds": 0.0,
    "native_bookkeeping_seconds": 0.0,
    "payload_tolist_seconds": 0.0,
    "payload_sync_seconds": 0.0,
    "async_bookkeeping_calls": 0,
    "controller_decision_seconds": 0.0,
    "controller_apply_seconds": 0.0,
    "trace_seconds": 0.0,
    "prepare_seconds": 0.0,
    "sample_seconds": 0.0,
    "stats_dump_seconds": 0.0,
    "trace_buffered_events": 0,
    "controller_seconds": 0.0,
    "shape_failures": 0,
    "first_forward_budget_assertions": 0,
    "first_forward_budget_assertion_failures": 0,
}

_FIRST_FORWARD_ASSERTED_REQUESTS: set[str] = set()


_METRIC_CHUNK_SIZE = 4096


@triton.jit
def _aris_metric_chunk_kernel(
    logits_ptr,
    workspace_ptr,
    logits_stride_row,
    workspace_stride_row,
    workspace_stride_chunk,
    vocab_size: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """One-pass partial logsumexp, entropy numerator and top-2."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    valid = offsets < vocab_size
    values = tl.load(
        logits_ptr + row * logits_stride_row + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    local_max = tl.max(values, axis=0)
    finite = valid & (values > -float("inf"))
    safe_values = tl.where(finite, values, 0.0)
    weights = tl.where(finite, tl.exp(values - local_max), 0.0)
    weight_sum = tl.sum(weights, axis=0)
    weighted_value_sum = tl.sum(weights * safe_values, axis=0)

    first_offset = tl.argmax(values, axis=0)
    first_value = tl.max(values, axis=0)
    first_id = chunk * chunk_size + first_offset
    second_values = tl.where(offsets == first_id, -float("inf"), values)
    second_offset = tl.argmax(second_values, axis=0)
    second_value = tl.max(second_values, axis=0)
    second_id = chunk * chunk_size + second_offset

    base = workspace_ptr + row * workspace_stride_row + chunk * workspace_stride_chunk
    tl.store(base + 0, local_max)
    tl.store(base + 1, weight_sum)
    tl.store(base + 2, weighted_value_sum)
    tl.store(base + 3, first_value)
    tl.store(base + 4, first_id.to(tl.float32))
    tl.store(base + 5, second_value)
    tl.store(base + 6, second_id.to(tl.float32))


@triton.jit
def _aris_metric_finalize_kernel(
    logits_ptr,
    selected_ids_ptr,
    tracked_ids_ptr,
    workspace_ptr,
    payload_ptr,
    logits_stride_row,
    workspace_stride_row,
    workspace_stride_chunk,
    payload_stride_row,
    num_chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    tracked_count: tl.constexpr,
    tracked_block: tl.constexpr,
    alternative_topk: tl.constexpr,
):
    """Merge partials and write the controller's compact host payload."""
    row = tl.program_id(0)
    chunk_offsets = tl.arange(0, chunk_block)
    chunk_valid = chunk_offsets < num_chunks
    chunk_base = (
        workspace_ptr
        + row * workspace_stride_row
        + chunk_offsets * workspace_stride_chunk
    )
    local_max = tl.load(chunk_base + 0, mask=chunk_valid, other=-float("inf"))
    global_max = tl.max(local_max, axis=0)
    rescale = tl.where(chunk_valid, tl.exp(local_max - global_max), 0.0)
    partial_sum = tl.load(chunk_base + 1, mask=chunk_valid, other=0.0)
    partial_weighted = tl.load(chunk_base + 2, mask=chunk_valid, other=0.0)
    total_sum = tl.sum(partial_sum * rescale, axis=0)
    total_weighted = tl.sum(partial_weighted * rescale, axis=0)
    log_z = global_max + tl.log(total_sum)
    entropy = log_z - total_weighted / total_sum

    candidate_offsets = tl.arange(0, 2 * chunk_block)
    candidate_chunks = candidate_offsets // 2
    candidate_slot = candidate_offsets % 2
    candidate_valid = candidate_chunks < num_chunks
    candidate_base = (
        workspace_ptr
        + row * workspace_stride_row
        + candidate_chunks * workspace_stride_chunk
    )
    candidate_values = tl.load(
        candidate_base + 3 + candidate_slot * 2,
        mask=candidate_valid,
        other=-float("inf"),
    )
    candidate_ids = tl.load(
        candidate_base + 4 + candidate_slot * 2,
        mask=candidate_valid,
        other=0.0,
    )

    out = payload_ptr + row * payload_stride_row
    top_values = candidate_values
    first_value = tl.max(top_values, axis=0)
    first_offset = tl.argmax(top_values, axis=0)
    first_id = tl.sum(tl.where(candidate_offsets == first_offset, candidate_ids, 0.0), axis=0)
    tl.store(out + 5, first_value)
    tl.store(out + 5 + alternative_topk, first_id)
    top_values = tl.where(candidate_offsets == first_offset, -float("inf"), top_values)
    second_value = tl.max(top_values, axis=0)
    second_offset = tl.argmax(top_values, axis=0)
    second_id = tl.sum(tl.where(candidate_offsets == second_offset, candidate_ids, 0.0), axis=0)
    tl.store(out + 6, second_value)
    tl.store(out + 6 + alternative_topk, second_id)

    selected_id = tl.load(selected_ids_ptr + row)
    selected_logit = tl.load(logits_ptr + row * logits_stride_row + selected_id).to(tl.float32)
    margin = tl.exp(first_value - log_z) - tl.exp(second_value - log_z)
    tl.store(out + 0, entropy)
    tl.store(out + 1, margin)
    tl.store(out + 2, log_z)
    tl.store(out + 3, selected_logit)
    # float32 exactly represents every Qwen token id (< 2**24).  Carrying the
    # sampled id in this compact payload lets the CPU controller support vLLM
    # 0.19 async scheduling without asking native bookkeeping for a sync copy.
    tl.store(out + 4, selected_id.to(tl.float32))

    tracked_offsets = tl.arange(0, tracked_block)
    tracked_valid = tracked_offsets < tracked_count
    tracked_ids = tl.load(tracked_ids_ptr + tracked_offsets, mask=tracked_valid, other=0)
    tracked_logits = tl.load(
        logits_ptr + row * logits_stride_row + tracked_ids,
        mask=tracked_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out + 5 + 2 * alternative_topk + tracked_offsets,
        tracked_logits,
        mask=tracked_valid,
    )


def _tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _is_trace_rank() -> bool:
    return _tp_rank() == int(os.environ.get("ARIS_DYNAMIC_TRACE_TP_RANK", "0"))


def _trace_path() -> Path | None:
    raw = os.environ.get("ARIS_DYNAMIC_TRACE_PATH")
    if not raw or not _is_trace_rank():
        return None
    return Path(raw.format(rank=_tp_rank()))


def _append_trace(payload: dict[str, Any]) -> None:
    trace_started = time.perf_counter()
    path = _trace_path()
    if path is None:
        return
    row = {
        "time": time.time(),
        "tp_rank": _tp_rank(),
        **payload,
    }
    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    global _TRACE_FD, _TRACE_FD_PATH, _TRACE_COMPLETED_PATH
    with _TRACE_LOCK:
        request_id = str(payload.get("request_id", ""))
        buffer_per_request = os.environ.get("ARIS_DYNAMIC_TRACE_BUFFER_PER_REQUEST", "1").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if buffer_per_request and request_id:
            request_buffer = _TRACE_BUFFER_BY_REQUEST.setdefault(request_id, bytearray())
            request_buffer.extend(encoded)
            _DYNAMIC_STATS["trace_buffered_events"] += 1
            if payload.get("event") != "request_finish":
                _DYNAMIC_STATS["trace_seconds"] += time.perf_counter() - trace_started
                return
            encoded = bytes(_TRACE_BUFFER_BY_REQUEST.pop(request_id))
            if _TRACE_COMPLETED_PATH is not None and _TRACE_COMPLETED_PATH != path:
                raise RuntimeError("ARIS trace path changed while a completion batch was buffered")
            _TRACE_COMPLETED_PATH = path
            _TRACE_COMPLETED_BUFFER.extend(encoded)
            _DYNAMIC_STATS["trace_seconds"] += time.perf_counter() - trace_started
            return
        if _TRACE_FD is None or _TRACE_FD_PATH != path:
            if _TRACE_FD is not None:
                os.close(_TRACE_FD)
            path.parent.mkdir(parents=True, exist_ok=True)
            _TRACE_FD = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            _TRACE_FD_PATH = path
        os.write(_TRACE_FD, encoded)
        fsync_items = os.environ.get("ARIS_DYNAMIC_TRACE_FSYNC", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        fsync_every_event = os.environ.get("ARIS_DYNAMIC_TRACE_FSYNC_EVERY_EVENT", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        # Per-item durability is sufficient for resumable evaluation.  Fsync
        # every event on a network filesystem destroys decode throughput.
        if fsync_items and fsync_every_event:
            os.fsync(_TRACE_FD)
    _DYNAMIC_STATS["trace_seconds"] += time.perf_counter() - trace_started


def _close_trace_fd() -> None:
    _flush_completed_trace_batch()
    global _TRACE_FD, _TRACE_FD_PATH
    with _TRACE_LOCK:
        if _TRACE_FD is not None:
            if os.environ.get("ARIS_DYNAMIC_TRACE_FSYNC", "0").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                os.fsync(_TRACE_FD)
            os.close(_TRACE_FD)
            _TRACE_FD = None
            _TRACE_FD_PATH = None


def _flush_completed_trace_batch() -> None:
    global _TRACE_FD, _TRACE_FD_PATH, _TRACE_COMPLETED_PATH
    with _TRACE_LOCK:
        if _TRACE_COMPLETED_BUFFER:
            if _TRACE_COMPLETED_PATH is None:
                raise RuntimeError("ARIS completed trace bytes have no destination path")
            path = _TRACE_COMPLETED_PATH
            if _TRACE_FD is None or _TRACE_FD_PATH != path:
                if _TRACE_FD is not None:
                    os.close(_TRACE_FD)
                path.parent.mkdir(parents=True, exist_ok=True)
                _TRACE_FD = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                _TRACE_FD_PATH = path
            os.write(_TRACE_FD, _TRACE_COMPLETED_BUFFER)
            _TRACE_COMPLETED_BUFFER.clear()
            _TRACE_COMPLETED_PATH = None
        if _TRACE_FD is not None and os.environ.get("ARIS_DYNAMIC_TRACE_FSYNC", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            os.fsync(_TRACE_FD)


def _stats_path() -> Path | None:
    raw = os.environ.get("ARIS_DYNAMIC_STATS_PATH")
    if not raw or not _is_trace_rank():
        return None
    return Path(raw.format(rank=_tp_rank()))


def dump_dynamic_stats() -> None:
    dump_started = time.perf_counter()
    path = _stats_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(_DYNAMIC_STATS)
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            # The API parent is also rank 0 outside a distributed context and
            # runs this atexit hook with zero counters. Preserve worker maxima
            # so that parent teardown cannot erase worker-local evidence.
            payload = {
                key: max(float(previous.get(key, 0)), float(value))
                if isinstance(value, (int, float))
                else value
                for key, value in payload.items()
            }
            for key, value in list(payload.items()):
                if isinstance(_DYNAMIC_STATS.get(key), int):
                    payload[key] = int(value)
        except Exception:
            payload = dict(_DYNAMIC_STATS)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _DYNAMIC_STATS["stats_dump_seconds"] += time.perf_counter() - dump_started


atexit.register(dump_dynamic_stats)
atexit.register(_close_trace_fd)


def get_dynamic_stats() -> dict[str, Any]:
    return json.loads(json.dumps(_DYNAMIC_STATS))


def _execution_mode() -> dict[str, Any]:
    """Return the runtime branch switches used by release-smoke auditing."""
    truthy = {"1", "true", "yes", "on"}

    def enabled(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in truthy

    return {
        "full_enabled": enabled("ARIS_FULL_ENABLE", True),
        "linear_enabled": enabled("ARIS_LINEAR_ENABLE", False),
        "conv_replacement_enabled": enabled("ARIS_LINEAR_CONV_REPLACE", False),
        "controller_skipped": enabled("ARIS_DYNAMIC_DIAGNOSTIC_SKIP_CONTROLLER", False),
        "forced_native_k0": enabled("ARIS_DIAGNOSTIC_FORCE_NATIVE_K0", False),
    }


def finalize_dynamic_states(runner, reason: str = "generate_return") -> dict[str, Any]:
    """Close every remaining request after the engine reports no unfinished work.

    vLLM can remove max-token requests without another ``_prepare_inputs``
    call.  A post-generate worker RPC invokes this function so the last batch
    receives request_finish events and its buffered trace is durable before a
    caller starts judging results.
    """
    states = getattr(runner, "_aris_dynamic_states", None)
    controller = getattr(runner, "_aris_dynamic_controller", None)
    if not isinstance(states, dict) or controller is None:
        return {
            "finalized": 0,
            "remaining": 0,
            "reason": reason,
            "mode": _execution_mode(),
            "stats": get_dynamic_stats(),
        }
    finalized = 0
    for req_id, state in list(states.items()):
        states.pop(req_id, None)
        _flush_events(state)
        _append_trace(controller.finish_event(state, reason))
        native._REQUEST_METADATA_BY_ID.pop(str(req_id), None)
        _DYNAMIC_STATS["states_finished"] += 1
        finalized += 1
    runner._aris_dynamic_last_req_ids = []
    _flush_completed_trace_batch()
    if os.environ.get("ARIS_DYNAMIC_STATS_ON_FINALIZE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        dump_dynamic_stats()
    return {
        "finalized": finalized,
        "remaining": len(states),
        "reason": reason,
        "mode": _execution_mode(),
        "stats": get_dynamic_stats(),
    }


def finalize_dynamic_request(
    runner,
    request_id: str,
    reason: str = "request_generate_return",
) -> dict[str, Any]:
    """Durably close one request without disturbing other live requests.

    Async continuous batching returns requests independently.  The former
    all-state finalizer is only safe after a synchronous batch has drained;
    using it per completion would incorrectly terminate controller state for
    requests that are still decoding.
    """
    request_id = str(request_id)
    resolved_request_id = request_id
    states = getattr(runner, "_aris_dynamic_states", None)
    controller = getattr(runner, "_aris_dynamic_controller", None)
    finalized = 0
    request_remaining = False
    if isinstance(states, dict) and controller is not None:
        state = states.pop(request_id, None)
        if state is None:
            # vLLM's V1 scheduler appends a short uniqueness suffix to the
            # worker-side request id, while AsyncLLMEngine exposes the public
            # id passed by the frontend.  Resolve that one-to-one mapping
            # explicitly so per-request finalization does not silently leave
            # controller state buffered until engine shutdown.
            matches = [
                str(candidate)
                for candidate in states
                if str(candidate).startswith(request_id + "-")
            ]
            if len(matches) > 1:
                raise RuntimeError(
                    "ARIS request id is ambiguous during async finalization: "
                    f"public={request_id!r}, matches={matches!r}"
                )
            if len(matches) == 1:
                resolved_request_id = matches[0]
                state = states.pop(resolved_request_id)
        if state is not None:
            _flush_events(state)
            _append_trace(controller.finish_event(state, reason))
            native._REQUEST_METADATA_BY_ID.pop(resolved_request_id, None)
            _DYNAMIC_STATS["states_finished"] += 1
            finalized = 1
        request_remaining = resolved_request_id in states
        remaining = len(states)
    else:
        remaining = 0
    # The scheduler may already have finalized the request before this RPC.
    # Flushing is still required so the frontend can safely read its complete
    # trajectory before exposing the prediction row to the streaming judge.
    _flush_completed_trace_batch()
    if os.environ.get("ARIS_DYNAMIC_STATS_ON_FINALIZE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        dump_dynamic_stats()
    return {
        "request_id": request_id,
        "resolved_request_id": resolved_request_id,
        "finalized": finalized,
        "request_remaining": request_remaining,
        "remaining": remaining,
        "reason": reason,
        "mode": _execution_mode(),
        "stats": get_dynamic_stats(),
    }


def finalize_llm_dynamic_states(llm, reason: str = "generate_return") -> list[dict[str, Any]]:
    reports = llm.collective_rpc(
        "aris_finalize_dynamic_states",
        kwargs={"reason": str(reason)},
    )
    if not reports or any(int(report.get("remaining", -1)) != 0 for report in reports):
        raise RuntimeError(f"ARIS worker finalization failed: {reports!r}")
    return reports


async def finalize_async_dynamic_request(
    engine,
    request_id: str,
    reason: str = "request_generate_return",
) -> list[dict[str, Any]]:
    reports = await engine.collective_rpc(
        "aris_finalize_dynamic_request",
        kwargs={"request_id": str(request_id), "reason": str(reason)},
    )
    if not reports or any(bool(report.get("request_remaining", True)) for report in reports):
        raise RuntimeError(
            f"ARIS request-specific worker finalization failed for {request_id!r}: "
            f"{reports!r}"
        )
    return reports


def _load_config() -> ControllerConfig:
    path = os.environ.get("ARIS_DYNAMIC_CONFIG_JSON")
    if not path:
        raise RuntimeError("ARIS_DYNAMIC_CONFIG_JSON is required")
    return ControllerConfig.from_json(path)


def initial_request_suppression_metadata(end_think_index: int = -1) -> dict[str, Any]:
    """Build the metadata that must reach vLLM before the first model forward.

    The controller state is created in the model runner, which is too late for
    the first ``_prepare_inputs`` call.  Encoding the configured initial K in
    the frontend request makes the request-id tag and the first attention
    metadata use the same value as ``Controller.new_state``.

    Diagnostic controller/K0 ablations retain their historical K=0 contract.
    """
    config = _load_config()
    budget = min(
        max(float(config.initial_suppression), 0.0),
        float(config.suppression_upper_bound),
    )
    if os.environ.get("ARIS_DIAGNOSTIC_FORCE_NATIVE_K0", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("ARIS_DYNAMIC_DIAGNOSTIC_SKIP_CONTROLLER", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        budget = 0.0
    return {
        "end_think_index": int(end_think_index),
        "enabled": True,
        "budget": round(float(budget), 6),
    }


def _runner_state(runner) -> tuple[DynamicSuppressionController, dict[str, RequestState]]:
    controller = getattr(runner, "_aris_dynamic_controller", None)
    states = getattr(runner, "_aris_dynamic_states", None)
    if controller is None:
        controller = DynamicSuppressionController(_load_config())
        runner._aris_dynamic_controller = controller
    if states is None:
        states = {}
        runner._aris_dynamic_states = states
        runner._aris_dynamic_last_req_ids = []
    return controller, states


def _end_index_from_request_id(request_id: str) -> int:
    parsed = native._parse_request_end_index(str(request_id))
    return -1 if parsed is None else int(parsed)


def _sync_native_metadata(request_id: str, state: RequestState) -> None:
    budget = float(state.current_effective_suppression)
    if os.environ.get("ARIS_DIAGNOSTIC_FORCE_NATIVE_K0", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        budget = 0.0
    native._REQUEST_METADATA_BY_ID[str(request_id)] = native.RequestSuppressionMetadata(
        end_think_index=_end_index_from_request_id(str(request_id)),
        enabled=True,
        budget=budget,
    )


def _flush_events(state: RequestState) -> None:
    events = state.drain_events()
    if not events:
        return
    heads_by_layer = native._heads_for_budget(float(state.current_effective_suppression))
    selected_head_count = sum(len(heads) for heads in heads_by_layer.values())
    for event in events:
        _append_trace(
            {
                "request_id": state.request_id,
                "step_index": state.step_index,
                "current_k_after_event": float(state.current_effective_suppression),
                "selected_head_count_after_event": int(selected_head_count),
                "selected_layer_count_after_event": int(len(heads_by_layer)),
                **event,
            }
        )


def _new_state(controller: DynamicSuppressionController, states: dict[str, RequestState], req_id: str) -> RequestState:
    state = controller.new_state(str(req_id))
    states[str(req_id)] = state
    _sync_native_metadata(str(req_id), state)
    _DYNAMIC_STATS["states_created"] += 1
    _DYNAMIC_STATS["max_live_states"] = max(_DYNAMIC_STATS["max_live_states"], len(states))
    _flush_events(state)
    return state


def _finish_states(runner, finished_req_ids, reason: str) -> None:
    if not finished_req_ids:
        return
    controller, states = _runner_state(runner)
    for req_id in finished_req_ids:
        state = states.pop(str(req_id), None)
        if state is None:
            continue
        _flush_events(state)
        _append_trace(controller.finish_event(state, reason))
        native._REQUEST_METADATA_BY_ID.pop(str(req_id), None)
        _DYNAMIC_STATS["states_finished"] += 1
    # All completed request buffers have been written at this point.  One
    # fsync makes every item in the scheduler completion batch durable before
    # generate() returns and before any streaming judge can observe it.
    _flush_completed_trace_batch()
    if os.environ.get("ARIS_DYNAMIC_STATS_ON_FINISH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        dump_dynamic_stats()


def _metric_payload(
    runner,
    logits: torch.Tensor,
    selected_ids: torch.Tensor,
    tracked_ids: torch.Tensor,
    alternative_topk: int,
) -> torch.Tensor:
    """Pack all controller inputs with two fused Triton reductions.

    Native vLLM synchronizes the sampled token immediately after this call;
    the payload's nonblocking pinned-host copy shares that synchronization.
    A cached partial workspace avoids the former sequence of full-vocabulary
    PyTorch logsumexp/exp/sum/topk kernels and their large temporary tensors.
    """
    if int(alternative_topk) != 2:
        raise RuntimeError(
            "ARIS fused metric kernel currently requires exactly one end-think token "
            f"(alternative_topk=2), got {alternative_topk}"
        )
    rows, vocab_size = (int(logits.shape[0]), int(logits.shape[1]))
    chunk_size = _METRIC_CHUNK_SIZE
    num_chunks = triton.cdiv(vocab_size, chunk_size)
    columns = 5 + 2 * int(alternative_topk) + int(tracked_ids.numel())

    workspace = getattr(runner, "_aris_dynamic_metric_workspace", None)
    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.shape[0] < rows
        or tuple(workspace.shape[1:]) != (num_chunks, 7)
        or workspace.device != logits.device
    ):
        workspace = torch.empty((rows, num_chunks, 7), dtype=torch.float32, device=logits.device)
        runner._aris_dynamic_metric_workspace = workspace
    workspace = workspace[:rows]
    payload = getattr(runner, "_aris_dynamic_metric_gpu", None)
    if (
        not isinstance(payload, torch.Tensor)
        or payload.shape[0] < rows
        or payload.shape[1] != columns
        or payload.device != logits.device
    ):
        payload = torch.empty((rows, columns), dtype=torch.float32, device=logits.device)
        runner._aris_dynamic_metric_gpu = payload
    payload = payload[:rows]

    _aris_metric_chunk_kernel[(rows, num_chunks)](
        logits,
        workspace,
        int(logits.stride(0)),
        int(workspace.stride(0)),
        int(workspace.stride(1)),
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        num_warps=8,
        num_stages=1,
    )
    chunk_block = triton.next_power_of_2(num_chunks)
    tracked_count = int(tracked_ids.numel())
    tracked_block = triton.next_power_of_2(max(1, tracked_count))
    _aris_metric_finalize_kernel[(rows,)](
        logits,
        selected_ids,
        tracked_ids,
        workspace,
        payload,
        int(logits.stride(0)),
        int(workspace.stride(0)),
        int(workspace.stride(1)),
        int(payload.stride(0)),
        num_chunks=num_chunks,
        chunk_block=chunk_block,
        tracked_count=tracked_count,
        tracked_block=tracked_block,
        alternative_topk=int(alternative_topk),
        num_warps=4,
        num_stages=1,
    )
    return payload


def _metric_layout(runner, controller: DynamicSuppressionController, device: torch.device):
    cached = getattr(runner, "_aris_dynamic_metric_layout", None)
    if cached is not None:
        return cached
    cfg = controller.config
    tracked = sorted(
        {
            int(token_id)
            for name in (
                "end_think_token_ids",
                "cue_token_ids",
                "rethink_cue_token_ids",
                "tail_signal_token_ids",
                "end_think_bridge_token_ids",
                "final_answer_prefix_token_ids",
            )
            for token_id in getattr(cfg, name)
        }
    )
    tracked_tensor = torch.tensor(tracked, dtype=torch.long, device=device)
    tracked_column = {token_id: idx for idx, token_id in enumerate(tracked)}
    alternative_topk = max(2, min(16, len(cfg.end_think_token_ids) + 1))
    cached = (tracked_tensor, tracked_column, alternative_topk)
    runner._aris_dynamic_metric_layout = cached
    return cached


def _pinned_payload_buffer(runner, rows: int, columns: int) -> torch.Tensor:
    buffer = getattr(runner, "_aris_dynamic_metric_cpu", None)
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.shape[0] < rows
        or buffer.shape[1] != columns
    ):
        buffer = torch.empty((rows, columns), dtype=torch.float32, device="cpu", pin_memory=True)
        runner._aris_dynamic_metric_cpu = buffer
    return buffer[:rows]


def _alternative_from_payload(
    row: list[float],
    *,
    top_values_start: int,
    top_ids_start: int,
    topk: int,
    forbidden_ids: list[int],
) -> tuple[int, float]:
    forbidden = {int(item) for item in forbidden_ids}
    for offset in range(topk):
        token_id = int(row[top_ids_start + offset])
        if token_id not in forbidden:
            return token_id, float(row[top_values_start + offset])
    raise RuntimeError("ARIS alternative-token top-k was exhausted by forbidden token ids")


def _install_dynamic_gpu_runner_patch() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aris_dynamic_patch_installed", False):
        return

    original_prepare_inputs = GPUModelRunner._prepare_inputs
    original_sample = GPUModelRunner._sample
    original_bookkeeping = GPUModelRunner._bookkeeping_sync

    def skip_controller() -> bool:
        return os.environ.get("ARIS_DYNAMIC_DIAGNOSTIC_SKIP_CONTROLLER", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def dynamic_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        if skip_controller():
            return original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)
        prepare_started = time.perf_counter()
        _finish_states(self, getattr(scheduler_output, "finished_req_ids", None), "scheduler_finished_or_aborted")
        controller, states = _runner_state(self)
        # Requests that ended at the last max-token step are removed from the
        # model runner before the next _prepare_inputs call, but vLLM does not
        # always repeat them in finished_req_ids.  Use the runner's canonical
        # request table to close those states without mistaking a merely
        # unscheduled/preempted request for a finished one.
        live_runner_ids = {str(req_id) for req_id in self.requests}
        stale_ids = [req_id for req_id in states if req_id not in live_runner_ids]
        for req_id in stale_ids:
            state = states.pop(req_id)
            _flush_events(state)
            _append_trace(controller.finish_event(state, "runner_request_removed"))
            native._REQUEST_METADATA_BY_ID.pop(req_id, None)
            _DYNAMIC_STATS["states_finished"] += 1
        if stale_ids:
            _flush_completed_trace_batch()
        result = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)
        # The canonical input batch is updated inside native _prepare_inputs.
        # Create new states afterwards so sampling/bookkeeping for the first
        # token already has a complete controller state.  The request-id tag
        # carries the configured initial K for that first attention step.
        req_ids = [str(item) for item in self.input_batch.req_ids[: self.input_batch.num_reqs]]
        previous = list(getattr(self, "_aris_dynamic_last_req_ids", []))
        if previous and req_ids != previous and set(req_ids) == set(previous):
            _DYNAMIC_STATS["batch_reorders"] += 1
        self._aris_dynamic_last_req_ids = req_ids
        for req_id in req_ids:
            state = states.get(req_id)
            if state is None:
                state = _new_state(controller, states, req_id)
            _sync_native_metadata(req_id, state)
        _DYNAMIC_STATS["prepare_seconds"] += time.perf_counter() - prepare_started
        return result

    def dynamic_sample(self, logits, spec_decode_metadata):
        if skip_controller():
            return original_sample(self, logits, spec_decode_metadata)
        sample_started = time.perf_counter()
        if spec_decode_metadata is not None:
            raise RuntimeError("ARIS dynamic suppression does not permit speculative decoding")
        if logits is not None:
            controller, states = _runner_state(self)
            req_ids = [str(item) for item in self.input_batch.req_ids[: self.input_batch.num_reqs]]
            if logits.shape[0] != len(req_ids):
                _DYNAMIC_STATS["shape_failures"] += 1
                raise RuntimeError(
                    f"ARIS logits/request shape mismatch before sampling: logits={tuple(logits.shape)}, reqs={len(req_ids)}; "
                    "the one-logit-row-per-live-request invariant was violated"
                )
            for row, req_id in enumerate(req_ids):
                state = states.get(req_id)
                if state is None:
                    continue
                forbidden = controller.token_ids_to_mask(state)
                if forbidden:
                    logits[row, forbidden] = -torch.inf
        # GPUModelRunner hands the raw model logits to bookkeeping, while the
        # vLLM sampler greedily selects from float32 logits after repetition /
        # presence / frequency penalties.  Transformers invokes our custom
        # LogitsProcessor after those same penalties.  Rebuild that processed
        # view before native sampling (which may mutate an already-float32
        # tensor in place).  vLLM 0.19 consolidates allowed-token, bad-word,
        # non-argmax processor and penalty handling in apply_logits_processors.
        metric_logits = None
        if logits is not None:
            metadata = self.input_batch.sampling_metadata
            logitsprocs = metadata.logitsprocs
            needs_processed_view = (
                not metadata.no_penalties
                or metadata.allowed_token_ids_mask is not None
                or bool(metadata.bad_words_token_ids)
                or bool(logitsprocs.non_argmax_invariant)
            )
            if needs_processed_view:
                ignore_non_argmax = os.environ.get(
                    "ARIS_DYNAMIC_DIAGNOSTIC_IGNORE_NON_ARGMAX_PROCESSORS", "0"
                ).lower() in {"1", "true", "yes", "on"}
                if ignore_non_argmax:
                    # Teacher-forced equivalence tests install a processor
                    # that replaces the complete vocabulary by one-hot logits.
                    # The real controller must nevertheless consume the
                    # *pre-forcing* model distribution.  This diagnostic mode
                    # is deliberately fail-closed when any normal sampling
                    # transformation is present, so it cannot silently alter
                    # a formal evaluation.
                    if (
                        not metadata.no_penalties
                        or metadata.allowed_token_ids_mask is not None
                        or bool(metadata.bad_words_token_ids)
                    ):
                        raise RuntimeError(
                            "diagnostic pre-forcing metrics require no penalties, "
                            "allowed-token mask, or bad-word processor"
                        )
                    metric_logits = logits.to(torch.float32).clone()
                else:
                    metric_logits = logits.to(torch.float32).clone()
                    metric_logits = self.sampler.apply_logits_processors(
                        metric_logits,
                        metadata,
                        predict_bonus_token=False,
                    )
                _DYNAMIC_STATS["processed_metric_steps"] += 1
        _DYNAMIC_STATS["sample_calls"] += 1
        result = original_sample(self, logits, spec_decode_metadata)
        self._aris_dynamic_metric_logits = metric_logits
        _DYNAMIC_STATS["sample_seconds"] += time.perf_counter() - sample_started
        return result

    def dynamic_bookkeeping(
        self,
        scheduler_output,
        sampler_output,
        logits,
        hidden_states,
        num_scheduled_tokens,
        spec_decode_metadata,
    ):
        if skip_controller():
            return original_bookkeeping(
                self,
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                num_scheduled_tokens,
                spec_decode_metadata,
            )
        if logits is None:
            raise RuntimeError("ARIS dynamic suppression requires logits on the sampling rank")
        controller, states = _runner_state(self)
        req_ids = [str(item) for item in self.input_batch.req_ids[: self.input_batch.num_reqs]]
        sampled = sampler_output.sampled_token_ids
        if sampled.ndim != 2 or sampled.shape[1] != 1:
            raise RuntimeError(f"ARIS requires one-token non-spec decode, got sampled shape={tuple(sampled.shape)}")
        num_rows = int(sampled.shape[0])
        if num_rows != len(req_ids) or logits.shape[0] != num_rows:
            _DYNAMIC_STATS["shape_failures"] += 1
            raise RuntimeError(
                f"ARIS bookkeeping shape mismatch: sampled={tuple(sampled.shape)}, logits={tuple(logits.shape)}, reqs={len(req_ids)}"
            )

        started = time.perf_counter()
        # Enqueue the controller payload before native bookkeeping.  vLLM
        # 0.19's async scheduler deliberately avoids a sampled-token CPU sync;
        # the compact payload event below is therefore the single explicit
        # synchronization required by the CPU-side dynamic controller.
        metric_started = time.perf_counter()
        dummy_payload = os.environ.get("ARIS_DYNAMIC_DIAGNOSTIC_DUMMY_PAYLOAD", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        candidate_tensor = sampled[:, 0]
        metric_logits = getattr(self, "_aris_dynamic_metric_logits", None)
        if metric_logits is None:
            metric_logits = logits
        tracked_tensor, tracked_column, alternative_topk = _metric_layout(
            self, controller, metric_logits.device
        )
        payload_cpu = None
        payload_ready = None
        if not dummy_payload:
            payload_gpu = _metric_payload(
                self, metric_logits, candidate_tensor, tracked_tensor, alternative_topk
            )
            payload_cpu = _pinned_payload_buffer(self, num_rows, int(payload_gpu.shape[1]))
            payload_cpu.copy_(payload_gpu, non_blocking=True)
            payload_ready = torch.cuda.Event()
            payload_ready.record()
        self._aris_dynamic_metric_logits = None
        _DYNAMIC_STATS["metric_seconds"] += time.perf_counter() - metric_started

        native_started = time.perf_counter()
        native_result = original_bookkeeping(
            self,
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            num_scheduled_tokens,
            spec_decode_metadata,
        )
        _DYNAMIC_STATS["native_bookkeeping_seconds"] += time.perf_counter() - native_started
        valid_sampled_token_ids = native_result[2]
        invalid_req_indices = {int(value) for value in native_result[6]}
        async_scheduling = bool(self.use_async_scheduling)
        _DYNAMIC_STATS["async_bookkeeping_calls"] += int(async_scheduling)
        tolist_started = time.perf_counter()
        if payload_cpu is None:
            # Performance attribution only: retain the controller/token-state
            # path while removing the full-vocabulary metric reduction.
            # Never enable this mode for correctness or formal evaluation.
            columns = 5 + 2 * alternative_topk + int(tracked_tensor.numel())
            selected_ids_cpu = sampled[:, 0].tolist()
            packed_rows = []
            for row in range(num_rows):
                candidate = (
                    int(valid_sampled_token_ids[row][0])
                    if not async_scheduling
                    and row < len(valid_sampled_token_ids)
                    and valid_sampled_token_ids[row]
                    else int(selected_ids_cpu[row])
                )
                packed = [0.0] * columns
                packed[4] = float(candidate)
                for offset in range(alternative_topk):
                    packed[5 + offset] = 0.0
                    packed[5 + alternative_topk + offset] = float(candidate)
                packed_rows.append(packed)
        else:
            sync_started = time.perf_counter()
            assert payload_ready is not None
            payload_ready.synchronize()
            _DYNAMIC_STATS["payload_sync_seconds"] += time.perf_counter() - sync_started
            packed_rows = payload_cpu.tolist()
        _DYNAMIC_STATS["payload_tolist_seconds"] += time.perf_counter() - tolist_started
        top_values_start = 5
        top_ids_start = top_values_start + alternative_topk
        tracked_start = top_ids_start + alternative_topk
        chosen_ids: list[int] = []
        forced_reasons: list[str | None] = []
        chosen_metrics: list[TokenMetric] = []
        candidate_ids: list[int] = []
        decision_started = time.perf_counter()
        for row, req_id in enumerate(req_ids):
            if row in invalid_req_indices or (
                not async_scheduling
                and (row >= len(valid_sampled_token_ids) or not valid_sampled_token_ids[row])
            ):
                # vLLM discarded this sample (for example after preemption).
                candidate_ids.append(-1)
                chosen_ids.append(-1)
                forced_reasons.append(None)
                chosen_metrics.append(TokenMetric(0.0, 0.0, 0.0))
                continue
            packed = packed_rows[row]
            candidate = int(packed[4]) if async_scheduling else int(valid_sampled_token_ids[row][0])
            candidate_ids.append(candidate)
            entropy = float(packed[0])
            margin = float(packed[1])
            log_z = float(packed[2])
            chosen_logit = float(packed[3])
            candidate_metric = TokenMetric(entropy, margin, chosen_logit - log_z)
            state = states.get(req_id)
            if state is None:
                state = _new_state(controller, states, req_id)
            if state.checkpoint_zero_pending:
                controller.apply_checkpoint_zero(
                    state,
                    candidate_metric,
                    candidate,
                )
            forced = controller.pop_forced(state)
            if forced is None and controller.intercept_end_think(state, candidate):
                controller.on_end_think_intercept(state)
                _DYNAMIC_STATS["end_think_intercepts"] += 1
                forced = controller.pop_forced(state)
                if forced is None:
                    candidate, chosen_logit = _alternative_from_payload(
                        packed,
                        top_values_start=top_values_start,
                        top_ids_start=top_ids_start,
                        topk=alternative_topk,
                        forbidden_ids=controller.config.end_think_token_ids,
                    )
                    forced_reason = "end_think_intercept_resample"
                else:
                    candidate = forced.token_id
                    forced_reason = forced.reason
            elif forced is not None:
                candidate = forced.token_id
                forced_reason = forced.reason
            else:
                forced_reason = None
            if candidate != candidate_ids[-1]:
                tracked_idx = tracked_column.get(int(candidate))
                if tracked_idx is not None:
                    chosen_logit = float(packed[tracked_start + tracked_idx])
                elif forced_reason != "end_think_intercept_resample":
                    raise RuntimeError(f"forced token {candidate} was not prepacked in the metric payload")
            chosen_ids.append(int(candidate))
            forced_reasons.append(forced_reason)
            chosen_metrics.append(TokenMetric(entropy, margin, chosen_logit - log_z))
        _DYNAMIC_STATS["controller_decision_seconds"] += time.perf_counter() - decision_started

        apply_started = time.perf_counter()
        for row, (req_id, token_id, original_token_id, reason, metric) in enumerate(
            zip(req_ids, chosen_ids, candidate_ids, forced_reasons, chosen_metrics)
        ):
            if token_id < 0:
                continue
            if async_scheduling:
                # Native async bookkeeping appended a -1 placeholder and kept
                # the GPU sampled tensor for the next iteration.  Repair all
                # canonical views now; AsyncGPUModelRunnerOutput is constructed
                # after this wrapper returns and therefore observes any forced
                # replacement in sampler_output.
                token_pos = int(self.input_batch.num_tokens_no_spec[row]) - 1
                self.input_batch.token_ids_cpu[row, token_pos] = int(token_id)
                request_state = self.requests[req_id]
                request_state.output_token_ids[-1] = int(token_id)
                sampler_output.sampled_token_ids[row, 0] = int(token_id)
            elif token_id != original_token_id:
                # Native bookkeeping has already appended the sampled token;
                # rewrite every canonical CPU view before ModelRunnerOutput is
                # constructed and before the next decode input is prepared.
                valid_sampled_token_ids[row][0] = int(token_id)
                token_pos = int(self.input_batch.num_tokens_no_spec[row]) - 1
                self.input_batch.token_ids_cpu[row, token_pos] = int(token_id)
                request_state = self.requests[req_id]
                request_state.output_token_ids[-1] = int(token_id)
                sampler_output.sampled_token_ids[row, 0] = int(token_id)
            state = states[req_id]
            controller.on_token(state, token_id, metric, forced_reason=reason)
            _sync_native_metadata(req_id, state)
            _flush_events(state)
            _DYNAMIC_STATS["processed_tokens"] += 1
            if reason is not None:
                _DYNAMIC_STATS["forced_tokens"] += 1
        _DYNAMIC_STATS["controller_apply_seconds"] += time.perf_counter() - apply_started
        _DYNAMIC_STATS["controller_seconds"] += time.perf_counter() - started
        _DYNAMIC_STATS["bookkeeping_calls"] += 1
        stats_interval = int(os.environ.get("ARIS_DYNAMIC_STATS_INTERVAL", "0") or "0")
        if stats_interval > 0 and _DYNAMIC_STATS["bookkeeping_calls"] % stats_interval == 0:
            dump_dynamic_stats()
        return native_result

    GPUModelRunner._prepare_inputs = dynamic_prepare_inputs
    GPUModelRunner._sample = dynamic_sample
    GPUModelRunner._bookkeeping_sync = dynamic_bookkeeping
    GPUModelRunner._aris_dynamic_patch_installed = True


def _install_first_forward_budget_assertion() -> None:
    """Fail closed unless the first real attention forward sees initial K.

    This checks the device-side budget id consumed by the fused attention
    path, not merely the controller state or frontend request dictionary.
    """
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    if getattr(FlashAttentionImpl, "_aris_dynamic_first_forward_assert_installed", False):
        return
    original_forward = FlashAttentionImpl.forward

    def dynamic_asserting_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        req_ids = list(getattr(attn_metadata, "aris_req_ids", []) or []) if attn_metadata is not None else []
        unseen_rows = [
            row
            for row, req_id in enumerate(req_ids)
            if str(req_id) not in _FIRST_FORWARD_ASSERTED_REQUESTS
        ]
        if unseen_rows:
            try:
                actual_budget_ids = native._tensor_or_cpu_list(attn_metadata.aris_budget_ids)
                actual_enabled = native._tensor_or_cpu_list(attn_metadata.aris_enabled)
                actual_end_positions = native._tensor_or_cpu_list(attn_metadata.aris_end_positions)
                expected_initial = float(initial_request_suppression_metadata()["budget"])
                expected_budget_id = int(native._budget_id_for_budget(expected_initial))
                for row in unseen_rows:
                    req_id = str(req_ids[row])
                    has_tag, tagged_budget = native._parse_request_budget(req_id)
                    actual_budget_id = int(actual_budget_ids[row])
                    enabled = int(actual_enabled[row])
                    end_position = int(actual_end_positions[row])
                    expected_enabled = int(expected_initial > 0.0 and end_position >= 0)
                    errors = []
                    if not has_tag or tagged_budget is None:
                        errors.append("request id has no numeric aris_budget tag")
                    elif abs(float(tagged_budget) - expected_initial) > 1e-6:
                        errors.append(
                            f"request tag K={float(tagged_budget):.6g}, expected K={expected_initial:.6g}"
                        )
                    if actual_budget_id != expected_budget_id:
                        errors.append(
                            f"device budget_id={actual_budget_id}, expected budget_id={expected_budget_id}"
                        )
                    if enabled != expected_enabled:
                        errors.append(f"device enabled={enabled}, expected enabled={expected_enabled}")
                    if errors:
                        raise RuntimeError(
                            "ARIS first-forward initial-K assertion failed for "
                            f"request {req_id}: " + "; ".join(errors)
                        )
                    _FIRST_FORWARD_ASSERTED_REQUESTS.add(req_id)
                    _DYNAMIC_STATS["first_forward_budget_assertions"] += 1
                    _append_trace(
                        {
                            "event": "first_forward_budget_asserted",
                            "request_id": req_id,
                            "expected_initial_k": expected_initial,
                            "request_tag_k": float(tagged_budget),
                            "actual_budget_id": actual_budget_id,
                            "expected_budget_id": expected_budget_id,
                            "actual_enabled": enabled,
                            "actual_end_think_index": end_position,
                        }
                    )
            except Exception:
                _DYNAMIC_STATS["first_forward_budget_assertion_failures"] += 1
                raise
        return original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    FlashAttentionImpl.forward = dynamic_asserting_forward
    FlashAttentionImpl._aris_dynamic_first_forward_assert_installed = True


def verify_native_fused_op() -> None:
    import vllm
    import vllm.vllm_flash_attn  # noqa: F401

    if str(vllm.__version__) != "0.19.1":
        raise RuntimeError(f"expected exact vLLM 0.19.1, got {vllm.__version__}")
    namespace = getattr(torch.ops, "_vllm_fa2_C", None)
    if namespace is None or not hasattr(namespace, "varlen_fwd"):
        raise RuntimeError("native torch.ops._vllm_fa2_C.varlen_fwd is unavailable")
    schema = str(torch._C._dispatch_find_schema_or_throw("_vllm_fa2_C::varlen_fwd", "").schema())
    required = (
        "aris_enabled",
        "aris_budget_ids",
        "aris_end_positions",
        "aris_head_mask_table",
        "aris_window_left",
        "aris_window_right",
    )
    if any(name not in schema for name in required):
        raise RuntimeError(f"vLLM FA2 op lacks the ARIS fused ABI: {schema}")


def install_dynamic_aris_patch(require_native_fused: bool = True) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if require_native_fused:
        verify_native_fused_op()
    native.configure_from_env()
    native.install_aris_vllm_v1_suppression()
    _install_dynamic_gpu_runner_patch()
    _install_first_forward_budget_assertion()
    _INSTALLED = True
