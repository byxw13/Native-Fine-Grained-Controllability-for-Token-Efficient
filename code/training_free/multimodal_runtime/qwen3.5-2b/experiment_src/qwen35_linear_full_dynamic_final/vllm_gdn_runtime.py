from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import torch

from vllm.triton_utils import tl, triton

from .head_selection import rank_linear_heads
from .linear_state import (
    boundary_contribution,
    boundary_step,
    normalize_qk_fp32,
    propagate_boundary_keys,
)


_BOUNDARY_LEFT_TOKENS = int(
    os.environ.get("ARIS_LINEAR_BOUNDARY_LEFT_TOKENS", "0")
)
_BOUNDARY_RIGHT_TOKENS = int(
    os.environ.get("ARIS_LINEAR_BOUNDARY_RIGHT_TOKENS", "1")
)
if _BOUNDARY_LEFT_TOKENS < 0 or _BOUNDARY_RIGHT_TOKENS < 0:
    raise ValueError("ARIS Linear boundary left/right token counts must be non-negative")
_RANK = _BOUNDARY_LEFT_TOKENS + _BOUNDARY_RIGHT_TOKENS + 1
if _RANK not in {2, 3}:
    raise ValueError(f"ARIS Linear boundary sidecar rank must be 2 or 3, got {_RANK}")
_FUSED_VALUE_BLOCK = 32
_INSTALL_LOCK = threading.Lock()
_METADATA_INSTALLED = False
_WARMUP_PATCHED = False
_THRESHOLD_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}
_TRACKED_HEAD_CACHE: dict[tuple[str, int, int, float, float], torch.Tensor] = {}
_FUSED_WARMUP_CACHE: set[tuple[str, int, int, int, int, int]] = set()
_GRAPH_BUDGET_BUFFER_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_MAX_EFFECTIVE_K_CACHE: float | None = None
_TRACE_COUNT = 0
_FUSED_TRACE_COUNT = 0
_FIRST_FORWARD_ASSERTED_REQUESTS: set[str] = set()
_STATS = {
    "decode_triton_calls": 0,
    "decode_triton_tokens": 0,
    "decode_torch_calls": 0,
    "decode_native_k0_bypass_calls": 0,
    "decode_native_k0_bypass_tokens": 0,
    "prefill_native_k0_bypass_calls": 0,
    "prefill_calls": 0,
    "profile_decode_skips": 0,
    "first_forward_budget_assertions": 0,
    "first_forward_budget_assertion_failures": 0,
}


def _canonical_mantissa_bits() -> int:
    raw = os.environ.get("ARIS_LINEAR_CANONICAL_MANTISSA_BITS")
    if raw is None:
        legacy = os.environ.get("ARIS_LINEAR_CANONICAL_BF16", "0").strip().lower()
        return 7 if legacy in {"1", "true", "yes", "on"} else 0
    try:
        bits = int(raw)
    except ValueError:
        _fail(f"invalid ARIS_LINEAR_CANONICAL_MANTISSA_BITS={raw!r}")
    if not 0 <= bits <= 23:
        _fail(f"canonical mantissa bits must be in [0,23], got {bits}")
    return bits


def _canonical_fp64_enabled() -> bool:
    return os.environ.get("ARIS_LINEAR_CANONICAL_FP64", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@triton.jit
def _canonical_round_fp32(value, DROP_BITS: tl.constexpr):
    if DROP_BITS > 0:
        raw = value.to(tl.int32, bitcast=True)
        lsb = (raw >> DROP_BITS) & 1
        bias = (1 << (DROP_BITS - 1)) - 1 + lsb
        rounded = (raw + bias) & ~((1 << DROP_BITS) - 1)
        return rounded.to(tl.float32, bitcast=True)
    else:
        return value


@triton.jit
def _aris_linear_decode_kernel(
    query_ptr,
    key_ptr,
    log_decay_ptr,
    beta_ptr,
    output_ptr,
    key_factor_ptr,
    value_factor_ptr,
    state_index_ptr,
    budget_ptr,
    threshold_ptr,
    tracked_head_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    g_stride_t,
    g_stride_h,
    b_stride_t,
    b_stride_h,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    kf_stride_s,
    kf_stride_h,
    kf_stride_r,
    kf_stride_d,
    vf_stride_s,
    vf_stride_h,
    vf_stride_r,
    vf_stride_d,
    threshold_stride_h,
    alpha,
    suppression_lambda,
    VALUE_HEADS: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    CANONICAL_DROP_BITS: tl.constexpr,
    CANONICAL_FP64: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.load(tracked_head_ptr + tl.program_id(1))
    key_head = head // (VALUE_HEADS // KEY_HEADS)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_v = tl.arange(0, BLOCK_V)
    mask_k = offsets_k < KEY_DIM
    mask_v = offsets_v < VALUE_DIM
    slot = tl.load(state_index_ptr + token)
    valid_slot = slot >= 0
    safe_slot = tl.maximum(slot, 0)
    threshold = tl.load(threshold_ptr + head * threshold_stride_h)
    query = tl.load(
        query_ptr + token * q_stride_t + key_head * q_stride_h + offsets_k * q_stride_d,
        mask=mask_k & valid_slot,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        key_ptr + token * k_stride_t + key_head * k_stride_h + offsets_k * k_stride_d,
        mask=mask_k & valid_slot,
        other=0.0,
    ).to(tl.float32)
    if CANONICAL_FP64:
        query64 = query.to(tl.float64)
        key64 = key.to(tl.float64)
        query = (
            query64 * tl.rsqrt(tl.sum(query64 * query64, axis=0) + 1e-6)
        ).to(tl.float32)
        key = (
            key64 * tl.rsqrt(tl.sum(key64 * key64, axis=0) + 1e-6)
        ).to(tl.float32)
    else:
        query = query * tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6)
        key = key * tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
    query = _canonical_round_fp32(query, CANONICAL_DROP_BITS)
    key = _canonical_round_fp32(key, CANONICAL_DROP_BITS)
    query = query * (KEY_DIM**-0.5)
    log_decay = tl.load(
        log_decay_ptr + token * g_stride_t + head * g_stride_h,
        mask=valid_slot,
        other=0.0,
    ).to(tl.float32)
    if CANONICAL_FP64:
        decay = tl.exp(log_decay.to(tl.float64))
    else:
        decay = tl.exp(log_decay)
    beta = tl.load(
        beta_ptr + token * b_stride_t + head * b_stride_h,
        mask=valid_slot,
        other=0.0,
    )
    contribution = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for factor in tl.static_range(0, RANK):
        key_factor_offsets = (
            safe_slot * kf_stride_s
            + head * kf_stride_h
            + factor * kf_stride_r
            + offsets_k * kf_stride_d
        )
        key_factor = tl.load(
            key_factor_ptr + key_factor_offsets,
            mask=mask_k & valid_slot,
            other=0.0,
        ).to(tl.float32)
        if CANONICAL_FP64:
            overlap = tl.sum(
                key_factor.to(tl.float64) * key.to(tl.float64), axis=0
            ).to(tl.float32)
        else:
            overlap = tl.sum(key_factor * key, axis=0)
        overlap = _canonical_round_fp32(overlap, CANONICAL_DROP_BITS)
        if CANONICAL_FP64:
            key_factor = (
                decay.to(tl.float64)
                * (
                    key_factor.to(tl.float64)
                    - beta.to(tl.float64)
                    * key.to(tl.float64)
                    * overlap.to(tl.float64)
                )
            ).to(tl.float32)
        else:
            key_factor = decay * (key_factor - beta * key * overlap)
        tl.store(
            key_factor_ptr + key_factor_offsets,
            key_factor,
            mask=mask_k & valid_slot,
        )
        if CANONICAL_FP64:
            coefficient = tl.sum(
                key_factor.to(tl.float64) * query.to(tl.float64), axis=0
            ).to(tl.float32)
        else:
            coefficient = tl.sum(key_factor * query, axis=0)
        coefficient = _canonical_round_fp32(coefficient, CANONICAL_DROP_BITS)
        value_factor = tl.load(
            value_factor_ptr
            + safe_slot * vf_stride_s
            + head * vf_stride_h
            + factor * vf_stride_r
            + offsets_v * vf_stride_d,
            mask=mask_v & valid_slot,
            other=0.0,
        ).to(tl.float32)
        term = _canonical_round_fp32(
            coefficient * value_factor, CANONICAL_DROP_BITS
        )
        contribution = _canonical_round_fp32(
            contribution + term, CANONICAL_DROP_BITS
        )
    # Defensive final boundary; idempotent under the production contract.
    contribution = _canonical_round_fp32(contribution, CANONICAL_DROP_BITS)
    budget = tl.load(budget_ptr + token, mask=valid_slot, other=0.0)
    # threshold is the cumulative mass *before* this ranked head.
    selected = (budget > 0.0) & (
        threshold < tl.minimum(1.0, budget * alpha)
    )
    output_offsets = token * out_stride_t + head * out_stride_h + offsets_v * out_stride_d
    old_output = tl.load(
        output_ptr + output_offsets,
        mask=mask_v & valid_slot,
        other=0.0,
    ).to(tl.float32)
    corrected = tl.where(
        selected,
        old_output - (1.0 - suppression_lambda) * contribution,
        old_output,
    )
    tl.store(
        output_ptr + output_offsets,
        corrected,
        mask=mask_v & valid_slot,
    )


def enabled() -> bool:
    return os.environ.get("ARIS_LINEAR_ENABLE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def exact_torch_graph_enabled() -> bool:
    """Use the audited ATen/cuBLAS arithmetic without forcing eager mode.

    vLLM captures this path in its decode CUDA graph, so Python does not run on
    every replay.  It is a release-candidate fallback for the case where a
    one-launch Triton recurrence cannot reproduce cuBLAS GEMV rounding closely
    enough over a long autoregressive trajectory.
    """
    return os.environ.get("ARIS_LINEAR_EXACT_TORCH_GRAPH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fail(message: str) -> None:
    raise RuntimeError(f"ARIS Linear fail-closed: {message}")


def get_linear_stats() -> dict[str, Any]:
    stats = dict(_STATS)
    if enabled():
        stats["max_effective_k_bound"] = _max_effective_k()
        stats["canonical_mantissa_bits"] = _canonical_mantissa_bits()
        stats["canonical_fp64_reductions"] = _canonical_fp64_enabled()
        stats["boundary_left_tokens"] = _BOUNDARY_LEFT_TOKENS
        stats["boundary_right_tokens"] = _BOUNDARY_RIGHT_TOKENS
        stats["boundary_rank"] = _RANK
    return stats


def _metadata_budgets_all_zero(metadata, tokens: int | None = None) -> bool:
    """Return whether this live metadata batch has no Linear suppression.

    ``ARIS_LINEAR_MAX_EFFECTIVE_K`` is a conservative *pre-tracking upper
    bound* used to decide which heads need a sidecar.  It is deliberately not
    the per-request K.  K=0 controls therefore must be decided from the
    request metadata installed by ``install_linear_metadata_patch``; otherwise
    a fixed-K=0 request still launches the Linear sidecar even though every
    selected mask is false.

    The release gates run with ``--enforce-eager``.  The small scalar reduction
    here is consequently safe and, for CUDA-graph profiling metadata without a
    live budget tensor, conservatively returns False so that profiling remains
    on the existing path.
    """
    value = getattr(metadata, "aris_budget_values", None)
    if not isinstance(value, torch.Tensor):
        return False
    # ``packed_fused_enabled`` is reached from the model forward while vLLM
    # captures its CUDA graphs.  A device-side reduction followed by
    # ``.item()`` is illegal during stream capture (and invalidates the whole
    # graph), even though the actual budget buffer is deliberately graph-safe
    # and updated in-place before each replay.  During capture we therefore
    # keep the integrated path enabled and let the Triton kernel consume the
    # captured buffer; an all-zero K=0 buffer naturally selects no heads,
    # while a later replay can carry the real per-request budgets.
    if value.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    if tokens is None:
        tokens = int(value.numel())
    tokens = max(0, min(int(tokens), int(value.numel())))
    if tokens == 0:
        return True
    return not bool(torch.any(value[:tokens] > 0.0).item())


def packed_fused_enabled(metadata=None, tokens: int | None = None) -> bool:
    """Whether the native packed GDN kernel should carry the sidecar."""
    if not enabled():
        return False
    if exact_torch_graph_enabled():
        return False
    # Diagnostic/release-candidate route: keep vLLM's stock packed GDN kernel
    # and apply the complete configured-rank boundary recurrence in the standalone
    # Triton kernel below.  This is intentionally separate from
    # ARIS_LINEAR_FORCE_TORCH: it contains no PyTorch fallback and lets the
    # long teacher-forced gate isolate whether a discrepancy comes from the
    # one-launch GDN integration or from the Triton boundary arithmetic itself.
    if os.environ.get("ARIS_LINEAR_FORCE_STANDALONE_TRITON", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    # A controller whose proven upper bound is exactly zero can never select
    # a Linear head.  Use vLLM's stock packed GDN kernel for this control arm:
    # it gives a genuinely native K=0 baseline and avoids compiling a
    # sidecar-free specialization of our fused kernel with a different
    # instruction schedule/rounding path.
    if _max_effective_k() <= 0.0:
        return False
    # Keep the global bound for head pre-tracking, but do not launch the
    # integrated sidecar for a live batch whose actual request budgets are all
    # exactly zero.  The caller then executes vLLM's stock GDN transition.
    if metadata is not None and _metadata_budgets_all_zero(metadata, tokens):
        return False
    return not (
        os.environ.get("ARIS_LINEAR_FORCE_TORCH", "0").lower()
        in {"1", "true", "yes", "on"}
        or bool(os.environ.get("ARIS_LINEAR_TRACE_PATH"))
    )


def packed_fused_inputs(layer, metadata, tokens: int) -> dict[str, Any]:
    if not packed_fused_enabled():
        _fail("packed_fused_inputs called while integrated path is disabled")
    if len(layer.kv_cache) != 4:
        _fail(f"layer {layer.layer_idx} has {len(layer.kv_cache)} cache tensors, expected 4")
    boundary_key, boundary_value = layer.kv_cache[2], layer.kv_cache[3]
    if boundary_key.ndim != 5 or int(boundary_key.shape[3]) != _RANK:
        _fail(f"integrated boundary-key cache has shape {tuple(boundary_key.shape)}")
    if boundary_value.ndim != 4 or int(boundary_value.shape[2]) != _RANK:
        _fail(f"integrated boundary-value cache has shape {tuple(boundary_value.shape)}")
    budgets = _metadata_tensor(metadata, "aris_budget_values", tokens, dtype=torch.float32)
    return {
        "boundary_key": boundary_key,
        "boundary_value": boundary_value,
        "budgets": budgets,
        "thresholds": _thresholds(layer, budgets.device)[int(layer.layer_idx)],
        "alpha": float(os.environ.get("ARIS_LINEAR_ALPHA", "1.0")),
        "suppression_lambda": float(os.environ.get("ARIS_LINEAR_LAMBDA", "0.0")),
        "max_effective_k": _max_effective_k(),
        "layer_idx": int(layer.layer_idx),
    }


def record_packed_fused_decode(tokens: int) -> None:
    _STATS["decode_integrated_calls"] = int(
        _STATS.get("decode_integrated_calls", 0)
    ) + 1
    _STATS["decode_integrated_tokens"] = int(
        _STATS.get("decode_integrated_tokens", 0)
    ) + int(tokens)


def _derived_controller_k_bound() -> float:
    """Return a conservative upper bound for every K emitted by the controller.

    Keeping the derivation next to the optimized kernel makes the optimization
    fail closed: a run cannot silently omit a boundary state for a head that a
    later controller step could select.
    """
    config_path = os.environ.get("ARIS_DYNAMIC_CONFIG_JSON")
    if not config_path:
        # Standalone operator replay has no controller.  Tracking every head is
        # the conservative behavior and keeps the replay utility self-contained.
        return 1.0
    path = Path(config_path)
    if not path.is_file():
        _fail(f"dynamic controller config does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"cannot read dynamic controller config {path}: {exc}")

    upper = float(payload.get("suppression_upper_bound", 1.0))
    scale = float(payload.get("suppression_scale", 1.0))
    initial = float(payload.get("initial_suppression", 0.0))
    if not all(math.isfinite(item) for item in (upper, scale, initial)):
        _fail("controller K bound fields must be finite")
    if upper < 0.0 or scale < 0.0 or initial < 0.0:
        _fail("controller K bound fields must be non-negative")

    # _mapped_suppression() can reach upper * scale and is then clamped.
    mapped_max = min(upper, upper * scale)
    effective_max = mapped_max
    if bool(payload.get("dynamic_question_lambda", False)):
        effective_max = min(
            upper,
            upper * float(payload.get("dynamic_lambda_late_scale_max", 0.80)),
        )
    elif bool(payload.get("high_low_high_question_lambda", False)):
        effective_max = min(
            upper,
            upper
            * max(
                float(payload.get("high_low_high_early_scale", 0.80)),
                float(payload.get("high_low_high_middle_scale", 0.40)),
                float(payload.get("high_low_high_late_scale_max", 0.80)),
            ),
        )
    elif bool(payload.get("linear_down_up_question_lambda", False)):
        effective_max = min(
            upper,
            upper
            * max(
                float(payload.get("linear_down_up_high_scale", 0.80)),
                float(payload.get("linear_down_up_low_scale", 0.40)),
            ),
        )
    else:
        calibration = payload.get("question_lambda_calibration")
        if isinstance(calibration, dict):
            effective_max = min(
                upper,
                mapped_max * float(calibration.get("lambda_max", 1.0)),
            )

    # The initial value is installed before any checkpoint/lambda update.
    return min(1.0, upper, max(initial, effective_max))


def _max_effective_k() -> float:
    global _MAX_EFFECTIVE_K_CACHE
    if _MAX_EFFECTIVE_K_CACHE is not None:
        return _MAX_EFFECTIVE_K_CACHE
    derived = _derived_controller_k_bound()
    declared_text = os.environ.get("ARIS_LINEAR_MAX_EFFECTIVE_K")
    if declared_text is None:
        bound = derived
    else:
        try:
            bound = float(declared_text)
        except ValueError:
            _fail(f"invalid ARIS_LINEAR_MAX_EFFECTIVE_K={declared_text!r}")
        if not math.isfinite(bound) or not 0.0 <= bound <= 1.0:
            _fail(f"ARIS_LINEAR_MAX_EFFECTIVE_K must be in [0,1], got {bound}")
        if bound + 1e-7 < derived:
            _fail(
                f"declared max effective K {bound:.9g} is below controller bound "
                f"{derived:.9g}"
            )
    _MAX_EFFECTIVE_K_CACHE = float(bound)
    return _MAX_EFFECTIVE_K_CACHE


def extend_state_shapes(layer, shapes):
    shapes = tuple(shapes)
    if not enabled():
        return shapes
    if len(shapes) != 2:
        _fail(f"expected two native GDN states, got {len(shapes)}")
    local_heads = int(layer.num_v_heads // layer.tp_size)
    value_tiles = math.ceil(int(layer.head_v_dim) / _FUSED_VALUE_BLOCK)
    return shapes + (
        (local_heads, value_tiles, _RANK, int(layer.head_k_dim)),
        (local_heads, _RANK, int(layer.head_v_dim)),
    )


def extend_state_dtypes(dtypes):
    dtypes = tuple(dtypes)
    if not enabled():
        return dtypes
    if len(dtypes) != 2:
        _fail(f"expected two native GDN dtypes, got {len(dtypes)}")
    # The ARIS sidecar stores the low-rank boundary state B, not the native
    # recurrent state.  Keeping B in the native BF16 dtype introduces a
    # quantisation point before the correction is read back; that drift is
    # amplified when the corrected hidden state is propagated through the
    # remaining layers.  The Transformers oracle keeps B in FP32, so the
    # effect-equivalence path uses FP32 sidecars by default.  An explicit
    # native override remains available for memory-constrained ablations.
    raw = os.environ.get("ARIS_LINEAR_SIDECAR_DTYPE", "float32").strip().lower()
    if raw in {"float32", "fp32", "torch.float32"}:
        sidecar_dtype = torch.float32
    elif raw in {"native", "model", "bf16", "bfloat16", "torch.bfloat16"}:
        sidecar_dtype = dtypes[1]
    else:
        _fail(
            "ARIS_LINEAR_SIDECAR_DTYPE must be float32 or native, "
            f"got {raw!r}"
        )
    return dtypes + (sidecar_dtype, sidecar_dtype)


def _repeat_to_value_heads(tensor: torch.Tensor, value_heads: int) -> torch.Tensor:
    key_heads = int(tensor.shape[-2])
    if value_heads % key_heads:
        _fail(f"value heads {value_heads} are not divisible by key heads {key_heads}")
    return tensor.repeat_interleave(value_heads // key_heads, dim=-2)


def _linear_summary_path() -> Path:
    value = os.environ.get("ARIS_LINEAR_SUMMARY_JSON")
    if not value:
        _fail("ARIS_LINEAR_SUMMARY_JSON is required")
    path = Path(value)
    if not path.is_file():
        _fail(f"linear summary does not exist: {path}")
    return path


def _thresholds(layer, device: torch.device) -> torch.Tensor:
    local_heads = int(layer.num_v_heads // layer.tp_size)
    key = (str(device), int(layer.tp_rank), local_heads)
    cached = _THRESHOLD_CACHE.get(key)
    if cached is not None:
        return cached
    ranked = rank_linear_heads(_linear_summary_path())
    table = torch.full(
        (int(layer.config.num_hidden_layers), local_heads),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    global_offset = int(layer.tp_rank) * local_heads
    cumulative_before = 0.0
    for item in ranked:
        local = int(item.head) - global_offset
        if 0 <= local < local_heads:
            table[int(item.layer), local] = float(cumulative_before)
        cumulative_before = float(item.cumulative)
    _THRESHOLD_CACHE[key] = table
    return table


def _tracked_heads(layer, device: torch.device) -> torch.Tensor:
    """Compact exact superset of heads that can be selected in this run."""
    alpha = float(os.environ.get("ARIS_LINEAR_ALPHA", "1.0"))
    max_k = _max_effective_k()
    key = (
        str(device),
        int(layer.tp_rank),
        int(layer.layer_idx),
        float(alpha),
        float(max_k),
    )
    cached = _TRACKED_HEAD_CACHE.get(key)
    if cached is not None:
        return cached
    threshold = _thresholds(layer, device)[int(layer.layer_idx)]
    target = min(1.0, max_k * alpha)
    tracked = torch.nonzero(threshold < target, as_tuple=False).flatten().to(torch.int32)
    tracked = tracked.contiguous()
    _TRACKED_HEAD_CACHE[key] = tracked
    return tracked


def _prewarm_threshold_tables(runner) -> None:
    """Materialize per-layer selection tensors before CUDA-graph capture.

    The first fused decode graph can call ``_thresholds`` while the stream is
    already being captured.  Building a CUDA tensor (and filling it from the
    Python-ranked summary) is not a graph-capturable operation.  vLLM gives us
    one ordinary warmup immediately before each capture, so walk the loaded
    model there and populate both the threshold and tracked-head caches.  The
    tensors are then stable graph inputs for every subsequent replay.
    """
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "modules"):
        print("[ARIS] threshold/fused warmup skipped: runner.model has no modules", flush=True)
        return
    threshold_seen: set[tuple[int, str]] = set()
    warmed_layers = 0
    for layer in model.modules():
        if not all(
            hasattr(layer, name)
            for name in (
                "num_v_heads",
                "num_k_heads",
                "head_k_dim",
                "head_v_dim",
                "tp_size",
                "tp_rank",
                "layer_idx",
            )
        ):
            continue
        local_heads = int(layer.num_v_heads) // int(layer.tp_size)
        if local_heads <= 0:
            continue
        device = None
        for parameter in layer.parameters():
            candidate = getattr(parameter, "device", None)
            if candidate is not None and candidate.type == "cuda":
                device = candidate
                break
        if device is None:
            continue
        cache_key = (int(layer.tp_rank), str(device))
        if cache_key not in threshold_seen:
            _thresholds(layer, device)
            threshold_seen.add(cache_key)
        # ``_tracked_heads`` is layer-specific even though the threshold
        # table itself is shared across all layers on a TP rank.  Warm it
        # for every layer before capture, not just the first one.
        _tracked_heads(layer, device)
        _prewarm_fused_boundary_kernel(layer, device)
        warmed_layers += 1
    print(
        f"[ARIS] threshold/fused warmup completed: layers={warmed_layers} "
        f"threshold_tables={len(_THRESHOLD_CACHE)} fused_specializations={len(_FUSED_WARMUP_CACHE)}",
        flush=True,
    )


def _prewarm_fused_boundary_kernel(layer, device: torch.device) -> None:
    """Load the integrated fused-GDN Triton binary outside CUDA Graph capture.

    vLLM's ordinary K=0 warmup intentionally takes the native bypass, so the
    ARIS fused kernel may otherwise be invoked for the first time from a graph
    capture.  Triton then attempts ``cuModuleLoad`` on the capturing stream,
    which CUDA rejects.  A one-token synthetic call with the actual layer
    dimensions materializes the same specialization and its module handle;
    it has no effect on model/cache state and is amortized per worker process.
    """
    from .fused_gdn_boundary import (
        fused_recurrent_gated_delta_rule_packed_decode_with_boundary,
    )

    tp_size = int(layer.tp_size)
    tp_rank = int(layer.tp_rank)
    key_heads = int(layer.num_k_heads) // tp_size
    value_heads = int(layer.num_v_heads) // tp_size
    key_dim = int(layer.head_k_dim)
    value_dim = int(layer.head_v_dim)
    rank = int(_RANK)
    cache_key = (
        str(device),
        tp_rank,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )
    if cache_key in _FUSED_WARMUP_CACHE:
        return
    # The runtime contract currently uses TP=1; retain explicit divisibility
    # checks so a future unsupported topology fails before graph capture.
    if key_heads <= 0 or value_heads <= 0:
        return
    model_dtype = getattr(getattr(layer, "model_config", None), "dtype", None)
    dtype = model_dtype if model_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

    # Triton specializes not only on H/HV/K/V but also on every constexpr
    # stride and tensor dtype.  The profiling KV cache is already installed
    # when this hook runs; use it as the exact layout oracle instead of
    # assuming a contiguous state or deriving the recurrent dtype from the
    # Transformers config (``mamba_ssm_dtype`` is a compute hint, while vLLM's
    # cache dtype is controlled by ``mamba_ssm_cache_dtype``).
    kv_cache = getattr(layer, "kv_cache", None)
    state_template = None
    boundary_key_template = None
    boundary_value_template = None
    if isinstance(kv_cache, (tuple, list)) and len(kv_cache) >= 4:
        candidates = (kv_cache[1], kv_cache[2], kv_cache[3])
        if all(isinstance(candidate, torch.Tensor) for candidate in candidates):
            state_template, boundary_key_template, boundary_value_template = candidates

    def _template_dtype(template, fallback):
        value = getattr(template, "dtype", None)
        return value if isinstance(value, torch.dtype) else fallback

    state_dtype = _template_dtype(state_template, dtype)
    sidecar_key_dtype = _template_dtype(boundary_key_template, torch.float32)
    sidecar_value_dtype = _template_dtype(boundary_value_template, torch.float32)
    value_block = min(1 << max(0, (value_dim - 1).bit_length()), 32)
    value_tiles = (value_dim + value_block - 1) // value_block
    qkv_width = 2 * key_heads * key_dim + value_heads * value_dim
    # Qwen3.5's mixed QKV view is sliced out of the wider QKVZ projection.
    # Preserve that parent row stride (2*K + 2*V), which is 12288 for 9B and
    # 8192 for 2B; a contiguous [B, QKV] warmup would compile a different
    # Triton specialization.
    qkvz_width = 2 * key_heads * key_dim + 2 * value_heads * value_dim
    mixed_qkv_stride0 = max(qkv_width, qkvz_width)
    mixed_qkv = torch.empty_strided(
        (1, qkv_width),
        (mixed_qkv_stride0, 1),
        dtype=dtype,
        device=device,
    ).zero_()
    a = torch.zeros((1, value_heads), dtype=dtype, device=device)
    b = torch.zeros((1, value_heads), dtype=dtype, device=device)
    A_log_dtype = _template_dtype(getattr(layer, "A_log", None), torch.float32)
    dt_bias_dtype = _template_dtype(getattr(layer, "dt_bias", None), dtype)
    A_log = torch.zeros((value_heads,), dtype=A_log_dtype, device=device)
    dt_bias = torch.zeros((value_heads,), dtype=dt_bias_dtype, device=device)

    def _cache_view(template, shape, fallback_strides, fallback_dtype):
        if isinstance(template, torch.Tensor) and template.ndim == len(shape):
            strides = tuple(int(x) for x in template.stride())
            # Replace the cache-slot extent by one while retaining its padded
            # slot stride and all inner strides exactly.
            view_shape = (1,) + tuple(int(x) for x in shape[1:])
            return torch.empty_strided(
                view_shape,
                strides,
                dtype=template.dtype,
                device=device,
            ).zero_()
        return torch.zeros(shape, dtype=fallback_dtype, device=device)

    initial_state = _cache_view(
        state_template,
        (1, value_heads, value_dim, key_dim),
        None,
        state_dtype,
    )
    out = torch.empty((1, 1, value_heads, value_dim), dtype=dtype, device=device)
    state_indices = torch.zeros((1,), dtype=torch.int32, device=device)
    boundary_key = _cache_view(
        boundary_key_template,
        (1, value_heads, value_tiles, rank, key_dim),
        None,
        sidecar_key_dtype,
    )
    boundary_value = _cache_view(
        boundary_value_template,
        (1, value_heads, rank, value_dim),
        None,
        sidecar_value_dtype,
    )
    budgets = torch.zeros((1,), dtype=torch.float32, device=device)
    thresholds = torch.full(
        (value_heads,), float("inf"), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        fused_recurrent_gated_delta_rule_packed_decode_with_boundary(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim ** -0.5,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=state_indices,
            boundary_key=boundary_key,
            boundary_value=boundary_value,
            budgets=budgets,
            thresholds=thresholds,
            alpha=1.0,
            suppression_lambda=0.0,
            max_effective_k=0.8,
            use_qk_l2norm_in_kernel=True,
            layer_idx=int(getattr(layer, "layer_idx", -1)),
        )
    torch.cuda.synchronize(device)
    _FUSED_WARMUP_CACHE.add(cache_key)


def _install_threshold_warmup_patch() -> None:
    """Prewarm CUDA tensors used by dynamic selection before graph capture."""
    global _WARMUP_PATCHED
    if _WARMUP_PATCHED:
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = getattr(GPUModelRunner, "_warmup_and_capture", None)
    if original is None:
        _WARMUP_PATCHED = True
        return
    if getattr(GPUModelRunner, "_aris_threshold_warmup_patch_installed", False):
        _WARMUP_PATCHED = True
        return

    def warmup_with_thresholds(self, *args, **kwargs):
        # This runs after model loading and immediately before vLLM starts the
        # ordinary warmup / CUDA-graph capture sequence.  All tensors created
        # by the dynamic selector therefore exist before the stream enters
        # capture; the wrapped method remains byte-for-byte native otherwise.
        _prewarm_threshold_tables(self)
        return original(self, *args, **kwargs)

    GPUModelRunner._warmup_and_capture = warmup_with_thresholds
    GPUModelRunner._aris_threshold_warmup_patch_installed = True
    _WARMUP_PATCHED = True


def _selection_mask(layer, budgets: torch.Tensor) -> torch.Tensor:
    alpha = float(os.environ.get("ARIS_LINEAR_ALPHA", "1.0"))
    target = budgets.float().clamp(0.0, 1.0).unsqueeze(-1) * alpha
    threshold = _thresholds(layer, budgets.device)[int(layer.layer_idx)].unsqueeze(0)
    return threshold < target.clamp_max(1.0)


def _metadata_tensor(metadata, name: str, length: int, *, dtype: torch.dtype) -> torch.Tensor:
    value = getattr(metadata, name, None)
    if not isinstance(value, torch.Tensor):
        _fail(f"metadata.{name} is missing")
    if value.dtype != dtype:
        _fail(f"metadata.{name} dtype {value.dtype} != {dtype}")
    if int(value.numel()) < int(length):
        _fail(f"metadata.{name} has {value.numel()} rows, expected at least {length}")
    return value[:length]


def _trace(
    *,
    layer: int,
    phase: str,
    contribution: torch.Tensor,
    key_factors: torch.Tensor,
    budgets: torch.Tensor,
    selected: torch.Tensor,
    corrected_output: torch.Tensor,
) -> None:
    global _TRACE_COUNT
    path_text = os.environ.get("ARIS_LINEAR_TRACE_PATH")
    trace_backend = "vllm"
    # The production fused path still initializes B during the ordinary
    # prefill replay.  Emit that one event into the fused trace without
    # setting ARIS_LINEAR_TRACE_PATH, because the latter intentionally routes
    # decode through the transparent diagnostic implementation.
    if (
        not path_text
        and phase == "prefill"
        and os.environ.get("ARIS_FUSED_LINEAR_TRACE_PATH")
    ):
        path_text = os.environ.get("ARIS_FUSED_LINEAR_TRACE_PATH")
        trace_backend = "vllm_fused"
    if not path_text:
        return
    limit = int(os.environ.get("ARIS_LINEAR_TRACE_MAX_EVENTS", "0") or "0")
    if limit > 0 and _TRACE_COUNT >= limit:
        return
    payload = {
        "backend": trace_backend,
        "event_index": int(_TRACE_COUNT),
        "layer": int(layer),
        "phase": str(phase),
        "budgets": budgets.detach().float().cpu().tolist(),
        "selected_heads": selected.detach().cpu().nonzero(as_tuple=False).tolist(),
        "boundary_key_norm": key_factors.detach().float().norm(dim=-1).cpu().tolist(),
        "contribution_norm": contribution.detach().float().norm(dim=-1).cpu().tolist(),
    }
    if os.environ.get("ARIS_NUMERIC_TRACE_VALUES", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        selected_pairs = selected.detach().cpu().nonzero(as_tuple=False).tolist()
        payload["selected_contribution_values"] = {
            f"{row}:{head}": contribution.detach().float()[row, head].cpu().tolist()
            for row, head in selected_pairs
        }
        payload["selected_boundary_key_values"] = {
            f"{row}:{head}": key_factors.detach().float()[row, head].cpu().tolist()
            for row, head in selected_pairs
        }
    if os.environ.get("ARIS_NUMERIC_TRACE_OUTPUT_VALUES", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        payload["corrected_output_values"] = (
            corrected_output.detach().float().cpu().tolist()
        )
    path = Path(path_text.format(rank=os.environ.get("LOCAL_RANK", "0")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _TRACE_COUNT += 1


def trace_packed_fused_state(
    *,
    mixed_qkv: torch.Tensor,
    state_indices: torch.Tensor,
    boundary_key: torch.Tensor,
    boundary_value: torch.Tensor,
    budgets: torch.Tensor,
    thresholds: torch.Tensor,
    alpha: float,
    layer_idx: int,
    key_heads: int,
    key_dim: int,
    value_heads: int,
    corrected_output: torch.Tensor,
) -> None:
    """Serialize the actual fused B/q^T B for a diagnostic engine run."""
    global _FUSED_TRACE_COUNT
    raw_path = os.environ.get("ARIS_FUSED_LINEAR_TRACE_PATH")
    if not raw_path:
        return
    limit = int(os.environ.get("ARIS_FUSED_LINEAR_TRACE_MAX_EVENTS", "0") or "0")
    if limit > 0 and _FUSED_TRACE_COUNT >= limit:
        return
    batch = int(mixed_qkv.shape[0])
    if bool((state_indices[:batch] < 0).any().item()) or not bool(
        budgets[:batch].float().gt(0).any().item()
    ):
        # CUDA/profile rows have no live request and no meaningful B state.
        return
    q = mixed_qkv[:, : key_heads * key_dim].reshape(batch, key_heads, key_dim)
    q = _repeat_to_value_heads(q, value_heads)
    q = normalize_qk_fp32(q) * (key_dim**-0.5)
    slots = state_indices[:batch].long()
    key_factors = boundary_key.index_select(0, slots)[:, :, 0].float()
    value_factors = boundary_value.index_select(0, slots).float()
    contribution = boundary_contribution(
        key_factors,
        value_factors,
        q,
        query_is_normalized_and_scaled=True,
    )
    selected = (
        thresholds.unsqueeze(0)
        < (budgets[:batch].float().unsqueeze(-1) * float(alpha)).clamp_max(1.0)
    ) & budgets[:batch].float().gt(0).unsqueeze(-1)
    pairs = selected.detach().cpu().nonzero(as_tuple=False).tolist()
    payload = {
        "backend": "vllm_fused",
        "event_index": int(_FUSED_TRACE_COUNT),
        "layer": int(layer_idx),
        "phase": "decode",
        "budgets": budgets[:batch].detach().float().cpu().tolist(),
        "selected_heads": pairs,
        "selected_contribution_values": {
            f"{row}:{head}": contribution[row, head].detach().float().cpu().tolist()
            for row, head in pairs
        },
        "selected_boundary_key_values": {
            f"{row}:{head}": key_factors[row, head].detach().float().cpu().tolist()
            for row, head in pairs
        },
    }
    if os.environ.get("ARIS_NUMERIC_TRACE_OUTPUT_VALUES", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        payload["corrected_output_values"] = (
            corrected_output[:batch].detach().float().cpu().tolist()
        )
    path = Path(raw_path.format(rank=os.environ.get("LOCAL_RANK", "0")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _FUSED_TRACE_COUNT += 1


def _apply_correction(
    *,
    layer,
    output: torch.Tensor,
    contribution: torch.Tensor,
    budgets: torch.Tensor,
    key_factors: torch.Tensor,
    phase: str,
) -> None:
    selected = _selection_mask(layer, budgets)
    active = budgets.float().gt(0).unsqueeze(-1)
    selected = selected & active
    if os.environ.get("ARIS_LINEAR_DIAGNOSTIC_TRACK_ONLY", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        mask = selected.unsqueeze(-1).to(contribution.dtype)
        suppression_lambda = float(os.environ.get("ARIS_LINEAR_LAMBDA", "0.0"))
        # Match both the Transformers numerical oracle and the integrated
        # packed-GDN kernel exactly: the native GDN result is already rounded
        # to ``output.dtype``; apply the boundary correction in FP32 and round
        # only the final corrected value.  Casting the contribution to BF16
        # before subtraction looks harmless in a one-layer operator check, but
        # its error is amplified by the remaining 24 GDN/MLP blocks and makes
        # the transparent and fused engines diverge from the first decode
        # token onward.
        correction = (1.0 - suppression_lambda) * contribution.float() * mask.float()
        output.copy_((output.float() - correction).to(output.dtype))
    _trace(
        layer=int(layer.layer_idx),
        phase=phase,
        contribution=contribution,
        key_factors=key_factors,
        budgets=budgets,
        selected=selected,
        corrected_output=output,
    )


def decode_boundary_correction(
    *,
    layer,
    metadata,
    query: torch.Tensor,
    key: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    if not enabled():
        return
    if len(layer.kv_cache) != 4:
        _fail(f"layer {layer.layer_idx} has {len(layer.kv_cache)} cache tensors, expected 4")
    # Generic GDN emits [1,T,HV,V], while its packed fast path emits
    # [B,1,HV,V].  The old ``shape[-3]`` rule silently treated packed B>1 as
    # one token and updated only the first request.
    packed_output = bool(
        output.ndim == 4
        and int(output.shape[1]) == 1
        and int(output.shape[0]) == int(state_indices.numel())
    )
    if output.ndim == 4:
        tokens = int(output.shape[0] if packed_output else output.shape[1])
    else:
        tokens = int(output.shape[0])
    if tokens <= 0:
        return
    if not isinstance(getattr(metadata, "aris_budget_values", None), torch.Tensor):
        # vLLM CUDA-graph profiling constructs synthetic attention metadata
        # without live request ids and runs decode before the normal metadata
        # builder.  Sidecars are zero in that path, so correction is a no-op.
        # Real requests always carry aris_req_ids; keep those fail-closed.
        if list(getattr(metadata, "aris_req_ids", ()) or ()):
            _fail("live decode metadata.aris_budget_values is missing")
        _STATS["profile_decode_skips"] += 1
        return
    states = state_indices[:tokens].long()
    exact_graph = exact_torch_graph_enabled()
    diagnostic_torch = os.environ.get("ARIS_LINEAR_FORCE_TORCH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or bool(os.environ.get("ARIS_LINEAR_TRACE_PATH")) or exact_graph
    if diagnostic_torch and not exact_graph and bool((states < 0).any().item()):
        _fail("decode state index contains padding/negative slot")
    raw_key_cache, value_cache = layer.kv_cache[2], layer.kv_cache[3]
    key_cache = raw_key_cache[:, :, 0] if raw_key_cache.ndim == 5 else raw_key_cache
    # A K=0 release/no-op process has no Linear correction budget at all.  In
    # that case return before touching the device budget tensor: vLLM may call
    # this hook from CUDA-graph profiling, where a host scalar reduction
    # (`torch.any(...).item()`) is illegal even though the native GDN path is
    # otherwise perfectly valid.  Positive-K requests retain the dynamic
    # per-request check below.
    if _max_effective_k() <= 0.0:
        _STATS["decode_native_k0_bypass_calls"] += 1
        _STATS["decode_native_k0_bypass_tokens"] += tokens
        return
    budgets = _metadata_tensor(metadata, "aris_budget_values", tokens, dtype=torch.float32)
    if not bool(torch.any(budgets > 0.0).item()):
        # The native GDN transition has already run in the caller.  Do not
        # propagate or read a Linear boundary sidecar when this request's
        # actual K is zero; the global pre-tracking bound may still be > 0 for
        # later dynamic requests in the same process.
        _STATS["decode_native_k0_bypass_calls"] += 1
        _STATS["decode_native_k0_bypass_tokens"] += tokens
        return
    output_view = (
        output[:, 0]
        if packed_output
        else output.squeeze(0) if output.ndim == 4 else output
    )
    if not diagnostic_torch:
        if query.device.type != "cuda" or key_cache.device.type != "cuda":
            _fail("formal fused decode requires CUDA tensors")
        if int(layer.head_k_dim) != 128 or int(layer.head_v_dim) != 128 or int(key_cache.shape[2]) != _RANK:
            _fail(
                f"unsupported fused shape K={layer.head_k_dim}, V={layer.head_v_dim}, rank={key_cache.shape[2]}"
            )
        q = query[:, :tokens].squeeze(0)
        k = key[:, :tokens].squeeze(0)
        g = log_decay[:, :tokens].squeeze(0)
        b = beta[:, :tokens].squeeze(0)
        thresholds = _thresholds(layer, query.device)[int(layer.layer_idx)]
        # Only launch programs for heads that can ever be selected under this
        # controller/alpha combination.  Other heads can never affect output,
        # so leaving their private B cache unpropagated is mathematically exact.
        tracked_heads = _tracked_heads(layer, query.device)
        tracked_count = int(tracked_heads.numel())
        if tracked_count == 0:
            # packed_fused_enabled() routed this exact-K0 request through the
            # stock vLLM kernel before entering this no-op bookkeeping hook.
            _STATS["decode_native_k0_bypass_calls"] += 1
            _STATS["decode_native_k0_bypass_tokens"] += tokens
            return
        _aris_linear_decode_kernel[(tokens, tracked_count)](
            q,
            k,
            g,
            b,
            output_view[:tokens],
            key_cache,
            value_cache,
            state_indices[:tokens],
            budgets,
            thresholds,
            tracked_heads,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            g.stride(0),
            g.stride(1),
            b.stride(0),
            b.stride(1),
            output_view.stride(0),
            output_view.stride(1),
            output_view.stride(2),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            thresholds.stride(0),
            float(os.environ.get("ARIS_LINEAR_ALPHA", "1.0")),
            float(os.environ.get("ARIS_LINEAR_LAMBDA", "0.0")),
            VALUE_HEADS=int(key_cache.shape[1]),
            KEY_HEADS=int(q.shape[1]),
            KEY_DIM=int(layer.head_k_dim),
            VALUE_DIM=int(layer.head_v_dim),
            RANK=_RANK,
            BLOCK_K=128,
            BLOCK_V=128,
            CANONICAL_DROP_BITS=23 - _canonical_mantissa_bits()
            if _canonical_mantissa_bits() > 0
            else 0,
            CANONICAL_FP64=_canonical_fp64_enabled(),
            # One warp matches the reduction topology used by each value tile
            # in the integrated packed kernel.  This standalone path is the
            # auditable Triton bridge between the PyTorch oracle and the
            # one-launch production kernel.
            num_warps=1,
        )
        _STATS["decode_triton_calls"] += 1
        _STATS["decode_triton_tokens"] += tokens
        _STATS["decode_triton_head_programs"] = int(
            _STATS.get("decode_triton_head_programs", 0)
        ) + tokens * tracked_count
        if raw_key_cache.ndim == 5 and int(raw_key_cache.shape[2]) > 1:
            raw_key_cache[:, :, 1:].copy_(
                key_cache.unsqueeze(2).expand_as(raw_key_cache[:, :, 1:])
            )
        return

    # CUDA-graph capture uses a negative dummy slot.  Clamp it to a real cache
    # row and mask the update back to the old value; real batch-1 replay then
    # substitutes the request's actual slot through the captured input tensor.
    # Formal MMMU evaluation runs one request per model instance, so there is
    # no mixed live/padded duplicate-slot write.  The eager diagnostic path
    # remains fail-closed for every negative slot above.
    valid_states = states.ge(0)
    safe_states = states.clamp_min(0) if exact_graph else states
    key_factors = key_cache.index_select(0, safe_states).float()
    value_factors = value_cache.index_select(0, safe_states).float()
    old_key_factors = key_factors
    value_heads = int(key_factors.shape[1])
    q = _repeat_to_value_heads(query[:, :tokens], value_heads).squeeze(0)
    k = _repeat_to_value_heads(key[:, :tokens], value_heads).squeeze(0)
    g = log_decay[:, :tokens].squeeze(0)
    b = beta[:, :tokens].squeeze(0)
    normalized_q = normalize_qk_fp32(q) * (q.shape[-1] ** -0.5)
    normalized_k = normalize_qk_fp32(k)
    key_factors = propagate_boundary_keys(
        key_factors,
        normalized_k,
        g,
        b,
        key_is_normalized=True,
    )
    if exact_graph:
        key_factors = torch.where(
            valid_states[:, None, None, None], key_factors, old_key_factors
        )
    contribution = boundary_contribution(
        key_factors,
        value_factors,
        normalized_q,
        query_is_normalized_and_scaled=True,
    )
    key_cache.index_copy_(0, safe_states, key_factors.to(key_cache.dtype))
    if raw_key_cache.ndim == 5:
        raw_key_cache.index_copy_(
            0,
            safe_states,
            key_factors
            .unsqueeze(2)
            .expand(-1, -1, int(raw_key_cache.shape[2]), -1, -1)
            .to(raw_key_cache.dtype),
        )
    _apply_correction(
        layer=layer,
        output=output_view[:tokens],
        contribution=contribution,
        budgets=budgets,
        key_factors=key_factors,
        phase="decode",
    )
    _STATS["decode_torch_calls"] += 1
    if exact_graph:
        _STATS["decode_exact_graph_capture_calls"] = int(
            _STATS.get("decode_exact_graph_capture_calls", 0)
        ) + 1


def prefill_boundary_correction(
    *,
    layer,
    metadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    if not enabled():
        return
    _STATS["prefill_calls"] += 1
    if len(layer.kv_cache) != 4:
        _fail(f"layer {layer.layer_idx} has {len(layer.kv_cache)} cache tensors, expected 4")
    computed = list(getattr(metadata, "aris_computed_lens_cpu", ()) or ())
    end_positions = list(getattr(metadata, "aris_found_end_idx_cpu", ()) or ())
    if not computed or len(computed) != len(end_positions):
        _fail("prefill requires aligned aris_computed_lens_cpu and aris_found_end_idx_cpu")
    q_starts = [int(x) for x in query_start_loc.detach().cpu().tolist()]
    requests = min(len(computed), len(q_starts) - 1, int(state_indices.numel()))
    raw_key_cache, value_cache = layer.kv_cache[2], layer.kv_cache[3]
    key_cache = raw_key_cache[:, :, 0] if raw_key_cache.ndim == 5 else raw_key_cache
    budget_values = _metadata_tensor(metadata, "aris_budget_values", requests, dtype=torch.float32)
    value_heads = int(value.shape[-2])

    req_ids = [str(x) for x in (getattr(metadata, "aris_req_ids", ()) or ())]
    for request_idx, req_id in enumerate(req_ids[:requests]):
        if req_id in _FIRST_FORWARD_ASSERTED_REQUESTS:
            continue
        try:
            from experiment_src.qwen35_vllm_dynamic_final.vendor import (
                aris_vllm_v1_suppression as native,
            )

            tagged, tagged_budget = native._parse_request_budget(req_id)
            actual_budget = float(budget_values[request_idx].item())
            if not tagged or tagged_budget is None:
                _fail(f"request {req_id} has no initial-K tag")
            if not math.isclose(
                actual_budget, float(tagged_budget), rel_tol=0.0, abs_tol=1e-6
            ):
                _fail(
                    f"request {req_id} first Linear K={actual_budget:.6g}, "
                    f"tagged K={float(tagged_budget):.6g}"
                )
            if int(end_positions[request_idx]) < 0:
                _fail(f"request {req_id} has no resolved </think> boundary")
            _FIRST_FORWARD_ASSERTED_REQUESTS.add(req_id)
            _STATS["first_forward_budget_assertions"] += 1
        except Exception:
            _STATS["first_forward_budget_assertion_failures"] += 1
            raise

    for request_idx in range(requests):
        q_start, q_end = q_starts[request_idx], q_starts[request_idx + 1]
        q_len = q_end - q_start
        if q_len <= 0:
            continue
        slot = int(state_indices[request_idx].item())
        if slot < 0:
            continue
        if float(budget_values[request_idx].item()) <= 0.0:
            # Native prefill has already performed the real recurrent
            # transition.  This hook owns only the ARIS boundary sidecar, so
            # an actual K=0 request must leave that sidecar/cache untouched.
            _STATS["prefill_native_k0_bypass_calls"] += 1
            continue
        has_initial = bool(metadata.has_initial_state[request_idx].item())
        if not has_initial:
            key_cache[slot].zero_()
            value_cache[slot].zero_()
        key_factors = key_cache[slot : slot + 1].float()
        value_factors = value_cache[slot : slot + 1].float()
        absolute_start = int(computed[request_idx])
        local_end = int(end_positions[request_idx]) - absolute_start
        local_boundary_start = local_end - _BOUNDARY_LEFT_TOKENS
        local_boundary_stop = local_end + _BOUNDARY_RIGHT_TOKENS + 1

        q_seq = query[:, q_start:q_end]
        k_seq = key[:, q_start:q_end]
        v_seq = value[:, q_start:q_end]
        g_seq = log_decay[:, q_start:q_end]
        b_seq = beta[:, q_start:q_end]
        start_token = 0
        main_state = initial_state[request_idx : request_idx + 1].float()

        if 0 <= local_boundary_start and local_boundary_stop <= q_len:
            if local_boundary_start > 0:
                prefix_cu = torch.tensor(
                    [0, local_boundary_start],
                    dtype=torch.int32,
                    device=query.device,
                )
                _, state_before = layer.chunk_gated_delta_rule(
                    q=q_seq[:, :local_boundary_start],
                    k=k_seq[:, :local_boundary_start],
                    v=v_seq[:, :local_boundary_start],
                    g=g_seq[:, :local_boundary_start],
                    beta=b_seq[:, :local_boundary_start],
                    initial_state=main_state,
                    output_final_state=True,
                    cu_seqlens=prefix_cu,
                    use_qk_l2norm_in_kernel=True,
                )
                main_state = state_before.float()
            start_token = local_boundary_start
        elif key_factors.abs().max().item() == 0.0:
            # The boundary is not in this chunk and has not been written yet.
            continue

        last_contribution = None
        for token_idx in range(start_token, q_len):
            q_t = normalize_qk_fp32(
                _repeat_to_value_heads(q_seq[:, token_idx], value_heads)
            )
            k_t = normalize_qk_fp32(
                _repeat_to_value_heads(k_seq[:, token_idx], value_heads)
            )
            absolute_position = absolute_start + token_idx
            boundary_absolute_start = (
                int(end_positions[request_idx]) - _BOUNDARY_LEFT_TOKENS
            )
            boundary_absolute_stop = (
                int(end_positions[request_idx]) + _BOUNDARY_RIGHT_TOKENS + 1
            )
            write_slot = (
                absolute_position - boundary_absolute_start
                if boundary_absolute_start
                <= absolute_position
                < boundary_absolute_stop
                else None
            )
            key_factors, value_factors, step = boundary_step(
                key_factors,
                value_factors,
                main_state,
                q_t,
                k_t,
                v_seq[:, token_idx],
                g_seq[:, token_idx],
                b_seq[:, token_idx],
                write_slot=write_slot,
                qk_are_normalized=True,
            )
            main_state = step.state
            last_contribution = boundary_contribution(
                key_factors,
                value_factors,
                step.query,
                query_is_normalized_and_scaled=True,
            )
        key_cache[slot].copy_(key_factors[0].to(key_cache.dtype))
        if raw_key_cache.ndim == 5:
            raw_key_cache[slot].copy_(
                key_factors[0]
                .unsqueeze(1)
                .expand_as(raw_key_cache[slot])
                .to(raw_key_cache.dtype)
            )
        value_cache[slot].copy_(value_factors[0].to(value_cache.dtype))
        if last_contribution is not None:
            _apply_correction(
                layer=layer,
                output=output[:, q_end - 1 : q_end].squeeze(0),
                contribution=last_contribution,
                budgets=budget_values[request_idx : request_idx + 1],
                key_factors=key_factors,
                phase="prefill",
            )


def _device_float_buffer(owner, name: str, values: list[float], device: torch.device) -> torch.Tensor:
    # CUDA graphs capture tensor addresses, not Python metadata objects.  Cache
    # one persistent budget buffer for every capture shape, exactly as the
    # audited Full-Attention path does for enabled/budget/end-position tensors.
    # Real metadata builds then update this same object in-place before replay.
    key = (str(device), len(values))
    tensor = _GRAPH_BUDGET_BUFFER_CACHE.get(key)
    if tensor is None:
        tensor = torch.empty((len(values),), dtype=torch.float32, device=device)
        _GRAPH_BUDGET_BUFFER_CACHE[key] = tensor
    setattr(owner, name, tensor)
    tensor.copy_(torch.tensor(values, dtype=torch.float32, device=device))
    return tensor


def install_linear_metadata_patch() -> None:
    global _METADATA_INSTALLED
    if not enabled() or _METADATA_INSTALLED:
        return
    with _INSTALL_LOCK:
        if _METADATA_INSTALLED:
            return
        from experiment_src.qwen35_vllm_dynamic_final.vendor import (
            aris_vllm_v1_suppression as native,
        )
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original = GPUModelRunner._build_attention_metadata

        def build_with_linear_metadata(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            attn_metadata = result[0]
            num_reqs = int(kwargs.get("num_reqs", self.input_batch.num_reqs))
            req_ids = [str(x) for x in self.input_batch.req_ids[:num_reqs]]
            budgets: list[float] = []
            for req_id in req_ids:
                meta = native._REQUEST_METADATA_BY_ID.get(req_id)
                if meta is not None and meta.budget is not None:
                    budgets.append(float(meta.budget))
                else:
                    tagged, value = native._parse_request_budget(req_id)
                    budgets.append(float(value) if tagged and value is not None else 0.0)
            computed = [int(x) for x in self.input_batch.num_computed_tokens_cpu[:num_reqs]]
            found_end_positions: list[int] = []
            for req_id in req_ids:
                if req_id in native._REQUEST_END_POSITION_CACHE:
                    found_end_positions.append(
                        int(native._REQUEST_END_POSITION_CACHE[req_id])
                    )
                    continue
                req_state = self.requests.get(req_id)
                prompt_ids = list(getattr(req_state, "prompt_token_ids", []) or [])
                found = (
                    native._find_subsequence(prompt_ids, native._END_THINK_TOKEN_IDS)
                    if native._END_THINK_TOKEN_IDS
                    else -1
                )
                native._REQUEST_END_POSITION_CACHE[req_id] = int(found)
                found_end_positions.append(int(found))
            metadata_dicts = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
            seen: set[int] = set()
            for metadata_dict in metadata_dicts:
                if not isinstance(metadata_dict, dict):
                    continue
                for meta_obj in metadata_dict.values():
                    query_start_loc = getattr(meta_obj, "query_start_loc", None)
                    if query_start_loc is None:
                        query_start_loc = getattr(
                            meta_obj, "non_spec_query_start_loc", None
                        )
                    if id(meta_obj) in seen or not isinstance(
                        query_start_loc, torch.Tensor
                    ):
                        continue
                    seen.add(id(meta_obj))
                    padded = max(0, int(query_start_loc.shape[0]) - 1)
                    used = min(padded, len(budgets))
                    fitted = budgets[:used] + [0.0] * (padded - used)
                    meta_obj.aris_req_ids = req_ids[:used]
                    meta_obj.aris_budget_values = _device_float_buffer(
                        meta_obj,
                        "_aris_linear_budget_values_buffer",
                        fitted,
                        query_start_loc.device,
                    )
                    meta_obj.aris_computed_lens_cpu = computed[:used] + [0] * (padded - used)
                    meta_obj.aris_found_end_idx_cpu = (
                        found_end_positions[:used] + [-1] * (padded - used)
                    )
            return result

        GPUModelRunner._build_attention_metadata = build_with_linear_metadata
        _install_threshold_warmup_patch()
        GPUModelRunner._aris_linear_metadata_patch_installed = True
        _METADATA_INSTALLED = True
