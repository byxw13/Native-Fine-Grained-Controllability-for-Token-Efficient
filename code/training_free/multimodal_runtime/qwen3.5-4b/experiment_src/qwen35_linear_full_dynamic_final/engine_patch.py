from __future__ import annotations

import inspect

from experiment_src.qwen35_vllm_dynamic_final import engine_patch as full_engine

from .vllm_gdn_runtime import enabled, get_linear_stats, install_linear_metadata_patch
from .selective_noop_config import enabled as selective_noop_enabled
from .vllm_selective_noop_runtime import get_selective_noop_stats
from .vllm_conv_replacement import (
    enabled as conv_replacement_enabled,
    get_conv_replacement_stats,
)


_FULL_DISABLE_PATCHED = False


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def full_enabled() -> bool:
    """Whether Full-Attention suppression is active for this process.

    The controller and request budgets remain active when this is false.  This
    provides a genuine Linear-only ablation instead of turning the complete
    method into K=0.
    """
    return _truthy(__import__("os").environ.get("ARIS_FULL_ENABLE"), default=True)


finalize_llm_dynamic_states = full_engine.finalize_llm_dynamic_states
initial_request_suppression_metadata = full_engine.initial_request_suppression_metadata


def get_dynamic_stats():
    """Return both controller/Full and Linear hot-path evidence."""
    return {
        "mode": {
            "full_enabled": full_enabled(),
            "linear_enabled": enabled(),
            "selective_noop_enabled": selective_noop_enabled(),
            "conv_replacement_enabled": conv_replacement_enabled(),
        },
        "full_and_controller": full_engine.get_dynamic_stats(),
        "linear": get_linear_stats(),
        "selective_noop": get_selective_noop_stats(),
        "conv_replacement": get_conv_replacement_stats(),
    }


def finalize_dynamic_states(runner, reason: str = "generate_return"):
    report = full_engine.finalize_dynamic_states(runner, reason=reason)
    report["linear_stats"] = get_linear_stats()
    report["selective_noop_stats"] = get_selective_noop_stats()
    report["conv_replacement_stats"] = get_conv_replacement_stats()
    return report


def finalize_dynamic_request(
    runner,
    request_id: str,
    reason: str = "request_generate_return",
):
    report = full_engine.finalize_dynamic_request(
        runner,
        request_id=request_id,
        reason=reason,
    )
    report["linear_stats"] = get_linear_stats()
    report["selective_noop_stats"] = get_selective_noop_stats()
    report["conv_replacement_stats"] = get_conv_replacement_stats()
    return report


async def finalize_async_dynamic_request(
    engine,
    request_id: str,
    reason: str = "request_generate_return",
):
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


def verify_linear_gdn_patch() -> None:
    if not enabled():
        raise RuntimeError("ARIS_LINEAR_ENABLE must be true for the Full+Linear engine")
    from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention

    shape_source = inspect.getsource(GatedDeltaNetAttention.get_state_shape)
    core_source = inspect.getsource(GatedDeltaNetAttention._forward_core)
    packed_source = inspect.getsource(GatedDeltaNetAttention._forward_core_decode_non_spec)
    required = {
        "state shape": "aris_extend_state_shapes" in shape_source,
        "prefill hook": "aris_prefill_boundary_correction" in core_source,
        "decode hook": "aris_decode_boundary_correction" in core_source,
        "packed fused decode": (
            "fused_recurrent_gated_delta_rule_packed_decode_with_boundary"
            in packed_source
        ),
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(f"Qwen3.5 GDN source patch is incomplete: {missing}")
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    planner_source = inspect.getsource(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    if "aris_linear_boundary_rank" not in planner_source:
        raise RuntimeError(
            "Qwen3.5 cache planner does not use the configured ARIS boundary rank"
        )


def verify_selective_noop_gdn_patch() -> None:
    from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention

    core_source = inspect.getsource(GatedDeltaNetAttention._forward_core)
    required = {
        "raw convolution input": "aris_raw_mixed_qkv_non_spec" in core_source,
        "selective no-op hook": "aris_prefill_selective_noop" in core_source,
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(
            f"Qwen3.5 selective no-op GDN source patch is incomplete: {missing}"
        )


def verify_conv_replacement_gdn_patch() -> None:
    from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention

    core_source = inspect.getsource(GatedDeltaNetAttention._forward_core)
    required = {
        "raw convolution input": "aris_raw_mixed_qkv_non_spec" in core_source,
        "conv replacement hook": "aris_apply_prefill_conv_replacement" in core_source,
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise RuntimeError(
            f"Qwen3.5 conv-replacement GDN source patch is incomplete: {missing}"
        )


def _install_full_disable_patch() -> None:
    """Zero only Full-Attention enable bits while retaining dynamic K.

    Linear Attention consumes ``aris_budget_values`` from the same controller.
    Full Attention consumes the separate ``aris_enabled`` tensor.  Clearing
    the latter after metadata construction creates an auditable Linear-only
    mode without changing request ids, K scheduling, or Linear sidecars.
    """
    global _FULL_DISABLE_PATCHED
    if full_enabled() or _FULL_DISABLE_PATCHED:
        return
    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    original_build = GPUModelRunner._build_attention_metadata

    def build_with_full_disabled(self, *args, **kwargs):
        result = original_build(self, *args, **kwargs)
        metadata = result[0]
        dictionaries = metadata if isinstance(metadata, list) else [metadata]
        for dictionary in dictionaries:
            if not isinstance(dictionary, dict):
                continue
            for meta in dictionary.values():
                flag = getattr(meta, "aris_enabled", None)
                if isinstance(flag, torch.Tensor):
                    flag.zero_()
        return result

    GPUModelRunner._build_attention_metadata = build_with_full_disabled

    # The inherited Full first-forward assertion expects enabled=1 whenever
    # K>0.  Linear-only intentionally violates that condition, so mark these
    # requests as checked by the Full branch immediately before delegating.
    original_forward = FlashAttentionImpl.forward

    def forward_with_full_disabled(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None and len(args) >= 6:
            attn_metadata = args[5]
        req_ids = list(getattr(attn_metadata, "aris_req_ids", ()) or ())
        full_engine._FIRST_FORWARD_ASSERTED_REQUESTS.update(str(x) for x in req_ids)
        return original_forward(self, *args, **kwargs)

    FlashAttentionImpl.forward = forward_with_full_disabled
    GPUModelRunner._aris_full_disabled_patch_installed = True
    _FULL_DISABLE_PATCHED = True


def install_dynamic_aris_patch(require_native_fused: bool = True) -> None:
    full_engine.install_dynamic_aris_patch(require_native_fused=require_native_fused)
    if selective_noop_enabled():
        verify_selective_noop_gdn_patch()
    if conv_replacement_enabled():
        verify_conv_replacement_gdn_patch()
    if enabled():
        verify_linear_gdn_patch()
        install_linear_metadata_patch()
    _install_full_disable_patch()
