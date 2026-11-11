from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


_STATS: dict[str, Any] = {
    "prefill_calls": 0,
    "requests": 0,
    "layers": {},
}


def enabled() -> bool:
    return os.environ.get("ARIS_LINEAR_CONV_REPLACE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _window() -> tuple[int, int, int]:
    left = int(os.environ.get("ARIS_LINEAR_BOUNDARY_LEFT_TOKENS", "0"))
    right = int(os.environ.get("ARIS_LINEAR_BOUNDARY_RIGHT_TOKENS", "1"))
    rank = left + right + 1
    if left < 0 or right < 0 or rank not in {2, 3}:
        raise RuntimeError(
            f"invalid ARIS Linear conv window left={left}, right={right}"
        )
    return left, right, rank


def _fail(message: str) -> None:
    raise RuntimeError(f"ARIS Linear conv replacement fail-closed: {message}")


def apply_prefill_conv_replacement(
    *,
    layer,
    metadata,
    raw_mixed_qkv: torch.Tensor,
    convolved_mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply the Transformer-oracle three-token causal-conv counterfactual.

    The main convolution remains native vLLM. For each fresh prefill request we
    run the same fused causal-conv operator once on the compacted raw stream,
    replace only the next W-1 affected outputs, and install its final live
    conv cache. Decode then stays on vLLM's native update kernel.
    """
    if not enabled():
        return convolved_mixed_qkv
    if state_indices.ndim != 1:
        _fail(f"APC/multi-cache state indices are unsupported: {state_indices.shape}")
    left, right, rank = _window()
    computed = list(getattr(metadata, "aris_computed_lens_cpu", ()) or ())
    end_positions = list(getattr(metadata, "aris_found_end_idx_cpu", ()) or ())
    req_ids = [str(item) for item in (getattr(metadata, "aris_req_ids", ()) or ())]
    starts = [int(item) for item in query_start_loc.detach().cpu().tolist()]
    requests = min(
        len(starts) - 1,
        len(computed),
        len(end_positions),
        len(req_ids),
        int(state_indices.numel()),
    )
    if requests <= 0:
        _fail("live prefill has no aligned request metadata")

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )

    kernel = int(layer.conv_kernel_size)
    if kernel != int(conv_weights.shape[1]) or kernel != 4:
        _fail(f"expected Qwen3.5 causal-conv width 4, got {kernel}")
    _STATS["prefill_calls"] += 1
    layer_idx = int(layer.layer_idx)
    layer_stats = _STATS["layers"].setdefault(
        str(layer_idx),
        {
            "calls": 0,
            "window_rank": rank,
            "left_tokens": left,
            "right_tokens": right,
            "max_output_delta": 0.0,
            "max_cache_delta": 0.0,
            "max_cache_install_delta": 0.0,
            "kernel": "vllm_triton_causal_conv1d_fn",
        },
    )

    for request_idx in range(requests):
        start, stop = starts[request_idx], starts[request_idx + 1]
        length = stop - start
        absolute_start = int(computed[request_idx])
        absolute_end = int(end_positions[request_idx])
        local_end = absolute_end - absolute_start
        window_start = local_end - left
        window_stop = local_end + right + 1
        # A mixed prefill/decode forward can include cached decode requests
        # alongside fresh prompts.  Their boundary window is wholly before
        # the current one-token span and was already corrected during the
        # original prefill.  Likewise, an early prefill chunk can precede the
        # boundary.  Neither case should replay the fixed replacement.
        if window_stop <= 0 or window_start >= length:
            continue
        # A window that only partially overlaps the current span really was
        # split across recurrent transitions.  We cannot reconstruct that
        # counterfactual from the post-convolution cache, so fail closed.
        if window_start < 0 or window_stop > length:
            _fail(
                f"request {req_ids[request_idx]} boundary window "
                f"[{absolute_end-left},{absolute_end+right+1}) straddles "
                f"current span [{absolute_start},{absolute_start+length})"
            )
        # Fixed compacted-stream replacement currently requires the complete
        # boundary-bearing prompt transition.  With chunked prefill disabled,
        # a nonzero absolute_start here is a real scheduler/ABI violation.
        if absolute_start != 0:
            _fail(
                f"request {req_ids[request_idx]} boundary-bearing prefill "
                f"is chunked (computed={absolute_start})"
            )
        if (
            window_start < kernel - 1
            or window_stop - window_start != rank
        ):
            _fail(
                f"request {req_ids[request_idx]} invalid window "
                f"[{window_start},{window_stop}) for prefill length {length}"
            )
        slot = int(state_indices[request_idx].item())
        if slot < 0 or slot >= int(conv_state.shape[0]):
            _fail(f"request {req_ids[request_idx]} invalid conv slot {slot}")

        raw = raw_mixed_qkv[start:stop]
        filtered = torch.cat(
            (raw[:window_start], raw[window_stop:]), dim=0
        ).contiguous()
        filtered_length = int(filtered.shape[0])
        temp_storage = torch.zeros(
            (1, kernel - 1, int(filtered.shape[1])),
            dtype=conv_state.dtype,
            device=filtered.device,
        )
        temp_state = temp_storage.transpose(-1, -2)
        filtered_start_loc = torch.tensor(
            [0, filtered_length], dtype=torch.int32, device=filtered.device
        )
        filtered_output = causal_conv1d_fn(
            filtered.transpose(0, 1),
            conv_weights,
            layer.conv1d.bias,
            temp_state,
            filtered_start_loc,
            cache_indices=torch.zeros(
                (1,), dtype=torch.int32, device=filtered.device
            ),
            has_initial_state=torch.zeros(
                (1,), dtype=torch.bool, device=filtered.device
            ),
            activation=layer.activation,
            metadata=None,
            validate_data=True,
        ).transpose(0, 1)

        first = window_stop
        affected = min(kernel - 1, length - first)
        if affected <= 0:
            _fail(
                f"request {req_ids[request_idx]} has no post-window conv output"
            )
        replacement = filtered_output[
            window_start : window_start + affected
        ]
        native_slice = convolved_mixed_qkv[start + first : start + first + affected]
        output_delta = float(
            (native_slice.float() - replacement.float()).abs().max().item()
        )
        cache_delta = float(
            (conv_state[slot].float() - temp_state[0].float()).abs().max().item()
        )
        # The next W-1 outputs must change.  The final cache may legitimately
        # be identical when at least W-1 ordinary suffix tokens follow the
        # removed window, because both histories then end in the same tokens.
        if (
            not math.isfinite(output_delta)
            or output_delta <= 0.0
            or not math.isfinite(cache_delta)
        ):
            _fail(
                f"layer {layer_idx} positive control failed: "
                f"output_delta={output_delta}, cache_delta={cache_delta}"
            )
        native_slice.copy_(replacement.to(native_slice.dtype))
        conv_state[slot].copy_(temp_state[0].to(conv_state.dtype))
        install_delta = float(
            (conv_state[slot].float() - temp_state[0].float()).abs().max().item()
        )
        if install_delta != 0.0:
            _fail(
                f"layer {layer_idx} corrected conv cache install delta "
                f"{install_delta}"
            )
        layer_stats["calls"] += 1
        layer_stats["max_output_delta"] = max(
            float(layer_stats["max_output_delta"]), output_delta
        )
        layer_stats["max_cache_delta"] = max(
            float(layer_stats["max_cache_delta"]), cache_delta
        )
        layer_stats["max_cache_install_delta"] = max(
            float(layer_stats["max_cache_install_delta"]), install_delta
        )
        _STATS["requests"] += 1
        _trace(
            {
                "backend": "vllm",
                "event": "linear_conv_replacement",
                "request_id": req_ids[request_idx],
                "layer": layer_idx,
                "end_think_index": absolute_end,
                "positions": list(
                    range(
                        absolute_start + window_start,
                        absolute_start + window_stop,
                    )
                ),
                "rank": rank,
                "affected_suffix_positions": affected,
                "output_max_abs_delta": output_delta,
                "cache_positive_control_max_abs_delta": cache_delta,
                "cache_install_max_abs_delta": install_delta,
                "kernel": "vllm_triton_causal_conv1d_fn",
            }
        )
    return convolved_mixed_qkv


def _trace(row: dict[str, Any]) -> None:
    raw_path = os.environ.get("ARIS_LINEAR_CONV_TRACE_PATH", "").strip()
    if not raw_path:
        return
    path = Path(raw_path.format(rank=os.environ.get("LOCAL_RANK", "0")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_conv_replacement_stats() -> dict[str, Any]:
    result = dict(_STATS)
    if enabled():
        left, right, rank = _window()
        result["config"] = {
            "left_tokens": left,
            "right_tokens": right,
            "rank": rank,
            "semantics": "global compacted-stream causal-conv replacement",
        }
    return result
