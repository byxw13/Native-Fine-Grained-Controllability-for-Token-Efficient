from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from .selective_noop_config import enabled, load_config


_STATS: dict[str, Any] = {
    "prefill_calls": 0,
    "requests": 0,
    "layers": {},
}


def _fail(message: str) -> None:
    raise RuntimeError(f"ARIS selective Linear no-op fail-closed: {message}")


def _local_linear_heads(layer, layer_idx: int) -> list[int]:
    config = load_config()
    global_heads = config.linear_heads.get(int(layer_idx), ())
    local_count = int(layer.num_v_heads // layer.tp_size)
    offset = int(layer.tp_rank) * local_count
    result = [head - offset for head in global_heads if offset <= head < offset + local_count]
    if len(result) != len(global_heads) and int(layer.tp_size) == 1:
        _fail(
            f"layer {layer_idx} has invalid Linear head(s) {list(global_heads)}; "
            f"valid range is [0,{local_count})"
        )
    return result


def _causal_conv(
    raw: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
) -> torch.Tensor:
    """Run the same causal-conv operator as vLLM on one [T,C] sequence."""
    from causal_conv1d import causal_conv1d_fn

    output = causal_conv1d_fn(
        raw.transpose(0, 1).unsqueeze(0).contiguous(),
        weight,
        bias,
        activation=activation,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


def _run_chunk(layer, q, k, v, g, beta, state):
    if int(q.shape[1]) == 0:
        return v, state
    cu = torch.tensor([0, int(q.shape[1])], dtype=torch.int32, device=q.device)
    return layer.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )


def _selected_conv_channels(layer, selected_value_heads: list[int]) -> torch.Tensor:
    key_heads = int(layer.num_k_heads // layer.tp_size)
    value_heads = int(layer.num_v_heads // layer.tp_size)
    group = value_heads // key_heads
    selected_key_heads = sorted({head // group for head in selected_value_heads})
    key_dim = int(layer.head_k_dim)
    value_dim = int(layer.head_v_dim)
    q_channels = [
        key_head * key_dim + dim
        for key_head in selected_key_heads
        for dim in range(key_dim)
    ]
    k_base = key_heads * key_dim
    k_channels = [
        k_base + key_head * key_dim + dim
        for key_head in selected_key_heads
        for dim in range(key_dim)
    ]
    v_base = 2 * key_heads * key_dim
    v_channels = [
        v_base + head * value_dim + dim
        for head in selected_value_heads
        for dim in range(value_dim)
    ]
    return torch.tensor(
        q_channels + k_channels + v_channels,
        dtype=torch.long,
        device=layer.conv1d.weight.device,
    )


def prefill_selective_noop(
    *,
    layer,
    metadata,
    raw_mixed_qkv: torch.Tensor,
    native_mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Install exact static-head counterfactual prefill outputs and live caches.

    Native vLLM output/state is left intact for unselected heads. Selected
    value heads are recomputed with every configured boundary transition
    replaced by an identity transition. The configured boundary tokens are
    also removed from the causal-convolution history used by the next W-1
    positions and from the installed decode conv cache.
    """
    if not enabled():
        return
    config = load_config()
    layer_idx = int(layer.layer_idx)
    selected = _local_linear_heads(layer, layer_idx)
    if not selected:
        return
    _STATS["prefill_calls"] += 1

    computed = list(getattr(metadata, "aris_computed_lens_cpu", ()) or ())
    end_positions = list(getattr(metadata, "aris_found_end_idx_cpu", ()) or ())
    req_ids = [str(item) for item in (getattr(metadata, "aris_req_ids", ()) or ())]
    starts = [int(x) for x in query_start_loc.detach().cpu().tolist()]
    requests = min(len(starts) - 1, len(computed), len(end_positions), len(req_ids))
    if requests <= 0:
        _fail("live prefill metadata has no requests")
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=output.device)
    conv_channels = _selected_conv_channels(layer, selected)
    kernel = int(layer.conv_kernel_size)
    layer_stats = _STATS["layers"].setdefault(
        str(layer_idx),
        {
            "calls": 0,
            "selected_heads": selected,
            "window": config.window,
            "offsets": list(config.offsets),
            "max_noop_identity_delta": 0.0,
            "max_native_state_delta": 0.0,
            "max_conv_replacement_delta": 0.0,
            "max_recurrent_cache_install_delta": 0.0,
            "max_conv_cache_install_delta": 0.0,
        },
    )

    for request_idx in range(requests):
        if int(computed[request_idx]) != 0:
            _fail(
                f"request {req_ids[request_idx]} uses chunked prefill "
                f"(computed={computed[request_idx]}); exact no-op requires fresh prefill"
            )
        start, stop = starts[request_idx], starts[request_idx + 1]
        length = stop - start
        slot = int(state_indices[request_idx].item())
        if slot < 0:
            _fail(f"request {req_ids[request_idx]} has invalid state slot {slot}")
        absolute_positions = config.positions(int(end_positions[request_idx]))
        local_positions = tuple(position - int(computed[request_idx]) for position in absolute_positions)
        window_start, window_stop = local_positions[0], local_positions[-1] + 1
        if (
            tuple(range(window_start, window_stop)) != local_positions
            or window_start < kernel - 1
            or window_stop > length
        ):
            _fail(
                f"request {req_ids[request_idx]} invalid contiguous boundary window "
                f"{local_positions} for prefill length {length}"
            )

        raw = raw_mixed_qkv[start:stop]
        native_conv = native_mixed_qkv[start:stop]
        filtered_raw = torch.cat((raw[:window_start], raw[window_stop:]), dim=0)
        filtered_conv = _causal_conv(
            filtered_raw,
            conv_weights,
            conv_bias,
            str(layer.activation),
        )
        counterfactual_conv = native_conv.clone()
        suffix_stop = min(length, window_stop + kernel - 1)
        replacement = filtered_conv[
            window_start : window_start + (suffix_stop - window_stop)
        ]
        counterfactual_conv[window_stop:suffix_stop] = replacement
        conv_delta = float(
            (
                native_conv[window_stop:suffix_stop].float()
                - replacement.float()
            )
            .abs()
            .max()
            .item()
        )
        if not torch.isfinite(torch.tensor(conv_delta)) or conv_delta <= 0.0:
            _fail(f"layer {layer_idx} convolution positive control failed: {conv_delta}")

        q_cf, k_cf, v_cf = layer.rearrange_mixed_qkv(counterfactual_conv)
        q_cf = q_cf[:, None] if q_cf.ndim == 3 else q_cf
        k_cf = k_cf[:, None] if k_cf.ndim == 3 else k_cf
        v_cf = v_cf[:, None] if v_cf.ndim == 3 else v_cf
        # rearrange_mixed_qkv returns [1,T,H,D] on the vLLM Qwen3.5 path.
        if int(q_cf.shape[0]) != 1:
            q_cf = q_cf.transpose(0, 1)
            k_cf = k_cf.transpose(0, 1)
            v_cf = v_cf.transpose(0, 1)
        g_req = log_decay[:, start:stop]
        beta_req = beta[:, start:stop]
        state0 = initial_state[request_idx : request_idx + 1]

        prefix_out, state_before = _run_chunk(
            layer,
            q_cf[:, :window_start],
            k_cf[:, :window_start],
            v_cf[:, :window_start],
            g_req[:, :window_start],
            beta_req[:, :window_start],
            state0,
        )
        native_boundary_out, native_boundary_state = _run_chunk(
            layer,
            q_cf[:, window_start:window_stop],
            k_cf[:, window_start:window_stop],
            v_cf[:, window_start:window_stop],
            g_req[:, window_start:window_stop],
            beta_req[:, window_start:window_stop],
            state_before.clone(),
        )
        boundary_q = torch.nn.functional.normalize(
            q_cf[:, window_start:window_stop].float(), p=2, dim=-1
        ) * (int(layer.head_k_dim) ** -0.5)
        noop_boundary_out = torch.einsum(
            "bthk,bhkv->bthv", boundary_q, state_before.float()
        ).to(value.dtype)
        identity_delta = 0.0
        native_state_delta = float(
            (native_boundary_state.float() - state_before.float()).abs().max().item()
        )
        if native_state_delta <= 0.0:
            _fail(
                f"layer {layer_idx} recurrent transition positive control is zero"
            )
        suffix_out, corrected_state = _run_chunk(
            layer,
            q_cf[:, window_stop:],
            k_cf[:, window_stop:],
            v_cf[:, window_stop:],
            g_req[:, window_stop:],
            beta_req[:, window_stop:],
            state_before,
        )
        cf_output = torch.cat((prefix_out, noop_boundary_out, suffix_out), dim=1)
        if tuple(cf_output.shape[1:]) != (length, int(value.shape[-2]), int(value.shape[-1])):
            _fail(
                f"layer {layer_idx} counterfactual output shape {tuple(cf_output.shape)}"
            )
        output[0, start:stop].index_copy_(
            1,
            selected_tensor,
            cf_output[0].index_select(1, selected_tensor).to(output.dtype),
        )
        final_state[request_idx].index_copy_(
            0,
            selected_tensor,
            corrected_state[0].index_select(0, selected_tensor).to(final_state.dtype),
        )

        # vLLM stores conv_state here as [slot, channel, kernel_width].
        cache_width = int(conv_state.shape[-1])
        corrected_conv_cache = filtered_raw[-cache_width:].transpose(0, 1)
        if int(corrected_conv_cache.shape[-1]) < cache_width:
            corrected_conv_cache = torch.nn.functional.pad(
                corrected_conv_cache, (cache_width - int(corrected_conv_cache.shape[-1]), 0)
            )
        conv_state[slot].index_copy_(
            0,
            conv_channels,
            corrected_conv_cache.index_select(0, conv_channels).to(conv_state.dtype),
        )
        recurrent_install_delta = float(
            (
                final_state[request_idx].index_select(0, selected_tensor).float()
                - corrected_state[0].index_select(0, selected_tensor).float()
            )
            .abs()
            .max()
            .item()
        )
        conv_install_delta = float(
            (
                conv_state[slot].index_select(0, conv_channels).float()
                - corrected_conv_cache.index_select(0, conv_channels).float()
            )
            .abs()
            .max()
            .item()
        )
        if recurrent_install_delta != 0.0 or conv_install_delta != 0.0:
            _fail(
                f"layer {layer_idx} cache install mismatch: recurrent="
                f"{recurrent_install_delta}, conv={conv_install_delta}"
            )
        layer_stats["calls"] += 1
        layer_stats["max_noop_identity_delta"] = max(
            float(layer_stats["max_noop_identity_delta"]), identity_delta
        )
        layer_stats["max_native_state_delta"] = max(
            float(layer_stats["max_native_state_delta"]), native_state_delta
        )
        layer_stats["max_conv_replacement_delta"] = max(
            float(layer_stats["max_conv_replacement_delta"]), conv_delta
        )
        layer_stats["max_recurrent_cache_install_delta"] = max(
            float(layer_stats["max_recurrent_cache_install_delta"]),
            recurrent_install_delta,
        )
        layer_stats["max_conv_cache_install_delta"] = max(
            float(layer_stats["max_conv_cache_install_delta"]), conv_install_delta
        )
        _STATS["requests"] += 1


def get_selective_noop_stats() -> dict[str, Any]:
    result = dict(_STATS)
    if enabled():
        config = load_config()
        result["config"] = {
            "path": config.source_path,
            "sha256": config.sha256,
            "window": config.window,
            "offsets": list(config.offsets),
            "full_heads": {str(k): list(v) for k, v in config.full_heads.items()},
            "linear_heads": {str(k): list(v) for k, v in config.linear_heads.items()},
        }
    return result


def dump_selective_noop_stats() -> None:
    path_text = os.environ.get("ARIS_SELECTIVE_NOOP_STATS_PATH", "").strip()
    if not path_text:
        return
    path = Path(path_text.format(rank=os.environ.get("LOCAL_RANK", "0")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(get_selective_noop_stats(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
