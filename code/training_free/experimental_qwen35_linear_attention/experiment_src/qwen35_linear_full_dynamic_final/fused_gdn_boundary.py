# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed Qwen3.5 GDN decode with the ARIS boundary sidecar fused in.

The native recurrent body is kept in the same Triton program.  ARIS updates
the rank-2 boundary state and subtracts its selected contribution before the
native core output is stored, avoiding one additional kernel launch for each
of the 24 Linear-Attention layers.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.op import exp


def _canonical_mantissa_bits() -> int:
    raw = os.environ.get("ARIS_LINEAR_CANONICAL_MANTISSA_BITS")
    if raw is None:
        legacy = os.environ.get("ARIS_LINEAR_CANONICAL_BF16", "0").strip().lower()
        return 7 if legacy in {"1", "true", "yes", "on"} else 0
    bits = int(raw)
    if not 0 <= bits <= 23:
        raise ValueError(f"canonical mantissa bits must be in [0,23], got {bits}")
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


@triton.jit(noinline=True)
def _apply_boundary_after_native_store(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ssm_state_indices,
    boundary_key,
    boundary_value,
    budgets,
    thresholds,
    scale,
    alpha,
    suppression_lambda,
    max_effective_k,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_bk_slot: tl.constexpr,
    stride_bk_head: tl.constexpr,
    stride_bk_tile: tl.constexpr,
    stride_bk_rank: tl.constexpr,
    stride_bk_dim: tl.constexpr,
    stride_bv_slot: tl.constexpr,
    stride_bv_head: tl.constexpr,
    stride_bv_rank: tl.constexpr,
    stride_bv_dim: tl.constexpr,
    stride_threshold_head: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RANK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    CANONICAL_DROP_BITS: tl.constexpr,
    CANONICAL_FP64: tl.constexpr,
):
    """Run ARIS after native stores without extending native live ranges.

    Triton restricts no-inline device functions to scalar arguments.  Every
    argument here is therefore a pointer, scalar, or constexpr; the function
    reconstructs its program-local vectors from program ids.  This keeps the
    large boundary recurrence out of the stock packed-GDN function's register
    allocation while preserving a single CUDA kernel launch.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    if state_idx < 0:
        return

    p_o = o + (i_n * HV + i_hv) * V + o_v
    threshold = tl.load(thresholds + i_hv * stride_threshold_head)
    tracked = threshold < tl.minimum(1.0, max_effective_k * alpha)
    boundary_o = tl.zeros([BV], dtype=tl.float32)
    if tracked:
        p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_off = i_h * K + o_k
        k_off = (H * K) + i_h * K + o_k
        boundary_q = tl.load(
            p_mixed + q_off, mask=mask_k, other=0
        ).to(tl.float32)
        boundary_k = tl.load(
            p_mixed + k_off, mask=mask_k, other=0
        ).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            if CANONICAL_FP64:
                boundary_q64 = boundary_q.to(tl.float64)
                boundary_k64 = boundary_k.to(tl.float64)
                boundary_q = (
                    boundary_q64
                    * tl.rsqrt(tl.sum(boundary_q64 * boundary_q64) + 1e-6)
                ).to(tl.float32)
                boundary_k = (
                    boundary_k64
                    * tl.rsqrt(tl.sum(boundary_k64 * boundary_k64) + 1e-6)
                ).to(tl.float32)
            else:
                boundary_q = boundary_q * tl.rsqrt(
                    tl.sum(boundary_q * boundary_q) + 1e-6
                )
                boundary_k = boundary_k * tl.rsqrt(
                    tl.sum(boundary_k * boundary_k) + 1e-6
                )
        boundary_q = _canonical_round_fp32(boundary_q, CANONICAL_DROP_BITS)
        boundary_k = _canonical_round_fp32(boundary_k, CANONICAL_DROP_BITS)
        boundary_q = boundary_q * scale

        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(
            x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x
        )
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

        for factor in tl.static_range(0, RANK):
            p_bk = (
                boundary_key
                + state_idx * stride_bk_slot
                + i_hv * stride_bk_head
                + i_v * stride_bk_tile
                + factor * stride_bk_rank
                + o_k * stride_bk_dim
            )
            factor_k = tl.load(p_bk, mask=mask_k, other=0).to(tl.float32)
            if CANONICAL_FP64:
                overlap = tl.sum(
                    factor_k.to(tl.float64) * boundary_k.to(tl.float64),
                    axis=0,
                ).to(tl.float32)
            else:
                overlap = tl.sum(factor_k * boundary_k, axis=0)
            overlap = _canonical_round_fp32(overlap, CANONICAL_DROP_BITS)
            if CANONICAL_FP64:
                boundary_decay64 = tl.exp(g_val.to(tl.float64))
                factor_k = (
                    boundary_decay64
                    * (
                        factor_k.to(tl.float64)
                        - beta_val.to(tl.float64)
                        * boundary_k.to(tl.float64)
                        * overlap.to(tl.float64)
                    )
                ).to(tl.float32)
            else:
                factor_k = exp(g_val) * (
                    factor_k - beta_val * boundary_k * overlap
                )
            tl.store(p_bk, factor_k, mask=mask_k)
            if CANONICAL_FP64:
                coefficient = tl.sum(
                    factor_k.to(tl.float64) * boundary_q.to(tl.float64),
                    axis=0,
                ).to(tl.float32)
            else:
                coefficient = tl.sum(factor_k * boundary_q, axis=0)
            coefficient = _canonical_round_fp32(coefficient, CANONICAL_DROP_BITS)
            p_bv = (
                boundary_value
                + state_idx * stride_bv_slot
                + i_hv * stride_bv_head
                + factor * stride_bv_rank
                + o_v * stride_bv_dim
            )
            factor_v = tl.load(p_bv, mask=mask_v, other=0).to(tl.float32)
            term = _canonical_round_fp32(
                coefficient * factor_v, CANONICAL_DROP_BITS
            )
            boundary_o = _canonical_round_fp32(
                boundary_o + term, CANONICAL_DROP_BITS
            )

    # Retain a final canonical boundary as a defensive invariant; for the
    # production contract the per-term/per-accumulator rounds above already
    # make this operation idempotent.
    boundary_o = _canonical_round_fp32(boundary_o, CANONICAL_DROP_BITS)

    native_rounded = tl.load(p_o, mask=mask_v, other=0.0).to(tl.float32)
    budget = tl.load(budgets + i_n)
    selected = (budget > 0.0) & (
        threshold < tl.minimum(1.0, budget * alpha)
    )
    corrected = tl.where(
        selected,
        native_rounded - (1.0 - suppression_lambda) * boundary_o,
        native_rounded,
    )
    tl.store(p_o, corrected.to(p_o.dtype.element_ty), mask=mask_v)


@triton.jit
def _packed_gdn_boundary_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    boundary_key,
    boundary_value,
    budgets,
    thresholds,
    scale,
    alpha,
    suppression_lambda,
    max_effective_k,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_bk_slot: tl.constexpr,
    stride_bk_head: tl.constexpr,
    stride_bk_tile: tl.constexpr,
    stride_bk_rank: tl.constexpr,
    stride_bk_dim: tl.constexpr,
    stride_bv_slot: tl.constexpr,
    stride_bv_head: tl.constexpr,
    stride_bv_rank: tl.constexpr,
    stride_bv_dim: tl.constexpr,
    stride_threshold_head: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    RANK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    CANONICAL_DROP_BITS: tl.constexpr,
    CANONICAL_FP64: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
    # Keep the stock packed-GDN body instruction-for-instruction equivalent.
    # Any decay needed by the ARIS section is recomputed after the native
    # output/state stores instead of extending a native temporary's live range.
    b_h *= exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # End the native output live range at the same point as the stock packed
    # GDN kernel as well.  Keeping the BV-wide ``b_o`` vector live across the
    # complete boundary recurrence changes register allocation in the native
    # body.  On a real 32-layer trajectory that stayed hidden for 39 decode
    # steps, then crossed a BF16 boundary at linear layer 5 and made the two
    # otherwise identical recurrent executions diverge.  The transparent
    # reference first materializes native GDN output in BF16 and then applies
    # its FP32 correction, so this early store is also the exact observable
    # ordering of the audited two-kernel path.  The final correction remains
    # inside this same Triton launch.
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # Commit the large [BV, BK] main state in the stock kernel's original
    # order, immediately after its BF16 output store.  The ARIS boundary
    # update depends only on q/k/g/beta, so neither native tensor remains live
    # across the sidecar section.
    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
    _apply_boundary_after_native_store(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        o,
        ssm_state_indices,
        boundary_key,
        boundary_value,
        budgets,
        thresholds,
        scale,
        alpha,
        suppression_lambda,
        max_effective_k,
        stride_mixed_qkv_tok,
        stride_a_tok,
        stride_b_tok,
        stride_indices_seq,
        stride_bk_slot,
        stride_bk_head,
        stride_bk_tile,
        stride_bk_rank,
        stride_bk_dim,
        stride_bv_slot,
        stride_bv_head,
        stride_bv_rank,
        stride_bv_dim,
        stride_threshold_head,
        H,
        HV,
        K,
        V,
        RANK,
        BK,
        BV,
        SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM_IN_KERNEL,
        CANONICAL_DROP_BITS,
        CANONICAL_FP64,
    )


def fused_recurrent_gated_delta_rule_packed_decode_with_boundary(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    boundary_key: torch.Tensor,
    boundary_value: torch.Tensor,
    budgets: torch.Tensor,
    thresholds: torch.Tensor,
    alpha: float,
    suppression_lambda: float,
    max_effective_k: float,
    use_qk_l2norm_in_kernel: bool = False,
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be contiguous [B,QKV]")
    if a.ndim != 2 or b.ndim != 2 or a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("a and b must be contiguous [B,HV]")
    if initial_state.ndim != 4 or boundary_key.ndim != 5 or boundary_value.ndim != 4:
        raise ValueError("unexpected main/boundary state rank")
    if ssm_state_indices.ndim != 1 or budgets.ndim != 1 or thresholds.ndim != 1:
        raise ValueError("indices, budgets and thresholds must be rank-1")

    batch = int(mixed_qkv.shape[0])
    hv, value_dim, key_dim = (int(x) for x in initial_state.shape[-3:])
    if out.shape != (batch, 1, hv, value_dim):
        raise ValueError(f"unexpected output shape {tuple(out.shape)}")
    if not all(
        tensor.device == mixed_qkv.device
        for tensor in (
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            out,
            ssm_state_indices,
            boundary_key,
            boundary_value,
            budgets,
            thresholds,
        )
    ):
        raise ValueError("all fused GDN boundary inputs must share one device")

    qkv_dim = int(mixed_qkv.shape[1])
    qk_dim = qkv_dim - hv * value_dim
    if qk_dim <= 0 or qk_dim % 2:
        raise ValueError("cannot infer Q/K dimensions")
    q_dim = qk_dim // 2
    if q_dim % key_dim:
        raise ValueError("Q dimension is not divisible by head key dimension")
    heads = q_dim // key_dim
    if heads <= 0 or hv % heads:
        raise ValueError("invalid key/value head mapping")

    block_k = triton.next_power_of_2(key_dim)
    if triton.cdiv(key_dim, block_k) != 1:
        raise ValueError("fused boundary kernel requires NK=1")
    # Frozen after the A100 operator sweep: BV=32, one warp and three stages
    # are the fastest numerically equivalent configuration for Qwen3.5-4B.
    block_v = min(triton.next_power_of_2(value_dim), 32)
    value_tiles = triton.cdiv(value_dim, block_v)
    if tuple(boundary_key.shape[1:]) != (hv, value_tiles, 2, key_dim):
        raise ValueError(f"unexpected boundary-key shape {tuple(boundary_key.shape)}")
    if tuple(boundary_value.shape[1:]) != (hv, 2, value_dim):
        raise ValueError(f"unexpected boundary-value shape {tuple(boundary_value.shape)}")
    if budgets.numel() < batch or thresholds.numel() != hv:
        raise ValueError("budget/threshold size does not match packed decode")
    grid = (value_tiles, batch * hv)
    _packed_gdn_boundary_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        initial_state,
        ssm_state_indices,
        boundary_key,
        boundary_value,
        budgets,
        thresholds,
        scale,
        alpha,
        suppression_lambda,
        max_effective_k,
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_bk_slot=boundary_key.stride(0),
        stride_bk_head=boundary_key.stride(1),
        stride_bk_tile=boundary_key.stride(2),
        stride_bk_rank=boundary_key.stride(3),
        stride_bk_dim=boundary_key.stride(4),
        stride_bv_slot=boundary_value.stride(0),
        stride_bv_head=boundary_value.stride(1),
        stride_bv_rank=boundary_value.stride(2),
        stride_bv_dim=boundary_value.stride(3),
        stride_threshold_head=thresholds.stride(0),
        H=heads,
        HV=hv,
        K=key_dim,
        V=value_dim,
        RANK=2,
        BK=block_k,
        BV=block_v,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        CANONICAL_DROP_BITS=23 - _canonical_mantissa_bits()
        if _canonical_mantissa_bits() > 0
        else 0,
        CANONICAL_FP64=_canonical_fp64_enabled(),
        num_warps=1,
        num_stages=3,
    )
    if os.environ.get("ARIS_FUSED_LINEAR_TRACE_PATH"):
        # Diagnostic-only readback of the state produced by the one-launch
        # kernel.  The formal path never sets this variable and therefore pays
        # no tensor materialization or host synchronization cost.
        from .vllm_gdn_runtime import trace_packed_fused_state

        trace_packed_fused_state(
            mixed_qkv=mixed_qkv,
            state_indices=ssm_state_indices,
            boundary_key=boundary_key,
            boundary_value=boundary_value,
            budgets=budgets,
            thresholds=thresholds,
            alpha=alpha,
            layer_idx=layer_idx,
            key_heads=heads,
            key_dim=key_dim,
            value_heads=hv,
            corrected_output=out[:, 0],
        )
    return out, initial_state
