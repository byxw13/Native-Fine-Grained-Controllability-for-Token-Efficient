from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class LinearStep:
    state: torch.Tensor
    output: torch.Tensor
    delta_value: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor


def canonical_mantissa_bits() -> int:
    """Return the FP32 mantissa width used at canonical reduction points.

    PyTorch lowers the two 128-wide inner products to cuBLAS GEMV, whereas the
    integrated vLLM kernel lowers them to Triton warp reductions.  Both are
    valid FP32 reductions, but their last few bits differ and the recurrent
    model can amplify that difference after several decode steps.  Rounding
    the normalized q/k vectors and the two scalar reductions at explicit BF16
    boundaries gives the ARIS sidecar a defined cross-backend arithmetic
    contract.  Unlike casting the whole value to BF16, mantissa rounding keeps
    the FP32 exponent range, which matters after a boundary state has decayed
    for hundreds of tokens.  Zero disables canonical rounding and leaves the
    high-precision implementation available as the diagnostic oracle.
    """
    raw = os.environ.get("ARIS_LINEAR_CANONICAL_MANTISSA_BITS")
    if raw is None:
        # Compatibility with the first diagnostic experiment.  Production
        # scripts use the explicit mantissa-width variable.
        legacy = os.environ.get("ARIS_LINEAR_CANONICAL_BF16", "0").strip().lower()
        return 7 if legacy in {"1", "true", "yes", "on"} else 0
    bits = int(raw)
    if not 0 <= bits <= 23:
        raise ValueError(
            "ARIS_LINEAR_CANONICAL_MANTISSA_BITS must be in [0,23], "
            f"got {bits}"
        )
    return bits


def canonical_bf16_reductions_enabled() -> bool:
    """Backward-compatible predicate used by existing tests/reports."""
    return canonical_mantissa_bits() > 0


def canonical_bf16(value: torch.Tensor) -> torch.Tensor:
    bits = canonical_mantissa_bits()
    if bits == 0:
        return value
    if value.dtype != torch.float32:
        value = value.float()
    drop = 23 - bits
    if drop == 0:
        return value
    raw = value.contiguous().view(torch.int32)
    lsb = (raw >> drop) & 1
    bias = (1 << (drop - 1)) - 1 + lsb
    rounded = (raw + bias) & ~((1 << drop) - 1)
    return rounded.view(torch.float32)


def canonical_fp64_reductions_enabled() -> bool:
    return os.environ.get("ARIS_LINEAR_CANONICAL_FP64", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def normalize_qk(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match Transformers Qwen3.5's dtype-preserving q/k L2 normalization."""
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


def normalize_qk_fp32(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match vLLM packed GDN's FP32 q/k L2 normalization.

    vLLM's packed GDN kernels load BF16 q/k into FP32 registers before the
    squared reduction.  Transformers instead performs this reduction in the
    input dtype.  Keeping the two helpers explicit prevents a vLLM numerical
    oracle from silently changing the Transformers/head-summary semantics.
    """
    work = value.double() if canonical_fp64_reductions_enabled() else value.float()
    normalized = work * torch.rsqrt(work.square().sum(dim=-1, keepdim=True) + eps)
    normalized = normalized.float()
    return canonical_bf16(normalized)


def recurrent_step(
    state_vk: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    qk_are_normalized: bool = False,
) -> LinearStep:
    """One exact Gated DeltaNet step using vLLM's [H,V,K] state layout."""
    if not qk_are_normalized:
        query = normalize_qk(query)
        key = normalize_qk(key)
    decay = log_decay.exp().unsqueeze(-1).unsqueeze(-1)
    decayed = state_vk.float() * decay.float()
    memory_value = torch.einsum("...hvk,...hk->...hv", decayed, key.float())
    delta_value = (value.float() - memory_value) * beta.float().unsqueeze(-1)
    updated = decayed + torch.einsum("...hv,...hk->...hvk", delta_value, key.float())
    scaled_query = query.float() * (query.shape[-1] ** -0.5)
    output = torch.einsum("...hvk,...hk->...hv", updated, scaled_query)
    return LinearStep(
        state=updated,
        output=output,
        delta_value=delta_value,
        query=scaled_query,
        key=key.float(),
    )


def propagate_boundary_keys(
    key_factors: torch.Tensor,
    key: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    key_is_normalized: bool = False,
) -> torch.Tensor:
    """Propagate rank factors through g*(I-beta*k*k^T), without new writes."""
    if not key_is_normalized:
        key = normalize_qk(key)
    fp64 = canonical_fp64_reductions_enabled()
    projection = torch.einsum(
        "...hrk,...hk->...hr",
        key_factors.double() if fp64 else key_factors.float(),
        key.double() if fp64 else key.float(),
    ).float()
    projection = canonical_bf16(projection)
    if fp64:
        removed = (
            beta.double().unsqueeze(-1).unsqueeze(-1)
            * key.double().unsqueeze(-2)
            * projection.double().unsqueeze(-1)
        )
        updated = (
            log_decay.double().exp().unsqueeze(-1).unsqueeze(-1)
            * (key_factors.double() - removed)
        )
        return updated.float()
    removed = beta.float().unsqueeze(-1).unsqueeze(-1) * key.float().unsqueeze(-2) * projection.unsqueeze(-1)
    return log_decay.float().exp().unsqueeze(-1).unsqueeze(-1) * (key_factors.float() - removed)


def boundary_step(
    key_factors: torch.Tensor,
    value_factors: torch.Tensor,
    main_state_before: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    *,
    write_slot: int | None = None,
    qk_are_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, LinearStep]:
    """Advance B and optionally write this token's exact rank-1 residual."""
    step = recurrent_step(
        main_state_before,
        query,
        key,
        value,
        log_decay,
        beta,
        qk_are_normalized=qk_are_normalized,
    )
    new_keys = propagate_boundary_keys(
        key_factors,
        step.key,
        log_decay,
        beta,
        key_is_normalized=True,
    )
    new_values = value_factors.float().clone()
    if write_slot is not None:
        if not 0 <= int(write_slot) < int(new_keys.shape[-2]):
            raise IndexError(f"boundary write slot {write_slot} outside rank {new_keys.shape[-2]}")
        new_keys[..., int(write_slot), :] = step.key
        new_values[..., int(write_slot), :] = step.delta_value
    return new_keys, new_values, step


def boundary_contribution(
    key_factors: torch.Tensor,
    value_factors: torch.Tensor,
    query: torch.Tensor,
    *,
    query_is_normalized_and_scaled: bool = False,
) -> torch.Tensor:
    """Return q^T B in output/value-head coordinates."""
    if not query_is_normalized_and_scaled:
        query = normalize_qk(query).float() * (query.shape[-1] ** -0.5)
    fp64 = canonical_fp64_reductions_enabled()
    coefficients = torch.einsum(
        "...hrk,...hk->...hr",
        key_factors.double() if fp64 else key_factors.float(),
        query.double() if fp64 else query.float(),
    ).float()
    coefficients = canonical_bf16(coefficients)
    if canonical_mantissa_bits() == 0:
        # Preserve the unrounded high-precision diagnostic oracle.
        return torch.einsum(
            "...hr,...hrv->...hv", coefficients, value_factors.float()
        )

    # Define the rank contraction as an explicit cross-backend arithmetic
    # contract.  ATen may lower the expression above to GEMV, while Triton may
    # use an FMA chain.  Rounding only their final sums leaves exact half-way
    # cases backend-dependent.  Canonicalizing every product and every
    # accumulator update makes the result independent of either lowering.
    contribution = torch.zeros_like(value_factors[..., 0, :], dtype=torch.float32)
    for factor in range(int(value_factors.shape[-2])):
        term = canonical_bf16(
            coefficients[..., factor, None]
            * value_factors[..., factor, :].float()
        )
        contribution = canonical_bf16(contribution + term)
    return contribution


def reconstruct_boundary_state(
    key_factors: torch.Tensor,
    value_factors: torch.Tensor,
) -> torch.Tensor:
    """Materialize B in vLLM [H,V,K] layout for diagnostics only."""
    return torch.einsum("...hrv,...hrk->...hvk", value_factors.float(), key_factors.float())
