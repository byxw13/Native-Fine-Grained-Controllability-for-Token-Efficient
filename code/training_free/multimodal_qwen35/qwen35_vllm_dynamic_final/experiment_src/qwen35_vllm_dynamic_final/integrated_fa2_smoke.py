from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
SELECTED_BY_BUDGET = ({0, 5}, {2, 15})


def _mask_word(heads: set[int]) -> int:
    value = 0
    for head in heads:
        value |= 1 << int(head)
    return value


def _oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: list[int],
    k_lens: list[int],
    end_positions: list[int],
    enabled: list[int],
    budget_ids: list[int],
    *,
    causal: bool,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(HEAD_DIM)
    outputs: list[torch.Tensor] = []
    q_start = 0
    k_start = 0
    groups = NUM_Q_HEADS // NUM_KV_HEADS
    for batch_idx, (q_len, k_len) in enumerate(zip(q_lens, k_lens, strict=True)):
        q_seq = q[q_start : q_start + q_len].float().transpose(0, 1)
        k_seq = k[k_start : k_start + k_len].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        v_seq = v[k_start : k_start + k_len].float().repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale
        if causal:
            q_positions = torch.arange(q_len, device=q.device) + (k_len - q_len)
            k_positions = torch.arange(k_len, device=q.device)
            scores.masked_fill_(k_positions[None, None, :] > q_positions[None, :, None], -torch.inf)
        if enabled[batch_idx]:
            heads = SELECTED_BY_BUDGET[budget_ids[batch_idx]]
            for head in heads:
                for position in (end_positions[batch_idx], end_positions[batch_idx] + 1):
                    if 0 <= position < k_len:
                        if q_len == 1:
                            scores[head, 0, position] = -torch.inf
                        else:
                            scores[head, q_len - 1, position] = -torch.inf
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_seq).transpose(0, 1)
        outputs.append(out)
        q_start += q_len
        k_start += k_len
    return torch.cat(outputs, dim=0).to(q.dtype)


def _run_case(name: str, q_lens: list[int], k_lens: list[int], *, causal: bool) -> dict[str, float | str]:
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    torch.manual_seed(20260720 + sum(q_lens))
    device = torch.device("cuda")
    q = torch.randn(sum(q_lens), NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(sum(k_lens), NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    enabled = torch.tensor([1, 1], dtype=torch.int32, device=device)
    disabled = torch.zeros_like(enabled)
    budget_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    end_positions = torch.tensor([3, 5], dtype=torch.int32, device=device)
    mask_table = torch.tensor(
        [_mask_word(set(SELECTED_BY_BUDGET[0])), _mask_word(set(SELECTED_BY_BUDGET[1]))],
        dtype=torch.int64,
        device=device,
    )
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        causal=causal,
        fa_version=2,
    )
    native = flash_attn_varlen_func(**kwargs)
    disabled_out = flash_attn_varlen_func(
        **kwargs,
        aris_enabled=disabled,
        aris_budget_ids=budget_ids,
        aris_end_positions=end_positions,
        aris_head_mask_table=mask_table,
        aris_layer_idx=0,
        aris_num_layers=1,
        aris_words_per_layer=1,
    )
    dynamic = flash_attn_varlen_func(
        **kwargs,
        aris_enabled=enabled,
        aris_budget_ids=budget_ids,
        aris_end_positions=end_positions,
        aris_head_mask_table=mask_table,
        aris_layer_idx=0,
        aris_num_layers=1,
        aris_words_per_layer=1,
    )
    oracle = _oracle(
        q,
        k,
        v,
        q_lens,
        k_lens,
        end_positions.tolist(),
        enabled.tolist(),
        budget_ids.tolist(),
        causal=causal,
    )
    k0_max_abs = float((native.float() - disabled_out.float()).abs().max().item())
    dynamic_max_abs = float((dynamic.float() - oracle.float()).abs().max().item())
    dynamic_mean_abs = float((dynamic.float() - oracle.float()).abs().mean().item())
    effect_max_abs = float((dynamic.float() - native.float()).abs().max().item())
    if k0_max_abs != 0.0:
        raise AssertionError(f"{name}: disabled metadata changed native output: {k0_max_abs}")
    if effect_max_abs <= 0.0:
        raise AssertionError(f"{name}: active suppression produced no numerical effect")
    if dynamic_max_abs > 0.035 or dynamic_mean_abs > 0.003:
        raise AssertionError(
            f"{name}: fused/oracle mismatch max={dynamic_max_abs:.6g}, mean={dynamic_mean_abs:.6g}"
        )
    return {
        "name": name,
        "k0_max_abs": k0_max_abs,
        "dynamic_max_abs": dynamic_max_abs,
        "dynamic_mean_abs": dynamic_mean_abs,
        "effect_max_abs": effect_max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "q_heads": NUM_Q_HEADS,
        "kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "cases": [
            _run_case("decode_gqa", [1, 1], [19, 23], causal=False),
            _run_case("first_forward_prefill_last_row", [17, 21], [17, 21], causal=True),
        ],
        "passed": True,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
