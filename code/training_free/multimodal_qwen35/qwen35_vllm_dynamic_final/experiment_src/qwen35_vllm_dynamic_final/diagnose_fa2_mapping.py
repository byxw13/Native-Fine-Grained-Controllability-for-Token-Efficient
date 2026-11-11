from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


Q_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 256


def _reference_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head: int, start: int) -> torch.Tensor:
    kv_head = head // (Q_HEADS // KV_HEADS)
    scores = torch.matmul(q[0, head].float(), k[:, kv_head].float().transpose(0, 1))
    scores.mul_(1.0 / math.sqrt(HEAD_DIM))
    scores[start : start + 2] = -torch.inf
    return torch.matmul(torch.softmax(scores, dim=-1), v[:, kv_head].float()).to(q.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    q = torch.randn(1, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(19, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 19], dtype=torch.int32, device=device)
    common = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=19,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        causal=False,
        fa_version=2,
    )
    native = flash_attn_varlen_func(**common)
    enabled = torch.ones(1, dtype=torch.int32, device=device)
    budget = torch.zeros(1, dtype=torch.int32, device=device)
    end_pos = torch.tensor([3], dtype=torch.int32, device=device)
    rows = []
    for requested_head in range(Q_HEADS):
        table = torch.tensor([1 << requested_head], dtype=torch.int64, device=device)
        dynamic = flash_attn_varlen_func(
            **common,
            aris_enabled=enabled,
            aris_budget_ids=budget,
            aris_end_positions=end_pos,
            aris_head_mask_table=table,
            aris_layer_idx=0,
            aris_num_layers=1,
            aris_words_per_layer=1,
        )
        per_head_effect = (dynamic.float() - native.float()).abs().amax(dim=-1)[0]
        changed_heads = [int(i) for i in torch.nonzero(per_head_effect > 0, as_tuple=False).flatten().tolist()]
        matches = []
        for actual_head in changed_heads:
            candidates = []
            for start in range(18):
                ref = _reference_head(q, k, v, actual_head, start)
                error = float((dynamic[0, actual_head].float() - ref.float()).abs().max().item())
                candidates.append((error, start))
            best_error, best_start = min(candidates)
            matches.append(
                {
                    "actual_head": actual_head,
                    "effect_max_abs": float(per_head_effect[actual_head].item()),
                    "best_mask_start": best_start,
                    "best_reference_max_abs": best_error,
                }
            )
        rows.append({"requested_head": requested_head, "changed_heads": changed_heads, "matches": matches})

    report = {"device": torch.cuda.get_device_name(0), "requested_end_position": 3, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
