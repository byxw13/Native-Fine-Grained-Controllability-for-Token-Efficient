from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from .vendor.aris_vllm_v1_suppression import _native_flash_correction


def main() -> None:
    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    tokens, heads, kv_heads, dim = 16, 8, 2, 128
    page, sequence = 16, 32
    blocks_per_request = sequence // page
    blocks = tokens * blocks_per_request
    dtype = torch.bfloat16
    query = torch.randn((tokens, heads, dim), dtype=dtype, device=device)
    key = torch.randn((blocks, page, kv_heads, dim), dtype=dtype, device=device)
    value = torch.randn_like(key)
    block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(tokens, blocks_per_request)
    seq_lens = torch.full((tokens,), sequence, dtype=torch.int32, device=device)
    enabled = torch.tensor([(index % 3) != 0 for index in range(tokens)], dtype=torch.int32, device=device)
    budgets = torch.zeros((tokens,), dtype=torch.int32, device=device)
    end_positions = torch.full((tokens,), 5, dtype=torch.int32, device=device)
    selected_heads = (1, 2, 4, 7)
    word = sum(1 << head for head in selected_heads)
    head_mask = torch.tensor([[[word]]], dtype=torch.int64, device=device)
    scale = dim**-0.5

    native = torch.empty_like(query)
    expected = torch.empty_like(query)
    lse = torch.empty((heads, tokens), dtype=torch.float32, device=device)
    for token in range(tokens):
        physical = block_table[token].long()
        keys = key.index_select(0, physical).reshape(sequence, kv_heads, dim).float()
        values = value.index_select(0, physical).reshape(sequence, kv_heads, dim).float()
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            scores = query[token, head].float() @ keys[:, kv_head].T * scale
            probabilities = torch.softmax(scores, dim=0)
            native[token, head] = probabilities @ values[:, kv_head]
            lse[head, token] = torch.logsumexp(scores, dim=0)
            if enabled[token] and head in selected_heads:
                keep = torch.ones(sequence, dtype=torch.bool, device=device)
                keep[5:7] = False
                masked_probabilities = torch.softmax(scores[keep], dim=0)
                expected[token, head] = masked_probabilities @ values[keep, kv_head]
            else:
                expected[token, head] = native[token, head]

    def run(grouped: bool) -> torch.Tensor:
        os.environ["ARIS_CORRECTION_GROUPED"] = "1" if grouped else "0"
        output = native.clone()
        _native_flash_correction(
            query=query,
            output=output,
            softmax_lse=lse,
            key_cache=key,
            value_cache=value,
            block_table=block_table,
            seq_lens=seq_lens,
            enabled=enabled,
            budget_ids=budgets,
            end_positions=end_positions,
            head_mask_table=head_mask,
            num_heads=heads,
            num_kv_heads=kv_heads,
            scale=scale,
        )
        torch.cuda.synchronize()
        return output

    grouped = run(True)
    scalar = run(False)
    grouped_error = (grouped.float() - expected.float()).abs().max().item()
    scalar_error = (scalar.float() - expected.float()).abs().max().item()
    grouped_scalar_error = (grouped.float() - scalar.float()).abs().max().item()

    timings = {}
    for name, flag in (("grouped", True), ("scalar", False)):
        os.environ["ARIS_CORRECTION_GROUPED"] = "1" if flag else "0"
        output = native.clone()
        for _ in range(10):
            output.copy_(native)
            _native_flash_correction(
                query=query,
                output=output,
                softmax_lse=lse,
                key_cache=key,
                value_cache=value,
                block_table=block_table,
                seq_lens=seq_lens,
                enabled=enabled,
                budget_ids=budgets,
                end_positions=end_positions,
                head_mask_table=head_mask,
                num_heads=heads,
                num_kv_heads=kv_heads,
                scale=scale,
            )
        torch.cuda.synchronize()
        started = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            output.copy_(native)
            _native_flash_correction(
                query=query,
                output=output,
                softmax_lse=lse,
                key_cache=key,
                value_cache=value,
                block_table=block_table,
                seq_lens=seq_lens,
                enabled=enabled,
                budget_ids=budgets,
                end_positions=end_positions,
                head_mask_table=head_mask,
                num_heads=heads,
                num_kv_heads=kv_heads,
                scale=scale,
            )
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - started) * 1000.0 / iterations

    report = {
        "grouped_max_abs_error": grouped_error,
        "scalar_max_abs_error": scalar_error,
        "grouped_vs_scalar_max_abs_error": grouped_scalar_error,
        "milliseconds_per_call": timings,
        "passed": bool(
            math.isfinite(grouped_error)
            and grouped_error <= 0.015625
            and grouped_scalar_error == 0.0
        ),
    }
    output_path = Path(os.environ["ARIS_CORRECTION_SMOKE_OUTPUT"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
