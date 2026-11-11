from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .engine_patch import _metric_payload


def main() -> None:
    torch.manual_seed(20260715)
    device = torch.device("cuda:0")
    # Qwen3.5-4B uses the expanded 248,320-token vocabulary.  Keep the smoke
    # on the real shape so the selected-id payload and the </think> token near
    # the vocabulary tail are both exercised.
    rows, vocab = 16, 248320
    logits = torch.randn((rows, vocab), dtype=torch.bfloat16, device=device)
    logits[0, 248069] = -torch.inf
    selected = torch.randint(0, vocab, (rows,), dtype=torch.long, device=device)
    tracked = torch.tensor(
        [13, 198, 220, 279, 1184, 1590, 1654, 1782, 3410, 4226, 248069],
        dtype=torch.long,
        device=device,
    )
    runner = SimpleNamespace()

    actual = _metric_payload(runner, logits, selected, tracked, 2).clone()
    torch.cuda.synchronize()
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    probabilities = torch.exp(values - log_z[:, None])
    entropy = log_z - (probabilities * finite_values).sum(dim=-1)
    top_values, top_ids = torch.topk(values, k=2, dim=-1)
    top_probabilities = torch.exp(top_values - log_z[:, None])
    expected = torch.cat(
        (
            entropy[:, None],
            (top_probabilities[:, 0] - top_probabilities[:, 1])[:, None],
            log_z[:, None],
            values.gather(1, selected[:, None]),
            selected[:, None].float(),
            top_values,
            top_ids.float(),
            values.index_select(1, tracked),
        ),
        dim=1,
    )
    torch.cuda.synchronize()

    scalar_error = (actual[:, :7] - expected[:, :7]).abs().max().item()
    tracked_delta = torch.where(
        actual[:, 9:] == expected[:, 9:],
        torch.zeros_like(actual[:, 9:]),
        (actual[:, 9:] - expected[:, 9:]).abs(),
    )
    tracked_error = tracked_delta.max().item()
    selected_ids_equal = torch.equal(actual[:, 4].long(), selected)
    top_ids_equal = torch.equal(actual[:, 7:9].long(), expected[:, 7:9].long())
    top_id_sets_equal = torch.equal(
        actual[:, 7:9].long().sort(dim=1).values,
        expected[:, 7:9].long().sort(dim=1).values,
    )

    for _ in range(10):
        _metric_payload(runner, logits, selected, tracked, 2)
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        _metric_payload(runner, logits, selected, tracked, 2)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "rows": rows,
        "vocab": vocab,
        "scalar_max_abs_error": scalar_error,
        "tracked_max_abs_error": tracked_error,
        "selected_ids_equal": selected_ids_equal,
        "top_ids_equal": top_ids_equal,
        "top_id_sets_equal": top_id_sets_equal,
        "actual_top_ids": actual[:, 7:9].long().tolist(),
        "expected_top_ids": expected[:, 7:9].long().tolist(),
        "milliseconds_per_call": elapsed * 1000.0 / iterations,
        # BF16 can produce tied top logits; Triton and torch.topk are allowed
        # to order the same tied pair differently, while the candidate set and
        # every scalar metric remain identical.
        "passed": bool(
            scalar_error < 5e-4
            and tracked_error == 0.0
            and selected_ids_equal
            and top_id_sets_equal
        ),
    }
    output = Path(__import__("os").environ["ARIS_METRIC_SMOKE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
