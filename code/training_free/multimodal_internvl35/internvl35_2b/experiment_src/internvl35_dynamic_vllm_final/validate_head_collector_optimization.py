from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def fsync_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.manual_seed(3407)
    heads, hidden, head_dim = 32, 2560, 128
    weights = (torch.randn(heads, hidden, head_dim, device=device, dtype=torch.float32) * 0.02)
    vector = torch.randn(heads, head_dim, device=device, dtype=torch.float32) * 0.01
    direct = torch.linalg.vector_norm(torch.einsum("hnd,hd->hn", weights, vector), dim=1)
    gram = torch.bmm(weights.transpose(1, 2), weights)
    optimized = torch.sqrt(
        torch.clamp(torch.einsum("hd,hde,he->h", vector, gram, vector), min=0.0)
    )
    absolute = (direct - optimized).abs()
    relative = absolute / direct.clamp_min(1e-12)

    logits = torch.randn(1, 151936, device=device, dtype=torch.float32)
    candidate = torch.argmax(logits[0])
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = log_z - (probs * logits).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    old = [
        int(candidate.item()),
        float(entropy.item()),
        float((top2[:, 0] - top2[:, 1]).item()),
        float((logits[:, int(candidate.item())] - log_z).item()),
    ]
    packed = torch.stack(
        (candidate.float(), entropy[0], top2[0, 0] - top2[0, 1], logits[0, candidate] - log_z[0])
    ).cpu().tolist()
    packed[0] = int(packed[0])
    metric_max_absolute_error = max(abs(float(left) - float(right)) for left, right in zip(old, packed))
    result = {
        "pass": float(relative.max()) <= args.threshold and metric_max_absolute_error == 0.0,
        "contribution_max_absolute_error": float(absolute.max()),
        "contribution_max_relative_error": float(relative.max()),
        "threshold": args.threshold,
        "packed_metric_max_absolute_error": metric_max_absolute_error,
        "shape": [heads, hidden, head_dim],
    }
    fsync_json(args.output, result)
    print(json.dumps(result))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
