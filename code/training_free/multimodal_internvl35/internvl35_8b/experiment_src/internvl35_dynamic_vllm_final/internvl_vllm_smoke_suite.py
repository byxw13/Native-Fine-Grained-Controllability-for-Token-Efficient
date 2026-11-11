#!/usr/bin/env python3
"""Small real-InternVL vLLM smoke matrix (32/128/512 output tokens)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment_src.internvl35_vllm_baseline import mmmu_thinking_vllm as intern_baseline

baseline = intern_baseline.baseline

from .internvl_mmmu_dynamic_vllm import CLOSED_THINK_PREFIX, build_method_request, build_native_request


def _load_rows(manifest: Path, indices: list[int]) -> list[dict[str, Any]]:
    wanted = set(indices)
    rows = [row for row in baseline.read_jsonl(manifest) if int(row.get("global_index", -1)) in wanted]
    rows.sort(key=lambda row: indices.index(int(row["global_index"])))
    if len(rows) != len(indices):
        raise RuntimeError(f"smoke indices missing: requested={indices}, got={[row.get('global_index') for row in rows]}")
    return rows


def _serialize_logprobs(rows) -> list[dict[str, Any]]:
    """Make vLLM top-k logprobs JSON-safe for numerical smoke checks."""
    serialized = []
    for row in rows or []:
        if row is None:
            serialized.append({})
            continue
        serialized.append(
            {
                str(int(token_id)): {
                    "logprob": float(value.logprob),
                    "rank": None if value.rank is None else int(value.rank),
                    "decoded_token": value.decoded_token,
                }
                for token_id, value in sorted(row.items(), key=lambda item: int(item[0]))
            }
        )
    return serialized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indices", default="0,30,55,73,203,740")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), required=True)
    parser.add_argument("--method-cue", default="So")
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Unmeasured generate calls used to amortize engine/kernel startup.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--logprobs", type=int, default=0,
                        help="Store top-k logprobs for distribution/decision comparisons.")
    parser.add_argument("--allow-unfused", action="store_true")
    args = parser.parse_args()

    dynamic = args.mode == "dynamic"
    finalize = None
    metadata = None
    if dynamic:
        from .engine_patch import finalize_llm_dynamic_states, initial_request_suppression_metadata, install_dynamic_aris_patch

        install_dynamic_aris_patch(require_native_fused=not args.allow_unfused)
        finalize = finalize_llm_dynamic_states
        metadata = initial_request_suppression_metadata

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    indices = [int(item) for item in args.indices.split(",") if item.strip()]
    rows = _load_rows(args.manifest, indices)
    requests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in rows:
        if args.mode == "native":
            request, audit = build_native_request(tokenizer, row)
        else:
            request, audit = build_method_request(tokenizer, row, args.method_cue)
        if args.mode == "dynamic":
            request["aris_suppression"] = metadata()
        requests.append(request)
        audits.append({
            "id": row.get("id"),
            "global_index": row.get("global_index"),
            "method_prefix_count": str(request["prompt"]).count(CLOSED_THINK_PREFIX + args.method_cue),
            "end_think_count": str(request["prompt"]).count("</think>"),
            **audit,
        })

    expected_prefix = 0 if args.mode == "native" else 1
    if any(int(audit["method_prefix_count"]) != expected_prefix for audit in audits):
        raise RuntimeError(f"prefix audit failed: {audits}")
    if dynamic and any(int(audit["end_think_count"]) != 1 for audit in audits):
        raise RuntimeError("dynamic requests do not contain exactly one closed-think boundary")

    kwargs: dict[str, Any] = {}
    if dynamic:
        kwargs["worker_extension_cls"] = "experiment_src.internvl35_dynamic_vllm_final.worker_extension.DynamicArisWorkerExtension"
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.max_images_per_prompt, "video": 0},
        **kwargs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for max_tokens in args.max_tokens:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            max_tokens=int(max_tokens),
            seed=0,
            skip_special_tokens=False,
            logprobs=args.logprobs or None,
        )
        runs: list[dict[str, Any]] = []
        saved: list[list[dict[str, Any]]] = []
        finalize_reports: list[dict[str, Any]] = []
        for warmup_idx in range(max(0, int(args.warmup))):
            llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            if finalize is not None:
                finalize_reports.append(finalize(llm, reason=f"internvl_smoke_t{max_tokens}_warmup{warmup_idx}"))
        for repeat in range(args.repeats):
            started = time.perf_counter()
            outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            generated = time.perf_counter() - started
            if finalize is not None:
                finalize_reports.append(finalize(llm, reason=f"internvl_smoke_t{max_tokens}_repeat{repeat}"))
            token_count = sum(len(item.outputs[0].token_ids) for item in outputs)
            runs.append({
                "repeat": repeat,
                "wall_seconds": generated,
                "output_tokens": token_count,
                "throughput_tokens_per_second": token_count / generated if generated > 0 else None,
            })
            saved.append([
                {
                    "global_index": int(audits[idx].get("global_index", -1)),
                    "id": str(audits[idx].get("id", "")),
                    "request_id": str(item.request_id),
                    "token_ids": [int(x) for x in item.outputs[0].token_ids],
                    "text": item.outputs[0].text,
                    "finish_reason": item.outputs[0].finish_reason,
                        "stop_reason": item.outputs[0].stop_reason,
                        "topk_logprobs": _serialize_logprobs(item.outputs[0].logprobs),
                }
                for idx, item in enumerate(outputs)
            ])
        report = {
            "model": args.model,
            "mode": args.mode,
            "max_tokens": int(max_tokens),
            "indices": indices,
            "audits": audits,
            "runs": runs,
            "median_throughput_tokens_per_second": statistics.median(
                float(item["throughput_tokens_per_second"]) for item in runs
            ),
            "outputs": saved,
            "worker_finalize_reports": finalize_reports,
            "method_prefix": None if args.mode == "native" else CLOSED_THINK_PREFIX + args.method_cue,
        }
        (args.output_dir / f"t{max_tokens}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
