from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .text_vllm_smoke import build_prompt, read_prompts, synchronize
from .prompting import CLOSED_THINK_PREFIX, FORMAL_METHOD_CUE


def _requests(
    tokenizer,
    rows: list[dict[str, Any]],
    mode: str,
    dynamic: bool,
    initial_request_metadata=None,
) -> tuple[list[dict[str, Any]], list[int]]:
    rendered = [build_prompt(tokenizer, row, mode) for row in rows]
    method_prefix = CLOSED_THINK_PREFIX + FORMAL_METHOD_CUE
    counts = [text.count(method_prefix) for text, _ in rendered]
    requests = []
    for _, token_ids in rendered:
        request: dict[str, Any] = {"prompt_token_ids": token_ids}
        if dynamic:
            request["aris_suppression"] = initial_request_metadata()
        requests.append(request)
    return requests, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    dynamic = args.mode == "dynamic"
    finalize_dynamic = None
    initial_request_metadata = None
    if dynamic:
        from .engine_patch import (
            finalize_llm_dynamic_states,
            initial_request_suppression_metadata,
            install_dynamic_aris_patch,
        )

        install_dynamic_aris_patch(require_native_fused=True)
        finalize_dynamic = finalize_llm_dynamic_states
        initial_request_metadata = initial_request_suppression_metadata

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = read_prompts(args.prompts)[: args.batch_size]
    requests, prefix_counts = _requests(
        tokenizer, rows, args.mode, dynamic, initial_request_metadata
    )
    expected = 0 if args.mode == "native" else 1
    if any(count != expected for count in prefix_counts):
        raise RuntimeError(f"unexpected method-prefix counts for {args.mode}: {prefix_counts}")

    engine_kwargs = {}
    if dynamic:
        engine_kwargs["worker_extension_cls"] = (
            "experiment_src.qwen35_vllm_dynamic_final.worker_extension.DynamicArisWorkerExtension"
        )
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **engine_kwargs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for max_tokens in args.max_tokens:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(max_tokens),
            repetition_penalty=1.0,
            skip_special_tokens=False,
        )
        for warmup_idx in range(args.warmup):
            llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            if finalize_dynamic is not None:
                finalize_dynamic(llm, reason=f"t{max_tokens}_warmup_{warmup_idx}_return")
        runs = []
        saved_outputs = []
        finalize_reports = []
        for repeat in range(args.repeats):
            synchronize()
            started = time.perf_counter()
            outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            if finalize_dynamic is not None:
                finalize_reports.append(
                    finalize_dynamic(llm, reason=f"t{max_tokens}_repeat_{repeat}_return")
                )
            synchronize()
            elapsed = time.perf_counter() - started
            count = sum(len(output.outputs[0].token_ids) for output in outputs)
            runs.append(
                {
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "output_tokens": count,
                    "throughput_tokens_per_second": count / elapsed,
                }
            )
            saved_outputs.append(
                [
                    {
                        "request_id": output.request_id,
                        "token_ids": [int(item) for item in output.outputs[0].token_ids],
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                    }
                    for output in outputs
                ]
            )
        report = {
            "mode": args.mode,
            "max_tokens": int(max_tokens),
            "batch_size": len(requests),
            "tensor_parallel_size": args.tensor_parallel_size,
            "prefix_counts": prefix_counts,
            "prompt_token_counts": [len(request["prompt_token_ids"]) for request in requests],
            "runs": runs,
            "median_throughput_tokens_per_second": statistics.median(
                run["throughput_tokens_per_second"] for run in runs
            ),
            "outputs": saved_outputs,
            "worker_finalize_reports": finalize_reports,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "vllm": vllm.__version__,
            },
        }
        target = args.output_dir / f"t{max_tokens}.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
