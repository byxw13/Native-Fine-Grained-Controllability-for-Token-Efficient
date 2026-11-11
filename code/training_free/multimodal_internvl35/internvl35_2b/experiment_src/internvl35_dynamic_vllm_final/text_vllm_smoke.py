from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def read_prompts(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            rows.append(value if isinstance(value, dict) else {"prompt": str(value)})
    return rows


def build_prompt(tokenizer, row: dict[str, Any]) -> tuple[str, list[int]]:
    question = str(row.get("prompt") or row.get("question") or "").strip()
    # Match evaluation_v3_8_qwen3_final exactly: the raw question is the user
    # message and the tokenizer chat template contributes the assistant/method
    # prefix.  No runner-side instruction wrapper is permitted.
    messages = row.get("messages") or [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text, tokenizer.encode(text, add_special_tokens=False)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), required=True)
    parser.add_argument("--model", default="/path/to/aris_data/Qwen3-8B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, choices=(32, 128, 512), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    finalize_dynamic = None
    initial_request_metadata = None
    if args.mode == "dynamic":
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
    rendered = [build_prompt(tokenizer, row) for row in rows]
    prefix_counts = [text.count("<think>\n\n</think>\n\nAlright") for text, _ in rendered]
    if args.mode in {"prefix", "dynamic"} and any(count != 1 for count in prefix_counts):
        raise RuntimeError(f"method prefix must appear exactly once: {prefix_counts}")

    requests: list[dict[str, Any]] = []
    for _, token_ids in rendered:
        request: dict[str, Any] = {"prompt_token_ids": token_ids}
        if args.mode == "dynamic":
            request["aris_suppression"] = initial_request_metadata()
        requests.append(request)

    engine_kwargs = {}
    if args.mode == "dynamic":
        engine_kwargs["worker_extension_cls"] = (
            "experiment_src.qwen3vl_vllm_dynamic_final.worker_extension.DynamicArisWorkerExtension"
        )
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
    )
    for warmup_idx in range(args.warmup):
        llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        if finalize_dynamic is not None:
            finalize_dynamic(llm, reason=f"warmup_{warmup_idx}_return")

    runs = []
    all_outputs = []
    finalize_reports = []
    for repeat in range(args.repeats):
        synchronize()
        started = time.perf_counter()
        outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
        if finalize_dynamic is not None:
            finalize_reports.append(finalize_dynamic(llm, reason=f"repeat_{repeat}_return"))
        synchronize()
        elapsed = time.perf_counter() - started
        output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
        runs.append(
            {
                "repeat": repeat,
                "wall_seconds": elapsed,
                "output_tokens": output_tokens,
                "throughput_tokens_per_second": output_tokens / elapsed,
            }
        )
        all_outputs.append(
            [
                {
                    "request_id": item.request_id,
                    "token_ids": list(item.outputs[0].token_ids),
                    "text": item.outputs[0].text,
                    "finish_reason": item.outputs[0].finish_reason,
                }
                for item in outputs
            ]
        )

    throughputs = sorted(run["throughput_tokens_per_second"] for run in runs)
    report = {
        "mode": args.mode,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "batch_size": len(requests),
        "tensor_parallel_size": args.tensor_parallel_size,
        "prefix_counts": prefix_counts,
        "runs": runs,
        "median_throughput_tokens_per_second": throughputs[len(throughputs) // 2],
        "outputs": all_outputs,
        "worker_finalize_reports": finalize_reports,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "dynamic_config": os.environ.get("ARIS_DYNAMIC_CONFIG_JSON"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
