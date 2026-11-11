#!/usr/bin/env python3
"""InternVL3.5 MMMUVal baseline with durable per-item output."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment_src.qwen35_vllm_dynamic_final.baseline_snapshot import (
    run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline,
)


R1_SYSTEM_PROMPT = """You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""

ASSISTANT_GENERATION_PREFIX = "<|im_start|>assistant\n"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None:
                completed.add(str(row["id"]))
    return completed


def build_request(
    tokenizer: Any,
    sample: dict[str, Any],
    thinking_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = baseline.load_images(sample.get("image_paths") or [])
    content, audit = baseline.build_interleaved_content(sample, images)
    if thinking_mode == "r1":
        system_prompt: str | None = R1_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        thinking_mode_label = "official_r1_system_prompt"
    elif thinking_mode == "off":
        system_prompt = None
        messages = [{"role": "user", "content": content}]
        thinking_mode_label = "native_non_thinking_no_r1_system_prompt"
    else:
        raise ValueError(f"Unsupported thinking mode: {thinking_mode}")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not prompt.endswith(ASSISTANT_GENERATION_PREFIX):
        raise RuntimeError(
            "InternVL assistant generation suffix mismatch: "
            f"{prompt[-160:]!r}"
        )

    occurrence_order = audit["image_occurrence_order"]
    ordered_images = [images[index - 1] for index in occurrence_order]
    if prompt.count("<image>") != len(ordered_images):
        raise RuntimeError(
            "InternVL image placeholder mismatch: "
            f"prompt={prompt.count('<image>')} images={len(ordered_images)}"
        )

    request: dict[str, Any] = {"prompt": prompt}
    if ordered_images:
        request["multi_modal_data"] = {
            "image": ordered_images if len(ordered_images) != 1 else ordered_images[0]
        }
    return request, {
        **audit,
        "image_count": len(images),
        "image_occurrence_count": len(occurrence_order),
        "image_order": "full_rendered_prompt_placeholder_order",
        "prompt_style": baseline.PROMPT_STYLE,
        "assistant_generation_prefix": ASSISTANT_GENERATION_PREFIX,
        "system_prompt": system_prompt,
        "thinking_mode": thinking_mode_label,
    }


def error_row(
    sample: dict[str, Any],
    args: argparse.Namespace,
    error: str,
) -> dict[str, Any]:
    return {
        "id": sample.get("id"),
        "trajectory_id": baseline.trajectory_id(sample.get("id"), args.seed),
        "sample_seed": args.seed,
        "global_index": sample.get("global_index"),
        "subject": sample.get("subject"),
        "question": sample.get("question"),
        "options": sample.get("options") or [],
        "ground_truth": sample.get("ground_truth"),
        "engine": "vllm",
        "model": args.model_path,
        "rank": args.shard_index,
        "error": error,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }


def run(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # MMMUVal is image-only. Explicitly disabling video also avoids the
        # vLLM 0.11 InternVL dummy-video profiler path, whose int64 synthetic
        # frames cannot be converted by PIL.
        limit_mm_per_prompt={
            "image": args.max_images_per_prompt,
            "video": 0,
        },
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        skip_special_tokens=False,
    )

    samples = list(baseline.read_jsonl(Path(args.manifest)))
    samples = [
        sample
        for sample in samples
        if int(sample.get("global_index", 0)) % args.shard_count
        == args.shard_index
    ]
    if args.limit is not None:
        samples = samples[: args.limit]

    output = Path(args.output)
    done = completed_ids(output) if args.resume else set()
    samples = [sample for sample in samples if str(sample.get("id")) not in done]
    if not args.resume and output.exists():
        output.unlink()

    for sample in tqdm(samples, desc=f"internvl-shard{args.shard_index:02d}"):
        try:
            request, meta = build_request(tokenizer, sample, args.thinking_mode)
            started = time.perf_counter()
            result = llm.generate(
                [request],
                sampling_params=[sampling],
                use_tqdm=False,
            )[0]
            elapsed = time.perf_counter() - started
            completion = result.outputs[0]
            raw_output = completion.text
            token_ids = list(getattr(completion, "token_ids", None) or [])
            prompt_token_ids = list(getattr(result, "prompt_token_ids", None) or [])
            output_tokens = (
                len(token_ids)
                if token_ids
                else len(tokenizer.encode(raw_output, add_special_tokens=False))
            )
            input_tokens = len(prompt_token_ids)
            visual_tokens = (
                sum(token == img_context_token_id for token in prompt_token_ids)
                if isinstance(img_context_token_id, int)
                else 0
            )
            thinking_tokens, thinking_status = baseline.count_reasoning_tokens(
                tokenizer, raw_output
            )
            prediction, prediction_source, parseable = baseline.extract_prediction(
                raw_output,
                sample.get("options") or [],
                str(sample.get("ground_truth") or ""),
            )
            generation_correct = baseline.is_correct_prediction(
                prediction,
                str(sample.get("ground_truth") or ""),
                sample.get("options") or [],
            )
            finish_reason = getattr(completion, "finish_reason", None)
            row = {
                "id": sample.get("id"),
                "trajectory_id": baseline.trajectory_id(
                    sample.get("id"), args.seed
                ),
                "sample_seed": args.seed,
                "global_index": sample.get("global_index"),
                "subject": sample.get("subject"),
                "question": sample.get("question"),
                "options": sample.get("options") or [],
                "ground_truth": sample.get("ground_truth"),
                "prediction": prediction,
                "prediction_source": prediction_source,
                "parseable": parseable,
                "raw_output": raw_output,
                "assistant_output_for_evaluation": raw_output,
                "is_correct": generation_correct,
                "input_tokens": input_tokens,
                "visual_tokens": visual_tokens,
                "image_count": meta["image_count"],
                "image_occurrence_count": meta["image_occurrence_count"],
                "image_order": meta["image_order"],
                "image_occurrence_order": meta["image_occurrence_order"],
                "referenced_image_indices": meta["referenced_image_indices"],
                "unused_image_indices": meta["unused_image_indices"],
                "rendered_prompt_template": meta["rendered_prompt_template"],
                "prompt_style": meta["prompt_style"],
                "assistant_generation_prefix": meta["assistant_generation_prefix"],
                "system_prompt": meta["system_prompt"],
                "thinking_mode": meta["thinking_mode"],
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "thinking_tokens": thinking_tokens,
                "thinking_status": thinking_status,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repetition_penalty": args.repetition_penalty,
                "hit_max_new_tokens": bool(
                    finish_reason == "length"
                    or output_tokens >= args.max_new_tokens
                ),
                "finish_reason": finish_reason,
                "stop_reason": getattr(completion, "stop_reason", None),
                "time_seconds": elapsed,
                "generate_time_seconds": elapsed,
                "tokens_per_second": (
                    output_tokens / elapsed if elapsed > 0 else None
                ),
                "rank": args.shard_index,
                "engine": "vllm",
                "model": args.model_path,
                "prompt_suffix": (
                    baseline.MULTIPLE_CHOICE_FINAL_FORMAT
                    if sample.get("options")
                    else None
                ),
                "error": None,
            }
        except Exception as exc:
            row = error_row(
                sample,
                args,
                f"{type(exc).__name__}: {exc}",
            )
        append_jsonl_fsync(output, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)
    parser.add_argument(
        "--thinking-mode",
        choices=("r1", "off"),
        default="r1",
        help="Use the official R1 system prompt or the native non-thinking template.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
