#!/usr/bin/env python3
"""MMMU Qwen3-VL Thinking multi-sample evaluation with P0 image handling.

The complete question/options prompt is rendered before ``<image N>`` markers
are replaced. This preserves image references inside answer options and ignores
unreferenced image fields while retaining them in the audit metadata.

It keeps the same output schema as scripts/run_baseline.py so that the existing
post-processing can reuse the single-sample fields. Each ``(sample, seed)`` is
flushed as an independent JSONL row and can be resumed without regeneration.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MULTIPLE_CHOICE_INSTRUCTION = "Please select the correct answer from the options above."
MULTIPLE_CHOICE_FINAL_FORMAT = (
    "At the end, output the final answer as: Answer: <option letter>."
)
PROMPT_STYLE = "vlmevalkit_mmmu_concise_mc_answer_suffix_multisample"
IMAGE_PLACEHOLDER_RE = re.compile(r"<image\s+(\d+)>", flags=re.I)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def normalize_gt(answer: Any) -> str:
    text = str(answer or "").strip()
    return text.upper() if re.fullmatch(r"[A-Z]", text, flags=re.I) else text


def format_options(options: list[str]) -> str:
    lines = []
    for idx, option in enumerate(options or []):
        label = LETTERS[idx]
        option_text = str(option).strip()
        if re.match(r"^[A-Z][\.\)]\s+", option_text):
            lines.append(option_text)
        else:
            lines.append(f"{label}. {option_text}")
    return "\n".join(lines)


def build_rendered_prompt(sample: dict[str, Any]) -> str:
    question = str(sample.get("question") or "")
    options = format_options(sample.get("options") or [])
    prompt = f"Question: {question}\n"
    if options:
        prompt += (
            f"Options:\n{options}\n{MULTIPLE_CHOICE_INSTRUCTION} \n"
            f"{MULTIPLE_CHOICE_FINAL_FORMAT}\n"
        )
    return prompt


def load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(path).convert("RGB") for path in paths]


def append_text_content(content: list[dict[str, Any]], text: str) -> None:
    if not text:
        return
    if content and content[-1]["type"] == "text":
        content[-1]["text"] += text
    else:
        content.append({"type": "text", "text": text})


def build_interleaved_content(
    sample: dict[str, Any],
    images: list[Image.Image],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rendered_prompt = build_rendered_prompt(sample)
    content: list[dict[str, Any]] = []
    image_occurrence_order: list[int] = []
    cursor = 0

    for match in IMAGE_PLACEHOLDER_RE.finditer(rendered_prompt):
        append_text_content(content, rendered_prompt[cursor:match.start()])
        image_index = int(match.group(1)) - 1
        if image_index < 0 or image_index >= len(images):
            raise ValueError(
                f"Image placeholder {match.group(0)!r} has no matching image; "
                f"received {len(images)} image(s)"
            )
        content.append({"type": "image", "image": images[image_index]})
        image_occurrence_order.append(image_index)
        cursor = match.end()

    append_text_content(content, rendered_prompt[cursor:])
    if images and not image_occurrence_order:
        content = [{"type": "image", "image": image} for image in images] + content
        image_occurrence_order = list(range(len(images)))

    referenced = sorted(set(image_occurrence_order))
    unused = sorted(set(range(len(images))) - set(referenced))
    return content, {
        "rendered_prompt_template": rendered_prompt,
        "image_occurrence_order": [index + 1 for index in image_occurrence_order],
        "referenced_image_indices": [index + 1 for index in referenced],
        "unused_image_indices": [index + 1 for index in unused],
    }


def count_reasoning_tokens(tokenizer, text: str) -> tuple[int, str]:
    if "<think>" in text and "</think>" in text:
        segment = text.split("<think>", 1)[1].split("</think>", 1)[0]
        return len(tokenizer.encode(segment, add_special_tokens=False)), "closed_with_start"
    if "</think>" in text:
        segment = text.split("</think>", 1)[0]
        return len(tokenizer.encode(segment, add_special_tokens=False)), "closed_no_start"
    return 0, "not_found"


def answer_region(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text[-2400:]


def extract_mc_prediction(text: str, option_count: int) -> tuple[str | None, str]:
    valid = LETTERS[:option_count]
    region = answer_region(text)
    patterns = [
        r"(?:the\s+answer|final\s+answer|answer|correct\s+answer|option)\s*(?:is|:|=)?\s*(?:option\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?",
        r"\\boxed\{\s*([A-Z])\s*\}",
        r"\b(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+answer\s+is\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?",
        r"\b(?:choose|select|selected)\s*(?:option\s*)?[\(\[]?\s*([A-Z])\s*[\)\]]?",
        r"\b([A-Z])\s*(?:is\s+the\s+answer|is\s+correct)\b",
        r"[\(\[]\s*([A-Z])\s*[\)\]]",
        r"^\s*([A-Z])\s*[\)\]\.\:]\s+",
        r"^\s*[\(\[]?\s*([A-Z])\s*[\)\]\.\:]?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, region, flags=re.I | re.M)
        for match in reversed(matches):
            letter = str(match).upper()
            if letter in valid:
                return letter, f"pattern:{pattern}"
    tail = region[-700:]
    matches = re.findall(r"(?<![A-Za-z])[\(\[]?([A-Z])[\)\]]?(?![A-Za-z])", tail)
    for match in reversed(matches):
        letter = match.upper()
        if letter in valid:
            return letter, "last_letter_heuristic"
    return None, "parse_failed"


def extract_option_text_prediction(text: str, options: list[str]) -> tuple[str | None, str]:
    region = answer_region(text)[-1500:]
    for idx, option in enumerate(options or []):
        option_text = str(option).strip()
        if not option_text:
            continue
        escaped = re.escape(option_text)
        patterns = [
            rf"(?:final\s+answer|answer|correct\s+answer)\s*(?:is|:)?\s*(?:option\s+[A-Z]\s*)?{escaped}",
            rf"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+answer\s+is\s*)?(?:option\s+[A-Z]\s*)?{escaped}",
            rf"{escaped}\s*(?:is\s+the\s+answer|is\s+correct)",
        ]
        for pattern in patterns:
            if re.search(pattern, region, flags=re.I):
                return LETTERS[idx], "option_text_answer_region"
    return None, "parse_failed"


def extract_prediction(text: str, options: list[str], ground_truth: str) -> tuple[str | None, str, bool]:
    gt = normalize_gt(ground_truth)
    if options and re.fullmatch(r"[A-Z]", gt):
        pred, source = extract_mc_prediction(text, len(options))
        if pred is not None:
            return pred, source, True
        pred, source = extract_option_text_prediction(text, options)
        return pred, source, pred is not None
    region = answer_region(text)
    # ``</think>\n\nAnswer`` can be injected by the controller as a bridge.
    # For open questions it frequently starts a new explanatory response, so
    # the leading occurrence is not itself a final-answer marker.
    region = re.sub(
        r"^\s*answer\b\s*(?::|=|is\b)?\s*",
        "",
        region,
        count=1,
        flags=re.I,
    )
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", region)
    if boxed:
        return boxed[-1].strip(), "boxed_text", True
    answer_match = re.findall(
        r"(?im)^\s*(?:[-*#>]\s*)*\**\s*"
        r"(?:final\s+answer|answer|final\s+value|conclusion)"
        r"\s*\**\s*(?:is|:|=)\s*(.+)$",
        region,
    )
    if answer_match:
        return answer_match[-1].strip(), "last_explicit_answer_line", True
    return region.strip().splitlines()[-1].strip() if region.strip() else None, "last_line", bool(region.strip())


def is_correct_prediction(prediction: str | None, ground_truth: str, options: list[str]) -> bool:
    if prediction is None:
        return False
    gt = normalize_gt(ground_truth)
    if options and re.fullmatch(r"[A-Z]", gt):
        return prediction.strip().upper() == gt
    return normalize_text(prediction) == normalize_text(gt)


def get_image_token_id(processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    token = getattr(tokenizer, "image_token", None) or "<|image_pad|>"
    if tokenizer is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id if isinstance(token_id, int) and token_id >= 0 else None


def build_processed_image_audit(
    processor,
    image_inputs: list[Image.Image] | None,
    proc_inputs,
    image_occurrence_order: list[int],
) -> list[dict[str, Any]]:
    images = list(image_inputs or [])
    grids = proc_inputs.get("image_grid_thw")
    grid_rows = grids.tolist() if grids is not None else [None] * len(images)
    if len(grid_rows) != len(images) or len(images) != len(image_occurrence_order):
        raise ValueError(
            "Image audit length mismatch: "
            f"images={len(images)}, grids={len(grid_rows)}, order={len(image_occurrence_order)}"
        )

    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    audit = []
    for occurrence, (image, grid, source_index) in enumerate(
        zip(images, grid_rows, image_occurrence_order), start=1
    ):
        width, height = image.size
        per_image_tokens = None
        if grid is not None:
            per_image_tokens = int(grid[0] * grid[1] * grid[2] // (merge_size**2))
        audit.append(
            {
                "occurrence": occurrence,
                "source_image_index": source_index,
                "processed_width": int(width),
                "processed_height": int(height),
                "visual_tokens": per_image_tokens,
                "image_grid_thw": grid,
            }
        )
    return audit


def build_vllm_request(
    processor,
    sample: dict[str, Any],
    image_token_id: int | None,
    generation_prefix: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    images = load_images(sample.get("image_paths") or [])
    content, content_audit = build_interleaved_content(sample, images)
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    native_generation_prefix = "<think>\n"
    if generation_prefix is not None:
        if not prompt.endswith(native_generation_prefix):
            raise ValueError(
                "Cannot override generation prefix because the rendered prompt does not "
                f"end with {native_generation_prefix!r}: {prompt[-120:]!r}"
            )
        prompt = prompt[: -len(native_generation_prefix)] + generation_prefix
    image_patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
        )
        proc_inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        mm_images = image_inputs
    else:
        mm_images = [
            images[index - 1]
            for index in content_audit["image_occurrence_order"]
        ]
        proc_inputs = processor(text=[prompt], images=mm_images, return_tensors="pt")
    input_ids = proc_inputs["input_ids"]
    input_tokens = int(input_ids.shape[-1])
    visual_tokens = int((input_ids == image_token_id).sum().item()) if image_token_id is not None else 0
    processed_images = build_processed_image_audit(
        processor,
        mm_images,
        proc_inputs,
        content_audit["image_occurrence_order"],
    )
    audited_visual_tokens = sum(
        image["visual_tokens"] or 0 for image in processed_images
    )
    if processed_images and audited_visual_tokens != visual_tokens:
        raise ValueError(
            f"Per-image visual token audit mismatch: {audited_visual_tokens} != {visual_tokens}"
        )
    request = {"prompt": prompt}
    if mm_images:
        request["multi_modal_data"] = {"image": mm_images if len(mm_images) != 1 else mm_images[0]}
    return request, {
        "input_tokens": input_tokens,
        "visual_tokens": visual_tokens,
        "image_patch_size": image_patch_size,
        "image_count": len(images),
        "image_occurrence_count": len(content_audit["image_occurrence_order"]),
        "image_order": "full_rendered_prompt_placeholder_order",
        "prompt_style": PROMPT_STYLE,
        "assistant_generation_prefix": generation_prefix or native_generation_prefix,
        "processed_images": processed_images,
        **content_audit,
    }


def trajectory_id(sample_id: Any, seed: int) -> str:
    return f"{sample_id}::seed{seed}"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain one or more unique integers")
    return seeds


def completed_trajectory_ids(path: Path) -> set[str]:
    completed = set()
    if not path.exists():
        return completed
    for row in read_jsonl(path):
        key = row.get("trajectory_id")
        if key:
            completed.add(str(key))
    return completed


def error_row(
    sample: dict[str, Any],
    args,
    error: str,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "id": sample.get("id"),
        "trajectory_id": trajectory_id(sample.get("id"), seed),
        "sample_seed": seed,
        "global_index": sample.get("global_index"),
        "subject": sample.get("subject"),
        "question": sample.get("question"),
        "options": sample.get("options") or [],
        "ground_truth": sample.get("ground_truth"),
        "prediction": None,
        "parseable": False,
        "raw_output": "",
        "is_correct": False,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "engine": "vllm",
        "prompt_style": PROMPT_STYLE,
        "rendered_prompt_template": build_rendered_prompt(sample),
        "error": error,
        "rank": rank,
    }


def run_shard(args) -> None:
    seeds = parse_seeds(args.seeds)
    processor_path = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    image_token_id = get_image_token_id(processor)
    llm = LLM(
        model=args.model_path,
        tokenizer=processor_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
    )
    candidate_rows = list(read_jsonl(Path(args.manifest)))
    if args.only_open_ended:
        candidate_rows = [row for row in candidate_rows if not (row.get("options") or [])]
    if args.limit_samples is not None:
        candidate_rows = candidate_rows[: args.limit_samples]
    tasks = []
    for sample in candidate_rows:
        global_index = int(sample.get("global_index", 0))
        for seed_position, seed in enumerate(seeds):
            task_index = global_index * len(seeds) + seed_position
            if task_index % args.shard_count == args.shard_index:
                tasks.append((sample, seed))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_completed = completed_trajectory_ids(out_path) if args.resume else set()
    tasks = [
        (sample, seed)
        for sample, seed in tasks
        if trajectory_id(sample.get("id"), seed) not in already_completed
    ]
    output_mode = "a" if args.resume else "w"
    with out_path.open(output_mode, encoding="utf-8") as f:
        for start in tqdm(range(0, len(tasks), args.batch_size), desc=f"vllm-shard{args.shard_index:02d}"):
            batch = tasks[start:start + args.batch_size]
            requests, metas, prepared_tasks, sampling_params = [], [], [], []
            for sample, seed in batch:
                try:
                    request, meta = build_vllm_request(
                        processor,
                        sample,
                        image_token_id,
                        generation_prefix=args.generation_prefix,
                    )
                    requests.append(request)
                    metas.append(meta)
                    prepared_tasks.append((sample, seed))
                    sampling_params.append(
                        SamplingParams(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            min_p=args.min_p,
                            presence_penalty=args.presence_penalty,
                            max_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty,
                            seed=seed,
                            skip_special_tokens=False,
                        )
                    )
                except Exception as exc:
                    f.write(json.dumps(error_row(sample, args, f"{type(exc).__name__}: {exc}", args.shard_index, seed), ensure_ascii=False) + "\n")
                    f.flush()
            if not requests:
                continue
            gen_start = time.perf_counter()
            try:
                outputs = llm.generate(requests, sampling_params=sampling_params, use_tqdm=False)
                gen_end = time.perf_counter()
            except Exception as exc:
                for sample, seed in prepared_tasks:
                    f.write(json.dumps(error_row(sample, args, f"{type(exc).__name__}: {exc}", args.shard_index, seed), ensure_ascii=False) + "\n")
                f.flush()
                continue
            per_sample_time = (gen_end - gen_start) / max(1, len(outputs))
            for (sample, seed), meta, output in zip(prepared_tasks, metas, outputs):
                completion = output.outputs[0]
                raw_output = completion.text
                assistant_output_for_evaluation = (
                    str(meta.get("assistant_generation_prefix") or "") + raw_output
                )
                token_ids = getattr(completion, "token_ids", None) or []
                output_tokens = len(token_ids) if token_ids else len(tokenizer.encode(raw_output, add_special_tokens=False))
                thinking_tokens, thinking_status = count_reasoning_tokens(tokenizer, raw_output)
                prediction, prediction_source, parseable = extract_prediction(
                    assistant_output_for_evaluation,
                    sample.get("options") or [],
                    str(sample.get("ground_truth") or ""),
                )
                correct = is_correct_prediction(prediction, str(sample.get("ground_truth") or ""), sample.get("options") or [])
                total_tokens = int(meta.get("input_tokens") or 0) + output_tokens
                row = {
                    "id": sample.get("id"),
                    "trajectory_id": trajectory_id(sample.get("id"), seed),
                    "sample_seed": seed,
                    "global_index": sample.get("global_index"),
                    "subject": sample.get("subject"),
                    "question": sample.get("question"),
                    "options": sample.get("options") or [],
                    "ground_truth": sample.get("ground_truth"),
                    "prediction": prediction,
                    "prediction_source": prediction_source,
                    "parseable": parseable,
                    "raw_output": raw_output,
                    "assistant_output_for_evaluation": assistant_output_for_evaluation,
                    "is_correct": correct,
                    "input_tokens": int(meta.get("input_tokens") or 0),
                    "visual_tokens": int(meta.get("visual_tokens") or 0),
                    "image_patch_size": int(meta.get("image_patch_size") or 0),
                    "image_count": int(meta.get("image_count") or 0),
                    "image_occurrence_count": int(meta.get("image_occurrence_count") or 0),
                    "image_order": meta.get("image_order"),
                    "image_occurrence_order": meta.get("image_occurrence_order") or [],
                    "referenced_image_indices": meta.get("referenced_image_indices") or [],
                    "unused_image_indices": meta.get("unused_image_indices") or [],
                    "processed_images": meta.get("processed_images") or [],
                    "rendered_prompt_template": meta.get("rendered_prompt_template"),
                    "prompt_style": meta.get("prompt_style"),
                    "assistant_generation_prefix": meta.get("assistant_generation_prefix"),
                    "output_tokens": output_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "repetition_penalty": args.repetition_penalty,
                    "hit_max_new_tokens": output_tokens >= args.max_new_tokens,
                    "thinking_tokens": thinking_tokens,
                    "thinking_status": thinking_status,
                    "total_tokens": total_tokens,
                    "time_seconds": per_sample_time,
                    "generate_time_seconds": per_sample_time,
                    "tokens_per_second": output_tokens / per_sample_time if per_sample_time > 0 else None,
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "rank": args.shard_index,
                    "engine": "vllm",
                    "prompt_suffix": (
                        MULTIPLE_CHOICE_FINAL_FORMAT
                        if sample.get("options")
                        else None
                    ),
                    "error": None,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/path/to/local_asset")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument(
        "--generation-prefix",
        default=None,
        help=(
            "Optional exact replacement for the native '<think>\\n' assistant "
            "generation prefix. Newlines must be passed as actual characters."
        ),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--only-open-ended", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    run_shard(args)


if __name__ == "__main__":
    main()
