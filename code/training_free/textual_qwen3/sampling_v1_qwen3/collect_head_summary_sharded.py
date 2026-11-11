from __future__ import annotations

import argparse
import gc
import json
import os
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_src.sampling_v1.io_utils import append_jsonl, read_jsonl, write_json
from experiment_src.sampling_v1_qwen3.collect_head_summary import (
    SCORE_DEFINITION,
    custom_qwen3_attention_forward,
    finalize_dataset_summary,
    read_subset,
    run_one_item,
    setup_seed,
    update_dataset_summary,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Qwen3 head summary with one model sharded across visible GPUs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--max-memory-per-gpu",
        default="42GiB",
        help="Max memory for each visible GPU, e.g. 42GiB for L20.",
    )
    parser.add_argument("--cpu-memory", default="120GiB")
    return parser.parse_args()


def _load_model_sharded(model_path: str, max_memory_per_gpu: str, cpu_memory: str):
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 1:
        raise RuntimeError("This collector requires at least one visible CUDA GPU.")
    max_memory = {gpu_id: max_memory_per_gpu for gpu_id in range(visible_gpu_count)}
    max_memory["cpu"] = cpu_memory
    common_kwargs = dict(
        attn_implementation="eager",
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            **common_kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            **common_kwargs,
        )


def _infer_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_subset(Path(args.input), args.start_index, args.end_index)
    manifest = {
        "model": args.model,
        "input": args.input,
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_total_items": len(rows),
        "mode": "single_process_device_map_auto",
        "visible_cuda_devices": torch.cuda.device_count(),
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "cpu_memory": args.cpu_memory,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "score_definition": SCORE_DEFINITION,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    text_out = output_dir / "text_results.jsonl"
    head_out = output_dir / "head_scores.jsonl"
    for path in (text_out, head_out):
        if path.exists():
            path.unlink()

    print("[LOAD] tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    print("[LOAD] sharded model", flush=True)
    model = _load_model_sharded(args.model, args.max_memory_per_gpu, args.cpu_memory)
    input_device = _infer_input_device(model)
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "input_device": str(input_device),
                "hf_device_map": getattr(model, "hf_device_map", None),
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )

    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(custom_qwen3_attention_forward, layer.self_attn)
    model.eval()

    summary_accum: dict[str, object] = {}
    valid_items_for_summary = 0
    for row in rows:
        setup_seed(args.seed)
        idx = int(row["_local_index"])
        question = str(row.get("question") or row.get("problem") or "").strip()
        if not question:
            continue
        standard_answer = row.get("answer")
        res = run_one_item(
            question=question,
            mdl=model,
            tok=tok,
            input_device=input_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        text_row = {
            "index": idx,
            "question": question[:2000],
            "input": res["user_prompt"],
            "generated_answer": res["generated_text"],
            "standard_answer": standard_answer,
            "total_token_consumed": int(res["total_token_consumed"]),
            "has_end_think": bool(res["has_end_think"]),
            "end_think_index": int(res["end_think_index"]),
        }
        head_row = {
            "index": idx,
            "has_end_think": bool(res["has_end_think"]),
            "end_think_index": int(res["end_think_index"]),
            "head_scores_by_layer": res["head_scores"],
        }
        append_jsonl(text_out, text_row)
        append_jsonl(head_out, head_row)
        if head_row["has_end_think"] and head_row["head_scores_by_layer"]:
            update_dataset_summary(summary_accum, head_row["head_scores_by_layer"])
            valid_items_for_summary += 1
        write_json(
            output_dir / "head_scores_summary.partial.json",
            {
                "model": manifest["model"],
                "input_file": manifest["input"],
                "num_total_items": len(rows),
                "num_processed_items": idx - args.start_index + 1,
                "num_valid_items_with_end_think": valid_items_for_summary,
                "score_definition": SCORE_DEFINITION,
                "summary_by_layer": finalize_dataset_summary(summary_accum),
            },
        )
        print(
            f"[ITEM] idx={idx} has_end={res['has_end_think']} "
            f"end={res['end_think_index']} tokens={res['total_token_consumed']}",
            flush=True,
        )
        del res
        gc.collect()
        torch.cuda.empty_cache()

    write_json(
        output_dir / "head_scores_summary.json",
        {
            "model": manifest["model"],
            "input_file": manifest["input"],
            "num_total_items": len(rows),
            "num_valid_items_with_end_think": valid_items_for_summary,
            "score_definition": SCORE_DEFINITION,
            "summary_by_layer": finalize_dataset_summary(summary_accum),
        },
    )


if __name__ == "__main__":
    main()
