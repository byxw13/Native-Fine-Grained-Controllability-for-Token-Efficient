from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from experiment_src.sampling_v1.io_utils import append_jsonl, read_jsonl, write_json
from experiment_src.sampling_v1.remote_sampling_runner import (
    CANONICAL_CLOSED_THINK_SUFFIX,
    build_prompt,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GLOBAL_END_THINK_INDEX = -1
GLOBAL_HEAD_RECORDS: Dict[int, Dict[str, object]] = {}

SCORE_DEFINITION = {
    "region": "{</think>, </think>+1}",
    "attention_score": "a_{l,h,</think>} + a_{l,h,</think>+1}",
    "contribution_score": "|| W_o^(l,h) (a_{l,h,</think>} * v_{l,h,</think>} + a_{l,h,</think>+1} * v_{l,h,</think>+1}) ||_2",
}


class Timer:
    def __init__(self) -> None:
        self.last = None

    def log(self, msg: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time

        current = time.perf_counter()
        delta = 0.0 if self.last is None else current - self.last
        self.last = current
        print(f"[TIME] {msg}: {delta:.3f}s", flush=True)


class DetectEndThinkLogitsProcessor(LogitsProcessor):
    def __init__(self, end_think_token_id: int):
        self.end_think_token_id = end_think_token_id
        reset_global_records()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global GLOBAL_END_THINK_INDEX
        if GLOBAL_END_THINK_INDEX != -1:
            return scores
        matches = (input_ids[0] == self.end_think_token_id).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            GLOBAL_END_THINK_INDEX = matches[0].item()
            print(f"[END_THINK] detected at token index {GLOBAL_END_THINK_INDEX}", flush=True)
        return scores


def setup_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_text_prompt(question: str) -> str:
    return (
        "Solve the problem with numbered steps in the exact format:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "Step 3: ...\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        + question
    )


def read_subset(path: Path, start_index: int, end_index: Optional[int]) -> list[dict]:
    rows = read_jsonl(path)
    start = max(0, int(start_index))
    end = len(rows) if end_index is None else min(len(rows), int(end_index))
    subset = rows[start:end]
    for local_idx, row in enumerate(subset, start=start):
        row["_local_index"] = local_idx
    return subset


def reset_global_records() -> None:
    global GLOBAL_END_THINK_INDEX, GLOBAL_HEAD_RECORDS
    GLOBAL_END_THINK_INDEX = -1
    GLOBAL_HEAD_RECORDS = {}


def init_layer_record(layer_idx: int, num_heads: int, device: torch.device) -> None:
    if layer_idx not in GLOBAL_HEAD_RECORDS:
        GLOBAL_HEAD_RECORDS[layer_idx] = {
            "attn_sum": torch.zeros(num_heads, dtype=torch.float32, device=device),
            "contrib_sum": torch.zeros(num_heads, dtype=torch.float32, device=device),
            "count": 0,
        }


def serialize_head_records() -> Dict[str, object]:
    result: Dict[str, object] = {}
    for layer_idx, rec in GLOBAL_HEAD_RECORDS.items():
        cnt = max(int(rec["count"]), 1)
        attn_sum = rec["attn_sum"].detach().cpu()
        contrib_sum = rec["contrib_sum"].detach().cpu()
        result[str(layer_idx)] = {
            "record_steps": int(rec["count"]),
            "attn_sum": attn_sum.tolist(),
            "attn_mean": (attn_sum / cnt).tolist(),
            "contrib_sum": contrib_sum.tolist(),
            "contrib_mean": (contrib_sum / cnt).tolist(),
        }
    return result


def custom_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states_for_attn = repeat_kv(key_states, self.num_key_value_groups)
    value_states_for_attn = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states.float(), key_states_for_attn.float().transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_for_attn.shape[-2]]
        attn_weights = attn_weights + causal_mask.float()
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

    global GLOBAL_END_THINK_INDEX
    if GLOBAL_END_THINK_INDEX != -1:
        seq_len = attn_weights.shape[-1]
        num_heads = attn_weights.shape[1]
        end_idx = GLOBAL_END_THINK_INDEX
        next_idx = GLOBAL_END_THINK_INDEX + 1
        if next_idx < seq_len:
            device = attn_weights.device
            attn_to_end = attn_weights[0, :, -1, end_idx].detach().float()
            attn_to_next = attn_weights[0, :, -1, next_idx].detach().float()
            attn_score = attn_to_end + attn_to_next

            value_end = value_states_for_attn[0, :, end_idx, :].detach().float()
            value_next = value_states_for_attn[0, :, next_idx, :].detach().float()
            token_head_vec = (
                attn_to_end.unsqueeze(-1) * value_end
                + attn_to_next.unsqueeze(-1) * value_next
            )
            wo_heads = (
                self.o_proj.weight.detach().float().view(self.o_proj.out_features, num_heads, self.head_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            projected = torch.einsum("hnd,hd->hn", wo_heads, token_head_vec)
            contrib_score = torch.norm(projected, p=2, dim=1)

            init_layer_record(self.layer_idx, num_heads, device)
            GLOBAL_HEAD_RECORDS[self.layer_idx]["attn_sum"] += attn_score
            GLOBAL_HEAD_RECORDS[self.layer_idx]["contrib_sum"] += contrib_score
            GLOBAL_HEAD_RECORDS[self.layer_idx]["count"] += 1

    attn_output = torch.matmul(attn_weights, value_states_for_attn.float()).to(query_states.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def update_dataset_summary(summary_accum: Dict[str, object], item_head_scores: Dict[str, object]) -> None:
    for layer_str, rec in item_head_scores.items():
        attn_mean = np.array(rec["attn_mean"], dtype=np.float64)
        contrib_mean = np.array(rec["contrib_mean"], dtype=np.float64)
        if layer_str not in summary_accum:
            summary_accum[layer_str] = {
                "attn_sum": np.zeros_like(attn_mean),
                "contrib_sum": np.zeros_like(contrib_mean),
                "count": 0,
            }
        summary_accum[layer_str]["attn_sum"] += attn_mean
        summary_accum[layer_str]["contrib_sum"] += contrib_mean
        summary_accum[layer_str]["count"] += 1


def finalize_dataset_summary(summary_accum: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for layer_str, rec in summary_accum.items():
        cnt = max(int(rec["count"]), 1)
        out[layer_str] = {
            "num_items": int(rec["count"]),
            "attn_mean_over_items": (rec["attn_sum"] / cnt).tolist(),
            "contrib_mean_over_items": (rec["contrib_sum"] / cnt).tolist(),
        }
    return out


def run_one_item(
    question: str,
    mdl,
    tok,
    input_device: torch.device,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
) -> dict[str, object]:
    reset_global_records()
    user_prompt = ensure_text_prompt(question)
    input_ids = build_prompt(tok, user_prompt, CANONICAL_CLOSED_THINK_SUFFIX).to(input_device)
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    input_len = input_ids.shape[1]
    end_think_token_ids = tok.encode("</think>", add_special_tokens=False)
    if not end_think_token_ids:
        raise ValueError("Tokenizer could not encode </think>.")
    trigger_processor = DetectEndThinkLogitsProcessor(int(end_think_token_ids[0]))
    logits_processor = LogitsProcessorList([trigger_processor])
    with torch.no_grad():
        out = mdl.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=repetition_penalty,
            logits_processor=logits_processor,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    gen_seq = out[0][input_len:]
    generated_text = tok.decode(gen_seq, skip_special_tokens=False).strip()
    return {
        "user_prompt": user_prompt,
        "generated_text": generated_text,
        "total_token_consumed": int(gen_seq.shape[0]),
        "has_end_think": GLOBAL_END_THINK_INDEX != -1,
        "end_think_index": int(GLOBAL_END_THINK_INDEX),
        "head_scores": serialize_head_records(),
    }


def merge_worker_outputs(output_dir: Path, num_workers: int, manifest: dict) -> None:
    text_rows = []
    head_rows = []
    summary_accum: Dict[str, object] = {}
    valid_items_for_summary = 0
    for worker_rank in range(num_workers):
        worker_dir = output_dir / f"worker_{worker_rank}"
        text_path = worker_dir / "text_results.jsonl"
        head_path = worker_dir / "head_scores.jsonl"
        if text_path.exists():
            text_rows.extend(read_jsonl(text_path))
        if head_path.exists():
            head_rows.extend(read_jsonl(head_path))
    text_rows.sort(key=lambda x: x["index"])
    head_rows.sort(key=lambda x: x["index"])
    final_text_path = output_dir / "text_results.jsonl"
    final_head_path = output_dir / "head_scores.jsonl"
    if final_text_path.exists():
        final_text_path.unlink()
    if final_head_path.exists():
        final_head_path.unlink()
    for row in text_rows:
        append_jsonl(final_text_path, row)
    for row in head_rows:
        append_jsonl(final_head_path, row)
        if row.get("has_end_think") and row.get("head_scores_by_layer"):
            update_dataset_summary(summary_accum, row["head_scores_by_layer"])
            valid_items_for_summary += 1
    write_json(
        output_dir / "head_scores_summary.json",
        {
            "model": manifest["model"],
            "input_file": manifest["input"],
            "num_total_items": len(text_rows),
            "num_valid_items_with_end_think": valid_items_for_summary,
            "score_definition": SCORE_DEFINITION,
            "summary_by_layer": finalize_dataset_summary(summary_accum),
        },
    )


def worker_main(worker_rank: int, device_id: int, args_dict: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    output_dir = Path(args_dict["output_dir"])
    worker_dir = output_dir / f"worker_{worker_rank}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    text_out = worker_dir / "text_results.jsonl"
    head_out = worker_dir / "head_scores.jsonl"
    for path in (text_out, head_out):
        if path.exists():
            path.unlink()

    rows = read_subset(Path(args_dict["input"]), args_dict["start_index"], args_dict["end_index"])
    assigned_rows = rows[worker_rank::args_dict["num_workers"]]

    timer = Timer()
    timer.log(f"worker{worker_rank}: start gpu={device_id}")
    tok = AutoTokenizer.from_pretrained(args_dict["model"])
    timer.log(f"worker{worker_rank}: tokenizer loaded")
    mdl = AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    timer.log(f"worker{worker_rank}: model loaded")
    for layer in mdl.model.layers:
        layer.self_attn.forward = types.MethodType(custom_qwen2_attention_forward, layer.self_attn)
    print(f">>> worker{worker_rank} summary hook mounted on cuda:{device_id}", flush=True)
    mdl.eval()

    for row in assigned_rows:
        setup_seed(args_dict["seed"])
        idx = int(row["_local_index"])
        question = str(row.get("question") or row.get("problem") or "").strip()
        if not question:
            continue
        standard_answer = row.get("answer")
        res = run_one_item(
            question=question,
            mdl=mdl,
            tok=tok,
            input_device=device,
            max_new_tokens=args_dict["max_new_tokens"],
            temperature=args_dict["temperature"],
            repetition_penalty=1.0,
        )
        append_jsonl(
            text_out,
            {
                "index": idx,
                "question": question[:2000],
                "input": res["user_prompt"],
                "generated_answer": res["generated_text"],
                "standard_answer": standard_answer,
                "total_token_consumed": int(res["total_token_consumed"]),
                "has_end_think": bool(res["has_end_think"]),
                "end_think_index": int(res["end_think_index"]),
            },
        )
        append_jsonl(
            head_out,
            {
                "index": idx,
                "has_end_think": bool(res["has_end_think"]),
                "end_think_index": int(res["end_think_index"]),
                "head_scores_by_layer": res["head_scores"],
            },
        )
        print(
            f"[ITEM] worker={worker_rank} gpu={device_id} idx={idx} "
            f"has_end={res['has_end_think']} end={res['end_think_index']} "
            f"tokens={res['total_token_consumed']}",
            flush=True,
        )
        del res
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device-ids", default="0,1,2,3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This collector requires CUDA.")
    device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
    if not device_ids:
        raise ValueError("No device ids provided.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "input": args.input,
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "num_total_items": len(read_subset(Path(args.input), args.start_index, args.end_index)),
        "device_ids": device_ids,
        "num_workers": len(device_ids),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "score_definition": SCORE_DEFINITION,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    ctx = mp.get_context("spawn")
    args_dict = {
        "model": args.model,
        "input": args.input,
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "num_workers": len(device_ids),
    }
    procs = []
    for worker_rank, device_id in enumerate(device_ids):
        proc = ctx.Process(target=worker_main, args=(worker_rank, device_id, args_dict))
        proc.start()
        procs.append(proc)
    failed = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.pid, proc.exitcode))
    if failed:
        raise RuntimeError(f"One or more workers failed: {failed}")
    merge_worker_outputs(output_dir, len(device_ids), manifest)


if __name__ == "__main__":
    main()
