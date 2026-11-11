from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SEED = 3407
DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.0


def set_seed(seed: int, include_cuda: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_think_open_tag(text: str) -> str:
    if "</think>" in text and "<think>" not in text:
        return "<think>\n" + text
    return text


def str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def install_qwen3_fp32_attention_patch() -> None:
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    def eager_fp32_attention_forward(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)

        query_fp32 = query.float()
        key_fp32 = key_states.float()
        value_fp32 = value_states.float()
        attn_weights = torch.matmul(query_fp32, key_fp32.transpose(2, 3)) * float(scaling)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]].float()
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_fp32)
        attn_output = attn_output.transpose(1, 2).contiguous().to(query.dtype)
        return attn_output, attn_weights.to(query.dtype)

    modeling_qwen3.eager_attention_forward = eager_fp32_attention_forward


def iter_shard_rows(rows: list[dict[str, Any]], args: argparse.Namespace):
    start = max(int(args.start_index), 0)
    end = len(rows) if args.end_index is None else min(int(args.end_index), len(rows))
    selected = list(enumerate(rows[start:end], start=start))
    if args.max_samples is not None:
        selected = selected[: int(args.max_samples)]
    shard_count = max(int(args.shard_count), 1)
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard {shard_index}/{shard_count}")
    for ordinal, row in selected:
        if ordinal % shard_count == shard_index:
            yield ordinal, row


@torch.no_grad()
def run_one_item_baseline(
    *,
    question: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    user_prompt = "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + question
    messages = [{"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
    ).to(device)
    input_len = input_ids.shape[1]
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_seq = out[0][input_len:]
    gen_ids = [int(token_id) for token_id in gen_seq.tolist()]
    generated_text = tokenizer.decode(gen_seq, skip_special_tokens=True).strip()
    if enable_thinking:
        generated_text = ensure_think_open_tag(generated_text)
    return {
        "generated_text": generated_text,
        "gen_ids": gen_ids,
        "total_token_consumed": len(gen_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 pure-thinking baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--attn-implementation", choices=["eager", "default", "eager_fp32_attn"], default="eager")
    parser.add_argument("--enable-thinking", type=str_to_bool, default=True)
    parser.add_argument("--reset-seed-per-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    device = torch.device(args.device)
    set_seed(args.seed, include_cuda=device.type == "cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.attn_implementation == "eager_fp32_attn":
        install_qwen3_fp32_attention_patch()
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    if args.attn_implementation in {"eager", "eager_fp32_attn"}:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    rows = read_jsonl(Path(args.input))
    selected_rows = list(iter_shard_rows(rows, args))
    manifest = {
        "model": args.model,
        "input": args.input,
        "out": str(output_path),
        "attn_implementation": args.attn_implementation,
        "enable_thinking": bool(args.enable_thinking),
        "seed": args.seed,
        "reset_seed_per_sample": args.reset_seed_per_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_sample_count": len(selected_rows),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mode_label = "think" if args.enable_thinking else "nothink"
    for ordinal, row in tqdm(selected_rows, desc=f"qwen3-{mode_label}-{args.attn_implementation} shard={args.shard_index}"):
        question = row.get("question") or row.get("problem") or ""
        if not isinstance(question, str) or not question.strip():
            continue
        if args.reset_seed_per_sample:
            set_seed(args.seed, include_cuda=device.type == "cuda")
        started_at = time.time()
        res = run_one_item_baseline(
            question=question,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            enable_thinking=bool(args.enable_thinking),
        )
        append_jsonl(
            output_path,
            {
                "event": "item_done",
                "ts": time.time(),
                "elapsed_seconds": time.time() - started_at,
                "index": int(ordinal),
                "run_id": 0,
                "seed": int(args.seed),
                "reset_seed_per_sample": bool(args.reset_seed_per_sample),
                "attn_implementation": args.attn_implementation,
                "enable_thinking": bool(args.enable_thinking),
                "question": question[:2000],
                "generated_answer": res["generated_text"],
                "generated_answer_id": [int(token_id) for token_id in res["gen_ids"]],
                "R_history": [],
                "mode_history": [],
                "standard_answer": row.get("answer"),
                "final_chain_tokens": int(res["total_token_consumed"]),
                "total_token_consumed": int(res["total_token_consumed"]),
            },
        )


if __name__ == "__main__":
    main()
