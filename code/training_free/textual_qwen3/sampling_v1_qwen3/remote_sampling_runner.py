from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from experiment_src.sampling_v1_qwen3.config import CHECKPOINT_NEWLINE_SPAN, DEFAULT_MAX_NEW_TOKENS, LEVEL_BY_ID
from experiment_src.sampling_v1.io_utils import append_jsonl, read_jsonl


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# These globals are the runtime knobs used by the monkey-patched attention
# forward. evaluation_v1 changes them when a sample enters a different level.
# In particular:
# - GLOBAL_END_THINK_INDEX: token position of the first "</think>" token in the
#   prompt/context. Suppression is applied to this position and the next token.
# - GLOBAL_SUPPRESSION_ACTIVE: master switch; when False, the custom attention
#   forward behaves like vanilla attention.
# - TARGET_SUPPRESS_HEADS_BY_LAYER: {layer_idx -> [head_idx, ...]} selected from
#   the summary json according to the configured suppression budget.
GLOBAL_END_THINK_INDEX = -1
GLOBAL_SUPPRESSION_ACTIVE = False
TARGET_SUPPRESS_HEADS_BY_LAYER: Dict[int, List[int]] = {}
# MASK_VALUE=0.0 means a hard multiplicative suppression. The selected
# attention mass is zeroed out first, then renormalized over the remaining
# positions. If this is changed to e.g. 0.2, the code becomes a soft scaling
# instead of full suppression.
MASK_VALUE = 0.0


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _infer_cache_seq_len(cache: Cache | None) -> int | None:
    if cache is None:
        return 0
    try:
        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            value = get_seq_length()
            if isinstance(value, int):
                return value
    except Exception:
        pass
    try:
        if isinstance(cache, (list, tuple)) and cache:
            first = cache[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1 and hasattr(first[0], "shape"):
                return int(first[0].shape[-2])
    except Exception:
        pass
    return None


_DEBUG_COUNTS = {
    "attn": 0,
    "suppression": 0,
    "heads": 0,
    "prompt": 0,
    "end_think": 0,
}


def _debug_log(counter_key: str, event: str, **payload: object) -> None:
    flag_by_counter = {
        "attn": "ARIS_QWEN3_DEBUG_ATTN",
        "suppression": "ARIS_QWEN3_DEBUG_ATTN",
        "heads": "ARIS_QWEN3_DEBUG_HEADS",
        "prompt": "ARIS_QWEN3_DEBUG_PROMPT",
        "end_think": "ARIS_QWEN3_DEBUG_PROMPT",
    }
    if not _env_flag(flag_by_counter.get(counter_key, "ARIS_QWEN3_DEBUG_LOGS")):
        return
    limit = _env_int("ARIS_QWEN3_DEBUG_MAX_EVENTS", 40)
    seen = _DEBUG_COUNTS.get(counter_key, 0)
    if seen >= limit:
        return
    _DEBUG_COUNTS[counter_key] = seen + 1
    print(
        json.dumps(
            {
                "qwen3_debug": True,
                "source": "sampling_runtime",
                "event": event,
                "counter_key": counter_key,
                "counter_index": seen,
                **payload,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

def custom_qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # Keep this path as close as possible to the native Qwen3 eager attention
    # implementation, and only inject the minimal post-softmax suppression hook.
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cache_seq_len_before = _infer_cache_seq_len(past_key_values)
    kwargs_has_singular_cache = "past_key_value" in kwargs and kwargs["past_key_value"] is not None

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states_for_attn = repeat_kv(key_states, self.num_key_value_groups)
    value_states_for_attn = repeat_kv(value_states, self.num_key_value_groups)
    cache_seq_len_after = _infer_cache_seq_len(past_key_values)

    attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * self.scaling
    _debug_log(
        "attn",
        "attention_forward",
        layer_idx=int(self.layer_idx),
        q_len=int(query_states.shape[-2]),
        kv_len=int(key_states_for_attn.shape[-2]),
        attention_mask_shape=None if attention_mask is None else list(attention_mask.shape),
        attention_mask_last_dim=None if attention_mask is None else int(attention_mask.shape[-1]),
        mask_minus_kv_len=None
        if attention_mask is None
        else int(attention_mask.shape[-1]) - int(key_states_for_attn.shape[-2]),
        cache_seq_len_before=cache_seq_len_before,
        cache_seq_len_after=cache_seq_len_after,
        cache_position_shape=None if cache_position is None else list(cache_position.shape),
        cache_position_tail=None
        if cache_position is None
        else [int(v) for v in cache_position[-8:].detach().cpu().tolist()],
        kwargs_keys=sorted(str(key) for key in kwargs.keys()),
        kwargs_has_past_key_value=bool(kwargs_has_singular_cache),
        past_key_values_is_none=bool(past_key_values is None),
    )
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    global GLOBAL_END_THINK_INDEX, GLOBAL_SUPPRESSION_ACTIVE, TARGET_SUPPRESS_HEADS_BY_LAYER
    target_heads = TARGET_SUPPRESS_HEADS_BY_LAYER.get(int(self.layer_idx), [])
    if target_heads and GLOBAL_END_THINK_INDEX != -1 and GLOBAL_SUPPRESSION_ACTIVE:
        seq_len = attn_weights.shape[-1]
        suppress_indices = [idx for idx in (GLOBAL_END_THINK_INDEX, GLOBAL_END_THINK_INDEX + 1) if idx < seq_len]
        if suppress_indices:
            head_idx_tensor = torch.tensor(target_heads, device=attn_weights.device, dtype=torch.long)
            selected_attn = torch.index_select(attn_weights, dim=1, index=head_idx_tensor).float()
            for idx in suppress_indices:
                selected_attn[:, :, :, idx] *= MASK_VALUE
            denom = selected_attn.sum(dim=-1, keepdim=True)
            _debug_log(
                "suppression",
                "suppression_stats",
                layer_idx=int(self.layer_idx),
                selected_head_count=int(len(target_heads)),
                suppress_indices=[int(idx) for idx in suppress_indices],
                seq_len=int(seq_len),
                denom_min=round(float(denom.min().item()), 8),
                denom_max=round(float(denom.max().item()), 8),
                denom_lt_1e_6=int((denom < 1e-6).sum().item()),
                mask_value=float(MASK_VALUE),
                global_end_think_index=int(GLOBAL_END_THINK_INDEX),
            )
            selected_attn = selected_attn / (denom + 1e-6)
            attn_weights.index_copy_(1, head_idx_tensor, selected_attn.to(attn_weights.dtype))

    attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states_for_attn)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def load_target_heads(summary_json: Path, budget: Optional[float]) -> Dict[int, List[int]]:
    if budget is None or float(budget) <= 0.0:
        _debug_log(
            "heads",
            "load_target_heads",
            summary_json=str(summary_json),
            requested_budget=None if budget is None else float(budget),
            selected_head_count=0,
            selected_layer_count=0,
            positive_head_count=0,
        )
        return {}
    # The summary json stores a per-head importance score. "budget" here does
    # not mean "multiply attention by budget". Instead, it means "take the top
    # heads until their normalized cumulative importance reaches this budget".
    # Example: budget=0.2 selects the most important heads whose total score
    # mass is at least 20% of the positive mass across all heads.
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    by_layer = summary["summary_by_layer"]
    flat: list[tuple[int, int, float]] = []
    for layer_str, rec in by_layer.items():
        for head_idx, score in enumerate(rec["attn_mean_over_items"]):
            flat.append((int(layer_str), int(head_idx), max(float(score), 0.0)))
    total = sum(score for _, _, score in flat)
    if total <= 0:
        _debug_log(
            "heads",
            "load_target_heads",
            summary_json=str(summary_json),
            requested_budget=float(budget),
            selected_head_count=0,
            selected_layer_count=0,
            positive_head_count=0,
            total_positive_mass=0.0,
        )
        return {}
    positive_head_count = sum(1 for _, _, score in flat if score > 0.0)
    flat.sort(key=lambda item: item[2], reverse=True)
    selected: Dict[int, List[int]] = {}
    cumulative = 0.0
    for layer_idx, head_idx, score in flat:
        selected.setdefault(layer_idx, []).append(head_idx)
        cumulative += score / total
        if cumulative >= budget:
            break
    selected_sorted = {layer_idx: sorted(heads) for layer_idx, heads in selected.items()}
    _debug_log(
        "heads",
        "load_target_heads",
        summary_json=str(summary_json),
        requested_budget=float(budget),
        selected_head_count=int(sum(len(heads) for heads in selected_sorted.values())),
        selected_layer_count=int(len(selected_sorted)),
        positive_head_count=int(positive_head_count),
        total_positive_mass=round(float(total), 8),
        realized_cumulative=round(float(cumulative), 8),
    )
    return selected_sorted


def format_user_prompt(question: str) -> str:
    raw_question = str(question)
    question = raw_question.strip()
    prompt_style = os.environ.get("ARIS_QWEN3_USER_PROMPT_STYLE", "default").strip().lower()
    if prompt_style == "human_eval_baseline":
        return (
            "Please reason step by step, and put your final Python code implementation "
            "within ```python and ``` markdown code blocks.\n\n"
            f"{raw_question}"
        )
    if prompt_style == "legacy_gpqa":
        return (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"{question}"
        )
    if prompt_style != "default":
        raise ValueError(f"Unsupported ARIS_QWEN3_USER_PROMPT_STYLE: {prompt_style!r}")
    return (
        "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        "Problem:\n"
        f"{question}"
    )


def build_prompt(tokenizer, question: str, assistant_suffix: str) -> torch.Tensor:
    messages = [{"role": "user", "content": format_user_prompt(question)}]
    apply_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
    }
    base_ids = tokenizer.apply_chat_template(messages, **apply_kwargs)
    if assistant_suffix:
        suffix_ids = tokenizer(assistant_suffix, add_special_tokens=False, return_tensors="pt").input_ids
        base_ids = torch.cat([base_ids, suffix_ids], dim=1)
    prompt_token_ids = base_ids[0].tolist()
    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    _debug_log(
        "prompt",
        "build_prompt",
        assistant_suffix=assistant_suffix,
        prompt_len=int(base_ids.shape[1]),
        prompt_end_think_literal_count=int(prompt_text.count("</think>")),
        prompt_tail_ids=[int(token_id) for token_id in prompt_token_ids[-48:]],
        prompt_tail_text_visible=prompt_text[-400:],
    )
    return base_ids


def tensor_to_list(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in tensor.detach().cpu().float().tolist()]


def safe_round(value: float, digits: int = 6) -> float:
    if value != value:
        return 0.0
    if value == float("inf"):
        return 1e9
    if value == float("-inf"):
        return -1e9
    if abs(value) < 10 ** (-digits):
        return 0.0
    return round(float(value), digits)


def compute_token_metric(logits: torch.Tensor, token_id: int) -> tuple[float, float, float]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    log_probs = torch.nan_to_num(log_probs, nan=-1e9, posinf=0.0, neginf=-1e9)
    nonzero_mask = probs > 0
    entropy_value = float(-(probs[nonzero_mask] * log_probs[nonzero_mask]).sum().item())
    top2_probs = torch.topk(probs, k=2).values
    margin_value = float((top2_probs[0] - top2_probs[1]).item())
    logprob_value = float(log_probs[token_id].item())
    return safe_round(entropy_value), safe_round(margin_value), safe_round(logprob_value)


def compute_checkpoints(tokenizer, generated_ids: Sequence[int], generated_last_hidden: Sequence[torch.Tensor]) -> tuple[list[int], list[list[float]], list[list[float]]]:
    checkpoint_end_token_idx: list[int] = []
    checkpoint_window_hidden_mean: list[list[float]] = []
    checkpoint_last_hidden: list[list[float]] = []
    newline_count = 0
    window_vectors: list[torch.Tensor] = []
    for token_offset, (token_id, last_layer) in enumerate(zip(generated_ids, generated_last_hidden)):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        newline_count += token_text.count("\n")
        window_vectors.append(last_layer)
        if newline_count >= CHECKPOINT_NEWLINE_SPAN:
            stacked = torch.stack(window_vectors, dim=0)
            checkpoint_end_token_idx.append(token_offset)
            checkpoint_window_hidden_mean.append(tensor_to_list(stacked.mean(dim=0)))
            checkpoint_last_hidden.append(tensor_to_list(last_layer))
            newline_count = 0
            window_vectors = []
    return checkpoint_end_token_idx, checkpoint_window_hidden_mean, checkpoint_last_hidden


def extract_question(row: dict) -> str:
    value = str(row.get("question") or row.get("problem") or "")
    if os.environ.get("ARIS_QWEN3_USER_PROMPT_STYLE", "").strip().lower() == "human_eval_baseline":
        return value
    return value.strip()


def extract_answer(row: dict) -> str:
    return str(row.get("answer") or row.get("solution") or row.get("final_answer") or "").strip()


def initialize_end_think_state(input_ids: torch.Tensor, end_think_token_id: int) -> None:
    global GLOBAL_END_THINK_INDEX, GLOBAL_SUPPRESSION_ACTIVE
    GLOBAL_END_THINK_INDEX = -1
    GLOBAL_SUPPRESSION_ACTIVE = False
    matches = (input_ids[0] == end_think_token_id).nonzero(as_tuple=True)[0]
    if len(matches) > 0:
        # We only record the first occurrence. In the current prompts that is
        # the closed-think boundary inserted by the chat template / suffix.
        GLOBAL_END_THINK_INDEX = matches[0].item()
        GLOBAL_SUPPRESSION_ACTIVE = True
    _debug_log(
        "end_think",
        "initialize_end_think_state",
        input_len=int(input_ids.shape[1]),
        end_think_token_id=int(end_think_token_id),
        detected_end_think_index=int(GLOBAL_END_THINK_INDEX),
        suppression_active=bool(GLOBAL_SUPPRESSION_ACTIVE),
    )


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generated_ids: Sequence[int] | None = None,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    sampling_logits = logits.float()
    penalty = float(presence_penalty)
    if penalty > 0.0 and generated_ids:
        unique_ids = sorted({int(token_id) for token_id in generated_ids if int(token_id) >= 0})
        if unique_ids:
            token_index = torch.tensor(unique_ids, device=sampling_logits.device, dtype=torch.long)
            sampling_logits[:, token_index] -= penalty
    if temperature > 0:
        sampling_logits = sampling_logits / temperature
    sampling_logits = apply_top_p(sampling_logits, top_p)
    sampling_probs = torch.softmax(sampling_logits, dim=-1)
    sampling_probs = torch.nan_to_num(sampling_probs, nan=0.0, posinf=0.0, neginf=0.0)
    total_prob = sampling_probs.sum(dim=-1, keepdim=True)
    if torch.any(total_prob <= 0):
        return torch.argmax(logits, dim=-1, keepdim=True)
    sampling_probs = sampling_probs / total_prob
    if temperature <= 0:
        return torch.argmax(sampling_probs, dim=-1, keepdim=True)
    return torch.multinomial(sampling_probs, num_samples=1)


def hidden_output_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.hidden.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--level-id", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-custom-attention-patch", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    level = LEVEL_BY_ID[args.level_id]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model)
    end_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
    if not end_think_tokens:
        raise ValueError("Tokenizer could not encode </think>.")
    end_think_token_id = end_think_tokens[0]

    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "eager",
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        input_device = model.model.embed_tokens.weight.device
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(args.device)
        input_device = torch.device(args.device)
    if not args.disable_custom_attention_patch:
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(custom_qwen3_attention_forward, layer.self_attn)
    model.eval()

    global TARGET_SUPPRESS_HEADS_BY_LAYER
    TARGET_SUPPRESS_HEADS_BY_LAYER = load_target_heads(Path(args.summary_json), level.suppression_budget)

    output_path = Path(args.output)
    hidden_path = hidden_output_path(output_path)
    if output_path.exists():
        output_path.unlink()
    if hidden_path.exists():
        hidden_path.unlink()

    for row_idx, row in enumerate(rows):
        question = extract_question(row)
        if not question:
            continue
        sample_id = row.get("id", row.get("index", row_idx))
        answer = extract_answer(row)
        input_ids = build_prompt(tokenizer, question, level.assistant_suffix).to(input_device)
        attention_mask = torch.ones_like(input_ids, device=input_device)
        initialize_end_think_state(input_ids, end_think_token_id)
        generated_ids: list[int] = []
        entropy: list[float] = []
        margin: list[float] = []
        logprob: list[float] = []
        generated_last_hidden: list[torch.Tensor] = []
        past_key_values = None
        with torch.no_grad():
            current_ids = input_ids
            current_attention_mask = attention_mask
            for _ in range(args.max_new_tokens):
                model_input_ids = current_ids if past_key_values is None else current_ids[:, -1:]
                output = model(
                    input_ids=model_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_key_values = output.past_key_values
                next_token_logits = output.logits[:, -1, :]
                next_token = sample_next_token(
                    next_token_logits,
                    args.temperature,
                    top_p=0.95,
                    generated_ids=generated_ids,
                    presence_penalty=args.presence_penalty,
                )
                token_id = int(next_token.item())
                entropy_value, margin_value, logprob_value = compute_token_metric(next_token_logits[0], token_id)
                entropy.append(entropy_value)
                margin.append(margin_value)
                logprob.append(logprob_value)
                generated_last_hidden.append(output.hidden_states[-1][0, -1, :].detach().cpu().float())
                generated_ids.append(token_id)
                current_ids = torch.cat([current_ids, next_token.to(current_ids.device)], dim=1)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                    dim=1,
                )
                if token_id == tokenizer.eos_token_id:
                    break
        final_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        checkpoint_end_token_idx, checkpoint_window_hidden_mean, checkpoint_last_hidden = compute_checkpoints(
            tokenizer,
            generated_ids,
            generated_last_hidden,
        )
        hidden_ref = {
            "file": hidden_path.name,
            "id": sample_id,
            "level_id": level.level_id,
        }
        append_jsonl(
            output_path,
            {
                "id": sample_id,
                "question": question,
                "answer": answer,
                "level_id": level.level_id,
                "level_label": level.label,
                "final_text": final_text,
                "n_tokens": len(generated_ids),
                "entropy": entropy,
                "margin": margin,
                "logprob": logprob,
                "checkpoint_end_token_idx": checkpoint_end_token_idx,
                "hidden_state_ref": hidden_ref,
            },
        )
        append_jsonl(
            hidden_path,
            {
                "id": sample_id,
                "level_id": level.level_id,
                "checkpoint_window_hidden_mean": checkpoint_window_hidden_mean,
                "checkpoint_last_hidden": checkpoint_last_hidden,
            },
        )


if __name__ == "__main__":
    main()
