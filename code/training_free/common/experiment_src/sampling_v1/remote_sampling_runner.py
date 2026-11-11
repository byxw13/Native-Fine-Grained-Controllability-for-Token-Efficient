from __future__ import annotations

import argparse
import os
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from experiment_src.sampling_v1.config import CHECKPOINT_NEWLINE_SPAN, DEFAULT_MAX_NEW_TOKENS, LEVEL_BY_ID
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
BOUNDARY_POSITION_MODE = "full_boundary"
GLOBAL_SUPPRESSION_MASK_CALLS = 0
GLOBAL_SUPPRESSION_POSITION_HITS = {"end": 0, "next": 0}
GLOBAL_MAX_SELECTED_HEAD_COUNT = 0
# MASK_VALUE=0.0 means a hard multiplicative suppression. The selected
# attention mass is zeroed out first, then renormalized over the remaining
# positions. If this is changed to e.g. 0.2, the code becomes a soft scaling
# instead of full suppression.
MASK_VALUE = 0.0

# DeepSeek/Qwen chat_template with add_generation_prompt=True already appends
# "<｜Assistant｜><think>\n\n</think>\n\n". When our level suffix also starts
# with the same closed-think scaffold, we must only append the residual cue
# (for example, "Alright") instead of duplicating the scaffold.
AUTO_TEMPLATE_THINK_SCAFFOLD = "<｜Assistant｜><think>\n\n</think>\n\n"
CANONICAL_CLOSED_THINK_SUFFIX = "<think>\n\n</think>\n\n"


def custom_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # This function is a copy of the Qwen2 attention forward with one extra
    # intervention point after softmax(attn_weights): selected heads can have
    # their attention to specific source positions suppressed.
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

    global GLOBAL_END_THINK_INDEX, GLOBAL_SUPPRESSION_ACTIVE, TARGET_SUPPRESS_HEADS_BY_LAYER
    global GLOBAL_SUPPRESSION_MASK_CALLS, GLOBAL_SUPPRESSION_POSITION_HITS, GLOBAL_MAX_SELECTED_HEAD_COUNT
    target_heads = TARGET_SUPPRESS_HEADS_BY_LAYER.get(int(self.layer_idx), [])
    if target_heads and GLOBAL_END_THINK_INDEX != -1 and GLOBAL_SUPPRESSION_ACTIVE:
        seq_len = attn_weights.shape[-1]
        # The current implementation suppresses attention to the first
        # "</think>" token and the token immediately after it. The latter is
        # included because the closed-think scaffold usually occupies two
        # positions in the chat template / prompt stream we care about.
        if BOUNDARY_POSITION_MODE == "end_think_only":
            candidate_indices = (GLOBAL_END_THINK_INDEX,)
        elif BOUNDARY_POSITION_MODE == "post_end_think_only":
            candidate_indices = (GLOBAL_END_THINK_INDEX + 1,)
        else:
            candidate_indices = (GLOBAL_END_THINK_INDEX, GLOBAL_END_THINK_INDEX + 1)
        suppress_indices = [idx for idx in candidate_indices if 0 <= idx < seq_len]
        if suppress_indices:
            GLOBAL_SUPPRESSION_MASK_CALLS += 1
            GLOBAL_MAX_SELECTED_HEAD_COUNT = max(GLOBAL_MAX_SELECTED_HEAD_COUNT, len(target_heads))
            if GLOBAL_END_THINK_INDEX in suppress_indices:
                GLOBAL_SUPPRESSION_POSITION_HITS["end"] += 1
            if GLOBAL_END_THINK_INDEX + 1 in suppress_indices:
                GLOBAL_SUPPRESSION_POSITION_HITS["next"] += 1
            head_idx_tensor = torch.tensor(target_heads, device=attn_weights.device, dtype=torch.long)
            selected_attn = torch.index_select(attn_weights, dim=1, index=head_idx_tensor)
            for idx in suppress_indices:
                # selected_attn shape: [batch, selected_heads, query_len, key_len]
                # We scale the attention weight assigned to the target source
                # position. With MASK_VALUE=0.0 this fully deletes that source
                # position from the selected heads.
                selected_attn[:, :, :, idx] *= MASK_VALUE
            # After modifying the distribution we must renormalize, otherwise
            # the attention rows would no longer sum to 1.
            selected_attn = selected_attn / (selected_attn.sum(dim=-1, keepdim=True) + 1e-6)
            attn_weights.index_copy_(1, head_idx_tensor, selected_attn)

    attn_output = torch.matmul(attn_weights, value_states_for_attn.float()).to(query_states.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def set_boundary_position_mode(mode: str) -> None:
    global BOUNDARY_POSITION_MODE
    normalized = str(mode or "full_boundary").strip().lower()
    allowed = {"full_boundary", "end_think_only", "post_end_think_only"}
    if normalized not in allowed:
        raise ValueError(f"unsupported boundary position mode: {mode!r}")
    BOUNDARY_POSITION_MODE = normalized


def reset_intervention_counters() -> None:
    global GLOBAL_SUPPRESSION_MASK_CALLS, GLOBAL_SUPPRESSION_POSITION_HITS, GLOBAL_MAX_SELECTED_HEAD_COUNT
    GLOBAL_SUPPRESSION_MASK_CALLS = 0
    GLOBAL_SUPPRESSION_POSITION_HITS = {"end": 0, "next": 0}
    GLOBAL_MAX_SELECTED_HEAD_COUNT = 0


def intervention_counters() -> dict[str, object]:
    return {
        "boundary_position_mode": BOUNDARY_POSITION_MODE,
        "suppression_mask_calls": int(GLOBAL_SUPPRESSION_MASK_CALLS),
        "suppression_position_hits": dict(GLOBAL_SUPPRESSION_POSITION_HITS),
        "max_selected_head_count": int(GLOBAL_MAX_SELECTED_HEAD_COUNT),
    }


def load_target_heads(summary_json: Path, budget: Optional[float]) -> Dict[int, List[int]]:
    import json

    if budget is None:
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
        return {}
    flat.sort(key=lambda item: item[2], reverse=True)
    selected: Dict[int, List[int]] = {}
    cumulative = 0.0
    for layer_idx, head_idx, score in flat:
        selected.setdefault(layer_idx, []).append(head_idx)
        cumulative += score / total
        if cumulative >= budget:
            break
    return {layer_idx: sorted(heads) for layer_idx, heads in selected.items()}


def _resolve_residual_assistant_suffix(base_prompt_text: str, assistant_suffix: str) -> str:
    if not assistant_suffix:
        return ""
    if (
        base_prompt_text.endswith(AUTO_TEMPLATE_THINK_SCAFFOLD)
        and assistant_suffix.startswith(CANONICAL_CLOSED_THINK_SUFFIX)
    ):
        return assistant_suffix[len(CANONICAL_CLOSED_THINK_SUFFIX) :]
    return assistant_suffix


def build_prompt(tokenizer, question: str, assistant_suffix: str) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    base_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    residual_suffix = _resolve_residual_assistant_suffix(base_text, assistant_suffix)
    prompt_text = base_text + residual_suffix
    return tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids


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
    return str(row.get("question") or row.get("problem") or "").strip()


def extract_answer(row: dict) -> str:
    return str(row.get("answer") or row.get("solution") or row.get("final_answer") or "").strip()


def initialize_end_think_state(input_ids: torch.Tensor, end_think_token_id: int) -> None:
    global GLOBAL_END_THINK_INDEX, GLOBAL_SUPPRESSION_ACTIVE
    GLOBAL_END_THINK_INDEX = -1
    GLOBAL_SUPPRESSION_ACTIVE = False
    reset_intervention_counters()
    matches = (input_ids[0] == end_think_token_id).nonzero(as_tuple=True)[0]
    if len(matches) > 0:
        # We only record the first occurrence. In the current prompts that is
        # the closed-think boundary inserted by the chat template / suffix.
        GLOBAL_END_THINK_INDEX = matches[0].item()
        GLOBAL_SUPPRESSION_ACTIVE = True


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


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    sampling_logits = logits.float()
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
    parser.add_argument("--device", default="cuda:0")
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
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(custom_qwen2_attention_forward, layer.self_attn)
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
                next_token = sample_next_token(next_token_logits, args.temperature, top_p=0.95)
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
