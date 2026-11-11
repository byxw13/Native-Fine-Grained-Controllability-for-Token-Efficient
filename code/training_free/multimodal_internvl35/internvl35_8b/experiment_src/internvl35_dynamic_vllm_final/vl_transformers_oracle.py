from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from remote_scripts.mmmu_qwen3vl.scripts import run_mmmu_vlmeval_prompt_p0_vllm as baseline

from .collect_qwen3vl_head_summary import prepare_inputs
from .controller import ControllerConfig, DynamicSuppressionController, TokenMetric
from .vendor.aris_vllm_v1_suppression import load_target_heads


ACTIVE_END_THINK_INDEX = -1
ACTIVE_HEADS_BY_LAYER: dict[int, list[int]] = {}


def _fsync_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def suppression_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    # Suppression starts on decode.  Keep the K=0 multimodal prefill on the
    # model's native FA2 path so long visual prompts do not materialize an
    # O(sequence^2) language-attention matrix.
    if hidden_states.shape[-2] != 1:
        return self._aris_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
    del kwargs
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key, value, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    heads = ACTIVE_HEADS_BY_LAYER.get(int(self.layer_idx), [])
    end = int(ACTIVE_END_THINK_INDEX)
    if heads and end >= 0 and end + 1 < weights.shape[-1]:
        index = torch.tensor(heads, dtype=torch.long, device=weights.device)
        selected = weights.index_select(1, index)
        selected[..., end] = 0
        selected[..., end + 1] = 0
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-6)
        weights.index_copy_(1, index, selected)
    weights = nn.functional.dropout(
        weights, p=0.0 if not self.training else self.attention_dropout, training=self.training
    )
    output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(output), weights


def _metric(scores: torch.Tensor, token_id: int) -> TokenMetric:
    values = scores.float()
    log_z = torch.logsumexp(values, dim=-1)
    probs = torch.softmax(values, dim=-1)
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    entropy = log_z - (probs * finite_values).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    logprob = values[:, token_id] - log_z
    return TokenMetric(float(entropy.item()), float(margin.item()), float(logprob.item()))


class DynamicOracleProcessor(LogitsProcessor):
    def __init__(self, controller: DynamicSuppressionController, state, summary: Path) -> None:
        self.controller = controller
        self.state = state
        self.summary = summary
        self.selected_ids: list[int] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global ACTIVE_HEADS_BY_LAYER
        forbidden = self.controller.token_ids_to_mask(self.state)
        if forbidden:
            scores[0, forbidden] = -torch.inf
        candidate = int(torch.argmax(scores[0]).item())
        if self.state.checkpoint_zero_pending:
            self.controller.apply_checkpoint_zero(self.state, _metric(scores, candidate), candidate)
        forced = self.controller.pop_forced(self.state)
        forced_reason = None
        if forced is None and self.controller.intercept_end_think(self.state, candidate):
            self.controller.on_end_think_intercept(self.state)
            forced = self.controller.pop_forced(self.state)
            if forced is None:
                scratch = scores[0].clone()
                scratch[self.controller.config.end_think_token_ids] = -torch.inf
                candidate = int(torch.argmax(scratch).item())
                forced_reason = "end_think_intercept_resample"
            else:
                candidate, forced_reason = int(forced.token_id), forced.reason
        elif forced is not None:
            candidate, forced_reason = int(forced.token_id), forced.reason
        token_metric = _metric(scores, candidate)
        self.controller.on_token(self.state, candidate, token_metric, forced_reason=forced_reason)
        ACTIVE_HEADS_BY_LAYER = load_target_heads(self.summary, self.state.current_effective_suppression)
        self.selected_ids.append(candidate)
        if candidate != int(torch.argmax(scores[0]).item()):
            scores.fill_(-torch.inf)
            scores[0, candidate] = 0
        return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indices", default="0,30,55,73,203,740")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config-pattern", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    wanted = [int(item) for item in args.indices.split(",") if item.strip()]
    rows = [row for row in baseline.read_jsonl(args.manifest) if int(row["global_index"]) in set(wanted)]
    rows.sort(key=lambda row: wanted.index(int(row["global_index"])))
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    # The language attention modules below are replaced by the exact eager
    # dynamic-suppression oracle.  Loading the container model with FA2 keeps
    # the independent vision encoder memory-linear for high-resolution MMMU
    # images; it does not change the patched language-attention semantics.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.vision_attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    for layer in model.model.language_model.layers:
        layer.self_attn._aris_original_forward = layer.self_attn.forward
        layer.self_attn.forward = types.MethodType(suppression_attention_forward, layer.self_attn)
    end_ids = processor.tokenizer.encode("</think>", add_special_tokens=False)
    if len(end_ids) != 1:
        raise RuntimeError(f"expected single </think> token, got {end_ids}")

    for max_tokens in args.max_tokens:
        config = ControllerConfig.from_json(args.config_pattern.format(max_tokens=max_tokens))
        for position, sample in enumerate(rows):
            if position % world != rank:
                continue
            target = args.output_dir / f"t{max_tokens}" / f"{int(sample['global_index']):04d}_{sample['id']}.json"
            try:
                inputs, prompt, audit = prepare_inputs(processor, sample, device)
                matches = (inputs["input_ids"][0] == end_ids[0]).nonzero(as_tuple=True)[0]
                if len(matches) != 1:
                    raise RuntimeError(f"expected one </think> token, found {len(matches)}")
                global ACTIVE_END_THINK_INDEX, ACTIVE_HEADS_BY_LAYER
                ACTIVE_END_THINK_INDEX = int(matches[0].item())
                ACTIVE_HEADS_BY_LAYER = {}
                controller = DynamicSuppressionController(config)
                state = controller.new_state(str(sample["id"]))
                logits_processor = DynamicOracleProcessor(controller, state, args.summary)
                input_len = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=int(max_tokens),
                        do_sample=False,
                        top_p=1.0,
                        repetition_penalty=args.repetition_penalty,
                        logits_processor=LogitsProcessorList([logits_processor]),
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                generated = [int(item) for item in output[0, input_len:].tolist()]
                item = {
                    "id": sample["id"],
                    "global_index": sample["global_index"],
                    "max_tokens": int(max_tokens),
                    "prompt": prompt,
                    "image_audit": audit,
                    "output_token_ids": generated,
                    "processor_selected_ids": logits_processor.selected_ids,
                    "text": processor.tokenizer.decode(generated, skip_special_tokens=False),
                    "events": state.events,
                    "final_k": state.current_effective_suppression,
                    "peak_k": state.peak_effective_suppression,
                    "rank": rank,
                    "vision_attn_implementation": args.vision_attn_implementation,
                    "language_attn_implementation": "custom_eager_dynamic_suppression",
                    "error": None,
                }
            except Exception as exc:
                item = {
                    "id": sample["id"],
                    "global_index": sample["global_index"],
                    "max_tokens": int(max_tokens),
                    "rank": rank,
                    "vision_attn_implementation": args.vision_attn_implementation,
                    "language_attn_implementation": "custom_eager_dynamic_suppression",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            _fsync_json(target, item)
            print(json.dumps({"saved": str(target), "error": item.get("error")}), flush=True)
            torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
