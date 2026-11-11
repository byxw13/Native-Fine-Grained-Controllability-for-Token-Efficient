#!/usr/bin/env python3
"""Independent Transformers oracle for InternVL3.5-2B Full-Attention ARIS."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from experiment_src.internvl35_vllm_baseline import mmmu_thinking_vllm as intern_baseline
from .controller import ControllerConfig, DynamicSuppressionController, TokenMetric
from .internvl_mmmu_dynamic_vllm import CLOSED_THINK_PREFIX, build_method_request
from .vendor.aris_vllm_v1_suppression import load_target_heads

baseline = intern_baseline.baseline


ACTIVE_END_INDEX = -1
ACTIVE_HEADS_BY_LAYER: dict[int, list[int]] = {}


class TopKLogprobCapture(LogitsProcessor):
    """Capture a compact top-k distribution after ARIS logits processing.

    This deliberately records only top-k log probabilities, not the full
    vocabulary tensor.  It is used to compare the numerical decision surface
    of the HF eager oracle with vLLM under the same fixed prompt/history.
    """

    def __init__(self, topk: int):
        self.topk = max(0, int(topk))
        self.rows: list[dict[str, Any]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.topk:
            values, ids = torch.topk(scores[0].float(), k=min(self.topk, scores.shape[-1]))
            logprob = torch.log_softmax(scores[0].float(), dim=-1)
            self.rows.append({
                str(int(token_id)): {
                    "logprob": float(logprob[int(token_id)].item()),
                    "rank": int(rank),
                    "decoded_token": None,
                }
                for rank, (token_id, _value) in enumerate(zip(ids.tolist(), values.tolist()), start=1)
            })
        return scores


def _attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    # The vLLM correction is applied during one-token decode against its KV
    # cache.  Prefill is intentionally left on the original HF path.
    if hidden_states.shape[-2] != 1:
        return self._aris_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key, value, self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    weights = torch.matmul(query.float(), key.float().transpose(2, 3)) * float(self.scaling)
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32)
    heads = ACTIVE_HEADS_BY_LAYER.get(int(self.layer_idx), [])
    end = int(ACTIVE_END_INDEX)
    if heads and end >= 0 and end + 1 < weights.shape[-1]:
        index = torch.tensor(heads, dtype=torch.long, device=weights.device)
        selected = weights.index_select(1, index)
        selected[..., end] = 0.0
        selected[..., end + 1] = 0.0
        selected = selected / (selected.sum(dim=-1, keepdim=True) + 1e-6)
        weights.index_copy_(1, index, selected)
    output = torch.matmul(weights.to(value.dtype), value).transpose(1, 2).contiguous()
    output = output.reshape(*input_shape, -1)
    return self.o_proj(output), None


def _metric(scores: torch.Tensor, token_id: int) -> TokenMetric:
    values = scores.float()
    log_z = torch.logsumexp(values, dim=-1)
    probs = torch.softmax(values, dim=-1)
    entropy = log_z - (probs * values).sum(dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    logprob = values[:, token_id] - log_z
    return TokenMetric(float(entropy.item()), float(margin.item()), float(logprob.item()))


class DynamicProcessor(LogitsProcessor):
    def __init__(self, controller: DynamicSuppressionController, state, summary: Path):
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
        reason = None
        if forced is not None:
            candidate, reason = int(forced.token_id), forced.reason
        token_metric = _metric(scores, candidate)
        self.controller.on_token(self.state, candidate, token_metric, forced_reason=reason)
        ACTIVE_HEADS_BY_LAYER = load_target_heads(self.summary, self.state.current_effective_suppression)
        self.selected_ids.append(candidate)
        if candidate != int(torch.argmax(scores[0]).item()):
            scores.fill_(-torch.inf)
            scores[0, candidate] = 0.0
        return scores


def _target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]:
    values = {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(values, key=lambda item: item[0] * item[1])


def _dynamic_preprocess(image: Image.Image, image_size: int = 448, max_dynamic_patch: int = 12) -> list[Image.Image]:
    ratios = _target_ratios(1, max_dynamic_patch)
    aspect = image.width / image.height
    best = (1, 1)
    best_diff = float("inf")
    area = image.width * image.height
    for ratio in ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]):
            best, best_diff = ratio, diff
    width, height = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((width, height))
    patches = []
    for index in range(blocks):
        x0 = (index % (width // image_size)) * image_size
        y0 = (index // (width // image_size)) * image_size
        patches.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if len(patches) != 1:
        patches.append(image.resize((image_size, image_size)))
    return patches


def _transform(image: Image.Image) -> torch.Tensor:
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image.convert("RGB"))


def prepare_inputs(tokenizer, model, sample: dict[str, Any], device: torch.device, cue: str):
    request, audit = build_method_request(tokenizer, sample, cue)
    images = baseline.baseline.load_images(sample.get("image_paths") or [])
    order = [int(index) - 1 for index in audit["image_occurrence_order"]]
    ordered = [images[index] for index in order]
    patches: list[torch.Tensor] = []
    patch_counts: list[int] = []
    expanded = str(request["prompt"])
    for image in ordered:
        image_patches = _dynamic_preprocess(image)
        patch_counts.append(len(image_patches))
        patches.extend(_transform(patch) for patch in image_patches)
        replacement = "<img>" + "<IMG_CONTEXT>" * (256 * len(image_patches)) + "</img>"
        expanded = expanded.replace("<image>", replacement, 1)
    encoded = tokenizer(expanded, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = torch.stack(patches).to(device=device, dtype=torch.bfloat16) if patches else None
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}, expanded, audit | {
        "patch_counts": patch_counts,
        "transformer_input_tokens": int(input_ids.shape[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices", default="0,30,55,73,203,740")
    parser.add_argument("--method-cue", default="The")
    parser.add_argument(
        "--mode",
        choices=("dynamic", "native"),
        default="dynamic",
        help="dynamic applies the ARIS attention correction; native is an unmodified HF control.",
    )
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--topk-logprobs", type=int, default=0,
                        help="Capture compact top-k logprobs for numerical comparisons.")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    if args.mode == "dynamic":
        for layer in model.language_model.model.layers:
            layer.self_attn._aris_original_forward = layer.self_attn.forward
            layer.self_attn.forward = types.MethodType(_attention_forward, layer.self_attn)

    rows_by_index = {int(row["global_index"]): row for row in baseline.read_jsonl(args.manifest)}
    wanted = [int(item) for item in args.indices.split(",") if item.strip()]
    rows = [rows_by_index[index] for index in wanted]
    config = ControllerConfig.from_json(args.config)
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(end_ids) != 1:
        raise RuntimeError(f"expected </think> to be one token, got {end_ids}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for max_tokens in args.max_tokens:
        target_dir = args.output_dir / f"t{max_tokens}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for sample in rows:
            target = target_dir / f"{int(sample['global_index']):04d}_{sample['id']}.json"
            try:
                inputs, prompt, audit = prepare_inputs(tokenizer, model, sample, device, args.method_cue)
                matches = (inputs["input_ids"][0] == int(end_ids[0])).nonzero(as_tuple=True)[0]
                if len(matches) != 1:
                    raise RuntimeError(f"expected one </think>, found {len(matches)}")
                global ACTIVE_END_INDEX, ACTIVE_HEADS_BY_LAYER
                ACTIVE_END_INDEX = int(matches[0].item())
                controller = DynamicSuppressionController(config) if args.mode == "dynamic" else None
                state = controller.new_state(str(sample["id"])) if controller is not None else None
                processor = DynamicProcessor(controller, state, args.summary) if controller is not None else None
                capture = TopKLogprobCapture(args.topk_logprobs)
                if controller is not None:
                    # vLLM receives initial K in the request metadata and
                    # applies it on the first real decode forward. Install
                    # the identical initial head set before HF generate().
                    ACTIVE_HEADS_BY_LAYER = load_target_heads(
                        args.summary, state.current_effective_suppression
                    )
                else:
                    ACTIVE_HEADS_BY_LAYER = {}
                processors = [item for item in (processor, capture) if item is not None]
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=int(max_tokens),
                        do_sample=False,
                        repetition_penalty=args.repetition_penalty,
                        logits_processor=(LogitsProcessorList(processors) if processors else None),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                output_ids = generated[0].tolist()
                input_len = int(inputs["input_ids"].shape[-1])
                if len(output_ids) > input_len and output_ids[:input_len] == inputs["input_ids"][0].tolist():
                    output_ids = output_ids[input_len:]
                item = {
                    "id": sample.get("id"),
                    "global_index": sample.get("global_index"),
                    "max_tokens": int(max_tokens),
                    "output_token_ids": [int(x) for x in output_ids],
                    "text": tokenizer.decode(output_ids, skip_special_tokens=False),
                    "events": [] if state is None else state.events,
                    "selected_ids": [] if processor is None else processor.selected_ids,
                    "topk_logprobs": capture.rows,
                    "input_tokens": input_len,
                    "prompt": prompt,
                    "audit": audit,
                    "engine": f"transformers_{args.mode}_oracle",
                    "error": None,
                }
            except Exception as exc:
                item = {"id": sample.get("id"), "global_index": sample.get("global_index"), "max_tokens": int(max_tokens), "engine": f"transformers_{args.mode}_oracle", "error": f"{type(exc).__name__}: {exc}"}
            target.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"saved": str(target), "tokens": len(item.get("output_token_ids", [])), "error": item.get("error")}), flush=True)


if __name__ == "__main__":
    main()
