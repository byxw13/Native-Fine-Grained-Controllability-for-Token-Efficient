#!/usr/bin/env python3
"""Collect a real InternVL3.5-8B language-head summary.

The collector mirrors the audited Qwen3-VL collector, but uses InternVL's
remote-code wrapper and the Qwen3 language decoder inside it.  It records the
attention and output-projection contribution of the two prompt positions
``{</think>, </think>+1}`` during one-token decode.  Vision layers are never
hooked.  Each item is fsynced before the next item starts, so a pre-empted
Slurm job can resume safely.

Important prompt invariant: the summary uses an ordinary user prompt plus the
closed-think scaffold ``<think>\n\n</think>\n\n`` and **no cue word**.  It does
not add the baseline R1 system prompt; doing both would create duplicate
``</think>`` markers and invalidate the head attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from experiment_src.internvl35_vllm_baseline import mmmu_thinking_vllm as intern_baseline
from .internvl_mmmu_dynamic_vllm import CLOSED_THINK_PREFIX
from .internvl_transformers_oracle import _dynamic_preprocess, _transform


SCORE_DEFINITION = {
    "region": "{</think>, </think>+1}",
    "attention_score": "a_end + a_end+1",
    "contribution_score": "||W_o(a_end*v_end + a_end+1*v_end+1)||_2",
    "contribution_implementation": "FP32 sqrt(v^T (W_o^T W_o) v), one cached Gram matrix per query head",
    "scope": "InternVL3.5-8B Qwen3 language decoder only; vision encoder is excluded",
}

GLOBAL_END_INDEX = -1
GLOBAL_HEAD_RECORDS: dict[int, dict[str, Any]] = {}


def prepare_head_inputs(tokenizer, model, sample: dict[str, Any], device: torch.device):
    """Build the prior Qwen3.5-style closed-think prompt without a cue."""
    request, audit = intern_baseline.build_request(tokenizer, sample, "off")
    if audit.get("system_prompt") is not None:
        raise RuntimeError("head summary prompt unexpectedly contains an R1 system prompt")
    prompt = str(request["prompt"])
    if not prompt.endswith(intern_baseline.ASSISTANT_GENERATION_PREFIX):
        raise RuntimeError("InternVL assistant generation suffix changed")
    prompt += CLOSED_THINK_PREFIX
    if prompt.count(CLOSED_THINK_PREFIX) != 1 or prompt.count("</think>") != 1:
        raise RuntimeError("head summary prompt must contain exactly one closed-think scaffold")

    # ``mmmu_thinking_vllm`` wraps the shared Qwen3-VL baseline as
    # ``mmmu_thinking_vllm.baseline``; it does not re-export ``load_images``.
    images = intern_baseline.baseline.load_images(sample.get("image_paths") or [])
    order = [int(index) - 1 for index in audit["image_occurrence_order"]]
    ordered = [images[index] for index in order]
    expanded = prompt
    patches: list[torch.Tensor] = []
    patch_counts: list[int] = []
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
    return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}, expanded, {
        **audit,
        "system_prompt": None,
        "thinking_mode": "ordinary_prompt_plus_closed_think_scaffold_no_cue",
        "prefix": CLOSED_THINK_PREFIX,
        "image_count": len(images),
        "image_occurrence_count": len(order),
        "image_order": "full_rendered_prompt_placeholder_order",
        "patch_counts": patch_counts,
        "transformer_input_tokens": int(input_ids.shape[-1]),
    }


def _fsync_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _reset_records(end_index: int) -> None:
    global GLOBAL_END_INDEX, GLOBAL_HEAD_RECORDS
    GLOBAL_END_INDEX = int(end_index)
    GLOBAL_HEAD_RECORDS = {}


def _record(layer_idx: int, heads: int, device: torch.device) -> dict[str, Any]:
    value = GLOBAL_HEAD_RECORDS.get(int(layer_idx))
    if value is None:
        value = {
            "attn_sum": torch.zeros(heads, dtype=torch.float32, device=device),
            "contrib_sum": torch.zeros(heads, dtype=torch.float32, device=device),
            "count": 0,
        }
        GLOBAL_HEAD_RECORDS[int(layer_idx)] = value
    return value


def _internvl_head_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    # Prefill is intentionally delegated to the native multimodal path.  The
    # eager row below is only used for one-token decode, where the attribution
    # region is unambiguous and the sequence length is small enough to inspect.
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
            key,
            value,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    key_attn = repeat_kv(key, self.num_key_value_groups)
    value_attn = repeat_kv(value, self.num_key_value_groups)
    weights = torch.matmul(query.float(), key_attn.float().transpose(2, 3)) * float(self.scaling)
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key_attn.shape[-2]]
    weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32)

    end_index = int(GLOBAL_END_INDEX)
    if end_index >= 0 and end_index + 1 < weights.shape[-1]:
        attn_end = weights[0, :, -1, end_index].detach()
        attn_next = weights[0, :, -1, end_index + 1].detach()
        score = attn_end + attn_next
        token_head = (
            attn_end.unsqueeze(-1) * value_attn[0, :, end_index, :].float().detach()
            + attn_next.unsqueeze(-1) * value_attn[0, :, end_index + 1, :].float().detach()
        )
        gram = getattr(self, "_aris_head_summary_wo_gram", None)
        num_heads = int(weights.shape[1])
        if gram is None or gram.device != token_head.device:
            wo_heads = (
                self.o_proj.weight.detach()
                .float()
                .view(self.o_proj.out_features, num_heads, self.head_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            gram = torch.bmm(wo_heads.transpose(1, 2), wo_heads)
            self._aris_head_summary_wo_gram = gram
        contribution_sq = torch.einsum("hd,hde,he->h", token_head, gram, token_head)
        contribution = torch.sqrt(torch.clamp(contribution_sq, min=0.0))
        rec = _record(int(self.layer_idx), num_heads, weights.device)
        rec["attn_sum"] += score.float()
        rec["contrib_sum"] += contribution.float()
        rec["count"] += 1

    output = torch.matmul(weights.to(value_attn.dtype), value_attn)
    output = output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(output), None


class SeedProcessor(LogitsProcessor):
    """Make sampling deterministic per MMMU item without altering logits."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.called = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # HF's generator is global; setting it once here occurs after prefill
        # and before the first sampled decode token.
        if not self.called:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            self.called = True
        return scores


def _serialize_records() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_idx, rec in sorted(GLOBAL_HEAD_RECORDS.items()):
        count = int(rec["count"])
        if count <= 0:
            continue
        attn = rec["attn_sum"].detach().cpu().numpy()
        contrib = rec["contrib_sum"].detach().cpu().numpy()
        result[str(layer_idx)] = {
            "record_steps": count,
            "attn_sum": attn.tolist(),
            "attn_mean": (attn / count).tolist(),
            "contrib_sum": contrib.tolist(),
            "contrib_mean": (contrib / count).tolist(),
        }
    return result


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("id") is not None and not row.get("error"):
            ids.add(str(row["id"]))
    return ids


def _merge(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_item: list[dict[str, Any]] = []
    accum: dict[str, dict[str, Any]] = {}
    for sample in rows:
        path = output_dir / "per_item_head_stats" / f"{int(sample['global_index']):04d}_{sample['id']}.json"
        if not path.exists():
            failures.append({"id": sample.get("id"), "global_index": sample.get("global_index"), "error": "missing"})
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        key = str(item.get("id"))
        if key in seen:
            failures.append({"id": key, "error": "duplicate"})
            continue
        seen.add(key)
        per_item.append(item)
        layer_stats = item.get("head_scores_by_layer") or {}
        if item.get("error") or len(layer_stats) != args.expected_layers:
            failures.append({"id": key, "global_index": item.get("global_index"), "error": item.get("error") or "layer_coverage"})
            continue
        for layer, rec in layer_stats.items():
            attn = np.asarray(rec["attn_mean"], dtype=np.float64)
            contrib = np.asarray(rec["contrib_mean"], dtype=np.float64)
            if attn.shape != (args.expected_heads,) or contrib.shape != (args.expected_heads,):
                failures.append({"id": key, "layer": layer, "error": "head_count"})
                continue
            target = accum.setdefault(layer, {"attn": np.zeros_like(attn), "contrib": np.zeros_like(contrib), "count": 0})
            target["attn"] += attn
            target["contrib"] += contrib
            target["count"] += 1
    summary_by_layer = {
        layer: {
            "num_items": int(value["count"]),
            "attn_mean_over_items": (value["attn"] / max(value["count"], 1)).tolist(),
            "contrib_mean_over_items": (value["contrib"] / max(value["count"], 1)).tolist(),
        }
        for layer, value in sorted(accum.items(), key=lambda item: int(item[0]))
    }
    _fsync_json(output_dir / "head_scores_summary.json", {
        "model": str(args.model),
        "model_language_architecture": "InternVL3.5-8B/InternVLChatModel/Qwen3ForCausalLM",
        "num_total_items": len(rows),
        "num_valid_items_with_end_think": sum(1 for item in per_item if not item.get("error") and len(item.get("head_scores_by_layer") or {}) == args.expected_layers),
        "expected_layers": args.expected_layers,
        "expected_query_heads": args.expected_heads,
        "score_definition": SCORE_DEFINITION,
        "summary_by_layer": summary_by_layer,
    })
    _fsync_json(output_dir / "head_summary_merge_report.json", {
        "expected": len(rows), "found": len(per_item), "unique": len(seen),
        "failures": failures, "complete": len(per_item) == len(rows) and not failures,
    })
    _fsync_json(output_dir / "head_summary_manifest.json", {
        "model": str(args.model),
        "manifest": str(args.manifest),
        "global_indices": [int(row.get("global_index")) for row in rows],
        "sample_ids": [str(row.get("id")) for row in rows],
        # The InternVL baseline snapshot exposes this label in the per-item
        # audit rather than as a module constant.  Keep a stable literal here
        # so validation does not depend on an incidental baseline import.
        "prompt_style": "vlmevalkit_mmmu_concise_mc_answer_suffix_multisample",
        "system_prompt": None,
        "thinking_mode": "ordinary_prompt_plus_closed_think_scaffold_no_cue",
        "prefix": CLOSED_THINK_PREFIX,
        "prefix_count": 1,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": "model_default",
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "language_layers": args.expected_layers,
        "query_heads_per_layer": args.expected_heads,
        "vision_layers_hooked": 0,
        "language_prefill_attn_implementation": args.attn_implementation,
        "language_decode_attn_implementation": "custom_eager_head_summary",
        "items": [
            {
                "id": item.get("id"), "global_index": item.get("global_index"),
                "input_tokens": item.get("input_tokens"), "output_tokens": item.get("output_tokens"),
                "end_think_index": item.get("end_think_index"), "prompt": item.get("prompt"),
                "error": item.get("error"),
            }
            for item in per_item
        ],
        "failures": failures,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--expected-layers", type=int, default=36)
    parser.add_argument("--expected-heads", type=int, default=32)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    try:
        layers = model.language_model.model.layers
    except AttributeError as exc:
        raise RuntimeError("InternVL language decoder path model.language_model.model.layers not found") from exc
    if len(layers) != args.expected_layers:
        raise RuntimeError(f"expected {args.expected_layers} language layers, found {len(layers)}")
    for layer in layers:
        layer.self_attn._aris_original_forward = layer.self_attn.forward
        layer.self_attn.forward = types.MethodType(_internvl_head_forward, layer.self_attn)

    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(end_ids) != 1:
        raise RuntimeError(f"expected </think> to be one token, got {end_ids}")
    all_rows = list(intern_baseline.baseline.read_jsonl(args.manifest))[: args.limit]
    rows = [row for position, row in enumerate(all_rows) if position % world == rank]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    done = _completed(output_dir / "completed.jsonl") if args.resume else set()
    remaining = [row for row in rows if str(row.get("id")) not in done]
    completed_path = output_dir / "completed.jsonl"
    if not args.resume and completed_path.exists():
        completed_path.unlink()

    for sample in remaining:
        target = output_dir / "per_item_head_stats" / f"{int(sample['global_index']):04d}_{sample['id']}.json"
        try:
            inputs, prompt, audit = prepare_head_inputs(tokenizer, model, sample, device)
            if prompt.count(CLOSED_THINK_PREFIX) != 1 or not prompt.endswith(CLOSED_THINK_PREFIX):
                raise RuntimeError("ordinary prompt does not end in exactly one closed-think scaffold")
            if prompt.count("</think>") != 1:
                raise RuntimeError("head summary prompt must contain exactly one </think>")
            matches = (inputs["input_ids"][0] == int(end_ids[0])).nonzero(as_tuple=True)[0]
            if len(matches) != 1:
                raise RuntimeError(f"expected one </think> token, found {len(matches)}")
            _reset_records(int(matches[0].item()))
            input_len = int(inputs["input_ids"].shape[-1])
            processor = LogitsProcessorList([SeedProcessor(args.seed + int(sample.get("global_index", 0)))])
            torch.manual_seed(args.seed + int(sample.get("global_index", 0)))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + int(sample.get("global_index", 0)))
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    logits_processor=processor,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            out_ids = generated[0].tolist()
            if len(out_ids) >= input_len and out_ids[:input_len] == inputs["input_ids"][0].tolist():
                out_ids = out_ids[input_len:]
            scores = _serialize_records()
            if len(scores) != args.expected_layers:
                raise RuntimeError(f"head summary layer coverage {len(scores)}/{args.expected_layers}")
            item = {
                "id": sample.get("id"), "global_index": sample.get("global_index"),
                "subject": sample.get("subject"), "input_tokens": input_len,
                "output_tokens": len(out_ids), "prompt": prompt,
                "image_audit": audit, "end_think_index": GLOBAL_END_INDEX,
                "head_scores_by_layer": scores,
                "generated_text": tokenizer.decode(out_ids, skip_special_tokens=False),
                "prefix": CLOSED_THINK_PREFIX, "system_prompt": None,
                "thinking_mode": "ordinary_prompt_plus_closed_think_scaffold_no_cue",
                "error": None,
            }
        except Exception as exc:
            item = {
                "id": sample.get("id"), "global_index": sample.get("global_index"),
                "prefix": CLOSED_THINK_PREFIX, "system_prompt": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _fsync_json(target, item)
        with completed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": sample.get("id"), "global_index": sample.get("global_index"), "error": item.get("error")}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"saved": str(target), "global_index": sample.get("global_index"), "tokens": item.get("output_tokens"), "error": item.get("error")}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()
    if rank == 0:
        _merge(output_dir, all_rows, args)
        print(json.dumps({"summary": str(output_dir / 'head_scores_summary.json'), "merge": str(output_dir / 'head_summary_merge_report.json')}, ensure_ascii=False), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
