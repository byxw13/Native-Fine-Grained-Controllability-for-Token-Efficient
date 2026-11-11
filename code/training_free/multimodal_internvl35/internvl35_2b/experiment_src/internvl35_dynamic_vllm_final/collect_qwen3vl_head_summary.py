from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

from remote_scripts.mmmu_qwen3vl.scripts import run_mmmu_vlmeval_prompt_p0_vllm as baseline

from .controller import ControllerConfig, DynamicSuppressionController, TokenMetric

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None


PREFIX_BY_PROFILE = {
    "method": "<think>\n\n</think>\n\nAlright",
    "closed_think": "<think>\n\n</think>\n\n",
}
SCORE_DEFINITION = {
    "region": "{</think>, </think>+1}",
    "attention_score": "a_end + a_end+1",
    "contribution_score": "||W_o(a_end*v_end + a_end+1*v_end+1)||_2",
    "contribution_implementation": "FP32 sqrt(v^T (W_o^T W_o) v), mathematically equivalent with one cached Gram matrix per head",
    "scope": "Qwen3-VL language decoder only",
}
GLOBAL_END_THINK_INDEX = -1
GLOBAL_HEAD_RECORDS: dict[int, dict[str, Any]] = {}


class DetectEndThinkAndCaptureSignal(LogitsProcessor):
    """Arm head collection after prefill and capture unsuppressed signals."""

    def __init__(self, end_think_token_id: int, config: ControllerConfig | None, request_id: str) -> None:
        self.end_think_token_id = int(end_think_token_id)
        self.controller = None if config is None else DynamicSuppressionController(config)
        self.state = None if self.controller is None else self.controller.new_state(request_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global GLOBAL_END_THINK_INDEX
        if GLOBAL_END_THINK_INDEX == -1:
            matches = (input_ids[0] == self.end_think_token_id).nonzero(as_tuple=True)[0]
            if len(matches):
                GLOBAL_END_THINK_INDEX = int(matches[0].item())
        if self.controller is not None and self.state is not None:
            values = scores.float()
            candidate_tensor = torch.argmax(values[0])
            log_z = torch.logsumexp(values, dim=-1)
            probs = torch.softmax(values, dim=-1)
            finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
            entropy = log_z - (probs * finite_values).sum(dim=-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            packed = torch.stack(
                (
                    candidate_tensor.float(),
                    entropy[0],
                    top2[0, 0] - top2[0, 1],
                    values[0, candidate_tensor] - log_z[0],
                )
            ).detach().cpu().tolist()
            candidate = int(packed[0])
            metric = TokenMetric(
                entropy=float(packed[1]),
                margin=float(packed[2]),
                logprob=float(packed[3]),
            )
            if self.state.checkpoint_zero_pending:
                self.controller.apply_checkpoint_zero(self.state, metric, candidate)
            # Calibration observes the native greedy trajectory and never
            # applies queued forced tokens or suppression.
            self.controller.on_token(self.state, candidate, metric)
        return scores

    def checkpoint_events(self) -> list[dict[str, Any]]:
        if self.state is None:
            return []
        return [
            event
            for event in self.state.events
            if event.get("event") == "checkpoint"
            and event.get("checkpoint_type") in {"increase_check_checkpoint0", "increase_check"}
        ]


def _fsync_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def reset_records(end_think_index: int = -1) -> None:
    global GLOBAL_END_THINK_INDEX, GLOBAL_HEAD_RECORDS
    GLOBAL_END_THINK_INDEX = int(end_think_index)
    GLOBAL_HEAD_RECORDS = {}


def _init_record(layer_idx: int, num_heads: int, device: torch.device) -> dict[str, Any]:
    record = GLOBAL_HEAD_RECORDS.get(layer_idx)
    if record is None:
        record = {
            "attn_sum": torch.zeros(num_heads, dtype=torch.float32, device=device),
            "contrib_sum": torch.zeros(num_heads, dtype=torch.float32, device=device),
            "count": 0,
        }
        GLOBAL_HEAD_RECORDS[layer_idx] = record
    return record


def custom_qwen3vl_head_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    # Head capture is armed only after prefill.  Delegate the multimodal
    # prefill to FA2; decode still uses the exact eager rows needed for the
    # </think> and </think>+1 head statistics.
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
    weights = torch.matmul(query, key_attn.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key_attn.shape[-2]]
    weights = nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)

    end_idx = GLOBAL_END_THINK_INDEX
    if end_idx >= 0 and end_idx + 1 < weights.shape[-1]:
        num_heads = int(weights.shape[1])
        attn_end = weights[0, :, -1, end_idx].detach().float()
        attn_next = weights[0, :, -1, end_idx + 1].detach().float()
        score = attn_end + attn_next
        token_head = (
            attn_end.unsqueeze(-1) * value_attn[0, :, end_idx, :].detach().float()
            + attn_next.unsqueeze(-1) * value_attn[0, :, end_idx + 1, :].detach().float()
        )
        wo_gram = getattr(self, "_aris_wo_gram", None)
        if wo_gram is None or wo_gram.device != token_head.device:
            wo_heads = (
                self.o_proj.weight.detach()
                .float()
                .view(self.o_proj.out_features, num_heads, self.head_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            wo_gram = torch.bmm(wo_heads.transpose(1, 2), wo_heads)
            self._aris_wo_gram = wo_gram
        contribution_sq = torch.einsum("hd,hde,he->h", token_head, wo_gram, token_head)
        contribution = torch.sqrt(torch.clamp(contribution_sq, min=0.0))
        record = _init_record(int(self.layer_idx), num_heads, weights.device)
        record["attn_sum"] += score
        record["contrib_sum"] += contribution
        record["count"] += 1

    weights = nn.functional.dropout(
        weights,
        p=0.0 if not self.training else self.attention_dropout,
        training=self.training,
    )
    output = torch.matmul(weights, value_attn).transpose(1, 2).contiguous().reshape(*input_shape, -1)
    return self.o_proj(output), weights


def serialize_records() -> dict[str, Any]:
    result = {}
    for layer_idx, record in GLOBAL_HEAD_RECORDS.items():
        count = max(int(record["count"]), 1)
        attn = record["attn_sum"].detach().cpu().numpy()
        contrib = record["contrib_sum"].detach().cpu().numpy()
        result[str(layer_idx)] = {
            "record_steps": int(record["count"]),
            "attn_sum": attn.tolist(),
            "attn_mean": (attn / count).tolist(),
            "contrib_sum": contrib.tolist(),
            "contrib_mean": (contrib / count).tolist(),
        }
    return result


def prepare_inputs(
    processor,
    sample: dict[str, Any],
    device: torch.device,
    expected_prefix: str = PREFIX_BY_PROFILE["method"],
):
    images = baseline.load_images(sample.get("image_paths") or [])
    content, audit = baseline.build_interleaved_content(sample, images)
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt.count(expected_prefix) != 1 or not prompt.endswith(expected_prefix):
        raise RuntimeError("requested generation prefix is not present exactly once at the prompt tail")
    patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=patch_size)
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
    else:
        ordered = [images[index - 1] for index in audit["image_occurrence_order"]]
        inputs = processor(text=[prompt], images=ordered, return_tensors="pt")
    return inputs.to(device), prompt, audit


def metadata_hash(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        if path.exists():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty quantile input")
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = q * (len(ordered) - 1)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    value = ordered[lo] if lo == hi else ordered[lo] * (hi - position) + ordered[hi] * (position - lo)
    return round(value, 6)


def _signal_calibration(items: list[dict[str, Any]]) -> dict[str, Any]:
    events = [event for item in items for event in (item.get("signal_checkpoint_events") or [])]
    if not events:
        raise RuntimeError("head-summary run captured no signal checkpoints")
    entropy = [float(event["avg_entropy"]) for event in events]
    margin = [float(event["avg_margin"]) for event in events]
    delta = [float(event["positive_entropy_delta"]) for event in events]
    summarize = lambda values: {
        "count": len(values),
        "min": round(min(values), 6),
        "q10": _quantile(values, 0.10),
        "q50": _quantile(values, 0.50),
        "q90": _quantile(values, 0.90),
        "max": round(max(values), 6),
    }
    ent, mar, dlt = summarize(entropy), summarize(margin), summarize(delta)
    return {
        "method": "evaluation_v3_8_composite_signal_lambdaq_v1",
        "dataset": "MMMU validation first 100",
        "selected_sample_count": len(items),
        "checkpoint_count": len(events),
        "avg_entropy": ent,
        "avg_margin": mar,
        "positive_entropy_delta": dlt,
        "recommended_signal_calibration": {
            "entropy_lo": ent["q10"],
            "entropy_hi": ent["q90"],
            "delta_entropy_hi": dlt["q90"],
            "margin_lo": mar["q10"],
            "margin_hi": mar["q90"],
        },
    }


def _question_lambda_calibration(
    items: list[dict[str, Any]], signal_report: dict[str, Any], config: ControllerConfig
) -> dict[str, Any]:
    cal = signal_report["recommended_signal_calibration"]

    def norm(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if value > upper else 0.0
        return min(max((value - lower) / (upper - lower), 0.0), 1.0)

    d_values = []
    for item in items:
        checkpoints = item.get("signal_checkpoint_events") or []
        base = []
        for event in checkpoints[:2]:
            z_ent = norm(float(event["avg_entropy"]), cal["entropy_lo"], cal["entropy_hi"])
            z_delta = norm(float(event["positive_entropy_delta"]), 0.0, cal["delta_entropy_hi"])
            z_margin = 1.0 - norm(float(event["avg_margin"]), cal["margin_lo"], cal["margin_hi"])
            signal = 0.5 * z_ent + 0.3 * z_delta + 0.2 * z_margin
            threshold = config.cue_threshold
            normalized = 0.0 if signal < threshold else norm(signal, threshold, config.entropy_upper_bound)
            if config.suppression_curve == "power":
                mapped = normalized**config.power_exponent
            else:
                mapped = (1.0 - math.exp(-config.exp_beta * normalized)) / (1.0 - math.exp(-config.exp_beta))
            base.append(min(config.suppression_upper_bound, config.suppression_scale * mapped))
        if len(base) == 2:
            d_values.append(0.7 * (sum(base) / 2.0) + 0.3 * min(base))
    if not d_values:
        raise RuntimeError("no items with two signal checkpoints for question-lambda calibration")
    q20, q90 = _quantile(d_values, 0.20), _quantile(d_values, 0.90)
    lambda_min, lambda_max, beta = 0.8, 1.05, 2.0
    target = (1.0 - lambda_min) / (lambda_max - lambda_min)
    u_at_one = -math.log(1.0 - target * (1.0 - math.exp(-beta))) / beta
    d_hi = q20 + ((q90 - q20) / u_at_one)
    return {
        "method": "evaluation_v3_8_question_lambda_q90_recal_v1",
        "source_sample_count": len(items),
        "recommended_question_lambda_calibration": {
            "d_lo": q20,
            "d_hi": round(d_hi, 6),
            "source_q20": q20,
            "source_q90": q90,
            "u_at_lambda_one": round(u_at_one, 6),
            "eligible_sample_count": len(d_values),
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "lambda_exp_beta": beta,
            "d_mean_weight": 0.7,
            "d_min_weight": 0.3,
        },
    }


def merge(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    per_item = []
    failures = []
    valid_items = []
    seen = set()
    accum: dict[str, dict[str, Any]] = {}
    for sample in rows:
        path = output_dir / "per_item_head_stats" / f"{sample['global_index']:04d}_{sample['id']}.json"
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
        if item.get("error") or not item.get("head_scores_by_layer"):
            failures.append({"id": key, "error": item.get("error") or "empty_head_scores"})
            continue
        valid_items.append(item)
        for layer, record in item["head_scores_by_layer"].items():
            attn = np.asarray(record["attn_mean"], dtype=np.float64)
            contrib = np.asarray(record["contrib_mean"], dtype=np.float64)
            target = accum.setdefault(
                layer,
                {"attn": np.zeros_like(attn), "contrib": np.zeros_like(contrib), "count": 0},
            )
            target["attn"] += attn
            target["contrib"] += contrib
            target["count"] += 1
    summary = {
        layer: {
            "num_items": value["count"],
            "attn_mean_over_items": (value["attn"] / max(value["count"], 1)).tolist(),
            "contrib_mean_over_items": (value["contrib"] / max(value["count"], 1)).tolist(),
        }
        for layer, value in sorted(accum.items(), key=lambda item: int(item[0]))
    }
    _fsync_json(
        output_dir / "head_scores_summary.json",
        {
            "model": str(args.model),
            "num_total_items": len(rows),
            "num_valid_items_with_end_think": len(valid_items),
            "score_definition": SCORE_DEFINITION,
            "summary_by_layer": summary,
        },
    )
    _fsync_json(
        output_dir / "head_summary_merge_report.json",
        {
            "expected": len(rows),
            "found": len(per_item),
            "unique": len(seen),
            "failures": failures,
            "complete": len(per_item) == len(rows) and not failures,
        },
    )
    _fsync_json(
        output_dir / "head_summary_manifest.json",
        {
            "model": str(args.model),
            "model_metadata_sha256": metadata_hash(
                args.model, ("config.json", "generation_config.json", "model.safetensors.index.json")
            ),
            "tokenizer_overlay": str(args.tokenizer),
            "processor_metadata_sha256": metadata_hash(
                args.tokenizer,
                (
                    "tokenizer_config.json",
                    "processor_config.json",
                    "preprocessor_config.json",
                    "chat_template.json",
                ),
            ),
            "collector_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest": str(args.manifest),
            "sample_ids": [str(row.get("id")) for row in rows],
            "global_indices": [int(row.get("global_index")) for row in rows],
            "prompt_style": baseline.PROMPT_STYLE,
            "prefix_profile": args.prefix_profile,
            "generation_prefix": PREFIX_BY_PROFILE[args.prefix_profile],
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "vision_attn_implementation": args.vision_attn_implementation,
            "language_prefill_attn_implementation": args.vision_attn_implementation,
            "language_decode_attn_implementation": "custom_eager_head_collector",
            "items": [
                {
                    "id": item.get("id"),
                    "global_index": item.get("global_index"),
                    "input_tokens": item.get("input_tokens"),
                    "visual_tokens": item.get("visual_tokens"),
                    "output_tokens": item.get("output_tokens"),
                    "prompt": item.get("prompt"),
                    "prompt_sha256": hashlib.sha256(str(item.get("prompt") or "").encode("utf-8")).hexdigest(),
                    "image_audit": item.get("image_audit"),
                    "error": item.get("error"),
                }
                for item in per_item
            ],
            "failures": failures,
        },
    )
    if valid_items and args.calibration_config is not None:
        signal_report = _signal_calibration(valid_items)
        _fsync_json(output_dir / "signal_calibration.json", signal_report)
        calibration_config = ControllerConfig.from_json(args.calibration_config)
        _fsync_json(
            output_dir / "question_lambda_calibration.json",
            _question_lambda_calibration(valid_items, signal_report, calibration_config),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/path/to/aris_data/Qwen3-VL-4B-Thinking"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--indices", default="")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--vision-attn-implementation", default="flash_attention_2")
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument(
        "--prefix-profile",
        choices=tuple(PREFIX_BY_PROFILE),
        default="method",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("gloo")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    all_rows = list(baseline.read_jsonl(args.manifest))
    if args.indices:
        wanted = [int(value) for value in args.indices.split(",") if value.strip()]
        by_index = {int(row["global_index"]): row for row in all_rows}
        missing = [value for value in wanted if value not in by_index]
        if missing:
            raise RuntimeError(f"requested head-summary smoke indices are missing: {missing}")
        rows = [by_index[value] for value in wanted][: args.limit]
    else:
        rows = all_rows[: args.limit]
    processor = AutoProcessor.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.vision_attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    layers = model.model.language_model.layers
    for layer in layers:
        layer.self_attn._aris_original_forward = layer.self_attn.forward
        layer.self_attn.forward = types.MethodType(custom_qwen3vl_head_forward, layer.self_attn)

    end_ids = processor.tokenizer.encode("</think>", add_special_tokens=False)
    calibration_config = (
        None if args.calibration_config is None else ControllerConfig.from_json(args.calibration_config)
    )
    if calibration_config is not None and not calibration_config.calibration_only:
        raise RuntimeError("--calibration-config must set calibration_only=true")
    for position, sample in enumerate(rows):
        if position % world != rank:
            continue
        target = args.output_dir / "per_item_head_stats" / f"{int(sample['global_index']):04d}_{sample['id']}.json"
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if not existing.get("error") and existing.get("head_scores_by_layer"):
                continue
        try:
            inputs, prompt, audit = prepare_inputs(
                processor,
                sample,
                device,
                PREFIX_BY_PROFILE[args.prefix_profile],
            )
            matches = (inputs["input_ids"][0] == end_ids[0]).nonzero(as_tuple=True)[0]
            if len(matches) != 1:
                raise RuntimeError(f"expected one </think> token, found {len(matches)}")
            # The reference text collector discovers the already-present prefix
            # in a logits processor after prefill, so prefill itself is excluded
            # from the per-head accumulation.
            reset_records()
            capture_processor = DetectEndThinkAndCaptureSignal(
                end_ids[0], calibration_config, str(sample.get("id"))
            )
            trigger_processor = LogitsProcessorList([capture_processor])
            input_len = int(inputs["input_ids"].shape[-1])
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "eos_token_id": processor.tokenizer.eos_token_id,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "logits_processor": trigger_processor,
            }
            if args.temperature > 0:
                generate_kwargs["temperature"] = args.temperature
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
            generated = output[0, input_len:].tolist()
            item = {
                "id": sample.get("id"),
                "global_index": sample.get("global_index"),
                "subject": sample.get("subject"),
                "visual_tokens": int((inputs["input_ids"] == getattr(model.config, "image_token_id", -1)).sum()),
                "input_tokens": input_len,
                "output_tokens": len(generated),
                "prompt": prompt,
                "image_audit": audit,
                "end_think_index": GLOBAL_END_THINK_INDEX,
                "head_scores_by_layer": serialize_records(),
                "signal_checkpoint_events": capture_processor.checkpoint_events(),
                "generated_text": processor.tokenizer.decode(generated, skip_special_tokens=False),
                "rank": rank,
                "vision_attn_implementation": args.vision_attn_implementation,
                "language_decode_attn_implementation": "custom_eager_head_collector",
                "error": None,
            }
        except Exception as exc:
            item = {
                "id": sample.get("id"),
                "global_index": sample.get("global_index"),
                "rank": rank,
                "vision_attn_implementation": args.vision_attn_implementation,
                "language_decode_attn_implementation": "custom_eager_head_collector",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _fsync_json(target, item)
        print(json.dumps({"saved": str(target), "error": item.get("error")}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    if world > 1:
        dist.barrier()
    if rank == 0:
        merge(args.output_dir, rows, args)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
