"""ARIS suppression patch for vLLM 0.10.2 V1 FlashAttention.

This module is intentionally opt-in. Importing it does not modify vLLM; call
``install_aris_vllm_v1_suppression`` before constructing an ``LLM``.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class RequestSuppressionMetadata:
    end_think_index: int
    enabled: bool = True
    budget: Optional[float] = None


_INSTALLED = False
_PENDING_REQUEST_METADATA: list[RequestSuppressionMetadata] = []
_REQUEST_METADATA_BY_ID: dict[str, RequestSuppressionMetadata] = {}
_ACTIVE_HEADS_BY_LAYER: Dict[int, List[int]] = {}
_SUMMARY_JSON_PATH: Optional[str] = None
_HEADS_BY_BUDGET: dict[float, Dict[int, List[int]]] = {}
_MASK_VALUE = 0.0
_AUTO_SUPPRESSION_ENABLED = False
_END_THINK_TOKEN_IDS: list[int] = []
_RECOMPUTE_ALL_HEADS = False
_RECOMPUTE_ALL_LAYERS = False
_DISABLE_SUP1_FAST_PATH = False
_NATIVE_FUSED_DECODE = os.getenv("ARIS_VLLM_V1_NATIVE_FUSED", "0").lower() in {"1", "true", "yes", "on"}
_SELECTED_FROM_FULL_HEADS = False
_FULL_LAYER_RECOMPUTE_SUBSET = False
_HF_STYLE_RECOMPUTE = False
_ACTIVE_BUDGET: Optional[float] = None
_DEBUG = os.getenv("ARIS_VLLM_V1_DEBUG", "0").lower() in {"1", "true", "yes"}
_STATS_PATH = os.getenv("ARIS_VLLM_V1_STATS_PATH")
_FALLBACK_DUMP_PATH = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_PATH")
_FALLBACK_DUMP_LAYER = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_LAYER")
_FALLBACK_DUMP_Q_LEN = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_Q_LEN")
_FALLBACK_DUMP_GENERATED = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_GENERATED")
_FALLBACK_DUMP_ONCE_DONE = False
_RUNTIME_COORD_DUMP_PATH = os.getenv("ARIS_VLLM_V1_RUNTIME_COORD_DUMP_PATH")
_RUNTIME_COORD_DUMP_ONLY_Q1 = os.getenv("ARIS_VLLM_V1_RUNTIME_COORD_DUMP_ONLY_Q1", "0").strip().lower() in {"1", "true", "yes", "on"}
_RUNTIME_COORD_DUMP_ONCE_DONE = False
_RUNTIME_TRACE_PATH = os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_PATH")
_RUNTIME_TRACE_ONLY_Q1 = os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_ONLY_Q1", "0").strip().lower() in {"1", "true", "yes", "on"}
_RUNTIME_TRACE_MAX_EVENTS = int(os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_MAX_EVENTS", "0") or "0")
_RUNTIME_TRACE_LAYER_FILTER = {
    int(item.strip())
    for item in os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_LAYER_FILTER", "").split(",")
    if item.strip()
}
_RUNTIME_TRACE_EVENT_COUNT = 0
_NATIVE_FULL_COMPARE_PATH = os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_PATH")
_NATIVE_FULL_COMPARE_ONLY_Q1 = os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_ONLY_Q1", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_NATIVE_FULL_COMPARE_MAX_EVENTS = int(os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_MAX_EVENTS", "0") or "0")
_NATIVE_FULL_COMPARE_LAYER_FILTER = {
    int(item.strip())
    for item in os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_LAYER_FILTER", "").split(",")
    if item.strip()
}
_NATIVE_FULL_COMPARE_EVENT_COUNT = 0
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)")
_REQUEST_BUDGET_RE = re.compile(r"(?:^|::)aris_budget=([^:]+)")
_REQUEST_END_RE = re.compile(r"(?:^|::)aris_end=(-?\d+)")
_REQUEST_ENABLED_RE = re.compile(r"(?:^|::)aris_enabled=([^:]+)")
_LAST_STATS_DUMP_TS = 0.0
_BUDGET_ID_BY_LABEL: dict[str, int] = {}
_BUDGET_HEADS_BY_ID: dict[int, Dict[int, List[int]]] = {}
_HEAD_MASK_TABLE_CACHE: dict[tuple, tuple[torch.Tensor, int, bool]] = {}

_STATS = {
    "prepare_inputs_calls": 0,
    "flash_calls": 0,
    "suppressed_flash_calls": 0,
    "selected_sequence_recomputes": 0,
    "selected_from_full_recomputes": 0,
    "full_layer_subset_recomputes": 0,
    "full_sequence_recomputes": 0,
    "all_head_fast_path_calls": 0,
    "all_head_fast_path_recomputes": 0,
    "all_head_varlen_fast_path_calls": 0,
    "all_head_varlen_fast_path_recomputes": 0,
    "native_fused_decode_calls": 0,
    "native_fused_decode_tokens": 0,
    "native_fused_decode_fallback_calls": 0,
    "native_fused_decode_fallback_reasons": {},
    "skipped_cascade_calls": 0,
    "skipped_fp8_calls": 0,
    "layers": {},
    "prepare_metadata_sources": {},
    "prepare_budget_counts": {},
    "prepare_head_counts": {},
    "prepare_enabled_budget_counts": {},
}


def get_debug_stats() -> dict:
    return json.loads(json.dumps(_STATS))


def _dump_debug_stats() -> None:
    if not _DEBUG:
        return
    text = json.dumps(_STATS, sort_keys=True)
    print(f"ARIS_VLLM_V1_STATS {text}", flush=True)
    if _STATS_PATH:
        try:
            path = Path(_STATS_PATH)
            stats_to_write = _STATS
            if path.exists():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    stats_to_write = _merge_stats_max(previous, _STATS)
                except Exception:
                    stats_to_write = _STATS
            path.write_text(json.dumps(stats_to_write, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"ARIS_VLLM_V1_STATS_WRITE_FAILED {exc}", flush=True)


def _merge_stats_max(previous, current):
    if isinstance(previous, dict) and isinstance(current, dict):
        keys = set(previous) | set(current)
        return {key: _merge_stats_max(previous.get(key), current.get(key)) for key in keys}
    if isinstance(previous, int) and isinstance(current, int):
        return max(previous, current)
    if previous is None:
        return current
    if current is None:
        return previous
    return current


def _maybe_dump_debug_stats(force: bool = False) -> None:
    global _LAST_STATS_DUMP_TS
    if not _DEBUG or not _STATS_PATH:
        return
    now = time.time()
    if force or now - _LAST_STATS_DUMP_TS >= 0.5:
        _LAST_STATS_DUMP_TS = now
        _dump_debug_stats()


def _maybe_dump_fallback_case(
    *,
    layer_idx: int,
    req_idx: int,
    q_start: int,
    q_end: int,
    query: torch.Tensor,
    key_seq: torch.Tensor,
    value_seq: torch.Tensor,
    end_pos: int,
    target_heads: Sequence[int],
    seq_len: int,
    attn_metadata=None,
) -> None:
    global _FALLBACK_DUMP_ONCE_DONE
    if _FALLBACK_DUMP_ONCE_DONE or not _FALLBACK_DUMP_PATH:
        return
    if _FALLBACK_DUMP_LAYER is not None and str(layer_idx) != str(_FALLBACK_DUMP_LAYER):
        return
    q_len = int(q_end) - int(q_start)
    if _FALLBACK_DUMP_Q_LEN not in (None, "") and q_len != int(_FALLBACK_DUMP_Q_LEN):
        return
    prompt_len = None
    generated_so_far = None
    if attn_metadata is not None:
        try:
            prompt_lens = list(getattr(attn_metadata, "aris_prompt_lens_cpu", []) or [])
            seq_lens = _tensor_or_cpu_list(getattr(attn_metadata, "seq_lens"))
            if int(req_idx) < len(prompt_lens) and int(req_idx) < len(seq_lens):
                prompt_len = int(prompt_lens[int(req_idx)])
                generated_so_far = int(seq_lens[int(req_idx)]) - prompt_len
        except Exception:
            prompt_len = None
            generated_so_far = None
    if _FALLBACK_DUMP_GENERATED not in (None, ""):
        if generated_so_far is None or generated_so_far != int(_FALLBACK_DUMP_GENERATED):
            return
    try:
        payload = {
            "layer_idx": int(layer_idx),
            "req_idx": int(req_idx),
            "q_start": int(q_start),
            "q_end": int(q_end),
            "seq_len": int(seq_len),
            "q_len": int(q_len),
            "prompt_len": prompt_len,
            "generated_so_far_vs_prompt": generated_so_far,
            "end_pos": int(end_pos),
            "target_heads": [int(x) for x in target_heads],
            "query_shape": list(query[q_start:q_end].shape),
            "key_shape": list(key_seq.shape),
            "value_shape": list(value_seq.shape),
            "query": query[q_start:q_end].detach().cpu(),
            "key": key_seq.detach().cpu(),
            "value": value_seq.detach().cpu(),
        }
        path = Path(_FALLBACK_DUMP_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        _FALLBACK_DUMP_ONCE_DONE = True
        if _DEBUG:
            print(
                "ARIS_VLLM_V1_FALLBACK_DUMP "
                + json.dumps(
                    {
                        "path": str(path),
                        "layer_idx": int(layer_idx),
                        "req_idx": int(req_idx),
                        "q_start": int(q_start),
                        "q_end": int(q_end),
                        "target_heads": [int(x) for x in target_heads],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as exc:
        if _DEBUG:
            print(f"ARIS_VLLM_V1_FALLBACK_DUMP_FAILED {exc}", flush=True)


def _maybe_dump_runtime_coord(*, attn_metadata, path_name: str, layer_idx: int, plan=None) -> None:
    global _RUNTIME_COORD_DUMP_ONCE_DONE
    if _RUNTIME_COORD_DUMP_ONCE_DONE or not _RUNTIME_COORD_DUMP_PATH:
        return
    try:
        max_query_len = int(getattr(attn_metadata, "max_query_len", 0))
        if _RUNTIME_COORD_DUMP_ONLY_Q1:
            plan_q_len = None
            if plan is not None:
                _, q_start, q_end, _ = plan
                plan_q_len = int(q_end - q_start)
            if max_query_len != 1 and plan_q_len != 1:
                return
        req_ids = list(getattr(attn_metadata, "aris_req_ids", []) or [])
        prompt_lens = list(getattr(attn_metadata, "aris_prompt_lens_cpu", []) or [])
        prompt_prefix8 = list(getattr(attn_metadata, "aris_prompt_prefix8_cpu", []) or [])
        found_end_idx = list(getattr(attn_metadata, "aris_found_end_idx_cpu", []) or [])
        meta_end_positions = _tensor_or_cpu_list(getattr(attn_metadata, "aris_end_positions"))
        enabled = _tensor_or_cpu_list(getattr(attn_metadata, "aris_enabled"))
        budget_ids = _tensor_or_cpu_list(getattr(attn_metadata, "aris_budget_ids"))
        query_start_loc = _tensor_or_cpu_list(getattr(attn_metadata, "query_start_loc"))
        seq_lens = _tensor_or_cpu_list(getattr(attn_metadata, "seq_lens"))
        slot_mapping = _tensor_or_cpu_list(getattr(attn_metadata, "slot_mapping"))
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", 0))
        payload = {
            "path_name": str(path_name),
            "layer_idx": int(layer_idx),
            "max_query_len": max_query_len,
            "max_seq_len": int(getattr(attn_metadata, "max_seq_len", 0)),
            "num_actual_tokens": num_actual_tokens,
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "slot_mapping_head32": slot_mapping[:32],
            "req_ids": [str(x) for x in req_ids],
            "prompt_lens": [int(x) for x in prompt_lens],
            "prompt_prefix8": [[int(y) for y in row] for row in prompt_prefix8],
            "found_end_idx_from_req_state": [int(x) for x in found_end_idx],
            "meta_end_positions": [int(x) for x in meta_end_positions],
            "enabled": [int(x) for x in enabled],
            "budget_ids": [int(x) for x in budget_ids],
        }
        if plan is not None:
            req_idx, q_start, q_end, target_heads = plan
            used_end_pos = int(meta_end_positions[req_idx]) if req_idx < len(meta_end_positions) else None
            payload["plan"] = {
                "req_idx": int(req_idx),
                "q_start": int(q_start),
                "q_end": int(q_end),
                "q_len": int(q_end - q_start),
                "target_heads_count": int(len(target_heads)),
                "target_heads_first16": [int(x) for x in list(target_heads)[:16]],
                "used_end_pos": used_end_pos,
                "used_target_key_positions": [int(pos) for pos in (used_end_pos, used_end_pos + 1) if used_end_pos is not None],
                "prompt_len_this_req": int(prompt_lens[req_idx]) if req_idx < len(prompt_lens) else None,
                "found_end_idx_this_req": int(found_end_idx[req_idx]) if req_idx < len(found_end_idx) else None,
                "prompt_prefix8_this_req": [int(x) for x in prompt_prefix8[req_idx]] if req_idx < len(prompt_prefix8) else None,
            }
        path = Path(_RUNTIME_COORD_DUMP_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _RUNTIME_COORD_DUMP_ONCE_DONE = True
        if _DEBUG:
            print(
                "ARIS_VLLM_V1_RUNTIME_COORD_DUMP "
                + json.dumps(
                    {
                        "path": str(path),
                        "path_name": str(path_name),
                        "layer_idx": int(layer_idx),
                        "max_query_len": int(getattr(attn_metadata, "max_query_len", 0)),
                        "num_actual_tokens": int(getattr(attn_metadata, "num_actual_tokens", 0)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as exc:
        if _DEBUG:
            print(f"ARIS_VLLM_V1_RUNTIME_COORD_DUMP_FAILED {exc}", flush=True)


def _maybe_append_runtime_trace(*, attn_metadata, path_name: str, layer_idx: int, plan=None) -> None:
    global _RUNTIME_TRACE_EVENT_COUNT
    if not _RUNTIME_TRACE_PATH:
        return
    if _RUNTIME_TRACE_MAX_EVENTS > 0 and _RUNTIME_TRACE_EVENT_COUNT >= _RUNTIME_TRACE_MAX_EVENTS:
        return
    if _RUNTIME_TRACE_LAYER_FILTER and int(layer_idx) not in _RUNTIME_TRACE_LAYER_FILTER:
        return
    try:
        max_query_len = int(getattr(attn_metadata, "max_query_len", 0))
        plan_q_len = None
        if plan is not None:
            _, q_start, q_end, _ = plan
            plan_q_len = int(q_end - q_start)
        if _RUNTIME_TRACE_ONLY_Q1 and max_query_len != 1 and plan_q_len != 1:
            return

        req_ids = list(getattr(attn_metadata, "aris_req_ids", []) or [])
        prompt_lens = list(getattr(attn_metadata, "aris_prompt_lens_cpu", []) or [])
        prompt_prefix8 = list(getattr(attn_metadata, "aris_prompt_prefix8_cpu", []) or [])
        found_end_idx = list(getattr(attn_metadata, "aris_found_end_idx_cpu", []) or [])
        meta_end_positions = _tensor_or_cpu_list(getattr(attn_metadata, "aris_end_positions"))
        enabled = _tensor_or_cpu_list(getattr(attn_metadata, "aris_enabled"))
        budget_ids = _tensor_or_cpu_list(getattr(attn_metadata, "aris_budget_ids"))
        query_start_loc = _tensor_or_cpu_list(getattr(attn_metadata, "query_start_loc"))
        seq_lens = _tensor_or_cpu_list(getattr(attn_metadata, "seq_lens"))
        slot_mapping = _tensor_or_cpu_list(getattr(attn_metadata, "slot_mapping"))
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", 0))

        payload = {
            "event_idx": int(_RUNTIME_TRACE_EVENT_COUNT),
            "path_name": str(path_name),
            "layer_idx": int(layer_idx),
            "max_query_len": max_query_len,
            "max_seq_len": int(getattr(attn_metadata, "max_seq_len", 0)),
            "num_actual_tokens": num_actual_tokens,
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "slot_mapping_head16": slot_mapping[:16],
            "req_ids": [str(x) for x in req_ids],
            "prompt_lens": [int(x) for x in prompt_lens],
            "prompt_prefix8": [[int(y) for y in row] for row in prompt_prefix8],
            "found_end_idx_from_req_state": [int(x) for x in found_end_idx],
            "meta_end_positions": [int(x) for x in meta_end_positions],
            "enabled": [int(x) for x in enabled],
            "budget_ids": [int(x) for x in budget_ids],
        }
        if plan is not None:
            req_idx, q_start, q_end, target_heads = plan
            used_end_pos = int(meta_end_positions[req_idx]) if req_idx < len(meta_end_positions) else None
            prompt_len_this_req = int(prompt_lens[req_idx]) if req_idx < len(prompt_lens) else None
            seq_len_this_req = int(seq_lens[req_idx]) if req_idx < len(seq_lens) else None
            payload["plan"] = {
                "req_idx": int(req_idx),
                "q_start": int(q_start),
                "q_end": int(q_end),
                "q_len": int(q_end - q_start),
                "target_heads_count": int(len(target_heads)),
                "target_heads_first16": [int(x) for x in list(target_heads)[:16]],
                "used_end_pos": used_end_pos,
                "used_target_key_positions": [int(pos) for pos in (used_end_pos, used_end_pos + 1) if used_end_pos is not None],
                "prompt_len_this_req": prompt_len_this_req,
                "seq_len_this_req": seq_len_this_req,
                "generated_so_far_vs_prompt": None
                if prompt_len_this_req is None or seq_len_this_req is None
                else int(seq_len_this_req - prompt_len_this_req),
                "found_end_idx_this_req": int(found_end_idx[req_idx]) if req_idx < len(found_end_idx) else None,
                "prompt_prefix8_this_req": [int(x) for x in prompt_prefix8[req_idx]] if req_idx < len(prompt_prefix8) else None,
            }

        path = Path(_RUNTIME_TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _RUNTIME_TRACE_EVENT_COUNT += 1
        if _DEBUG and (_RUNTIME_TRACE_EVENT_COUNT <= 8 or _RUNTIME_TRACE_EVENT_COUNT % 64 == 0):
            print(
                "ARIS_VLLM_V1_RUNTIME_TRACE "
                + json.dumps(
                    {
                        "event_idx": int(payload["event_idx"]),
                        "path_name": str(path_name),
                        "layer_idx": int(layer_idx),
                        "max_query_len": max_query_len,
                        "plan": payload.get("plan"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    except Exception as exc:
        if _DEBUG:
            print(f"ARIS_VLLM_V1_RUNTIME_TRACE_FAILED {exc}", flush=True)


def _maybe_append_native_full_compare(
    *,
    attn_metadata,
    layer_idx: int,
    plan,
    native_heads: torch.Tensor,
    full_heads: torch.Tensor,
    stage: str = "pre_merge",
) -> None:
    global _NATIVE_FULL_COMPARE_EVENT_COUNT
    if not _NATIVE_FULL_COMPARE_PATH:
        return
    if _NATIVE_FULL_COMPARE_MAX_EVENTS > 0 and _NATIVE_FULL_COMPARE_EVENT_COUNT >= _NATIVE_FULL_COMPARE_MAX_EVENTS:
        return
    if _NATIVE_FULL_COMPARE_LAYER_FILTER and int(layer_idx) not in _NATIVE_FULL_COMPARE_LAYER_FILTER:
        return
    try:
        req_idx, q_start, q_end, target_heads = plan
        q_len = int(q_end - q_start)
        max_query_len = int(getattr(attn_metadata, "max_query_len", 0))
        if _NATIVE_FULL_COMPARE_ONLY_Q1 and max_query_len != 1 and q_len != 1:
            return

        native_slice = native_heads[q_start:q_end].detach().float()
        full_slice = full_heads.detach().float()
        delta = (native_slice - full_slice).abs()
        per_head_max = delta.amax(dim=(0, 2)).detach().cpu().tolist()
        per_head_mean = delta.mean(dim=(0, 2)).detach().cpu().tolist()
        target_set = {int(x) for x in target_heads}
        selected = [idx for idx in range(len(per_head_max)) if idx in target_set]
        unselected = [idx for idx in range(len(per_head_max)) if idx not in target_set]

        def _summary(indices: list[int]) -> dict:
            if not indices:
                return {"count": 0, "max": None, "mean_of_means": None}
            return {
                "count": len(indices),
                "max": max(float(per_head_max[idx]) for idx in indices),
                "mean_of_means": sum(float(per_head_mean[idx]) for idx in indices) / len(indices),
            }

        seq_lens = _tensor_or_cpu_list(getattr(attn_metadata, "seq_lens"))
        prompt_lens = list(getattr(attn_metadata, "aris_prompt_lens_cpu", []) or [])
        prompt_len = int(prompt_lens[req_idx]) if req_idx < len(prompt_lens) else None
        seq_len = int(seq_lens[req_idx]) if req_idx < len(seq_lens) else None
        top_heads = sorted(
            (
                {
                    "head": int(idx),
                    "max_abs": float(per_head_max[idx]),
                    "mean_abs": float(per_head_mean[idx]),
                    "selected": bool(idx in target_set),
                }
                for idx in range(len(per_head_max))
            ),
            key=lambda item: item["max_abs"],
            reverse=True,
        )[:8]
        payload = {
            "event_idx": int(_NATIVE_FULL_COMPARE_EVENT_COUNT),
            "stage": str(stage),
            "layer_idx": int(layer_idx),
            "max_query_len": max_query_len,
            "q_start": int(q_start),
            "q_end": int(q_end),
            "q_len": q_len,
            "seq_len": seq_len,
            "prompt_len": prompt_len,
            "generated_so_far_vs_prompt": None
            if prompt_len is None or seq_len is None
            else int(seq_len - prompt_len),
            "target_heads": [int(x) for x in target_heads],
            "selected_summary": _summary(selected),
            "unselected_summary": _summary(unselected),
            "all_summary": _summary(list(range(len(per_head_max)))),
            "top_heads_by_max_abs": top_heads,
        }
        path = Path(_NATIVE_FULL_COMPARE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _NATIVE_FULL_COMPARE_EVENT_COUNT += 1
    except Exception as exc:
        if _DEBUG:
            print(f"ARIS_VLLM_V1_NATIVE_FULL_COMPARE_FAILED {exc}", flush=True)


atexit.register(_dump_debug_stats)


def load_target_heads(summary_json: Path | str, budget: Optional[float]) -> Dict[int, List[int]]:
    if budget is None:
        return {}
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    flat: list[tuple[int, int, float]] = []
    for layer_str, rec in summary["summary_by_layer"].items():
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
        if cumulative >= float(budget):
            break
    return {layer_idx: sorted(heads) for layer_idx, heads in selected.items()}


def _budget_key(budget: float) -> float:
    return round(float(budget), 6)


def _heads_for_budget(budget: Optional[float]) -> Dict[int, List[int]]:
    if budget is None or not _SUMMARY_JSON_PATH:
        return {}
    key = _budget_key(float(budget))
    if key not in _HEADS_BY_BUDGET:
        _HEADS_BY_BUDGET[key] = load_target_heads(_SUMMARY_JSON_PATH, float(budget))
    return _HEADS_BY_BUDGET[key]


def set_active_heads(heads_by_layer: Dict[int, Sequence[int]], mask_value: float = 0.0) -> None:
    global _ACTIVE_HEADS_BY_LAYER, _MASK_VALUE
    _ACTIVE_HEADS_BY_LAYER = {
        int(layer): sorted({int(head) for head in heads})
        for layer, heads in heads_by_layer.items()
        if heads
    }
    _MASK_VALUE = float(mask_value)


def clear_active_heads() -> None:
    set_active_heads({})


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _parse_request_budget(req_id: str) -> tuple[bool, Optional[float]]:
    match = _REQUEST_BUDGET_RE.search(str(req_id))
    if match is None:
        return False, None
    raw_value = match.group(1).strip().lower()
    if raw_value in {"", "off", "none", "false", "nan"}:
        return True, None
    return True, float(raw_value)


def _parse_request_end_index(req_id: str) -> Optional[int]:
    match = _REQUEST_END_RE.search(str(req_id))
    if match is None:
        return None
    return int(match.group(1))


def _parse_request_enabled(req_id: str) -> tuple[bool, bool]:
    match = _REQUEST_ENABLED_RE.search(str(req_id))
    if match is None:
        return False, True
    return True, _truthy(match.group(1))


def _format_request_id_tag(meta: RequestSuppressionMetadata) -> str:
    parts = [
        f"aris_end={int(meta.end_think_index)}",
        f"aris_enabled={1 if meta.enabled else 0}",
    ]
    if meta.budget is not None:
        parts.insert(0, f"aris_budget={float(meta.budget):.6g}")
    return "::" + "::".join(parts)


def _record_prepare_request(
    *,
    source: str,
    budget: Optional[float],
    heads_by_layer: Dict[int, List[int]],
    enabled: int,
) -> None:
    if not _DEBUG:
        return
    sources = _STATS.setdefault("prepare_metadata_sources", {})
    sources[source] = int(sources.get(source, 0)) + 1
    budget_key = "none" if budget is None else f"{float(budget):.6g}"
    budgets = _STATS.setdefault("prepare_budget_counts", {})
    budgets[budget_key] = int(budgets.get(budget_key, 0)) + 1
    head_count = sum(len(v) for v in heads_by_layer.values())
    head_key = str(int(head_count))
    head_counts = _STATS.setdefault("prepare_head_counts", {})
    head_counts[head_key] = int(head_counts.get(head_key, 0)) + 1
    if enabled:
        enabled_counts = _STATS.setdefault("prepare_enabled_budget_counts", {})
        enabled_counts[budget_key] = int(enabled_counts.get(budget_key, 0)) + 1


def _record_native_fused_fallback(reason: str) -> None:
    if not _DEBUG:
        return
    _STATS["native_fused_decode_fallback_calls"] += 1
    reasons = _STATS.setdefault("native_fused_decode_fallback_reasons", {})
    reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1


def _register_budget_heads(
    budget: Optional[float],
    heads_by_layer: Dict[int, List[int]],
) -> int:
    """Return a stable small budget id for CUDA-side head-mask lookup."""
    if budget is None:
        label = "active" if heads_by_layer else "none"
    else:
        label = f"budget:{float(budget):.6g}"
    if label not in _BUDGET_ID_BY_LABEL:
        _BUDGET_ID_BY_LABEL[label] = len(_BUDGET_ID_BY_LABEL)
    budget_id = int(_BUDGET_ID_BY_LABEL[label])
    _BUDGET_HEADS_BY_ID[budget_id] = {
        int(layer): sorted({int(head) for head in heads})
        for layer, heads in heads_by_layer.items()
        if heads
    }
    return budget_id


def _device_int32_buffer(owner, attr: str, values: Sequence[int], device: torch.device) -> torch.Tensor:
    """Reuse metadata buffers so CUDA graph captures see stable tensor pointers."""
    num_values = len(values)
    buf = getattr(owner, attr, None)
    if (
        not isinstance(buf, torch.Tensor)
        or buf.device != device
        or buf.dtype != torch.int32
        or tuple(buf.shape) != (num_values,)
    ):
        buf = torch.empty((num_values,), dtype=torch.int32, device=device)
        setattr(owner, attr, buf)
    cpu_values = torch.tensor([int(v) for v in values], dtype=torch.int32)
    buf.copy_(cpu_values, non_blocking=False)
    return buf


def _find_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int:
    if not pattern:
        return -1
    width = len(pattern)
    for idx in range(0, len(values) - width + 1):
        if [int(x) for x in values[idx : idx + width]] == [int(x) for x in pattern]:
            return idx
    return -1


def configure_from_env() -> None:
    """Configure child-process suppression from environment variables."""
    global _AUTO_SUPPRESSION_ENABLED, _END_THINK_TOKEN_IDS, _RECOMPUTE_ALL_HEADS, _RECOMPUTE_ALL_LAYERS, _DISABLE_SUP1_FAST_PATH, _NATIVE_FUSED_DECODE, _SELECTED_FROM_FULL_HEADS, _FULL_LAYER_RECOMPUTE_SUBSET, _HF_STYLE_RECOMPUTE, _ACTIVE_BUDGET, _DEBUG, _STATS_PATH, _SUMMARY_JSON_PATH
    global _FALLBACK_DUMP_PATH, _FALLBACK_DUMP_LAYER, _FALLBACK_DUMP_Q_LEN, _FALLBACK_DUMP_GENERATED, _RUNTIME_COORD_DUMP_PATH, _RUNTIME_COORD_DUMP_ONLY_Q1, _RUNTIME_TRACE_PATH, _RUNTIME_TRACE_ONLY_Q1, _RUNTIME_TRACE_MAX_EVENTS, _RUNTIME_TRACE_LAYER_FILTER, _RUNTIME_TRACE_EVENT_COUNT
    global _NATIVE_FULL_COMPARE_PATH, _NATIVE_FULL_COMPARE_ONLY_Q1, _NATIVE_FULL_COMPARE_MAX_EVENTS, _NATIVE_FULL_COMPARE_LAYER_FILTER, _NATIVE_FULL_COMPARE_EVENT_COUNT
    _DEBUG = os.getenv("ARIS_VLLM_V1_DEBUG", "0").lower() in {"1", "true", "yes"}
    _STATS_PATH = os.getenv("ARIS_VLLM_V1_STATS_PATH")
    _FALLBACK_DUMP_PATH = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_PATH")
    _FALLBACK_DUMP_LAYER = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_LAYER")
    _FALLBACK_DUMP_Q_LEN = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_Q_LEN")
    _FALLBACK_DUMP_GENERATED = os.getenv("ARIS_VLLM_V1_FALLBACK_DUMP_GENERATED")
    _RUNTIME_COORD_DUMP_PATH = os.getenv("ARIS_VLLM_V1_RUNTIME_COORD_DUMP_PATH")
    _RUNTIME_COORD_DUMP_ONLY_Q1 = _truthy(os.getenv("ARIS_VLLM_V1_RUNTIME_COORD_DUMP_ONLY_Q1"))
    _RUNTIME_TRACE_PATH = os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_PATH")
    _RUNTIME_TRACE_ONLY_Q1 = _truthy(os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_ONLY_Q1"))
    _RUNTIME_TRACE_MAX_EVENTS = int(os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_MAX_EVENTS", "0") or "0")
    _RUNTIME_TRACE_LAYER_FILTER = {
        int(item.strip())
        for item in os.getenv("ARIS_VLLM_V1_RUNTIME_TRACE_LAYER_FILTER", "").split(",")
        if item.strip()
    }
    _RUNTIME_TRACE_EVENT_COUNT = 0
    _NATIVE_FULL_COMPARE_PATH = os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_PATH")
    _NATIVE_FULL_COMPARE_ONLY_Q1 = _truthy(os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_ONLY_Q1", "1"))
    _NATIVE_FULL_COMPARE_MAX_EVENTS = int(os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_MAX_EVENTS", "0") or "0")
    _NATIVE_FULL_COMPARE_LAYER_FILTER = {
        int(item.strip())
        for item in os.getenv("ARIS_VLLM_V1_NATIVE_FULL_COMPARE_LAYER_FILTER", "").split(",")
        if item.strip()
    }
    _NATIVE_FULL_COMPARE_EVENT_COUNT = 0
    _AUTO_SUPPRESSION_ENABLED = _truthy(
        os.getenv("ARIS_VLLM_V1_SUPPRESSION_ENABLED", os.getenv("ARIS_VLLM_V1_ENABLE", "0"))
    )
    _RECOMPUTE_ALL_HEADS = _truthy(os.getenv("ARIS_VLLM_V1_RECOMPUTE_ALL_HEADS"))
    _RECOMPUTE_ALL_LAYERS = _truthy(os.getenv("ARIS_VLLM_V1_RECOMPUTE_ALL_LAYERS"))
    _DISABLE_SUP1_FAST_PATH = _truthy(os.getenv("ARIS_VLLM_V1_DISABLE_SUP1_FAST_PATH"))
    _NATIVE_FUSED_DECODE = _truthy(os.getenv("ARIS_VLLM_V1_NATIVE_FUSED"))
    _SELECTED_FROM_FULL_HEADS = _truthy(os.getenv("ARIS_VLLM_V1_SELECTED_FROM_FULL_HEADS"))
    _FULL_LAYER_RECOMPUTE_SUBSET = _truthy(os.getenv("ARIS_VLLM_V1_FULL_LAYER_RECOMPUTE_SUBSET"))
    _HF_STYLE_RECOMPUTE = _truthy(os.getenv("ARIS_VLLM_V1_HF_STYLE_RECOMPUTE"))
    _ACTIVE_BUDGET = None
    raw_end = os.getenv("ARIS_VLLM_V1_END_THINK_TOKEN_IDS", "")
    _END_THINK_TOKEN_IDS = [int(item) for item in raw_end.split(",") if item.strip()] if raw_end else []
    if not _AUTO_SUPPRESSION_ENABLED:
        clear_active_heads()
        return
    summary_json = os.getenv("ARIS_VLLM_V1_SUMMARY_JSON") or os.getenv("ARIS_VLLM_V1_HEAD_SCORES")
    _SUMMARY_JSON_PATH = summary_json
    raw_budget = os.getenv("ARIS_VLLM_V1_BUDGET")
    if not summary_json or raw_budget in (None, ""):
        clear_active_heads()
        return
    _ACTIVE_BUDGET = float(raw_budget)
    set_active_heads(
        load_target_heads(summary_json, _ACTIVE_BUDGET),
        mask_value=float(os.getenv("ARIS_VLLM_V1_MASK_VALUE", "0.0")),
    )


@contextlib.contextmanager
def request_metadata_scope(metadata: Sequence[RequestSuppressionMetadata]) -> Iterator[None]:
    global _PENDING_REQUEST_METADATA
    previous = _PENDING_REQUEST_METADATA
    _PENDING_REQUEST_METADATA = list(metadata)
    try:
        yield
    finally:
        _PENDING_REQUEST_METADATA = previous


def _bias_value(dtype: torch.dtype) -> float:
    if _MASK_VALUE <= 0.0:
        return -1e4
    return float(math.log(_MASK_VALUE))


def _parse_layer_idx(layer) -> Optional[int]:
    cached = getattr(layer, "aris_layer_idx", None)
    if cached is not None:
        return int(cached)
    name = str(getattr(layer, "layer_name", ""))
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    idx = int(match.group(1))
    layer.aris_layer_idx = idx
    return idx


def _local_target_heads(layer, layer_idx: int, num_heads: int) -> list[int]:
    return _local_target_heads_from_map(layer, layer_idx, num_heads, _ACTIVE_HEADS_BY_LAYER)


def _local_target_heads_from_map(
    layer,
    layer_idx: int,
    num_heads: int,
    heads_by_layer: Dict[int, Sequence[int]],
) -> list[int]:
    global_heads = heads_by_layer.get(int(layer_idx), [])
    if not global_heads:
        return []
    offset = int(getattr(layer, "aris_global_head_offset", 0))
    return [head - offset for head in global_heads if offset <= head < offset + num_heads]


def _native_head_mask_table(
    *,
    layer,
    layer_idx: int,
    num_heads: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], int, bool]:
    """Build/cache [budget_id, 1, words] int64 bitmask table for current layer.

    The fused CUDA path treats layer_idx as zero because each forward call only
    needs the current layer's row. The tensor is cached to keep its device
    pointer stable across decode steps and CUDA graph replay.
    """
    if not _BUDGET_HEADS_BY_ID:
        return None, 0, False
    words = max(1, (int(num_heads) + 63) // 64)
    offset = int(getattr(layer, "aris_global_head_offset", 0))
    signature = []
    for budget_id, heads_by_layer in sorted(_BUDGET_HEADS_BY_ID.items()):
        local_heads = tuple(
            sorted(_local_target_heads_from_map(layer, layer_idx, num_heads, heads_by_layer))
        )
        signature.append((int(budget_id), local_heads))
    cache_key = (str(device), int(layer_idx), int(num_heads), int(offset), tuple(signature))
    cached = _HEAD_MASK_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    max_budget_id = max(int(item[0]) for item in signature)
    rows = [[0 for _ in range(words)] for _ in range(max_budget_id + 1)]
    has_any = False
    for budget_id, local_heads in signature:
        for head in local_heads:
            if 0 <= int(head) < int(num_heads):
                rows[int(budget_id)][int(head) >> 6] |= 1 << (int(head) & 63)
                has_any = True
    for row in rows:
        for idx, word in enumerate(row):
            if word >= (1 << 63):
                row[idx] = word - (1 << 64)
    table = torch.tensor(rows, dtype=torch.int64, device=device).view(max_budget_id + 1, 1, words).contiguous()
    result = (table, words, has_any)
    _HEAD_MASK_TABLE_CACHE[cache_key] = result
    return result


def _is_all_local_heads(target_heads: Sequence[int], num_heads: int) -> bool:
    if len(target_heads) != int(num_heads):
        return False
    return sorted(int(head) for head in target_heads) == list(range(int(num_heads)))


def _cache_flash_kv(
    *,
    layer,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    kv_cache_dtype: str,
) -> None:
    if key is None or value is None:
        return
    from vllm.attention.utils.fa_utils import reshape_and_cache_flash

    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        attn_metadata.slot_mapping,
        kv_cache_dtype,
        layer._k_scale,
        layer._v_scale,
    )


def _selected_kv_indices(
    *,
    target_heads: Sequence[int],
    num_heads: int,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if num_kv_heads == num_heads:
        indices = [int(head) for head in target_heads]
    else:
        repeats = num_heads // num_kv_heads
        indices = [int(head) // repeats for head in target_heads]
    return torch.tensor(indices, dtype=torch.long, device=device)


def _bottom_right_causal_mask(
    *,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    query_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    key_pos = torch.arange(kv_len, device=device)[None, :]
    mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    mask.masked_fill_(key_pos > query_pos, -1e4)
    return mask


def _hf_style_attention_from_repeated_kv(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_heads: Sequence[int],
    num_heads: int,
    end_pos: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    q_len = int(q.shape[-2])
    kv_len = int(k.shape[-2])
    attn_weights = torch.matmul(q.float(), k.float().transpose(2, 3)) * float(scale)
    if causal:
        mask = _bottom_right_causal_mask(
            q_len=q_len,
            kv_len=kv_len,
            dtype=torch.float32,
            device=query.device,
        )
        attn_weights = attn_weights + mask.view(1, 1, q_len, kv_len)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)

    valid_heads = [int(head) for head in target_heads if 0 <= int(head) < int(num_heads)]
    if valid_heads:
        head_idx = torch.tensor(valid_heads, dtype=torch.long, device=query.device)
        selected_attn = torch.index_select(attn_weights, dim=1, index=head_idx)
        for pos in (int(end_pos), int(end_pos) + 1):
            if 0 <= pos < kv_len:
                selected_attn[:, :, :, pos] *= float(_MASK_VALUE)
        selected_attn = selected_attn / (selected_attn.sum(dim=-1, keepdim=True) + 1e-6)
        attn_weights.index_copy_(1, head_idx, selected_attn)

    out = torch.matmul(attn_weights, v.float()).to(query.dtype)
    return out.squeeze(0).transpose(0, 1).contiguous()


def _selected_sequence_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_heads: Sequence[int],
    num_heads: int,
    end_pos: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    head_idx = torch.tensor([int(head) for head in target_heads], dtype=torch.long, device=query.device)
    kv_idx = _selected_kv_indices(
        target_heads=target_heads,
        num_heads=num_heads,
        num_kv_heads=int(key.shape[1]),
        device=query.device,
    )
    if _HF_STYLE_RECOMPUTE:
        q = query.index_select(1, head_idx)
        k = key.index_select(1, kv_idx)
        v = value.index_select(1, kv_idx)
        out = _hf_style_attention_from_repeated_kv(
            query=q,
            key=k,
            value=v,
            target_heads=list(range(len(target_heads))),
            num_heads=len(target_heads),
            end_pos=end_pos,
            scale=scale,
            causal=causal,
        )
        return out
    q = query.index_select(1, head_idx).transpose(0, 1).unsqueeze(0)
    k = key.index_select(1, kv_idx).transpose(0, 1).unsqueeze(0)
    v = value.index_select(1, kv_idx).transpose(0, 1).unsqueeze(0)
    q_len = int(q.shape[-2])
    kv_len = int(k.shape[-2])
    mask = torch.zeros((1, len(target_heads), q_len, kv_len), dtype=query.dtype, device=query.device)
    if causal:
        mask += _bottom_right_causal_mask(q_len=q_len, kv_len=kv_len, dtype=query.dtype, device=query.device).view(
            1, 1, q_len, kv_len
        )
    bias = _bias_value(query.dtype)
    for pos in (int(end_pos), int(end_pos) + 1):
        if 0 <= pos < kv_len:
            mask[:, :, :, pos] += bias
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(scale),
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


def _repeat_kv(key: torch.Tensor, value: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(key.shape[1]) == int(num_heads):
        return key, value
    repeats = int(num_heads) // int(key.shape[1])
    return key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)


def _full_sequence_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_heads: Sequence[int],
    num_heads: int,
    end_pos: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    key, value = _repeat_kv(key, value, num_heads)
    if _HF_STYLE_RECOMPUTE:
        return _hf_style_attention_from_repeated_kv(
            query=query,
            key=key,
            value=value,
            target_heads=target_heads,
            num_heads=num_heads,
            end_pos=end_pos,
            scale=scale,
            causal=causal,
        )
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    q_len = int(q.shape[-2])
    kv_len = int(k.shape[-2])
    mask = torch.zeros((1, num_heads, q_len, kv_len), dtype=query.dtype, device=query.device)
    if causal:
        mask += _bottom_right_causal_mask(q_len=q_len, kv_len=kv_len, dtype=query.dtype, device=query.device).view(
            1, 1, q_len, kv_len
        )
    valid_heads = [int(head) for head in target_heads if 0 <= int(head) < int(num_heads)]
    if valid_heads:
        head_idx = torch.tensor(valid_heads, dtype=torch.long, device=query.device)
        bias = _bias_value(query.dtype)
        for pos in (int(end_pos), int(end_pos) + 1):
            if 0 <= pos < kv_len:
                mask[:, head_idx, :, pos] += bias
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=float(scale),
    )
    return out.squeeze(0).transpose(0, 1).contiguous()


def _gather_flash_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = int(key_cache.shape[1])
    keys = []
    values = []
    remaining = int(seq_len)
    for block_id in block_table.detach().cpu().tolist():
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        block_idx = int(block_id)
        keys.append(key_cache[block_idx, :take])
        values.append(value_cache[block_idx, :take])
        remaining -= take
    if not keys:
        raise RuntimeError("ARIS could not gather any KV cache blocks.")
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def _drop_target_positions(
    key: torch.Tensor,
    value: torch.Tensor,
    end_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    kv_len = int(key.shape[0])
    drop_positions = [pos for pos in (int(end_pos), int(end_pos) + 1) if 0 <= pos < kv_len]
    if not drop_positions:
        return key, value, kv_len
    keep = torch.ones(kv_len, dtype=torch.bool, device=key.device)
    keep[torch.tensor(drop_positions, dtype=torch.long, device=key.device)] = False
    return key[keep], value[keep], int(keep.sum().item())


def _all_head_varlen_drop_attention(
    *,
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    plans: Sequence[tuple[int, int, int, list[int]] | None],
    seq_lens: Sequence[int],
    end_positions: Sequence[int],
    attn_metadata,
    layer,
    scale: float,
    num_kv_heads: int,
    sliding_window,
    logits_soft_cap: float,
    alibi_slopes,
    vllm_flash_attn_version: int,
    sinks,
) -> bool:
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func

    packed_keys = []
    packed_values = []
    cu_k = [0]
    max_k = 0
    total_q = 0
    for plan in plans:
        if plan is None:
            return False
        req_idx, q_start, q_end, _target_heads = plan
        q_len = int(q_end) - int(q_start)
        seq_len = int(seq_lens[req_idx])
        key_seq, value_seq = _gather_flash_cache(key_cache, value_cache, block_table[req_idx], seq_len)
        key_seq, value_seq, packed_len = _drop_target_positions(key_seq, value_seq, int(end_positions[req_idx]))
        # The packed-K/V fast path is correct for decode/suffix chunks where Q
        # tokens are still the last q_len positions after dropping prompt keys.
        if q_len <= 0 or q_len > packed_len:
            return False
        packed_keys.append(key_seq)
        packed_values.append(value_seq)
        cu_k.append(cu_k[-1] + packed_len)
        max_k = max(max_k, packed_len)
        total_q += q_len

    if not packed_keys or total_q <= 0:
        return False

    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", total_q))
    if total_q != num_actual_tokens:
        return False

    k = torch.cat(packed_keys, dim=0)
    v = torch.cat(packed_values, dim=0)
    cu_seqlens_k = torch.tensor(cu_k, dtype=torch.int32, device=query.device)
    cu_seqlens_q = attn_metadata.query_start_loc
    descale_shape = (cu_seqlens_q.shape[0] - 1, int(num_kv_heads))
    flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=k,
        v=v,
        out=output[:num_actual_tokens],
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=attn_metadata.max_query_len,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_k,
        softmax_scale=float(scale),
        causal=bool(getattr(attn_metadata, "causal", True)),
        alibi_slopes=alibi_slopes,
        window_size=sliding_window,
        softcap=logits_soft_cap,
        q_descale=layer._q_scale.expand(descale_shape),
        k_descale=layer._k_scale.expand(descale_shape),
        v_descale=layer._v_scale.expand(descale_shape),
        num_splits=attn_metadata.max_num_splits,
        fa_version=vllm_flash_attn_version,
        s_aux=sinks,
    )
    return True


def _tensor_or_cpu_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]


def _request_id_sort_key(request_id) -> int:
    text = str(request_id)
    prefix = text.split("::", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        match = re.match(r"\d+", text)
        if match is not None:
            return int(match.group(0))
        return 0


def _install_run_engine_patch(LLM) -> None:
    if getattr(LLM, "_aris_v1_run_engine_patch_installed", False):
        return

    def aris_run_engine(self, *, use_tqdm=True):
        from tqdm.auto import tqdm
        from vllm.outputs import RequestOutput

        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"),
            )

        outputs = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        return sorted(outputs, key=lambda x: _request_id_sort_key(x.request_id))

    LLM._run_engine = aris_run_engine
    LLM._aris_v1_run_engine_patch_installed = True


def _install_request_patch() -> None:
    from vllm.entrypoints.llm import LLM

    _install_run_engine_patch(LLM)

    if getattr(LLM, "_aris_v1_request_patch_installed", False):
        return
    original_add_request = LLM._add_request

    def aris_add_request(self, prompt, params, tokenization_kwargs=None, lora_request=None, priority=0):
        meta = None
        if isinstance(prompt, dict) and "aris_suppression" in prompt:
            prompt = dict(prompt)
            raw = prompt.pop("aris_suppression")
            raw_budget = raw.get("budget", raw.get("suppression_budget"))
            meta = RequestSuppressionMetadata(
                end_think_index=int(raw.get("end_think_index", -1)),
                enabled=bool(raw.get("enabled", True)),
                budget=None if raw_budget in (None, "") else float(raw_budget),
            )
        elif _PENDING_REQUEST_METADATA:
            meta = _PENDING_REQUEST_METADATA.pop(0)

        if meta is None:
            return original_add_request(
                self,
                prompt,
                params,
                tokenization_kwargs=tokenization_kwargs,
                lora_request=lora_request,
                priority=priority,
            )

        request_id = str(next(self.request_counter)) + _format_request_id_tag(meta)
        _REQUEST_METADATA_BY_ID[request_id] = meta
        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

    LLM._add_request = aris_add_request
    LLM._aris_v1_request_patch_installed = True


def _request_id_sort_key(request_id) -> int:
    """Return the numeric vLLM request counter prefix.

    vLLM's public ``LLM.generate`` sorts finished outputs with
    ``int(output.request_id)``.  ARIS stores suppression metadata in the
    request id suffix so worker subprocesses can recover it without relying on
    inherited parent-process state.  Sorting by the numeric prefix preserves
    vLLM's output order while allowing ids like ``0::aris_budget=1``.
    """
    prefix = str(request_id).split("::", 1)[0]
    return int(prefix)


def _install_llm_run_engine_sort_patch() -> None:
    from vllm.entrypoints.llm import LLM

    if getattr(LLM, "_aris_v1_run_engine_sort_patch_installed", False):
        return

    def aris_run_engine(self, *, use_tqdm=True):
        import vllm.entrypoints.llm as llm_module
        from vllm.outputs import RequestOutput

        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else llm_module.tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"),
            )

        outputs = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, " f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        return sorted(outputs, key=lambda x: _request_id_sort_key(x.request_id))

    LLM._run_engine = aris_run_engine
    LLM._aris_v1_run_engine_sort_patch_installed = True


def _install_gpu_runner_patch() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_aris_v1_prepare_inputs_patch_installed", False):
        return
    original_prepare_inputs = GPUModelRunner._prepare_inputs

    def aris_prepare_inputs(self, scheduler_output):
        result = original_prepare_inputs(self, scheduler_output)
        attn_metadata = result[0]
        if not isinstance(attn_metadata, dict):
            return result

        num_reqs = int(self.input_batch.num_reqs)
        req_ids = list(self.input_batch.req_ids[:num_reqs])
        end_positions = []
        enabled = []
        budget_ids = []
        heads_by_req: list[Dict[int, List[int]]] = []
        prompt_lens: list[int] = []
        prompt_prefix8: list[list[int]] = []
        found_end_idx_by_req: list[int] = []
        for req_id in req_ids:
            meta = _REQUEST_METADATA_BY_ID.get(str(req_id))
            req_state = self.requests.get(req_id)
            req_prompt_ids = list(getattr(req_state, "prompt_token_ids", []) or [])
            prompt_lens.append(len(req_prompt_ids))
            prompt_prefix8.append([int(x) for x in req_prompt_ids[:8]])
            found_end_idx = _find_subsequence(req_prompt_ids, _END_THINK_TOKEN_IDS) if _END_THINK_TOKEN_IDS else -1
            found_end_idx_by_req.append(int(found_end_idx))
            if meta is None:
                tagged_end_idx = _parse_request_end_index(str(req_id))
                end_idx = int(found_end_idx)
                if end_idx < 0 and tagged_end_idx is not None:
                    end_idx = int(tagged_end_idx)
                has_budget_tag, request_budget = _parse_request_budget(str(req_id))
                if has_budget_tag:
                    request_heads = _heads_for_budget(request_budget)
                    source = "request_id"
                else:
                    request_heads = _ACTIVE_HEADS_BY_LAYER
                    request_budget = _ACTIVE_BUDGET
                    source = "env"
                has_enabled_tag, tagged_enabled = _parse_request_enabled(str(req_id))
                request_enabled = tagged_enabled if has_enabled_tag else _AUTO_SUPPRESSION_ENABLED
                budget_id = _register_budget_heads(request_budget, request_heads)
                enabled_value = 1 if request_enabled and request_heads and end_idx >= 0 else 0
                end_positions.append(end_idx)
                enabled.append(enabled_value)
                budget_ids.append(budget_id)
                heads_by_req.append(request_heads)
                _record_prepare_request(
                    source=source,
                    budget=request_budget,
                    heads_by_layer=request_heads,
                    enabled=enabled_value,
                )
            else:
                if meta.budget is None:
                    request_heads = _ACTIVE_HEADS_BY_LAYER
                    request_budget = _ACTIVE_BUDGET
                else:
                    request_heads = _heads_for_budget(float(meta.budget))
                    request_budget = float(meta.budget)
                end_idx = int(found_end_idx) if int(found_end_idx) >= 0 else int(meta.end_think_index)
                budget_id = _register_budget_heads(request_budget, request_heads)
                enabled_value = 1 if meta.enabled and request_heads and end_idx >= 0 else 0
                end_positions.append(end_idx)
                enabled.append(enabled_value)
                budget_ids.append(budget_id)
                heads_by_req.append(request_heads)
                _record_prepare_request(
                    source="metadata",
                    budget=request_budget,
                    heads_by_layer=request_heads,
                    enabled=enabled_value,
                )

        seen: set[int] = set()
        for meta_obj in attn_metadata.values():
            if id(meta_obj) in seen:
                continue
            seen.add(id(meta_obj))
            device = meta_obj.query_start_loc.device
            meta_obj.aris_req_ids = req_ids
            meta_obj.aris_end_positions = _device_int32_buffer(meta_obj, "_aris_end_positions_buffer", end_positions, device)
            meta_obj.aris_enabled = _device_int32_buffer(meta_obj, "_aris_enabled_buffer", enabled, device)
            meta_obj.aris_budget_ids = _device_int32_buffer(meta_obj, "_aris_budget_ids_buffer", budget_ids, device)
            meta_obj.aris_any_enabled = any(enabled)
            meta_obj.aris_heads_by_req = heads_by_req
            meta_obj.aris_query_start_loc_cpu = _tensor_or_cpu_list(meta_obj.query_start_loc)
            meta_obj.aris_seq_lens_cpu = _tensor_or_cpu_list(meta_obj.seq_lens)
            meta_obj.aris_prompt_lens_cpu = [int(x) for x in prompt_lens]
            meta_obj.aris_prompt_prefix8_cpu = [[int(y) for y in row] for row in prompt_prefix8]
            meta_obj.aris_found_end_idx_cpu = [int(x) for x in found_end_idx_by_req]

        for req_id in getattr(scheduler_output, "finished_req_ids", []) or []:
            _REQUEST_METADATA_BY_ID.pop(str(req_id), None)

        if _DEBUG:
            _STATS["prepare_inputs_calls"] += 1
        return result

    GPUModelRunner._prepare_inputs = aris_prepare_inputs
    GPUModelRunner._aris_v1_prepare_inputs_patch_installed = True


def _install_flash_attention_patch() -> None:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    if getattr(FlashAttentionImpl, "_aris_v1_flash_patch_installed", False):
        return
    original_forward = FlashAttentionImpl.forward

    def aris_flash_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if _DEBUG:
            _STATS["flash_calls"] += 1
            _maybe_dump_debug_stats()

        if attn_metadata is None or not getattr(attn_metadata, "aris_any_enabled", False):
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        layer_idx = _parse_layer_idx(layer)
        if layer_idx is None:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if getattr(attn_metadata, "use_cascade", False):
            _STATS["skipped_cascade_calls"] += 1
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if str(self.kv_cache_dtype).startswith("fp8"):
            _STATS["skipped_fp8_calls"] += 1
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        if output is None or output_scale is not None or output_block_scale is not None:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
            return original_forward(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        if _NATIVE_FUSED_DECODE:
            if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
                _record_native_fused_fallback("non_decode")
            elif getattr(attn_metadata, "block_table", None) is None:
                _record_native_fused_fallback("missing_block_table")
            elif int(self.vllm_flash_attn_version) != 2:
                _record_native_fused_fallback("fa_version_not_2")
            else:
                mask_table, words_per_layer, has_any_heads = _native_head_mask_table(
                    layer=layer,
                    layer_idx=layer_idx,
                    num_heads=int(self.num_heads),
                    device=query.device,
                )
                if not has_any_heads or mask_table is None:
                    _record_native_fused_fallback("empty_head_mask")
                else:
                    from vllm.attention.utils.fa_utils import flash_attn_varlen_func

                    _cache_flash_kv(
                        layer=layer,
                        key=key,
                        value=value,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        attn_metadata=attn_metadata,
                        kv_cache_dtype=self.kv_cache_dtype,
                    )
                    num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", int(query.shape[0])))
                    cu_seqlens_q = attn_metadata.query_start_loc
                    descale_shape = (cu_seqlens_q.shape[0] - 1, int(self.num_kv_heads))
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=attn_metadata.max_query_len,
                        seqused_k=attn_metadata.seq_lens,
                        max_seqlen_k=attn_metadata.max_seq_len,
                        softmax_scale=float(self.scale),
                        causal=bool(getattr(attn_metadata, "causal", True)),
                        alibi_slopes=self.alibi_slopes,
                        window_size=self.sliding_window,
                        block_table=attn_metadata.block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=attn_metadata.scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                        aris_enabled=attn_metadata.aris_enabled,
                        aris_budget_ids=attn_metadata.aris_budget_ids,
                        aris_end_positions=attn_metadata.aris_end_positions,
                        aris_head_mask_table=mask_table,
                        aris_layer_idx=0,
                        aris_num_layers=1,
                        aris_words_per_layer=words_per_layer,
                    )
                    req_idx_for_dump = 0
                    native_target_heads: list[int] = []
                    enabled_rows = _tensor_or_cpu_list(attn_metadata.aris_enabled)
                    budget_rows = _tensor_or_cpu_list(attn_metadata.aris_budget_ids)
                    for cand_idx, cand_enabled in enumerate(enabled_rows):
                        if cand_enabled:
                            req_idx_for_dump = int(cand_idx)
                            heads_by_layer = _BUDGET_HEADS_BY_ID.get(int(budget_rows[cand_idx]), {})
                            native_target_heads = _local_target_heads_from_map(
                                layer,
                                layer_idx,
                                int(self.num_heads),
                                heads_by_layer,
                            )
                            break
                    native_plan = (
                        int(req_idx_for_dump),
                        int(cu_seqlens_q[0].item()) if cu_seqlens_q.numel() > 0 else 0,
                        int(cu_seqlens_q[1].item()) if cu_seqlens_q.numel() > 1 else 0,
                        native_target_heads,
                    )
                    _maybe_dump_runtime_coord(
                        attn_metadata=attn_metadata,
                        path_name="native_fused_decode",
                        layer_idx=int(layer_idx),
                        plan=native_plan,
                    )
                    _maybe_append_runtime_trace(
                        attn_metadata=attn_metadata,
                        path_name="native_fused_decode",
                        layer_idx=int(layer_idx),
                        plan=native_plan,
                    )
                    _STATS["suppressed_flash_calls"] += 1
                    _STATS["native_fused_decode_calls"] += 1
                    _STATS["native_fused_decode_tokens"] += num_actual_tokens
                    layer_key = str(int(layer_idx))
                    layer_stats = _STATS["layers"].setdefault(layer_key, {"calls": 0, "sequence_recomputes": 0})
                    layer_stats["calls"] += 1
                    layer_stats["sequence_recomputes"] += 0
                    _maybe_dump_debug_stats(force=_STATS["suppressed_flash_calls"] == 1)
                    return output

        query_start = list(getattr(attn_metadata, "aris_query_start_loc_cpu"))
        seq_lens = list(getattr(attn_metadata, "aris_seq_lens_cpu"))
        end_positions = _tensor_or_cpu_list(attn_metadata.aris_end_positions)
        enabled = _tensor_or_cpu_list(attn_metadata.aris_enabled)
        heads_by_req = getattr(attn_metadata, "aris_heads_by_req", None)
        if heads_by_req is None:
            heads_by_req = [_ACTIVE_HEADS_BY_LAYER for _ in enabled]
        block_table = attn_metadata.block_table

        num_heads = int(self.num_heads)
        plans: list[tuple[int, int, int, list[int]] | None] = []
        can_use_all_head_fast_path = not _DISABLE_SUP1_FAST_PATH
        has_recompute_work = False
        for req_idx, is_enabled in enumerate(enabled):
            if not is_enabled:
                can_use_all_head_fast_path = False
                plans.append(None)
                continue
            request_heads_by_layer = heads_by_req[req_idx] if req_idx < len(heads_by_req) else _ACTIVE_HEADS_BY_LAYER
            target_heads = _local_target_heads_from_map(layer, layer_idx, num_heads, request_heads_by_layer)
            if not target_heads and not _RECOMPUTE_ALL_LAYERS:
                can_use_all_head_fast_path = False
                plans.append(None)
                continue
            if _RECOMPUTE_ALL_HEADS or _RECOMPUTE_ALL_LAYERS:
                target_heads = list(range(num_heads))
            if not _is_all_local_heads(target_heads, num_heads):
                can_use_all_head_fast_path = False
            q_start = int(query_start[req_idx])
            q_end = int(query_start[req_idx + 1])
            if q_end <= q_start:
                plans.append(None)
                continue
            has_recompute_work = True
            plans.append((req_idx, q_start, q_end, target_heads))

        if can_use_all_head_fast_path and has_recompute_work:
            _cache_flash_kv(
                layer=layer,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                attn_metadata=attn_metadata,
                kv_cache_dtype=self.kv_cache_dtype,
            )
            varlen_replaced = sum(1 for plan in plans if plan is not None)
            if _all_head_varlen_drop_attention(
                query=query,
                output=output,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                plans=plans,
                seq_lens=seq_lens,
                end_positions=end_positions,
                attn_metadata=attn_metadata,
                layer=layer,
                scale=float(self.scale),
                num_kv_heads=int(self.num_kv_heads),
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=self.alibi_slopes,
                vllm_flash_attn_version=self.vllm_flash_attn_version,
                sinks=self.sinks,
            ):
                _STATS["suppressed_flash_calls"] += 1
                _STATS["all_head_varlen_fast_path_calls"] += 1
                _STATS["all_head_varlen_fast_path_recomputes"] += varlen_replaced
                layer_key = str(int(layer_idx))
                layer_stats = _STATS["layers"].setdefault(layer_key, {"calls": 0, "sequence_recomputes": 0})
                layer_stats["calls"] += 1
                layer_stats["sequence_recomputes"] += varlen_replaced
                _maybe_dump_debug_stats(force=_STATS["suppressed_flash_calls"] == 1)
                return output

            num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", int(query.shape[0])))
            output_heads = output[:num_actual_tokens].view(-1, num_heads, int(self.head_size))
            replaced = 0
            for plan in plans:
                if plan is None:
                    continue
                req_idx, q_start, q_end, target_heads = plan
                seq_len = int(seq_lens[req_idx])
                key_seq, value_seq = _gather_flash_cache(key_cache, value_cache, block_table[req_idx], seq_len)
                full = _full_sequence_attention(
                    query=query[q_start:q_end],
                    key=key_seq,
                    value=value_seq,
                    target_heads=target_heads,
                    num_heads=num_heads,
                    end_pos=int(end_positions[req_idx]),
                    scale=float(self.scale),
                    causal=bool(getattr(attn_metadata, "causal", True)),
                )
                output_heads[q_start:q_end].copy_(full)
                replaced += 1
            if replaced:
                if plans:
                    for plan in plans:
                        if plan is not None:
                            _maybe_append_runtime_trace(
                                attn_metadata=attn_metadata,
                                path_name="all_head_full_sequence",
                                layer_idx=int(layer_idx),
                                plan=plan,
                            )
                _STATS["suppressed_flash_calls"] += 1
                _STATS["all_head_fast_path_calls"] += 1
                _STATS["all_head_fast_path_recomputes"] += replaced
                _STATS["full_sequence_recomputes"] += replaced
                layer_key = str(int(layer_idx))
                layer_stats = _STATS["layers"].setdefault(layer_key, {"calls": 0, "sequence_recomputes": 0})
                layer_stats["calls"] += 1
                layer_stats["sequence_recomputes"] += replaced
                _maybe_dump_debug_stats(force=_STATS["suppressed_flash_calls"] == 1)
            return output

        native_output = original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        output_heads = native_output.view(-1, int(self.num_heads), int(self.head_size))

        replaced = 0
        for plan in plans:
            if plan is None:
                continue
            req_idx, q_start, q_end, target_heads = plan
            seq_len = int(seq_lens[req_idx])
            key_seq, value_seq = _gather_flash_cache(key_cache, value_cache, block_table[req_idx], seq_len)
            _maybe_dump_runtime_coord(
                attn_metadata=attn_metadata,
                path_name="fallback_or_recompute",
                layer_idx=int(layer_idx),
                plan=plan,
            )
            _maybe_dump_fallback_case(
                layer_idx=int(layer_idx),
                req_idx=int(req_idx),
                q_start=int(q_start),
                q_end=int(q_end),
                query=query,
                key_seq=key_seq,
                value_seq=value_seq,
                end_pos=int(end_positions[req_idx]),
                target_heads=target_heads,
                seq_len=seq_len,
                attn_metadata=attn_metadata,
            )
            _maybe_append_runtime_trace(
                attn_metadata=attn_metadata,
                path_name="fallback_or_recompute",
                layer_idx=int(layer_idx),
                plan=plan,
            )
            full_for_compare = None
            if _NATIVE_FULL_COMPARE_PATH:
                full_for_compare = _full_sequence_attention(
                    query=query[q_start:q_end],
                    key=key_seq,
                    value=value_seq,
                    target_heads=target_heads,
                    num_heads=int(self.num_heads),
                    end_pos=int(end_positions[req_idx]),
                    scale=float(self.scale),
                    causal=bool(getattr(attn_metadata, "causal", True)),
                )
                _maybe_append_native_full_compare(
                    attn_metadata=attn_metadata,
                    layer_idx=int(layer_idx),
                    plan=plan,
                    native_heads=output_heads,
                    full_heads=full_for_compare,
                    stage="pre_merge",
                )
            if _RECOMPUTE_ALL_HEADS or _RECOMPUTE_ALL_LAYERS:
                full = full_for_compare
                if full is None:
                    full = _full_sequence_attention(
                        query=query[q_start:q_end],
                        key=key_seq,
                        value=value_seq,
                        target_heads=target_heads,
                        num_heads=num_heads,
                        end_pos=int(end_positions[req_idx]),
                        scale=float(self.scale),
                        causal=bool(getattr(attn_metadata, "causal", True)),
                    )
                output_heads[q_start:q_end].copy_(full)
                _STATS["full_sequence_recomputes"] += 1
            elif _FULL_LAYER_RECOMPUTE_SUBSET:
                full = full_for_compare
                if full is None:
                    full = _full_sequence_attention(
                        query=query[q_start:q_end],
                        key=key_seq,
                        value=value_seq,
                        target_heads=target_heads,
                        num_heads=num_heads,
                        end_pos=int(end_positions[req_idx]),
                        scale=float(self.scale),
                        causal=bool(getattr(attn_metadata, "causal", True)),
                    )
                output_heads[q_start:q_end].copy_(full)
                _STATS["full_layer_subset_recomputes"] += 1
            else:
                head_idx = torch.tensor(target_heads, dtype=torch.long, device=query.device)
                if _SELECTED_FROM_FULL_HEADS:
                    full = full_for_compare
                    if full is None:
                        full = _full_sequence_attention(
                            query=query[q_start:q_end],
                            key=key_seq,
                            value=value_seq,
                            target_heads=target_heads,
                            num_heads=int(self.num_heads),
                            end_pos=int(end_positions[req_idx]),
                            scale=float(self.scale),
                            causal=bool(getattr(attn_metadata, "causal", True)),
                        )
                    selected = full.index_select(1, head_idx)
                    _STATS["selected_from_full_recomputes"] += 1
                else:
                    selected = _selected_sequence_attention(
                        query=query[q_start:q_end],
                        key=key_seq,
                        value=value_seq,
                        target_heads=target_heads,
                        num_heads=int(self.num_heads),
                        end_pos=int(end_positions[req_idx]),
                        scale=float(self.scale),
                        causal=bool(getattr(attn_metadata, "causal", True)),
                    )
                output_heads[q_start:q_end].index_copy_(1, head_idx, selected)
            if full_for_compare is not None:
                _maybe_append_native_full_compare(
                    attn_metadata=attn_metadata,
                    layer_idx=int(layer_idx),
                    plan=plan,
                    native_heads=output_heads,
                    full_heads=full_for_compare,
                    stage="post_merge",
                )
            replaced += 1

        if replaced:
            _STATS["suppressed_flash_calls"] += 1
            _STATS["selected_sequence_recomputes"] += replaced
            layer_key = str(int(layer_idx))
            layer_stats = _STATS["layers"].setdefault(layer_key, {"calls": 0, "sequence_recomputes": 0})
            layer_stats["calls"] += 1
            layer_stats["sequence_recomputes"] += replaced
            _maybe_dump_debug_stats(force=_STATS["suppressed_flash_calls"] == 1)
        return native_output

    FlashAttentionImpl.forward = aris_flash_forward
    FlashAttentionImpl._aris_v1_flash_patch_installed = True


def install_aris_vllm_v1_suppression() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_request_patch()
    _install_llm_run_engine_sort_patch()
    _install_gpu_runner_patch()
    _install_flash_attention_patch()
    _INSTALLED = True


def patch_vllm_model_attention_metadata(inference_engine) -> int:
    """Set TP-aware local head offsets when the model object is directly visible."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
    except Exception:
        get_tensor_model_parallel_rank = None

    try:
        tp_rank = int(get_tensor_model_parallel_rank()) if get_tensor_model_parallel_rank else 0
    except Exception:
        tp_rank = 0

    models = []
    try:
        executor = getattr(getattr(inference_engine, "llm_engine", None), "model_executor", None)
        for attr in ("worker", "driver_worker"):
            worker = getattr(executor, attr, None)
            model = getattr(getattr(worker, "model_runner", None), "model", None)
            if model is not None:
                models.append(model)
    except Exception:
        pass

    patched = 0
    for model in models:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            continue
        for layer_idx, layer in enumerate(layers):
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is None:
                continue
            attn.aris_layer_idx = int(layer_idx)
            attn.aris_global_head_offset = tp_rank * int(attn.impl.num_heads)
            patched += 1
    return patched
