from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np


RolloutVariant = tuple[str, Optional[float], bool, str]

DEFAULT_TRAIN_ROLLOUT_VARIANTS: tuple[RolloutVariant, ...] = (
    ("Alright+sup0.1", 0.10, True, "default"),
    ("Alright+sup0.2", 0.20, True, "default"),
    ("Alright+sup0.3", 0.30, True, "default"),
    ("Alright+sup0.4", 0.40, True, "default"),
    ("Alright+sup0.5", 0.50, True, "default"),
    ("Alright+sup0.6", 0.60, True, "default"),
    ("Alright+sup0.7", 0.70, True, "default"),
    ("Alright+sup1.0", 1.00, True, "default"),
)

VALIDATION_ROLLOUT_VARIANT: RolloutVariant = ("Alright+sup1.0", 1.00, True, "default")
_DEBUG_DUMPED_SAMPLES = 0


def _parse_budget(value: str) -> Optional[float]:
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    return float(normalized)


def _variant_from_budget(budget: Optional[float]) -> RolloutVariant:
    if budget is None:
        return ("Alright_only", None, True, "default")
    budget_label = "1.0" if budget == 1.0 else f"{budget:g}"
    return (f"Alright+sup{budget_label}", budget, True, "default")


def _variant_from_name(value: str) -> RolloutVariant:
    normalized = value.strip().lower()
    if normalized in {"think", "thinking", "pure", "pure_think", "pure-thinking", "base", "nosupp", "no_supp"}:
        return ("Think", None, False, "think")
    if normalized in {"off", "none", "false"}:
        return ("NoSupp", None, False, "default")
    if normalized.startswith("sup"):
        budget = float(normalized.removeprefix("sup"))
        budget_label = "1.0" if budget == 1.0 else f"{budget:g}"
        return (f"Sup{budget_label}", budget, False, "sup")
    return _variant_from_budget(_parse_budget(normalized))


def train_rollout_variants(rollout_n: int) -> tuple[RolloutVariant, ...]:
    raw_variants = os.getenv("ARIS_TRAIN_ROLLOUT_VARIANTS", "").strip()
    if raw_variants:
        variants = tuple(_variant_from_name(part) for part in raw_variants.split(","))
        if len(variants) != int(rollout_n):
            raise ValueError(
                "ARIS_TRAIN_ROLLOUT_VARIANTS must contain exactly "
                f"{rollout_n} comma-separated entries, got {len(variants)}: {raw_variants!r}."
            )
        return variants

    raw_budgets = os.getenv("ARIS_TRAIN_ROLLOUT_BUDGETS", "").strip()
    if not raw_budgets:
        return DEFAULT_TRAIN_ROLLOUT_VARIANTS

    budgets = tuple(_parse_budget(part) for part in raw_budgets.split(","))
    if len(budgets) != int(rollout_n):
        raise ValueError(
            "ARIS_TRAIN_ROLLOUT_BUDGETS must contain exactly "
            f"{rollout_n} comma-separated entries, got {len(budgets)}: {raw_budgets!r}."
        )
    return tuple(_variant_from_budget(budget) for budget in budgets)


def validation_rollout_variant() -> RolloutVariant:
    raw_budget = os.getenv("ARIS_VALIDATION_SUPPRESSION_BUDGET", "").strip()
    if not raw_budget:
        raw_budget = os.getenv("ARIS_VLLM_V1_BUDGET", "").strip()
    if not raw_budget:
        return VALIDATION_ROLLOUT_VARIANT

    budget = _parse_budget(raw_budget)
    if budget is None:
        return ("Alright_only", None, True, "default")
    budget_label = "1.0" if budget == 1.0 else f"{budget:g}"
    return (f"Alright+sup{budget_label}", budget, True, "default")


def rollout_suppression_enabled(config: Any) -> bool:
    if os.getenv("ARIS_ROLLOUT_SUPPRESSION_ENABLE", "").lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        rollout_config = config.actor_rollout_ref.rollout
        return bool(rollout_config.get("aris_rollout_suppression_enable", False))
    except Exception:
        return False


def annotate_training_rollouts(batch: Any, rollout_n: int) -> Any:
    variants_config = train_rollout_variants(int(rollout_n))
    if int(rollout_n) != len(variants_config):
        raise ValueError(
            "ARIS suppression rollout replacement expects rollout.n="
            f"{len(variants_config)}, got {rollout_n}."
        )
    rollout_indices = np.arange(len(batch), dtype=np.int64) % int(rollout_n)
    variants = [variants_config[int(idx)][0] for idx in rollout_indices]
    budgets = [variants_config[int(idx)][1] for idx in rollout_indices]
    alright_prompts = [variants_config[int(idx)][2] for idx in rollout_indices]
    tokenizer_modes = [variants_config[int(idx)][3] for idx in rollout_indices]
    batch.non_tensor_batch["aris_alright_prompt"] = np.array(alright_prompts, dtype=object)
    batch.non_tensor_batch["aris_rollout_index"] = np.array(rollout_indices.tolist(), dtype=object)
    batch.non_tensor_batch["aris_rollout_variant"] = np.array(variants, dtype=object)
    batch.non_tensor_batch["aris_suppression_budget"] = np.array(budgets, dtype=object)
    batch.non_tensor_batch["aris_rollout_tokenizer"] = np.array(tokenizer_modes, dtype=object)
    return batch


def annotate_validation_alright_only(batch: Any) -> Any:
    variant, budget, alright_prompt, tokenizer_mode = validation_rollout_variant()
    batch.non_tensor_batch["aris_alright_prompt"] = np.array([alright_prompt] * len(batch), dtype=object)
    batch.non_tensor_batch["aris_rollout_index"] = np.array([7] * len(batch), dtype=object)
    batch.non_tensor_batch["aris_rollout_variant"] = np.array([variant] * len(batch), dtype=object)
    batch.non_tensor_batch["aris_suppression_budget"] = np.array([budget] * len(batch), dtype=object)
    batch.non_tensor_batch["aris_rollout_tokenizer"] = np.array([tokenizer_mode] * len(batch), dtype=object)
    return batch


def alright_suffix_text() -> str:
    return os.getenv("ARIS_ALRIGHT_SUFFIX_TEXT", "\n</think>\n\nAlright")


def append_suffix_text(
    tokenizer: Any,
    prompt_ids: list[int],
    suffix_text: str,
    prompt_length: Optional[int] = None,
    label: str = "suffix",
) -> list[int]:
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    suffix_ids = [int(token_id) for token_id in suffix_ids]
    suffix_id_candidates = [suffix_ids]
    stripped_suffix_text = suffix_text.lstrip("\n")
    if stripped_suffix_text != suffix_text:
        stripped_suffix_ids = tokenizer.encode(stripped_suffix_text, add_special_tokens=False)
        stripped_suffix_ids = [int(token_id) for token_id in stripped_suffix_ids]
        if stripped_suffix_ids:
            suffix_id_candidates.append(stripped_suffix_ids)
    output_ids = [int(token_id) for token_id in prompt_ids]
    has_suffix = any(
        candidate and len(output_ids) >= len(candidate) and output_ids[-len(candidate) :] == candidate
        for candidate in suffix_id_candidates
    )
    if suffix_ids and not has_suffix:
        output_ids = output_ids + suffix_ids
    if prompt_length is not None and len(output_ids) > int(prompt_length):
        raise ValueError(
            f"{label} is longer than rollout.prompt_length: "
            f"{len(output_ids)} > {prompt_length}. Increase data.max_prompt_length."
        )
    return output_ids


def append_alright_suffix(tokenizer: Any, prompt_ids: list[int], prompt_length: Optional[int] = None) -> list[int]:
    return append_suffix_text(
        tokenizer,
        prompt_ids,
        alright_suffix_text(),
        prompt_length=prompt_length,
        label="ARIS Alright prompt",
    )


def sup_rollout_train_prompt_rewrite_enabled() -> bool:
    return os.getenv("ARIS_SUP_ROLLOUT_REWRITE_TRAIN_PROMPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def sup_train_rewrite_suffix_text() -> str:
    return os.getenv("ARIS_SUP_TRAIN_REWRITE_SUFFIX_TEXT", "Alright")


def _sequence_count(values: list[int], pattern: list[int]) -> int:
    if not pattern:
        return 0
    width = len(pattern)
    return sum(1 for idx in range(0, len(values) - width + 1) if values[idx : idx + width] == pattern)


def _safe_encode(tokenizer: Any, text: str) -> list[int]:
    try:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
    except Exception:
        return []


def _safe_decode(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except Exception as exc:
        return f"<decode_failed:{type(exc).__name__}:{exc}>"


def maybe_dump_rollout_debug_sample(
    *,
    tokenizer: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    raw_prompt: Any,
    variant: Any,
    budget: Any,
    tokenizer_mode: Any,
    tokenizer_path: str,
    request_id: str,
    end_think_index: Optional[int],
    train_tokenizer: Any = None,
    train_prompt_ids: Optional[list[int]] = None,
    train_tokenizer_path: str = "",
    train_rewrite_applied: bool = False,
    train_response_ids: Optional[list[int]] = None,
    train_response_mask: Optional[list[int]] = None,
    train_response_logprobs_available: Optional[bool] = None,
) -> None:
    dump_path = os.getenv("ARIS_ROLLOUT_DEBUG_DUMP_PATH", "").strip()
    max_samples = int(os.getenv("ARIS_ROLLOUT_DEBUG_MAX_SAMPLES_PER_PROCESS", "0") or "0")
    if not dump_path or max_samples <= 0:
        return

    global _DEBUG_DUMPED_SAMPLES
    if _DEBUG_DUMPED_SAMPLES >= max_samples:
        return
    _DEBUG_DUMPED_SAMPLES += 1

    prompt_values = [int(token_id) for token_id in prompt_ids]
    response_values = [int(token_id) for token_id in response_ids]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    bos_token = getattr(tokenizer, "bos_token", None)
    end_think_ids = _safe_encode(tokenizer, "</think>")
    alright_ids = _safe_encode(tokenizer, "Alright")
    prompt_text = _safe_decode(tokenizer, prompt_values)
    response_text = _safe_decode(tokenizer, response_values)
    chat_template = str(getattr(tokenizer, "chat_template", ""))
    train_prompt_values = [int(token_id) for token_id in train_prompt_ids] if train_prompt_ids is not None else None
    train_response_values = (
        [int(token_id) for token_id in train_response_ids] if train_response_ids is not None else None
    )
    train_response_mask_values = (
        [int(mask_value) for mask_value in train_response_mask] if train_response_mask is not None else None
    )
    train_prompt_text = None
    train_response_text = None
    train_bos_id = None
    train_bos_token = None
    train_end_think_ids: list[int] = []
    train_alright_ids: list[int] = []
    active_train_tokenizer = train_tokenizer if train_tokenizer is not None else tokenizer
    if train_prompt_values is not None:
        train_bos_id = getattr(active_train_tokenizer, "bos_token_id", None)
        train_bos_token = getattr(active_train_tokenizer, "bos_token", None)
        train_end_think_ids = _safe_encode(active_train_tokenizer, "</think>")
        train_alright_ids = _safe_encode(active_train_tokenizer, "Alright")
        train_prompt_text = _safe_decode(active_train_tokenizer, train_prompt_values)
    if train_response_values is not None:
        train_response_text = _safe_decode(active_train_tokenizer, train_response_values)
    record = {
        "request_id": request_id,
        "variant": variant,
        "budget": budget,
        "tokenizer_mode": tokenizer_mode,
        "tokenizer_path": tokenizer_path,
        "bos_token_id": bos_id,
        "bos_token": bos_token,
        "bos_id_count": prompt_values.count(int(bos_id)) if bos_id is not None else None,
        "bos_text_count": prompt_text.count(str(bos_token)) if bos_token else None,
        "end_think_index": end_think_index,
        "end_think_token_ids": end_think_ids,
        "end_think_id_count": _sequence_count(prompt_values, end_think_ids),
        "end_think_text_count": prompt_text.count("</think>"),
        "alright_token_ids": alright_ids,
        "alright_id_count": _sequence_count(prompt_values, alright_ids),
        "alright_text_count": prompt_text.count("Alright"),
        "prompt_len": len(prompt_values),
        "response_len": len(response_values),
        "prompt_ids": prompt_values,
        "response_ids": response_values,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "raw_prompt": raw_prompt,
        "chat_template_tail": chat_template[-512:],
        "sampling_prompt_tail": prompt_text[-512:],
        "train_rewrite_applied": bool(train_rewrite_applied),
        "train_tokenizer_path": train_tokenizer_path,
    }
    if train_prompt_values is not None:
        record.update(
            {
                "train_bos_token_id": train_bos_id,
                "train_bos_token": train_bos_token,
                "train_bos_id_count": (
                    train_prompt_values.count(int(train_bos_id)) if train_bos_id is not None else None
                ),
                "train_bos_text_count": train_prompt_text.count(str(train_bos_token)) if train_bos_token else None,
                "train_end_think_token_ids": train_end_think_ids,
                "train_end_think_id_count": _sequence_count(train_prompt_values, train_end_think_ids),
                "train_end_think_text_count": train_prompt_text.count("</think>"),
                "train_alright_token_ids": train_alright_ids,
                "train_alright_id_count": _sequence_count(train_prompt_values, train_alright_ids),
                "train_alright_text_count": train_prompt_text.count("Alright"),
                "train_prompt_len": len(train_prompt_values),
                "train_prompt_ids": train_prompt_values,
                "train_prompt_text": train_prompt_text,
                "train_prompt_tail": train_prompt_text[-512:],
            }
        )
    if train_response_values is not None:
        record.update(
            {
                "train_response_logprobs_available": train_response_logprobs_available,
                "train_response_len": len(train_response_values),
                "train_response_mask_len": len(train_response_mask_values) if train_response_mask_values else 0,
                "train_response_mask_sum": sum(train_response_mask_values) if train_response_mask_values else 0,
                "train_response_alright_id_count": _sequence_count(train_response_values, train_alright_ids),
                "train_response_alright_text_count": train_response_text.count("Alright")
                if train_response_text is not None
                else None,
                "train_response_ids": train_response_values,
                "train_response_mask_head": train_response_mask_values[:16] if train_response_mask_values else [],
                "train_response_text": train_response_text,
                "train_response_head": train_response_text[:512] if train_response_text is not None else None,
                "train_response_tail": train_response_text[-512:] if train_response_text is not None else None,
                "train_full_boundary_text": (
                    f"{train_prompt_text[-256:] if train_prompt_text is not None else ''}"
                    f"{train_response_text[:256] if train_response_text is not None else ''}"
                ),
            }
        )
    summary = {
        key: record[key]
        for key in (
            "request_id",
            "variant",
            "budget",
            "tokenizer_mode",
            "tokenizer_path",
            "bos_token_id",
            "bos_id_count",
            "end_think_id_count",
            "end_think_text_count",
            "alright_id_count",
            "alright_text_count",
            "prompt_len",
            "response_len",
            "end_think_index",
        )
    }
    if train_prompt_values is not None:
        summary.update(
            {
                "train_rewrite_applied": record["train_rewrite_applied"],
                "train_tokenizer_path": record["train_tokenizer_path"],
                "train_prompt_len": record["train_prompt_len"],
                "train_bos_id_count": record["train_bos_id_count"],
                "train_end_think_id_count": record["train_end_think_id_count"],
                "train_end_think_text_count": record["train_end_think_text_count"],
                "train_alright_id_count": record["train_alright_id_count"],
                "train_alright_text_count": record["train_alright_text_count"],
            }
        )
    if train_response_values is not None:
        summary.update(
            {
                "train_response_logprobs_available": record["train_response_logprobs_available"],
                "train_response_len": record["train_response_len"],
                "train_response_mask_sum": record["train_response_mask_sum"],
                "train_response_alright_id_count": record["train_response_alright_id_count"],
                "train_response_alright_text_count": record["train_response_alright_text_count"],
                "train_response_head": record["train_response_head"],
                "train_full_boundary_text": record["train_full_boundary_text"],
            }
        )
    import json

    print(f"ARIS_ROLLOUT_DEBUG_SUMMARY {json.dumps(summary, ensure_ascii=False, sort_keys=True)}", flush=True)
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "a", encoding="utf-8") as fout:
        try:
            import fcntl

            fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
        except Exception:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _has_budget(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "off", "false", "nan"}
    try:
        return not np.isnan(value)
    except TypeError:
        return True


def _end_think_token_ids() -> list[int]:
    raw = os.getenv("ARIS_VLLM_V1_END_THINK_TOKEN_IDS", "151649")
    return [int(item) for item in raw.split(",") if item.strip()]


def find_end_think_index(prompt_ids: list[int]) -> int:
    """在已经 tokenized 的 prompt 中定位 </think> 起始位置。

    vLLM worker 子进程有时拿不到父进程的自动定位环境变量。把位置写入
    request_id 可以让 worker 只依赖请求自身 metadata，避免出现“配置了
    budget 但实际没有抑制”的假 sup 路径。
    """
    pattern = _end_think_token_ids()
    if not pattern:
        return -1
    width = len(pattern)
    values = [int(token_id) for token_id in prompt_ids]
    for idx in range(0, len(values) - width + 1):
        if values[idx : idx + width] == pattern:
            return idx
    return -1


def build_request_id(
    base_request_id: str,
    variant: Any = None,
    budget: Any = None,
    end_think_index: Optional[int] = None,
    enabled: bool = True,
) -> str:
    if not variant and not _has_budget(budget):
        return base_request_id
    if not _has_budget(budget):
        return f"{base_request_id}::aris_budget=off::aris_enabled=0"
    end_tag = "" if end_think_index is None else f"::aris_end={int(end_think_index)}"
    enabled_tag = f"::aris_enabled={1 if enabled else 0}"
    return f"{base_request_id}::aris_budget={float(budget):.2f}{end_tag}{enabled_tag}"


def copy_request_id_tags(base_request_id: str, source_request_id: str) -> str:
    source = str(source_request_id)
    marker = "::aris_"
    marker_index = source.find(marker)
    if marker_index < 0:
        return base_request_id
    return f"{base_request_id}{source[marker_index:]}"
