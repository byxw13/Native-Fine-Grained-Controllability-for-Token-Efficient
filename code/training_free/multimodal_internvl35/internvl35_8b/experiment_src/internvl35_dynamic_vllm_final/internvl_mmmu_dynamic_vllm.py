#!/usr/bin/env python3
"""InternVL3.5-2B MMMUVal runner for the isolated ARIS vLLM port.

The request construction is deliberately inherited from the already audited
InternVL baseline.  The only method-side prompt change is the closed-think
scaffold, which makes ``</think>`` and its following token fixed prompt
positions that the Full-Attention correction can remove.  Every completed
item is appended and fsynced immediately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment_src.internvl35_vllm_baseline import mmmu_thinking_vllm as intern_baseline

baseline = intern_baseline.baseline


CLOSED_THINK_PREFIX = "<think>\n\n</think>\n\n"
# The formal InternVL run uses the most frequent pure-thinking first word.
# Keep this default explicit so a submission cannot silently fall back to the
# older smoke-only ``The`` cue.  This branch deliberately calls the baseline
# builder in ``off`` mode and therefore never adds the R1 system prompt.
DEFAULT_METHOD_CUE = "Okay"
DEFAULT_SELECTOR_WORDS = ("So", "Okay")


def _append_fsync(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None and not row.get("error"):
                result.add(str(row["id"]))
    return result


def build_method_request(tokenizer, sample: dict[str, Any], cue: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # InternVL's native chat template has no thinking switch.  As in the
    # Qwen3-VL closed-think method, use the ordinary user prompt and append one
    # explicit scaffold; keeping the R1 instruction would add another literal
    # ``</think>`` and make the suppression target ambiguous.
    request, audit = intern_baseline.build_request(tokenizer, sample, "off")
    prompt = str(request["prompt"])
    if not prompt.endswith(intern_baseline.ASSISTANT_GENERATION_PREFIX):
        raise RuntimeError("InternVL baseline assistant suffix changed")
    prompt += CLOSED_THINK_PREFIX + str(cue)
    request = dict(request)
    request["prompt"] = prompt
    if prompt.count("</think>") != 1:
        raise RuntimeError("method prompt must contain exactly one </think>")
    return request, {
        **audit,
        "thinking_mode": "native_non_thinking_template_plus_closed_think_method_prefix",
        "method_prefix": CLOSED_THINK_PREFIX + str(cue),
        "assistant_generation_prefix": intern_baseline.ASSISTANT_GENERATION_PREFIX + CLOSED_THINK_PREFIX + str(cue),
        "method_prefix_count": prompt.count(CLOSED_THINK_PREFIX + str(cue)),
    }


def build_native_request(tokenizer, sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return intern_baseline.build_request(tokenizer, sample, "r1")


def build_non_thinking_request(
    tokenizer, sample: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ordinary (no R1 system prompt) request used by the selector.

    The selector must inspect the model's first-token distribution before the
    closed-think scaffold is appended.  Reusing the baseline's ``off`` branch
    makes this distinction explicit and prevents the selector from silently
    measuring a duplicated thinking prompt.
    """

    return intern_baseline.build_request(tokenizer, sample, "off")


def build_closed_think_selector_request(
    tokenizer, sample: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact context immediately before the selectable cue.

    The selector is intentionally evaluated after the closed-think scaffold,
    not after the ordinary non-thinking assistant prefix.  This makes its
    ``So``/``Okay`` comparison correspond to the token position that will be
    used by the method request.
    """

    request, audit = build_non_thinking_request(tokenizer, sample)
    prompt = str(request["prompt"])
    if not prompt.endswith(intern_baseline.ASSISTANT_GENERATION_PREFIX):
        raise RuntimeError("InternVL baseline assistant suffix changed")
    prompt += CLOSED_THINK_PREFIX
    request = dict(request)
    request["prompt"] = prompt
    if prompt.count("</think>") != 1:
        raise RuntimeError("selector prompt must contain exactly one </think>")
    return request, {
        **audit,
        "thinking_mode": "native_non_thinking_template_plus_closed_think_selector_prefix",
        "selector_prefix": CLOSED_THINK_PREFIX,
        "assistant_generation_prefix": intern_baseline.ASSISTANT_GENERATION_PREFIX + CLOSED_THINK_PREFIX,
        "selector_prefix_count": prompt.count(CLOSED_THINK_PREFIX),
    }


def _logprob_value(value: Any) -> float:
    """Read a vLLM Logprob object (or a compatible test double)."""

    raw = getattr(value, "logprob", value)
    if isinstance(raw, dict):
        raw = raw.get("logprob")
    if raw is None:
        raise RuntimeError(f"selector logprob entry has no logprob value: {value!r}")
    return float(raw)


def _select_pair_index(
    pair_probabilities: list[float],
    second_word_threshold: float,
) -> int:
    """Choose the second candidate only when it clears its configured gate."""

    if len(pair_probabilities) != 2:
        raise ValueError(f"expected two pair probabilities, got {pair_probabilities!r}")
    threshold = float(second_word_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"second-word threshold must be in [0, 1], got {threshold!r}")
    return 1 if float(pair_probabilities[1]) >= threshold else 0


def select_method_cue(
    llm: LLM,
    tokenizer: Any,
    sample: dict[str, Any],
    words: tuple[str, str],
    seed: int,
    metadata_factory=None,
    finalize=None,
    second_word_threshold: float = 0.5,
) -> tuple[str, dict[str, Any]]:
    """Select So/Okay from the token distribution after closed-think.

    vLLM exposes only a bounded top-logprob list in this deployment.  We
    therefore construct ``<think>\\n\\n</think>\\n\\n`` first and issue two
    one-token probes in one batch, each allowing exactly one candidate.  The
    sampled candidate is always included in its own logprob map; the returned
    value is its original (unbiased) logprob.  The two values can consequently
    be compared directly.
    """

    if len(words) != 2 or any(not str(word).strip() for word in words):
        raise ValueError(f"selector requires exactly two non-empty words, got {words!r}")
    candidate_ids: list[int] = []
    for word in words:
        encoded = list(tokenizer.encode(str(word), add_special_tokens=False))
        if len(encoded) != 1:
            raise RuntimeError(f"selector word must be one tokenizer token: {word!r} -> {encoded!r}")
        candidate_ids.append(int(encoded[0]))
    if candidate_ids[0] == candidate_ids[1]:
        raise RuntimeError(f"selector words map to the same token: {words!r} -> {candidate_ids!r}")

    request, prompt_audit = build_closed_think_selector_request(tokenizer, sample)
    probe_requests = []
    probe_samplings = []
    for token_id in candidate_ids:
        probe_request = dict(request)
        if metadata_factory is not None:
            # The dynamic worker expects the same initial request metadata as
            # a normal method request.  The selector prompt already contains
            # the closed-think scaffold, but has no selected cue yet; the
            # fused suppression path remains disabled for this one-token probe.
            probe_request["aris_suppression"] = metadata_factory()
        probe_requests.append(probe_request)
        probe_samplings.append(
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                max_tokens=1,
                min_tokens=1,
                seed=seed,
                allowed_token_ids=[int(token_id)],
                logprobs=1,
                skip_special_tokens=False,
            )
        )
    started = time.perf_counter()
    results = llm.generate(probe_requests, sampling_params=probe_samplings, use_tqdm=False)
    elapsed = time.perf_counter() - started
    if finalize is not None:
        finalize(llm, reason="selector_probe_return")
    completions = [result.outputs[0] for result in results]

    def find_sampled_logprob(completion: Any, token_id: int) -> float:
        token_logprobs = list(getattr(completion, "logprobs", None) or [])
        if not token_logprobs:
            raise RuntimeError("selector probe returned no token logprobs")
        first = token_logprobs[0] or {}
        for key in (token_id, str(token_id)):
            if key in first:
                return _logprob_value(first[key])
        raise RuntimeError(
            f"selector probe omitted candidate token {token_id}; available={list(first)[:10]!r}"
        )

    logprobs = [
        find_sampled_logprob(completion, token_id)
        for completion, token_id in zip(completions, candidate_ids)
    ]
    max_logprob = max(logprobs)
    weights = [math.exp(value - max_logprob) for value in logprobs]
    normalizer = sum(weights)
    pair_probabilities = [weight / normalizer for weight in weights]
    raw_argmax_index = 0 if logprobs[0] >= logprobs[1] else 1
    winner_index = _select_pair_index(pair_probabilities, second_word_threshold)
    selected_word = str(words[winner_index])
    probe_token_ids = [
        int((list(getattr(completion, "token_ids", None) or []) or [token_id])[0])
        for completion, token_id in zip(completions, candidate_ids)
    ]
    return selected_word, {
        "enabled": True,
        "mode": "closed_think_pair_logprob_forced_candidates",
        "candidate_words": [str(word) for word in words],
        "candidate_token_ids": candidate_ids,
        "candidate_logprobs": [round(value, 8) for value in logprobs],
        "candidate_pair_probabilities": [round(value, 8) for value in pair_probabilities],
        "selection_rule": "second_candidate_pair_probability_threshold",
        "second_word_threshold": float(second_word_threshold),
        "raw_argmax_word": str(words[raw_argmax_index]),
        "threshold_overrode_argmax": winner_index != raw_argmax_index,
        "selected_word": selected_word,
        "selected_token_id": candidate_ids[winner_index],
        "winner_margin_logprob": round(abs(logprobs[0] - logprobs[1]), 8),
        "probe_output_token_ids": probe_token_ids,
        "probe_count": 2,
        "probe_elapsed_seconds": elapsed,
        "probe_prompt_audit": prompt_audit,
    }


def _error(sample: dict[str, Any], args: argparse.Namespace, message: str) -> dict[str, Any]:
    return {
        "id": sample.get("id"),
        "global_index": sample.get("global_index"),
        "ground_truth": sample.get("ground_truth"),
        "engine": "vllm",
        "method": args.mode,
        "model": args.model_path,
        "error": message,
        "max_new_tokens": args.max_new_tokens,
    }


def run(args: argparse.Namespace) -> None:
    if args.mode == "dynamic":
        from .engine_patch import install_dynamic_aris_patch

        install_dynamic_aris_patch(require_native_fused=not args.allow_unfused)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    rows = list(baseline.read_jsonl(Path(args.manifest)))
    rows = [row for row in rows if int(row.get("global_index", 0)) % args.shard_count == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output)
    done = _completed(output) if args.resume else set()
    rows = [row for row in rows if str(row.get("id")) not in done]
    if not args.resume and output.exists():
        output.unlink()

    engine_kwargs: dict[str, Any] = {}
    finalize = None
    metadata_factory = None
    if args.mode == "dynamic":
        from .engine_patch import finalize_llm_dynamic_states, initial_request_suppression_metadata

        engine_kwargs["worker_extension_cls"] = (
            "experiment_src.internvl35_dynamic_vllm_final.worker_extension.DynamicArisWorkerExtension"
        )
        finalize = finalize_llm_dynamic_states
        metadata_factory = initial_request_suppression_metadata

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.max_images_per_prompt, "video": 0},
        **engine_kwargs,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
        skip_special_tokens=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for sample in rows:
            try:
                selector_audit: dict[str, Any] | None = None
                selected_cue_for_row: str | None = None
                if args.mode == "native":
                    request, audit = build_native_request(tokenizer, sample)
                else:
                    if args.selector_words:
                        selected_cue, selector_audit = select_method_cue(
                            llm,
                            tokenizer,
                            sample,
                            args.selector_words,
                            args.seed,
                            second_word_threshold=args.selector_second_word_threshold,
                            metadata_factory=metadata_factory if args.mode == "dynamic" else None,
                            finalize=finalize if args.mode == "dynamic" else None,
                        )
                    else:
                        selected_cue = args.method_cue
                    selected_cue_for_row = selected_cue
                    request, audit = build_method_request(tokenizer, sample, selected_cue)
                    # Keep this invariant at the actual per-item execution
                    # boundary as well as in the builder: dynamic InternVL
                    # must be ordinary user content plus one scaffold, never
                    # the R1 system prompt plus another scaffold.
                    if audit.get("system_prompt") is not None:
                        raise RuntimeError("dynamic method unexpectedly carries the R1 system prompt")
                    if str(request.get("prompt", "")).count("</think>") != 1:
                        raise RuntimeError("dynamic method prompt must contain exactly one </think>")
                    if selector_audit is not None:
                        audit["first_token_selector"] = selector_audit
                if args.mode == "dynamic":
                    request["aris_suppression"] = metadata_factory()
                started = time.perf_counter()
                result = llm.generate([request], sampling_params=[sampling], use_tqdm=False)[0]
                if finalize is not None:
                    finalize(llm, reason="internvl_item_return")
                elapsed = time.perf_counter() - started
                completion = result.outputs[0]
                token_ids = [int(value) for value in completion.token_ids]
                raw_output = completion.text
                prediction, prediction_source, parseable = baseline.extract_prediction(
                    raw_output, sample.get("options") or [], str(sample.get("ground_truth") or "")
                )
                correct = baseline.is_correct_prediction(
                    prediction, str(sample.get("ground_truth") or ""), sample.get("options") or []
                )
                prompt_token_ids = list(getattr(result, "prompt_token_ids", None) or [])
                row = {
                    "id": sample.get("id"),
                    "global_index": sample.get("global_index"),
                    "subject": sample.get("subject"),
                    "question": sample.get("question"),
                    "options": sample.get("options") or [],
                    "ground_truth": sample.get("ground_truth"),
                    "prediction": prediction,
                    "prediction_source": prediction_source,
                    "parseable": parseable,
                    "is_correct": correct,
                    "raw_output": raw_output,
                    "output_token_ids": token_ids,
                    "output_tokens": len(token_ids),
                    "input_tokens": len(prompt_token_ids),
                    "total_tokens": len(prompt_token_ids) + len(token_ids),
                    "prompt_token_ids": prompt_token_ids,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "hit_max_new_tokens": bool(completion.finish_reason == "length" or len(token_ids) >= args.max_new_tokens),
                    "time_seconds": elapsed,
                    "tokens_per_second": len(token_ids) / elapsed if elapsed > 0 else None,
                    "method": args.mode,
                    "method_cue": selected_cue_for_row,
                    "selected_method_cue": audit.get("first_token_selector", {}).get(
                        "selected_word", selected_cue_for_row
                    ),
                    "first_token_selector": audit.get("first_token_selector"),
                    "model": args.model_path,
                    "engine": "vllm",
                    "rank": args.shard_index,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "max_new_tokens": args.max_new_tokens,
                    "max_model_len": args.max_model_len,
                    "audit": audit,
                    "error": None,
                }
            except Exception as exc:
                row = _error(sample, args, f"{type(exc).__name__}: {exc}")
            _append_fsync(handle, row)
            print(json.dumps({"id": row.get("id"), "tokens": row.get("output_tokens"), "error": row.get("error")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), default="dynamic")
    parser.add_argument("--method-cue", default=DEFAULT_METHOD_CUE)
    parser.add_argument(
        "--selector-words",
        default="",
        help="Comma-separated two-word non-thinking first-token selector, e.g. So,Okay.",
    )
    parser.add_argument(
        "--selector-second-word-threshold",
        type=float,
        default=0.5,
        help="Select the second selector word only when its pair probability reaches this threshold.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-images-per-prompt", type=int, default=8)
    parser.add_argument("--allow-unfused", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.selector_words:
        parsed_words = tuple(item.strip() for item in str(args.selector_words).split(",") if item.strip())
        if len(parsed_words) != 2:
            parser.error("--selector-words must contain exactly two comma-separated words")
        args.selector_words = parsed_words
    else:
        args.selector_words = None
    if not 0.0 <= args.selector_second_word_threshold <= 1.0:
        parser.error("--selector-second-word-threshold must be in [0, 1]")
    run(args)


if __name__ == "__main__":
    main()
