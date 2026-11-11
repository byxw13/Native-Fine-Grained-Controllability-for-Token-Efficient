from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers
import vllm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from .baseline_snapshot import run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline
from .prompting import CLOSED_THINK_PREFIX, FORMAL_METHOD_CUE, build_qwen35_request


METHOD_PREFIX = CLOSED_THINK_PREFIX + FORMAL_METHOD_CUE


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _serialize_logprobs(rows) -> list[dict[str, Any]]:
    serialized = []
    for row in rows or []:
        if row is None:
            serialized.append({})
            continue
        serialized.append(
            {
                str(int(token_id)): {
                    "logprob": float(value.logprob),
                    "rank": None if value.rank is None else int(value.rank),
                    "decoded_token": value.decoded_token,
                }
                for token_id, value in sorted(row.items(), key=lambda item: int(item[0]))
            }
        )
    return serialized


def _prepare(processor, manifest: Path, indices: list[int], mode: str, initial_request_metadata=None):
    wanted = set(indices)
    rows = [row for row in baseline.read_jsonl(manifest) if int(row["global_index"]) in wanted]
    rows.sort(key=lambda row: indices.index(int(row["global_index"])))
    image_token_id = baseline.get_image_token_id(processor)
    requests = []
    audits = []
    for row in rows:
        request, audit = build_qwen35_request(
            processor,
            row,
            image_token_id,
            enable_thinking=mode == "native",
            assistant_cue="" if mode == "native" else FORMAL_METHOD_CUE,
        )
        if mode == "dynamic":
            request["aris_suppression"] = initial_request_metadata()
        requests.append(request)
        audits.append(
            {
                "id": row["id"],
                "global_index": row["global_index"],
                "question_type": row.get("question_type"),
                "method_prefix_count": str(request["prompt"]).count(METHOD_PREFIX),
                "prompt_sha256": hashlib.sha256(
                    str(request["prompt"]).encode("utf-8")
                ).hexdigest(),
                "prompt_token_ids": [
                    int(value)
                    for value in processor.tokenizer.encode(
                        str(request["prompt"]), add_special_tokens=False
                    )
                ],
                **audit,
            }
        )
    return rows, requests, audits


def _lifecycle_smoke(
    llm: LLM,
    requests: list[dict[str, Any]],
    output: Path,
    finalize_dynamic=None,
    initial_request_metadata=None,
) -> None:
    if len(requests) < 2:
        raise RuntimeError("lifecycle smoke requires at least two requests")
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=16, skip_special_tokens=False)
    initial_budget = float(initial_request_metadata()["budget"]) if initial_request_metadata else 0.0
    ids = [
        f"900000::aris_budget={initial_budget:.6g}::aris_end=-1::aris_enabled=1",
        f"900001::aris_budget={initial_budget:.6g}::aris_end=-1::aris_enabled=1",
    ]
    for request_id, request in zip(ids, requests[:2]):
        clean = dict(request)
        clean.pop("aris_suppression", None)
        llm.llm_engine.add_request(request_id, clean, sampling)
    step = 0
    finished: dict[str, str | None] = {}
    aborted = False
    while llm.llm_engine.has_unfinished_requests():
        for item in llm.llm_engine.step():
            if item.finished:
                finished[str(item.request_id)] = item.outputs[0].finish_reason
        step += 1
        if step == 3 and not aborted:
            llm.llm_engine.abort_request([ids[1]])
            aborted = True
        if step > 64:
            raise RuntimeError("lifecycle smoke exceeded 64 engine steps")
    finalize_report = (
        finalize_dynamic(llm, reason="lifecycle_smoke_return") if finalize_dynamic is not None else None
    )
    report = {
        "request_ids": ids,
        "steps": step,
        "aborted_request_id": ids[1],
        "abort_called": aborted,
        "normal_finished": ids[0] in finished,
        "normal_finish_reason": finished.get(ids[0]),
        "aborted_absent_from_finished_outputs": ids[1] not in finished,
        "worker_finalize_report": finalize_report,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("native", "prefix", "dynamic"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--indices", default="0,30,55,73,203,740")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, nargs="+", default=(32, 128, 512))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--logprobs", type=int, default=0)
    parser.add_argument("--lifecycle-smoke", action="store_true")
    args = parser.parse_args()

    dynamic = args.mode == "dynamic"
    finalize_dynamic = None
    initial_request_metadata = None
    if dynamic:
        from .engine_patch import (
            finalize_llm_dynamic_states,
            initial_request_suppression_metadata,
            install_dynamic_aris_patch,
        )

        install_dynamic_aris_patch(require_native_fused=True)
        finalize_dynamic = finalize_llm_dynamic_states
        initial_request_metadata = initial_request_suppression_metadata
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    indices = [int(item) for item in args.indices.split(",") if item.strip()]
    rows, requests, audits = _prepare(
        processor,
        args.manifest,
        indices,
        args.mode,
        initial_request_metadata,
    )
    expected = 0 if args.mode == "native" else 1
    counts = [int(audit["method_prefix_count"]) for audit in audits]
    if any(count != expected for count in counts):
        raise RuntimeError(f"unexpected method-prefix counts for {args.mode}: {counts}")

    engine_kwargs = {}
    if dynamic:
        engine_kwargs["worker_extension_cls"] = (
            "experiment_src.qwen35_vllm_dynamic_final.worker_extension.DynamicArisWorkerExtension"
        )
    llm = LLM(
        model=args.model,
        tokenizer=args.processor,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 8},
        **engine_kwargs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for max_tokens in args.max_tokens:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(max_tokens),
            repetition_penalty=1.05,
            logprobs=args.logprobs or None,
            skip_special_tokens=False,
        )
        for warmup_idx in range(args.warmup):
            llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            if finalize_dynamic is not None:
                finalize_dynamic(llm, reason=f"vl_t{max_tokens}_warmup_{warmup_idx}_return")
        runs = []
        saved = []
        finalize_reports = []
        for repeat in range(args.repeats):
            _sync()
            start = time.perf_counter()
            outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=False)
            _sync()
            generate_returned = time.perf_counter()
            if finalize_dynamic is not None:
                finalize_reports.append(
                    finalize_dynamic(llm, reason=f"vl_t{max_tokens}_repeat_{repeat}_return")
                )
            finalized = time.perf_counter()
            _sync()
            elapsed = time.perf_counter() - start
            output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            generate_elapsed = generate_returned - start
            runs.append(
                {
                    "repeat": repeat,
                    "wall_seconds": generate_elapsed,
                    "generate_seconds": generate_elapsed,
                    "finalize_seconds": finalized - generate_returned,
                    "end_to_end_wall_seconds": elapsed,
                    "output_tokens": output_tokens,
                    "throughput_tokens_per_second": output_tokens / generate_elapsed,
                    "end_to_end_throughput_tokens_per_second": output_tokens / elapsed,
                }
            )
            saved.append(
                [
                    {
                        "request_id": output.request_id,
                        "token_ids": [int(item) for item in output.outputs[0].token_ids],
                        "text": output.outputs[0].text,
                        "finish_reason": output.outputs[0].finish_reason,
                        "top5_logprobs": _serialize_logprobs(output.outputs[0].logprobs),
                    }
                    for output in outputs
                ]
            )
        report = {
            "mode": args.mode,
            "max_tokens": int(max_tokens),
            "indices": indices,
            "ids": [row["id"] for row in rows],
            "audits": audits,
            "runs": runs,
            "median_throughput_tokens_per_second": statistics.median(
                run["throughput_tokens_per_second"] for run in runs
            ),
            "median_end_to_end_throughput_tokens_per_second": statistics.median(
                run["end_to_end_throughput_tokens_per_second"] for run in runs
            ),
            "outputs": saved,
            "worker_finalize_reports": finalize_reports,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "vllm": vllm.__version__,
            },
        }
        (args.output_dir / f"t{max_tokens}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2), flush=True)
    if args.lifecycle_smoke:
        _lifecycle_smoke(
            llm,
            requests,
            args.output_dir / "lifecycle_abort.json",
            finalize_dynamic=finalize_dynamic,
            initial_request_metadata=initial_request_metadata,
        )


if __name__ == "__main__":
    main()
