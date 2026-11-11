from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


EXPECTED_BASELINE = {
    "model_path": "/path/to/aris_data/Qwen3.5-4B",
    "dataset": "MMMU/MMMU",
    "split": "validation",
    "expected_rows": 900,
    "enable_thinking": True,
    "native_generation_prefix": "<think>\n",
    "manual_generation_prefix": None,
    "max_new_tokens": 32768,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "sample_seed": 0,
    "samples_per_question": 1,
    "max_model_len": 40960,
    "dtype": "bfloat16",
    "do_sample": True,
    "engine": "vllm",
    "prompt_style": "vlmevalkit_mmmu_concise_mc_answer_suffix_sampled",
    "multiple_choice_prompt_suffix": "At the end, output the final answer as: Answer: <option letter>.",
    "open_prompt_suffix": None,
    "image_order": "full_rendered_prompt_placeholder_order",
    "allow_unused_manifest_images": True,
    "shard_count": 4,
    "batch_size": 1,
    "tensor_parallel_size": 1,
    "gpu_deployment": "one independent model instance per GPU",
    "trajectory_persistence": "one flushed JSONL row per completed sample",
    "multiple_choice_judge": "mc_6356 broad extractor",
    "open_judge": "rule judge followed by one API judge call on rule failures",
    "judge_execution_site": "remote login node",
}

EXPECTED_MANIFEST = "/path/to/aris_data/VLM/data/mmmuval/samples.jsonl"

EXPECTED_DYNAMIC = {
    "max_new_tokens": 32768,
    "initial_suppression": 0.1,
    "suppression_upper_bound": 1.0,
    "cue_threshold": 0.1,
    "cue_gate_threshold": 0.1,
    "min_tokens_before_cue": 0,
    "enable_runtime_cue": False,
    "entropy_upper_bound": 1.0,
    "suppression_curve": "exp_saturating",
    "power_exponent": 0.65,
    "exp_beta": 2.0,
    "suppression_scale": 0.8,
    "raw_history_size": 10,
    "k_smoothing_initial_rho": 1.0,
    "k_smoothing_late_rho": 1.0,
    "k_smoothing_initial_checkpoints": 6,
    "upgrade_newline_span": 1,
    "downgrade_newline_span": 10,
    "downgrade_confirm_windows": 5,
    "post_downgrade_upgrade_freeze_newlines": 0,
    "tail_soft_ratio": 0.99999,
    "tail_full_ratio": 1.0,
    "tail_exp_beta": 3.0,
    "tail_threshold_max": 1.0,
    "tail_signal_trigger_ratio": 0.8,
    "tail_signal_wait_token_limit": 300,
    "tail_signal_next_checkpoint_only": False,
    "final_answer_prefix_trigger_ratio": 0.98,
    "rethink_cue_trigger_tokens": 0,
    "end_think_to_l1_lock": True,
    "end_think_mask_newlines": 0,
    "end_think_mask_tokens": 1,
    "end_think_mask_forever": False,
    "checkpoint_zero": False,
    "assistant_prefix_checkpoint": False,
    "calibration_only": False,
    "dynamic_question_lambda": False,
    "high_low_high_question_lambda": False,
    "linear_down_up_question_lambda": False,
    "question_lambda_calibration": None,
}

EXPECTED_FORCED_TEXTS = {
    "cue_text": "",
    "rethink_cue_text": "",
    "tail_signal_text": "</think>\n",
    "end_think_bridge_text": ".",
    "final_answer_prefix_text": "the answer is ",
}

EXPECTED_BASELINE_RESULT = {
    "expected_rows": 900,
    "judged_rows": 900,
    "correct": 684,
    "accuracy": 0.76,
    "multiple_choice_rows": 847,
    "multiple_choice_correct": 649,
    "open_rows": 53,
    "open_correct": 35,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def metadata_record(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    files = [file_record(root / name) for name in names if (root / name).is_file()]
    if not files:
        raise RuntimeError(f"no metadata files found under {root}")
    digest = hashlib.sha256()
    for row in files:
        digest.update(row["sha256"].encode("ascii"))
    return {"root": str(root.resolve()), "metadata_sha256": digest.hexdigest(), "files": files}


def source_tree_record(root: Path) -> dict[str, Any]:
    suffixes = {".py", ".sh", ".sbatch", ".patch", ".md"}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
    )
    if not paths:
        raise RuntimeError(f"no method source files found under {root}")
    digest = hashlib.sha256()
    files = []
    for path in paths:
        relative = str(path.relative_to(root))
        row = {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        digest.update(relative.encode("utf-8") + b"\0" + row["sha256"].encode("ascii") + b"\n")
        files.append(row)
    return {
        "root": str(root.resolve()),
        "source_tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def environment_record() -> dict[str, Any]:
    distributions = ("torch", "transformers", "vllm", "flash-attn", "triton", "qwen-vl-utils")
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "packages": versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-judge-summary", type=Path, required=True)
    parser.add_argument("--baseline-runner", type=Path, required=True)
    parser.add_argument("--baseline-judge", type=Path, required=True)
    parser.add_argument("--baseline-merge", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--signal-calibration", type=Path, required=True)
    parser.add_argument("--question-lambda-calibration", type=Path, required=True)
    parser.add_argument("--dynamic-config", type=Path, required=True)
    parser.add_argument("--method-runner", type=Path, required=True)
    parser.add_argument("--method-slurm", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--native-fa2-library", type=Path, required=True)
    parser.add_argument("--formal-shard-count", type=int, choices=(4, 8), default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    comparisons = []
    for key, expected in EXPECTED_BASELINE.items():
        actual = baseline.get(key)
        comparisons.append({"field": key, "baseline": actual, "method": expected, "match": actual == expected})
    if args.manifest.resolve() != Path(EXPECTED_MANIFEST).resolve():
        raise RuntimeError(f"unexpected manifest path: {args.manifest}")
    if args.model.resolve() != Path(EXPECTED_BASELINE["model_path"]).resolve():
        raise RuntimeError(f"unexpected model path: {args.model}")
    if not all(row["match"] for row in comparisons):
        raise RuntimeError(f"formal config differs from baseline: {[row for row in comparisons if not row['match']]}")

    baseline_result = json.loads(args.baseline_judge_summary.read_text(encoding="utf-8"))
    baseline_result_comparisons = [
        {
            "field": key,
            "actual": baseline_result.get(key),
            "expected": expected,
            "match": baseline_result.get(key) == expected,
        }
        for key, expected in EXPECTED_BASELINE_RESULT.items()
    ]
    if not all(row["match"] for row in baseline_result_comparisons):
        raise RuntimeError(
            "baseline result is not the frozen Qwen3.5 684/900 run: "
            f"{[row for row in baseline_result_comparisons if not row['match']]}"
        )

    dynamic = json.loads(args.dynamic_config.read_text(encoding="utf-8"))
    dynamic_comparisons = [
        {
            "field": key,
            "actual": dynamic.get(key),
            "expected": expected,
            "match": dynamic.get(key) == expected,
        }
        for key, expected in EXPECTED_DYNAMIC.items()
    ]
    dynamic_comparisons.extend(
        {
            "field": f"forced_texts.{key}",
            "actual": (dynamic.get("forced_texts") or {}).get(key),
            "expected": expected,
            "match": (dynamic.get("forced_texts") or {}).get(key) == expected,
        }
        for key, expected in EXPECTED_FORCED_TEXTS.items()
    )
    dynamic_comparisons.append(
        {
            "field": "signal_calibration",
            "actual": "present" if dynamic.get("signal_calibration") else None,
            "expected": "present",
            "match": bool(dynamic.get("signal_calibration")),
        }
    )
    if not all(row["match"] for row in dynamic_comparisons):
        raise RuntimeError(
            "dynamic config differs from Qwen3-4B Math500 target: "
            f"{[row for row in dynamic_comparisons if not row['match']]}"
        )

    artifacts = {
        "baseline_config": file_record(args.baseline_config),
        "baseline_judge_summary": file_record(args.baseline_judge_summary),
        "baseline_runner": file_record(args.baseline_runner),
        "baseline_judge": file_record(args.baseline_judge),
        "baseline_merge": file_record(args.baseline_merge),
        "manifest": file_record(args.manifest),
        "head_summary": file_record(args.head_summary),
        "signal_calibration": file_record(args.signal_calibration),
        "question_lambda_calibration": file_record(args.question_lambda_calibration),
        "dynamic_config": file_record(args.dynamic_config),
        "method_runner": file_record(args.method_runner),
        "method_slurm": file_record(args.method_slurm),
        "method_source_tree": source_tree_record(args.method_root),
        "native_fa2_library": file_record(args.native_fa2_library),
    }
    report = {
        "status": "ready_for_human_confirmation",
        "environment": environment_record(),
        "baseline_alignment": comparisons,
        "baseline_result_alignment": baseline_result_comparisons,
        "dynamic_alignment": dynamic_comparisons,
        "execution_topology": {
            "jobs": [
                {
                    "partition": "YOUR_A100_PARTITION",
                    "gpus": args.formal_shard_count,
                    "global_index_range": [0, 900],
                    "worker_ranks": list(range(args.formal_shard_count)),
                    "tensor_parallel_size_per_worker": 1,
                }
            ],
            "expected_rows_per_job": 900,
            "expected_rows_total": 900,
        },
        "allowed_method_differences": [
            "the model's own non-thinking chat template emits <think>\\n\\n</think>\\n\\n, then the method appends To",
            "per-request dynamic suppression controller and fused attention path",
            "Qwen3.5 MMMUVal first-100 head summary and signal calibration; question-lambda artifact is audit-only and disabled",
            f"one {args.formal_shard_count}-GPU A100 job with TP=1 per model instance",
            "prefix-cache setting held identical (disabled) in native/method performance smoke",
            "trace fsync precedes prediction-row fsync and streaming judge consumption",
        ],
        "model": metadata_record(
            args.model,
            (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                *(path.name for path in sorted(args.model.glob("*.safetensors"))),
            ),
        ),
        "processor": metadata_record(
            args.processor,
            (
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.json",
            ),
        ),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
