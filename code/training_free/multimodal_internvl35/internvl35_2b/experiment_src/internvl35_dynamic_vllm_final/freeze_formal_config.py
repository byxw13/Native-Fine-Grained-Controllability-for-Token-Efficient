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
    "model_path": "/path/to/aris_data/Qwen3-VL-4B-Thinking",
    "manifest": "/path/to/aris_data/VLM/data/mmmuval/samples.jsonl",
    "expected_rows": 900,
    "max_new_tokens": 32768,
    "repetition_penalty": 1.05,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "sample_seed": 0,
    "samples_per_question": 1,
    "max_model_len": 40960,
    "dtype": "bfloat16",
    "shard_count": 4,
    "batch_size": 1,
}

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
    "suppression_scale": 0.3,
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
    "correct": 631,
    "accuracy": 0.7011111111111111,
    "multiple_choice_rows": 847,
    "multiple_choice_correct": 602,
    "open_rows": 53,
    "open_correct": 29,
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
    parser.add_argument("--baseline-watchdog", type=Path, required=True)
    parser.add_argument("--baseline-merge", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--signal-calibration", type=Path, required=True)
    parser.add_argument("--question-lambda-calibration", type=Path, required=True)
    parser.add_argument("--dynamic-config", type=Path, required=True)
    parser.add_argument("--method-runner", type=Path, required=True)
    parser.add_argument("--method-slurm", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--native-fa2-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    comparisons = []
    for key, expected in EXPECTED_BASELINE.items():
        actual = baseline.get(key)
        comparisons.append({"field": key, "baseline": actual, "method": expected, "match": actual == expected})
    if args.manifest.resolve() != Path(EXPECTED_BASELINE["manifest"]).resolve():
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
            "baseline result is not the frozen 631/900 run: "
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
        "baseline_watchdog": file_record(args.baseline_watchdog),
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
                    "partition": "YOUR_A6000_PARTITION",
                    "gpus": 4,
                    "global_index_range": [0, 450],
                    "worker_ranks": [0, 1, 2, 3],
                },
                {
                    "partition": "YOUR_L20_PARTITION",
                    "gpus": 4,
                    "global_index_range": [450, 900],
                    "worker_ranks": [4, 5, 6, 7],
                },
            ],
            "expected_rows_per_job": 450,
            "expected_rows_total": 900,
        },
        "allowed_method_differences": [
            "tokenizer overlay inserts <think>\\n\\n</think>\\n\\nAlright before generation",
            "per-request dynamic suppression controller and fused attention path",
            "Qwen3-VL first-100 head summary and signal calibration; question-lambda artifact is audit-only and disabled",
            "two independent four-GPU jobs split indices [0,450) and [450,900)",
            "prefix cache enabled identically in native/method performance smoke",
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
        "overlay": metadata_record(
            args.overlay,
            (
                "tokenizer_config.json",
                "tokenizer_config.original.json",
                "tokenizer_config.effective.json",
                "tokenizer_config.diff",
                "chat_template.json",
                "chat_template.original.json",
                "chat_template.diff",
                "overlay_manifest.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "processor_config.json",
            ),
        ),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
