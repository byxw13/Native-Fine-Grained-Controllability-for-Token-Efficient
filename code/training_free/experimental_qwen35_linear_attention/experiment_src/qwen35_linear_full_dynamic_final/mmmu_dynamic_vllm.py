from __future__ import annotations

import os
import sys

from experiment_src.qwen35_vllm_dynamic_final import mmmu_dynamic_vllm as base

from . import engine_patch as linear_engine_patch


_OriginalLLM = base.LLM


def _linear_llm(*args, **kwargs):
    if kwargs.get("worker_extension_cls"):
        kwargs["worker_extension_cls"] = (
            "experiment_src.qwen35_linear_full_dynamic_final.worker_extension."
            "DynamicArisLinearWorkerExtension"
        )
    if os.environ.get("ARIS_LINEAR_FORCE_TORCH", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("ARIS_LINEAR_TRACE_PATH"):
        # The diagnostic implementation performs CPU trace serialization and
        # must never be captured into a CUDA graph.  The formal Triton path
        # leaves eager mode disabled and remains graph compatible.
        kwargs["enforce_eager"] = True
    return _OriginalLLM(*args, **kwargs)


def install_frontend_overrides() -> None:
    # base.run imports its engine patch lazily. Redirect that exact package key
    # while preserving the mature prompt/streaming/judge implementation.
    sys.modules["experiment_src.qwen35_vllm_dynamic_final.engine_patch"] = linear_engine_patch
    base.LLM = _linear_llm


def main() -> None:
    install_frontend_overrides()
    base.main()


if __name__ == "__main__":
    main()
