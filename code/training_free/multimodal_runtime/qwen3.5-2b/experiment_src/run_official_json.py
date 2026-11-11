#!/usr/bin/env python3
"""Run Qwen3.5-2B ARIS with the official JSON multiple-choice prompt.

Only the multiple-choice prompt/output contract is overridden.  The dynamic
controller, method prefix, sampling settings, Full/Linear suppression, causal
convolution replacement, tail signal, and Answer bridge remain owned by the
existing formal ARIS frontend.
"""

from __future__ import annotations

import re

from experiment_src.qwen35_vllm_dynamic_final.baseline_snapshot import (
    run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline,
)


OFFICIAL_JSON_FINAL_FORMAT = (
    'At the end, please show your choice in the `answer` field with only the '
    'choice letter, e.g., {"answer": "C"}.'
)
OFFICIAL_JSON_PROMPT_STYLE = "qwen35_official_json_mc_answer_suffix"

baseline.MULTIPLE_CHOICE_FINAL_FORMAT = OFFICIAL_JSON_FINAL_FORMAT
baseline.PROMPT_STYLE = OFFICIAL_JSON_PROMPT_STYLE

_original_extract_mc_prediction = baseline.extract_mc_prediction


def _extract_mc_prediction_json_first(
    text: str, option_count: int
) -> tuple[str | None, str]:
    valid = baseline.LETTERS[:option_count]
    region = baseline.answer_region(text)
    matches = re.findall(
        r'["\']answer["\']\s*:\s*["\']([A-Z])["\']',
        region,
        flags=re.I,
    )
    for match in reversed(matches):
        letter = match.upper()
        if letter in valid:
            return letter, "official_json_answer"
    return _original_extract_mc_prediction(text, option_count)


baseline.extract_mc_prediction = _extract_mc_prediction_json_first

# Import only after applying the baseline-module override: both the ARIS
# frontend and prompting module retain the same shared module object.
from experiment_src.qwen35_linear_full_dynamic_final import (  # noqa: E402
    mmmu_dynamic_async_vllm,
)


def _preflight() -> None:
    sample = {
        "question": "Which option is correct?",
        "options": ["first", "second"],
    }
    rendered = baseline.build_rendered_prompt(sample)
    if OFFICIAL_JSON_FINAL_FORMAT not in rendered:
        raise RuntimeError("official JSON instruction is absent from rendered prompt")
    if "Answer: <option letter>" in rendered:
        raise RuntimeError("legacy concise answer suffix remains in rendered prompt")
    if baseline.PROMPT_STYLE != OFFICIAL_JSON_PROMPT_STYLE:
        raise RuntimeError("official JSON prompt style override was not installed")


if __name__ == "__main__":
    _preflight()
    mmmu_dynamic_async_vllm.main()
