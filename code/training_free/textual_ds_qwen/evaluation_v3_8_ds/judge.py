from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from experiment_src.evaluation_v1.evaluate_budget_results_attn_alright import (
    DEFAULT_REQUEST_TIMEOUT,
    resolve_primary_api_settings,
    resolve_review_model,
)
from experiment_src.evaluation_v1.evaluate_cascade_results_llm import evaluate_one_item


@dataclass(frozen=True)
class JudgeConfig:
    max_retries: int = 2
    retry_sleep: float = 3.0
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


class CascadeDualJudge:
    def __init__(self, config: JudgeConfig | None = None) -> None:
        self.config = config or JudgeConfig()
        primary_api_key, primary_base_url, primary_model = resolve_primary_api_settings()
        self.primary_model = primary_model
        self.review_model = resolve_review_model()
        self.primary_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)
        self.review_client = OpenAI(api_key=primary_api_key, base_url=primary_base_url)

    def evaluate_episode(self, run_name: str, item: dict[str, Any]) -> dict[str, Any]:
        result = evaluate_one_item(
            primary_client=self.primary_client,
            primary_model=self.primary_model,
            review_client=self.review_client,
            review_model=self.review_model,
            run_name=run_name,
            item=item,
            max_retries=self.config.max_retries,
            retry_sleep=self.config.retry_sleep,
            request_timeout=self.config.request_timeout,
        )
        if result.get("is_correct") is None:
            result["is_correct"] = False
            result["judge_failed"] = True
        else:
            result["judge_failed"] = False
        return result
