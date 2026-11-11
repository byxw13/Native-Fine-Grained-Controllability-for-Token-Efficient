# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils import hf_tokenizer
from verl.utils.aris_rollout_suppression import (
    append_alright_suffix,
    build_request_id,
    find_end_think_index,
    maybe_dump_rollout_debug_sample,
    sup_rollout_train_prompt_rewrite_enabled,
    sup_train_rewrite_suffix_text,
)
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.rollout_sup_tokenizer_path = os.getenv("ARIS_ROLLOUT_SUP_TOKENIZER_PATH", "").strip()
        self.rollout_sup_tokenizer = None
        if self.rollout_sup_tokenizer_path:
            trust_remote_code = self.data_config.get("trust_remote_code", False)
            self.rollout_sup_tokenizer = hf_tokenizer(
                self.rollout_sup_tokenizer_path,
                trust_remote_code=trust_remote_code,
            )

    async def _apply_text_chat_template(self, tokenizer, messages: list[dict]) -> list[int]:
        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                tokenizer,
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return normalize_token_ids(tokenized_prompt)

    async def _apply_rollout_chat_template(
        self,
        messages: list[dict],
        images,
        videos,
        tokenizer_mode: str,
    ) -> tuple[list[int], Any, str]:
        if tokenizer_mode == "sup":
            if images is not None or videos is not None:
                raise ValueError("ARIS sup rollout tokenizer override only supports text-only prompts.")
            if self.rollout_sup_tokenizer is None:
                raise ValueError("ARIS_ROLLOUT_SUP_TOKENIZER_PATH must be set for aris_rollout_tokenizer='sup'.")
            prompt_ids = await self._apply_text_chat_template(self.rollout_sup_tokenizer, messages)
            return prompt_ids, self.rollout_sup_tokenizer, self.rollout_sup_tokenizer_path

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )
        return prompt_ids, self.tokenizer, os.getenv("ARIS_TRAIN_TOKENIZER_PATH", "")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        aris_variant = kwargs.get("aris_rollout_variant", None)
        aris_budget = kwargs.get("aris_suppression_budget", None)
        aris_tokenizer_mode = str(kwargs.get("aris_rollout_tokenizer", "default") or "default")
        prompt_ids, active_tokenizer, active_tokenizer_path = await self._apply_rollout_chat_template(
            messages,
            images=images,
            videos=videos,
            tokenizer_mode=aris_tokenizer_mode,
        )
        if kwargs.get("aris_alright_prompt", False):
            prompt_ids = append_alright_suffix(active_tokenizer, prompt_ids, prompt_length=self.prompt_length)
        aris_end_think_index = find_end_think_index(prompt_ids) if aris_budget is not None else None
        request_id = build_request_id(
            uuid4().hex,
            variant=aris_variant,
            budget=aris_budget,
            end_think_index=aris_end_think_index,
            enabled=aris_end_think_index is not None and aris_end_think_index >= 0,
        )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        train_prompt_ids = prompt_ids
        train_tokenizer = active_tokenizer
        train_tokenizer_path = active_tokenizer_path
        train_rewrite_applied = False
        response_ids = output.token_ids[: self.response_length]
        response_logprobs = output.log_probs[: self.response_length] if output.log_probs else None
        train_response_prefix_text = ""
        if aris_tokenizer_mode == "sup" and sup_rollout_train_prompt_rewrite_enabled():
            train_prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
            )
            train_tokenizer = self.tokenizer
            train_tokenizer_path = os.getenv("ARIS_TRAIN_TOKENIZER_PATH", "")
            train_response_prefix_text = sup_train_rewrite_suffix_text()
            response_text = await self.loop.run_in_executor(
                None,
                lambda: active_tokenizer.decode(response_ids),
            )
            train_response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                    f"{train_response_prefix_text}{response_text}",
                    add_special_tokens=False,
                ),
            )
            response_ids = normalize_token_ids(train_response_ids)[: self.response_length]
            response_logprobs = None
            train_rewrite_applied = True

        response_mask = [1] * len(response_ids)
        maybe_dump_rollout_debug_sample(
            tokenizer=active_tokenizer,
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            raw_prompt=messages,
            variant=aris_variant,
            budget=aris_budget,
            tokenizer_mode=aris_tokenizer_mode,
            tokenizer_path=active_tokenizer_path,
            request_id=request_id,
            end_think_index=aris_end_think_index,
            train_tokenizer=train_tokenizer,
            train_prompt_ids=train_prompt_ids,
            train_tokenizer_path=train_tokenizer_path,
            train_rewrite_applied=train_rewrite_applied,
            train_response_ids=response_ids,
            train_response_mask=response_mask,
            train_response_logprobs_available=response_logprobs is not None,
        )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        metrics["aris_sampling_prompt_len"] = len(prompt_ids)
        metrics["aris_train_prompt_len"] = len(train_prompt_ids)
        metrics["aris_train_response_len"] = len(response_ids)
        metrics["aris_train_prompt_rewritten"] = 1 if train_rewrite_applied else 0
        metrics["aris_response_logprobs_available"] = 1 if response_logprobs is not None else 0
        routed_experts = None
        if output.routed_experts is not None and not train_rewrite_applied:
            routed_experts = output.routed_experts[: len(train_prompt_ids) + self.response_length]

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=train_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        if aris_variant:
            output.extra_fields.update(
                {
                    "request_id": request_id,
                    "aris_rollout_variant": aris_variant,
                    "aris_suppression_budget": aris_budget,
                    "aris_rollout_tokenizer": aris_tokenizer_mode,
                    "aris_sampling_prompt_len": len(prompt_ids),
                    "aris_train_prompt_len": len(train_prompt_ids),
                    "aris_train_response_len": len(response_ids),
                    "aris_train_prompt_rewritten": train_rewrite_applied,
                    "aris_response_logprobs_available": response_logprobs is not None,
                }
            )

        return output
