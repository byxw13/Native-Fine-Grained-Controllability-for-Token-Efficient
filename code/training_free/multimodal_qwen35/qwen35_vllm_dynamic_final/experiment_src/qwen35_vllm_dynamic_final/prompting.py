from __future__ import annotations

from typing import Any

from .baseline_snapshot import run_mmmu_vlmeval_prompt_p0_multisample_vllm as baseline


CLOSED_THINK_PREFIX = "<think>\n\n</think>\n\n"
FORMAL_METHOD_CUE = "To"


def build_qwen35_request(
    processor,
    sample: dict[str, Any],
    image_token_id: int | None,
    *,
    enable_thinking: bool,
    assistant_cue: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Baseline-identical MMMU input with the model's own thinking switch.

    No tokenizer file or rendered prompt is rewritten.  For the method path the
    model chat template emits its native closed-think scaffold and only the
    frozen cue text is appended afterwards.
    """
    images = baseline.load_images(sample.get("image_paths") or [])
    content, content_audit = baseline.build_interleaved_content(sample, images)
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    expected = "<think>\n" if enable_thinking else CLOSED_THINK_PREFIX
    if not prompt.endswith(expected):
        raise RuntimeError(
            f"Qwen3.5 template suffix mismatch: expected {expected!r}, got {prompt[-120:]!r}"
        )
    if assistant_cue:
        if enable_thinking:
            raise ValueError("assistant cue is only valid with the non-thinking template")
        prompt += assistant_cue

    image_patch_size = int(getattr(processor.image_processor, "patch_size", 16))
    if baseline.process_vision_info is not None:
        image_inputs, video_inputs = baseline.process_vision_info(
            messages,
            image_patch_size=image_patch_size,
        )
        proc_inputs = processor(
            text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt"
        )
        mm_images = image_inputs
    else:
        mm_images = [
            images[index - 1]
            for index in content_audit["image_occurrence_order"]
        ]
        proc_inputs = processor(text=[prompt], images=mm_images, return_tensors="pt")

    input_ids = proc_inputs["input_ids"]
    input_tokens = int(input_ids.shape[-1])
    visual_tokens = (
        int((input_ids == image_token_id).sum().item())
        if image_token_id is not None
        else 0
    )
    processed_images = baseline.build_processed_image_audit(
        processor,
        mm_images,
        proc_inputs,
        content_audit["image_occurrence_order"],
    )
    audited_visual_tokens = sum(item["visual_tokens"] or 0 for item in processed_images)
    if processed_images and audited_visual_tokens != visual_tokens:
        raise ValueError(
            f"Per-image visual token audit mismatch: {audited_visual_tokens} != {visual_tokens}"
        )

    request: dict[str, Any] = {"prompt": prompt}
    if mm_images:
        request["multi_modal_data"] = {
            "image": mm_images if len(mm_images) != 1 else mm_images[0]
        }
    return request, {
        "input_tokens": input_tokens,
        "visual_tokens": visual_tokens,
        "image_patch_size": image_patch_size,
        "image_count": len(images),
        "image_occurrence_count": len(content_audit["image_occurrence_order"]),
        "image_order": "full_rendered_prompt_placeholder_order",
        "prompt_style": baseline.PROMPT_STYLE,
        "assistant_generation_prefix": expected + assistant_cue,
        "template_enable_thinking": enable_thinking,
        "assistant_cue": assistant_cue,
        "processed_images": processed_images,
        **content_audit,
    }
