from __future__ import annotations

import argparse
import difflib
import json
import shutil
from pathlib import Path
from typing import Any


GENERATION_CONTENT_BY_PROFILE = {
    "method": "<think>\n\n</think>\n\nAlright",
    "closed_think": "<think>\n\n</think>\n\n",
}


def _replace_generation_branch(template: str, generation_content: str) -> str:
    marker = "{%- if add_generation_prompt %}"
    start = template.rfind(marker)
    if start < 0:
        raise RuntimeError("chat_template has no add_generation_prompt branch")
    tail = template[start:]
    if tail.count(marker) != 1 or not tail.rstrip().endswith("{%- endif %}"):
        raise RuntimeError("unable to identify a unique final generation branch")
    full_prefix = "<|im_start|>assistant\n" + generation_content
    jinja_literal = full_prefix.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
    replacement = (
        "{%- if add_generation_prompt %}\n"
        f"    {{{{- '{jinja_literal}' }}}}\n"
        "{%- endif %}"
    )
    result = template[:start] + replacement
    if not result.endswith(replacement):
        raise RuntimeError("effective template does not end in the requested generation branch")
    return result


def build_overlay(model_dir: Path, overlay_dir: Path, generation_profile: str = "method") -> dict[str, Any]:
    if generation_profile not in GENERATION_CONTENT_BY_PROFILE:
        raise ValueError(f"unsupported generation profile: {generation_profile!r}")
    generation_content = GENERATION_CONTENT_BY_PROFILE[generation_profile]
    generation_prefix = "<|im_start|>assistant\n" + generation_content
    model_dir = model_dir.resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)
    source_config = model_dir / "tokenizer_config.json"
    original = json.loads(source_config.read_text(encoding="utf-8"))
    effective = dict(original)
    effective["chat_template"] = _replace_generation_branch(
        str(original["chat_template"]), generation_content
    )
    source_chat_template = model_dir / "chat_template.json"
    original_chat_payload: dict[str, Any] | None = None
    effective_chat_payload: dict[str, Any] | None = None
    if source_chat_template.is_file():
        original_chat_payload = json.loads(source_chat_template.read_text(encoding="utf-8"))
        effective_chat_payload = dict(original_chat_payload)
        # AutoProcessor gives this standalone file precedence over the copy in
        # tokenizer_config.json.  Keep both metadata sources byte-equivalent in
        # meaning so the requested tokenizer-level overlay is actually used.
        effective_chat_payload["chat_template"] = effective["chat_template"]

    copied: list[str] = []
    for source in sorted(model_dir.iterdir()):
        if not source.is_file():
            continue
        name = source.name
        if name in {"tokenizer_config.json", "chat_template.json"}:
            continue
        if (
            name.endswith(".json")
            or name.endswith(".jinja")
            or name.startswith(("tokenizer", "vocab", "merges", "preprocessor", "video_preprocessor"))
        ):
            shutil.copy2(source, overlay_dir / name)
            copied.append(name)

    original_text = json.dumps(original, ensure_ascii=False, indent=2) + "\n"
    effective_text = json.dumps(effective, ensure_ascii=False, indent=2) + "\n"
    (overlay_dir / "tokenizer_config.original.json").write_text(original_text, encoding="utf-8")
    (overlay_dir / "tokenizer_config.effective.json").write_text(effective_text, encoding="utf-8")
    (overlay_dir / "tokenizer_config.json").write_text(effective_text, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            effective_text.splitlines(keepends=True),
            fromfile=str(source_config),
            tofile=str(overlay_dir / "tokenizer_config.json"),
        )
    )
    (overlay_dir / "tokenizer_config.diff").write_text(diff, encoding="utf-8")
    if original_chat_payload is not None and effective_chat_payload is not None:
        original_chat_text = json.dumps(original_chat_payload, ensure_ascii=False, indent=2) + "\n"
        effective_chat_text = json.dumps(effective_chat_payload, ensure_ascii=False, indent=2) + "\n"
        (overlay_dir / "chat_template.original.json").write_text(
            original_chat_text, encoding="utf-8"
        )
        (overlay_dir / "chat_template.json").write_text(effective_chat_text, encoding="utf-8")
        chat_diff = "".join(
            difflib.unified_diff(
                original_chat_text.splitlines(keepends=True),
                effective_chat_text.splitlines(keepends=True),
                fromfile=str(source_chat_template),
                tofile=str(overlay_dir / "chat_template.json"),
            )
        )
        (overlay_dir / "chat_template.diff").write_text(chat_diff, encoding="utf-8")

    manifest = {
        "source_model_dir": str(model_dir),
        "overlay_dir": str(overlay_dir.resolve()),
        "generation_profile": generation_profile,
        "generation_content": generation_content,
        "generation_prefix": generation_prefix,
        "generation_branch_ends_with_prefix": effective["chat_template"].rstrip().endswith(
            "{%- endif %}"
        ),
        "standalone_chat_template_patched": effective_chat_payload is not None,
        "copied_files": copied,
    }
    (overlay_dir / "overlay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-profile",
        choices=tuple(GENERATION_CONTENT_BY_PROFILE),
        default="method",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_overlay(args.model_dir, args.overlay_dir, args.generation_profile),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
