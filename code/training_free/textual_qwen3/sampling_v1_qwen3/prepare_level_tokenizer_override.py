from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TOKENIZER_FILE_NAMES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


def patch_chat_template(template: str, assistant_suffix: str) -> str:
    if not template or not assistant_suffix.strip():
        return template
    if assistant_suffix in template:
        return template
    if "</think>\\n\\n" in template:
        prefix, suffix = template.rsplit("</think>\\n\\n", 1)
        return prefix + "</think>\\n\\n" + assistant_suffix.removeprefix("<think>\n\n</think>\n\n") + suffix
    return template + assistant_suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assistant-suffix", required=True)
    args = parser.parse_args()

    base_model_dir = Path(args.base_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in TOKENIZER_FILE_NAMES:
        src = base_model_dir / file_name
        if src.exists():
            shutil.copy2(src, output_dir / file_name)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        payload["chat_template"] = patch_chat_template(str(payload.get("chat_template", "")), args.assistant_suffix)
        tokenizer_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
