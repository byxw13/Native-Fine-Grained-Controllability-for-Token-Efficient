from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .freeze_formal_config import environment_record, sha256, source_tree_record


def _verify_file(record: dict[str, Any], errors: list[str], label: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        errors.append(f"{label}: missing {path}")
        return
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_bytes != int(record["bytes"]):
        errors.append(f"{label}: byte count changed for {path}")
    if actual_sha256 != record["sha256"]:
        errors.append(f"{label}: sha256 changed for {path}")


def _verify_metadata(record: dict[str, Any], errors: list[str], label: str) -> None:
    for index, file_record in enumerate(record.get("files") or []):
        _verify_file(file_record, errors, f"{label}.files[{index}]")


def _fsync_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.report.read_text(encoding="utf-8"))
    errors: list[str] = []
    if frozen.get("status") != "ready_for_human_confirmation":
        errors.append("freeze report is not ready_for_human_confirmation")
    baseline_alignment = frozen.get("baseline_alignment") or []
    if not baseline_alignment:
        errors.append("freeze report is missing baseline configuration alignment")
    elif not all(row.get("match") for row in baseline_alignment):
        errors.append("freeze report contains a baseline mismatch")
    baseline_result_alignment = frozen.get("baseline_result_alignment") or []
    if not baseline_result_alignment:
        errors.append("freeze report is missing the 631/900 baseline result alignment")
    elif not all(row.get("match") for row in baseline_result_alignment):
        errors.append("freeze report contains a baseline-result mismatch")
    dynamic_alignment = frozen.get("dynamic_alignment") or []
    if not dynamic_alignment:
        errors.append("freeze report is missing dynamic-controller alignment")
    elif not all(row.get("match") for row in dynamic_alignment):
        errors.append("freeze report contains a dynamic-controller mismatch")
    if frozen.get("environment") != environment_record():
        errors.append("Python executable or package versions changed after freeze")

    for label in ("model", "processor", "overlay"):
        record = frozen.get(label)
        if not isinstance(record, dict):
            errors.append(f"missing metadata record: {label}")
        else:
            _verify_metadata(record, errors, label)

    artifacts = frozen.get("artifacts") or {}
    for label, record in artifacts.items():
        if label == "method_source_tree":
            root = Path(record["root"])
            try:
                actual = source_tree_record(root)
            except Exception as exc:
                errors.append(f"method_source_tree: {type(exc).__name__}: {exc}")
                continue
            if actual["source_tree_sha256"] != record["source_tree_sha256"]:
                errors.append("method_source_tree: source_tree_sha256 changed")
            if actual["file_count"] != record["file_count"]:
                errors.append("method_source_tree: file_count changed")
        elif isinstance(record, dict) and "path" in record:
            _verify_file(record, errors, f"artifacts.{label}")
        else:
            errors.append(f"unrecognized artifact record: {label}")

    result = {
        "pass": not errors,
        "freeze_report": str(args.report.resolve()),
        "freeze_report_sha256": sha256(args.report),
        "errors": errors,
    }
    _fsync_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
