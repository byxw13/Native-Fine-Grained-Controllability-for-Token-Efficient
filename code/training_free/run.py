"""Select recorded model/task settings for an existing evaluation entry point."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory


def prepare_namespace(directory: Path, root: Path) -> None:
    namespace = directory / "experiment_src"
    namespace.mkdir()
    (namespace / "__init__.py").touch()
    packages = [
        root / "textual_ds_qwen/evaluation_v3_8_ds",
        root / "textual_qwen3/evaluation_v3_8_qwen3_alright_precue",
        root / "textual_qwen3/sampling_v1_qwen3",
        root / "common/experiment_src/sampling_v1",
        root / "common/experiment_src/evaluation_v1",
    ]
    for package in packages:
        (namespace / package.name).symlink_to(package.resolve(), target_is_directory=True)


def build_command(profile: dict, entrypoint: Path, extra: list[str]) -> list[str]:
    tree = ast.parse(entrypoint.read_text(encoding="utf-8"))
    supported = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        and arg.value.startswith("--")
    }
    supplied = {arg.split("=", 1)[0] for arg in extra if arg.startswith("--")}
    arguments = []
    for key, value in profile["arguments"].items():
        flag = "--" + key
        if value is None or value is False or flag in supplied:
            continue
        if flag not in supported:
            raise ValueError(f"{flag} is not supported by this entry point; use the matching evaluator version.")
        arguments.append(flag)
        if value is not True:
            arguments.append(str(value))
    return [sys.executable, str(entrypoint), *arguments, *extra]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id")
    parser.add_argument("--dataset")
    parser.add_argument("--entrypoint", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Enable the existing API judge during text evaluation.")
    head, extra = _split_arguments(sys.argv[1:])
    args = parser.parse_args(head)
    config_dir = Path(__file__).parent / "configs"
    profiles = json.loads((config_dir / "index.json").read_text(encoding="utf-8"))["models"]
    if args.list:
        for model, tasks in sorted(profiles.items()):
            print(model + ": " + ", ".join(sorted(tasks)))
        return
    dataset = {"amc2023": "amc", "human_eval": "humaneval", "gpqa_diamond": "gpqa"}.get(args.dataset, args.dataset)
    try:
        relative_path = profiles[args.model_id][dataset]
    except KeyError:
        parser.error("No recorded configuration for this model/dataset; no cross-model or cross-task fallback is applied.")
    group = json.loads((config_dir / relative_path).read_text(encoding="utf-8"))
    if args.list_runs:
        for name in group["runs"]:
            print(name)
        return
    run_id = args.run_id or group["default_run"]
    if run_id is None:
        parser.error("Multiple historical configurations exist. Select --run-id from --list-runs; a final-paper default is not yet established.")
    if run_id not in group["runs"]:
        parser.error("Unknown --run-id for this model/dataset.")
    profile = group["runs"][run_id]
    if args.judge:
        if dataset == "humaneval":
            parser.error("HumanEval requires execution of its official tests, not an API correctness judge.")
        if "--skip-judge" in extra:
            parser.error("--judge and --skip-judge cannot be combined.")
        if "arguments" in profile:
            profile = {**profile, "arguments": {**profile["arguments"], "skip-judge": False}}
        if not args.dry_run and not args.show_config and not os.environ.get("ARIS_PRIMARY_API_KEY"):
            parser.error("Set ARIS_PRIMARY_API_KEY before enabling the API judge.")
    if args.show_config:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return
    if group.get("interface") == "runtime_record":
        from multimodal import launch
        return launch(args.model_id, profile, extra, args.dry_run, args.judge)
    use_bundled_entrypoint = args.entrypoint is None
    if use_bundled_entrypoint:
        family = "textual_ds_qwen/evaluation_v3_8_ds" if args.model_id.startswith("ds-qwen-") else "textual_qwen3/evaluation_v3_8_qwen3_alright_precue"
        args.entrypoint = Path(__file__).parent / family / "evaluate.py"
    if not args.entrypoint.is_file():
        parser.error("Evaluation entry point does not exist.")
    if "--dataset" in extra or any(arg.startswith("--dataset=") for arg in extra):
        parser.error("Specify the dataset only before --.")
    try:
        command = build_command(profile, args.entrypoint.resolve(), ["--dataset", dataset, *extra])
    except ValueError as error:
        parser.error(str(error))
    environment = os.environ.copy()
    for key, value in profile.get("environment", {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        if use_bundled_entrypoint:
            with TemporaryDirectory(prefix="aris_imports_") as directory:
                prepare_namespace(Path(directory), Path(__file__).resolve().parent)
                environment["PYTHONPATH"] = os.pathsep.join(
                    [directory, environment.get("PYTHONPATH", "")]
                )
                returncode = subprocess.run(command, env=environment).returncode
        else:
            returncode = subprocess.run(command, env=environment).returncode
        raise SystemExit(returncode)


def _split_arguments(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in argv:
        return argv, []
    position = argv.index("--")
    return argv[:position], argv[position + 1:]


if __name__ == "__main__":
    main()
