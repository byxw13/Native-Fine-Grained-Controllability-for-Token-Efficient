"""Run the bundled head collector or rebuild signals from calibration rollouts."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

from run import prepare_namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["heads", "signals"])
    parser.add_argument("--family", choices=["ds-qwen", "qwen3"], default="ds-qwen")
    parser.add_argument("--dry-run", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]
    if not extra:
        parser.error("Pass collector arguments after --; use -- --help to inspect them.")
    root = Path(__file__).resolve().parent
    if args.stage == "heads":
        package = "sampling_v1" if args.family == "ds-qwen" else "sampling_v1_qwen3"
        module = f"experiment_src.{package}.collect_head_summary"
    else:
        module = "experiment_src.evaluation_v3_8_ds.rebuild_signal_calibration_from_rollout"
    command = [sys.executable, "-m", module, *extra]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    with TemporaryDirectory(prefix="aris_calibration_") as directory:
        prepare_namespace(Path(directory), root)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([directory, env.get("PYTHONPATH", "")])
        raise SystemExit(subprocess.run(command, env=env).returncode)


if __name__ == "__main__":
    main()
