"""Install the archived patches into isolated, local source checkouts."""
from pathlib import Path
import shutil
import subprocess
import sys


VERL_REVISION = "c20d5049a1678b56a07d79bc887c4dbeec3988b1"


def main():
    root = Path(__file__).resolve().parent
    runtime = root / ".runtime"
    verl = runtime / "verl"
    smartthinker = runtime / "SmartThinker"
    if verl.exists() or smartthinker.exists():
        raise SystemExit("Runtime source directories already exist; refusing to overwrite them.")
    runtime.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/volcengine/verl.git", str(verl)], check=True)
    subprocess.run(["git", "-C", str(verl), "checkout", VERL_REVISION], check=True)
    for source in (root / "aris_patch/verl/verl").rglob("*.py"):
        target = verl / "verl" / source.relative_to(root / "aris_patch/verl/verl")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(root / "smartthinker_upstream", smartthinker)
    for source in (root / "aris_patch/smartthinker_src").rglob("*.py"):
        target = smartthinker / "src" / source.relative_to(root / "aris_patch/smartthinker_src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Preserve the versions used by the archived V1 attention patch.
    constraints = runtime / "constraints.txt"
    constraints.write_text(
        "torch==2.8.0\nvllm==0.10.2\ntransformers==4.56.2\nray==2.55.1\n",
        encoding="ascii",
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(constraints), "-e", str(verl)], check=True)
    print("Installed VERL overlays and SmartThinker source overlays.")


if __name__ == "__main__":
    main()
