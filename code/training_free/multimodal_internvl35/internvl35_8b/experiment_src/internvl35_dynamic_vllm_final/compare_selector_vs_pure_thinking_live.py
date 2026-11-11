#!/usr/bin/env python3
"""InternVL3.5-8B 当前 So/Okay selector 版本的同题子集对比入口。

完整实现位于当前实验包的 ``tools/`` 目录；这个入口方便在远程代码副本中
直接使用 ``python -m experiment_src.internvl35_dynamic_vllm_final...`` 运行。
默认目录指向当前 selector v4 任务和 8B 纯思考 baseline，支持 ``--once``、
``--interval``、``--current-run`` 与 ``--baseline-run`` 覆盖默认值。
"""

from pathlib import Path
import runpy


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (
    _ROOT
    / "project-context/experiments/internvl35-8b-dynamic-vllm/tools/"
    / "compare_8b_dynamic_vs_pure_thinking_live.py"
)
# The remote code-only deployment does not carry the control-plane
# ``project-context`` tree.  In that layout the full implementation is copied
# beside this compatibility entry point.
if not _SCRIPT.is_file():
    _SCRIPT = Path(__file__).with_name("compare_8b_dynamic_vs_pure_thinking_live.py")
if not _SCRIPT.is_file():
    raise FileNotFoundError(f"8B selector compare script not found: {_SCRIPT}")

runpy.run_path(str(_SCRIPT), run_name="__main__")
