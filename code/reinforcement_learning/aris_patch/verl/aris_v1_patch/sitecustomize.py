"""Optional ARIS vLLM V1 auto patch for spawned engine-core processes."""

from __future__ import annotations

import os
import sys


def _cmdline_text() -> str:
    try:
        with open("/proc/self/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode(errors="ignore")
    except OSError:
        return " ".join(sys.argv)


def _should_skip_auto_install() -> bool:
    cmdline = _cmdline_text()
    skip_markers = (
        "multiprocessing.resource_tracker",
        "ray/_private/workers/default_worker.py",
        "ray/_private/log_monitor.py",
        "ray/_private/runtime_env/agent/main.py",
        "ray/autoscaler/_private/monitor.py",
        "ray/dashboard/agent.py",
        "ray/dashboard/dashboard.py",
        "vllm.model_executor.models.registry",
    )
    return any(marker in cmdline for marker in skip_markers)


if (
    os.getenv("ARIS_VLLM_V1_AUTO_INSTALL", "").lower() in {"1", "true", "yes", "on"}
    and not _should_skip_auto_install()
):
    try:
        import aris_vllm_v1_suppression as _aris_v1

        _aris_v1.configure_from_env()
        _aris_v1.install_aris_vllm_v1_suppression()
        if os.getenv("ARIS_HIDDEN_TRACE_ENABLED", "").lower() in {"1", "true", "yes", "on"}:
            import aris_hidden_trace_autohook as _aris_hidden_trace

            _aris_hidden_trace.install_from_env()
    except Exception as exc:  # pragma: no cover
        print(f"ARIS_VLLM_V1_AUTOINSTALL_FAILED {exc}", flush=True)
