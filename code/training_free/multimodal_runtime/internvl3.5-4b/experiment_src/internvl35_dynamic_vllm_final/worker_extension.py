"""vLLM worker import hook for the dynamic ARIS patch.

vLLM resolves ``worker_extension_cls`` inside every engine worker before the
worker instance and GPUModelRunner are constructed. Importing this module is
therefore the auditable point where each TP worker installs the native fused
attention and per-request controller patches.
"""

import os

from .engine_patch import finalize_dynamic_states, install_dynamic_aris_patch


_require_native = os.environ.get("ARIS_REQUIRE_NATIVE_FUSED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
install_dynamic_aris_patch(require_native_fused=_require_native)


class DynamicArisWorkerExtension:
    """Worker RPCs needed for audited request lifecycle completion."""

    def aris_finalize_dynamic_states(self, reason: str = "generate_return"):
        return finalize_dynamic_states(self.model_runner, reason=reason)
