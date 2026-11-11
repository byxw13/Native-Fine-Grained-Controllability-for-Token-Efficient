"""Worker entrypoint for audited Qwen3.5 Full+Linear suppression."""

from .engine_patch import (
    finalize_dynamic_request,
    finalize_dynamic_states,
    install_dynamic_aris_patch,
)


install_dynamic_aris_patch(require_native_fused=True)


class DynamicArisLinearWorkerExtension:
    def aris_finalize_dynamic_states(self, reason: str = "generate_return"):
        return finalize_dynamic_states(self.model_runner, reason=reason)

    def aris_finalize_dynamic_request(
        self,
        request_id: str,
        reason: str = "request_generate_return",
    ):
        return finalize_dynamic_request(
            self.model_runner,
            request_id=request_id,
            reason=reason,
        )
