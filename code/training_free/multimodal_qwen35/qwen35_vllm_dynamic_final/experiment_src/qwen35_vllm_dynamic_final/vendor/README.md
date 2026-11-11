# Qwen3.5 / vLLM 0.19.1 fused suppression vendor patch

The production path extends the exact vLLM 0.19.1 FlashAttention-2 source
commit `29210221863736a08f71a866459e368ad1ac4a95`. It adds four request-local
device tensors and masks `</think>` plus its following token only for selected
query heads inside the decode kernel. Patch-off passes `None` metadata through
the same ABI.

`patches/0001-aris-vllm019-fa2-integrated-dynamic-suppression.patch` contains the
complete source delta, including GQA decode-row and empty-tile fast rejection.
The build must replace both `_vllm_fa2_C.abi3.so` and
`vllm/vllm_flash_attn/flash_attn_interface.py` in the independent environment;
the baseline environment is never modified.

