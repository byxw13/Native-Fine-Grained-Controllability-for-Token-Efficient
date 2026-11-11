#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=(--dry-run)
    shift
fi
if [[ $# -lt 5 ]]; then
    echo "Usage: bash $0 [--dry-run] OUTPUT MODEL TRAIN_PARQUET VAL_PARQUET HEAD_SUMMARY [HYDRA_ARGS...]" >&2
    exit 2
fi
output="$(realpath -m -- "$1")"
model="$(realpath -m -- "$2")"
train="$(realpath -m -- "$3")"
val="$(realpath -m -- "$4")"
summary="$(realpath -m -- "$5")"
shift 5
trainer="${root}/.runtime/SmartThinker"
if [[ ${#dry_run[@]} == 0 ]]; then
    for path in "${model}" "${train}" "${val}" "${summary}" "${trainer}/scripts/SmartThinker_Distill_7B.sh"; do
        [[ -e "${path}" ]] || { echo "Missing: ${path}" >&2; exit 2; }
    done
fi
export PYTHONPATH="${trainer}:${root}/aris_patch/verl/aris_v1_patch${PYTHONPATH:+:${PYTHONPATH}}"
export ARIS_VLLM_V1_SUMMARY_JSON="${summary}"
export ARIS_VLLM_V1_BUDGET=1.0
export ARIS_VLLM_V1_END_THINK_TOKEN_IDS=151649
export ARIS_VLLM_V1_NATIVE_FUSED=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
exec bash "${root}/run_75mixed_75think.sh" "${dry_run[@]}" "${output}" -- \
    bash "${trainer}/scripts/SmartThinker_Distill_7B.sh" --model "${model}" \
    "data.train_files=${train}" "data.val_files=${val}" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.ARIS_VLLM_V1_SUMMARY_JSON='${summary}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.ARIS_VLLM_V1_BUDGET='1.0'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.ARIS_VLLM_V1_END_THINK_TOKEN_IDS='151649'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.ARIS_VLLM_V1_NATIVE_FUSED='0'" \
    "${@}"
