#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
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
checkpoints="${output}/checkpoints"
if [[ "${dry_run}" == 0 ]]; then
    for path in "${model}" "${train}" "${val}" "${summary}" "${trainer}/scripts/SmartThinker_Distill_1.5B.sh"; do
        [[ -e "${path}" ]] || { echo "Missing: ${path}" >&2; exit 2; }
    done
    if [[ -e "${output}" && ( ! -d "${output}" || -n "$(find "${output}" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
        echo "Use a new or empty OUTPUT directory; existing training files will not be overwritten." >&2
        exit 2
    fi
    mkdir -p "${output}" "${checkpoints}"
fi

export PYTHONPATH="${trainer}/src:${trainer}/src/trainer:${trainer}:${root}/aris_patch/verl/aris_v1_patch${PYTHONPATH:+:${PYTHONPATH}}"
export ARIS_VLLM_V1_SUMMARY_JSON="${summary}"
export ARIS_VLLM_V1_BUDGET="${ARIS_VLLM_V1_BUDGET:-0.9}"
export ARIS_VLLM_V1_END_THINK_TOKEN_IDS=151649
export ARIS_VLLM_V1_NATIVE_FUSED="${ARIS_VLLM_V1_NATIVE_FUSED:-0}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export ARIS_TRAIN_ROLLOUT_VARIANTS="${ARIS_TRAIN_ROLLOUT_VARIANTS:-think,think,think,think,sup0.9,sup0.9,sup0.9,sup0.9}"
export ARIS_ROLLOUT_SUPPRESSION_ENABLE="${ARIS_ROLLOUT_SUPPRESSION_ENABLE:-True}"
export ARIS_SUP_ROLLOUT_REWRITE_TRAIN_PROMPT="${ARIS_SUP_ROLLOUT_REWRITE_TRAIN_PROMPT:-True}"
export ARIS_SUP_TRAIN_REWRITE_SUFFIX_TEXT="${ARIS_SUP_TRAIN_REWRITE_SUFFIX_TEXT:-Alright}"
export ARIS_VLLM_V1_SUPPRESSION_ENABLED="${ARIS_VLLM_V1_SUPPRESSION_ENABLED:-True}"
export ARIS_VLLM_V1_AUTO_INSTALL="${ARIS_VLLM_V1_AUTO_INSTALL:-True}"
export ARIS_ROLLOUT_RATIO_FILTER="${ARIS_ROLLOUT_RATIO_FILTER:-1}"
export ARIS_ROLLOUT_RATIO_LOW="${ARIS_ROLLOUT_RATIO_LOW:-0.8}"
export ARIS_ROLLOUT_RATIO_HIGH="${ARIS_ROLLOUT_RATIO_HIGH:-1.2}"

args=(
    "data.train_files=${train}"
    "data.val_files=${val}"
    "actor_rollout_ref.rollout.n=8"
    "++actor_rollout_ref.rollout.aris_rollout_suppression_enable=True"
    "++actor_rollout_ref.rollout.calculate_log_probs=True"
    "trainer.total_training_steps=150"
    "trainer.save_freq=25"
    "trainer.default_local_dir=${checkpoints}"
    "trainer.default_hdfs_dir=null"
    "trainer.resume_mode=disable"
    "trainer.del_local_ckpt_after_load=False"
    "trainer.use_legacy_worker_impl=disable"
    "++actor_rollout_ref.rollout.n_gpus_per_node=4"
    "++actor_rollout_ref.rollout.nnodes=1"
    "++distillation.enabled=false"
    "++actor_rollout_ref.rollout.checkpoint_engine._target_=verl.workers.config.rollout.CheckpointEngineConfig"
    "++actor_rollout_ref.rollout.checkpoint_engine.backend=naive"
    "++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048"
    "++actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=null"
    "++reward.custom_reward_function.path=${trainer}/src/utils/rewards/math_compute_score.py"
    "++reward.custom_reward_function.name=math_compute_score"
    "++reward.reward_manager.source=importlib"
    "++reward.reward_manager.name=SmartThinkerRewardManager"
    "++reward.reward_manager.module.path=${trainer}/src/workers/reward_manager/SmartThinker.py"
    "++reward.reward_manager.module.name=SmartThinkerRewardManager"
)

ray_env_vars=(
    PYTHONPATH
    ARIS_VLLM_V1_SUMMARY_JSON
    ARIS_VLLM_V1_BUDGET
    ARIS_VLLM_V1_END_THINK_TOKEN_IDS
    ARIS_VLLM_V1_NATIVE_FUSED
    ARIS_TRAIN_ROLLOUT_VARIANTS
    ARIS_ROLLOUT_SUPPRESSION_ENABLE
    ARIS_SUP_ROLLOUT_REWRITE_TRAIN_PROMPT
    ARIS_SUP_TRAIN_REWRITE_SUFFIX_TEXT
    ARIS_VLLM_V1_SUPPRESSION_ENABLED
    ARIS_VLLM_V1_AUTO_INSTALL
    ARIS_ROLLOUT_RATIO_FILTER
    ARIS_ROLLOUT_RATIO_LOW
    ARIS_ROLLOUT_RATIO_HIGH
)
env_assignments=()
for var_name in "${ray_env_vars[@]}"; do
    env_assignments+=("${var_name}=${!var_name}")
    args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${var_name}='${!var_name}'")
done
if [[ -n "${ARIS_ROLLOUT_SUP_TOKENIZER_PATH:-}" ]]; then
    env_assignments+=("ARIS_ROLLOUT_SUP_TOKENIZER_PATH=${ARIS_ROLLOUT_SUP_TOKENIZER_PATH}")
    args+=("++ray_kwargs.ray_init.runtime_env.env_vars.ARIS_ROLLOUT_SUP_TOKENIZER_PATH='${ARIS_ROLLOUT_SUP_TOKENIZER_PATH}'")
fi

command=(
    env -u ARIS_TRAIN_ROLLOUT_BUDGETS
    "${env_assignments[@]}"
    bash "${trainer}/scripts/SmartThinker_Distill_1.5B.sh" --model "${model}"
    "${args[@]}"
    "${@}"
)

if [[ "${dry_run}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

"${command[@]}" 2>&1 | tee "${output}/train.log"

if [[ ! -f "${checkpoints}/latest_checkpointed_iteration.txt" || \
      "$(cat "${checkpoints}/latest_checkpointed_iteration.txt")" != 150 ]]; then
    echo "Training did not reach checkpoint step150." >&2
    exit 1
fi
