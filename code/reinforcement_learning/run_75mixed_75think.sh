#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ $# -lt 3 || "${2:-}" != "--" ]]; then
    echo "Usage: bash $0 [--dry-run] OUTPUT_DIR -- PREPARED_TRAIN_COMMAND [ARGS...]" >&2
    exit 2
fi
output_dir="$(realpath -m -- "$1")"
shift 2
command=("$@")
checkpoints="${output_dir}/checkpoints"

if [[ "${dry_run}" == 0 ]]; then
    if [[ -e "${output_dir}" && ( ! -d "${output_dir}" || -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ) ]]; then
        echo "Use a new or empty OUTPUT_DIR; existing training files will not be overwritten." >&2
        exit 2
    fi
    mkdir -p "${output_dir}"
fi

run_stage() {
    local label="$1" total="$2" enabled="$3" variants="$4" resume="$5" resume_path="$6"
    local args=(
        "actor_rollout_ref.rollout.n=8"
        "++actor_rollout_ref.rollout.aris_rollout_suppression_enable=${enabled}"
        "trainer.total_training_steps=${total}"
        "trainer.save_freq=75"
        "trainer.default_local_dir=${checkpoints}"
        "trainer.default_hdfs_dir=null"
        "trainer.resume_mode=${resume}"
        "trainer.resume_from_path=${resume_path}"
        "trainer.del_local_ckpt_after_load=False"
    )
    local environment=(
        env -u ARIS_TRAIN_ROLLOUT_BUDGETS
        "ARIS_TRAIN_ROLLOUT_VARIANTS=${variants}"
        "ARIS_ROLLOUT_SUPPRESSION_ENABLE=${enabled}"
        "ARIS_SUP_ROLLOUT_REWRITE_TRAIN_PROMPT=${enabled}"
        "ARIS_VLLM_V1_SUPPRESSION_ENABLED=${enabled}"
        "ARIS_VLLM_V1_AUTO_INSTALL=${enabled}"
    )
    # A pre-existing Ray head must not retain the preceding stage's flags.
    local variable
    for variable in "${environment[@]:3}"; do
        args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${variable%%=*}='${variable#*=}'")
    done
    echo "${label}"
    if [[ "${dry_run}" == 1 ]]; then
        printf '%q ' "${environment[@]}" "${command[@]}" "${args[@]}"
        printf '\n'
    else
        "${environment[@]}" "${command[@]}" "${args[@]}" 2>&1 | tee "${output_dir}/${label}.log"
    fi
}

run_stage stage1_mixed_1_to_75 75 True \
    'think,think,think,think,sup1.0,sup1.0,sup1.0,sup1.0' disable null

if [[ "${dry_run}" == 0 ]]; then
    checkpoint="${checkpoints}/global_step_75"
    if [[ ! -d "${checkpoint}/actor" || ! -f "${checkpoint}/data.pt" || \
          ! -f "${checkpoints}/latest_checkpointed_iteration.txt" || \
          "$(cat "${checkpoints}/latest_checkpointed_iteration.txt")" != 75 ]]; then
        echo "Stage 1 did not save step75 actor and dataloader state; refusing stage 2." >&2
        exit 1
    fi
fi

run_stage stage2_thinking_76_to_150 150 False \
    'think,think,think,think,think,think,think,think' \
    resume_path "${checkpoints}/global_step_75"

if [[ "${dry_run}" == 0 ]]; then
    if [[ ! -f "${checkpoints}/latest_checkpointed_iteration.txt" || \
          "$(cat "${checkpoints}/latest_checkpointed_iteration.txt")" != 150 ]]; then
        echo "Stage 2 did not reach checkpoint step150." >&2
        exit 1
    fi
fi
