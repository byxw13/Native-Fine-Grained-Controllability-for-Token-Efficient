#!/bin/bash
set -euo pipefail

: "${ARIS_FORMAL_CONFIG_CONFIRMED:?set ARIS_FORMAL_CONFIG_CONFIRMED=YES only after human confirmation}"
[[ "$ARIS_FORMAL_CONFIG_CONFIRMED" == YES ]]

WORK=/path/to/aris_data/VLM/ARIS_qwen3vl_dynamic_vllm_final
RUN_DIR=${RUN_DIR:-/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/mmmu900_final}
ENV=/path/to/conda/envs/aris_qwen3vl_dynamic_vllm011
JUDGE_PY=${JUDGE_PY:-/path/to/conda/envs/aris_sup_fix/bin/python}
BASE="$WORK/code/remote_scripts/mmmu_qwen3vl"
METHOD="$WORK/code/experiment_src/qwen3vl_vllm_dynamic_final"
SBATCH=sbatch
SCANCEL=scancel

mkdir -p "$RUN_DIR"/{raw,logs,stream_judge,traces,stats}
export PYTHONPATH="$WORK/code:${PYTHONPATH:-}"

if find "$RUN_DIR/raw" -type f -name 'predictions.shard*.jsonl' -size +0c | grep -q .; then
  echo "Refusing to reuse a run directory containing predictions: $RUN_DIR" >&2
  exit 1
fi
if find "$RUN_DIR" -maxdepth 1 -type f \( -name '*COMPLETE*' -o -name '*FAILED*' \) | grep -q .; then
  echo "Refusing to reuse a run directory containing terminal markers: $RUN_DIR" >&2
  exit 1
fi

"$ENV/bin/python" -m experiment_src.qwen3vl_vllm_dynamic_final.verify_formal_freeze \
  --report "$WORK/configs/qwen3vl_full900_freeze.json" \
  --output "$RUN_DIR/formal_freeze_verification.prelaunch.json"

cleanup_helpers() {
  for pid in "${JUDGE_PID:-}" "${WATCHDOG_PID:-}" "${FINALIZER_PID:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
}

nohup "$JUDGE_PY" -u "$BASE/scripts/judge_mmmu_streaming.py" \
  --raw-glob "$RUN_DIR/raw/predictions.shard*.jsonl" \
  --output-dir "$RUN_DIR/stream_judge" \
  --expected-rows 900 \
  --poll-seconds 2 \
  >"$RUN_DIR/stream_judge/judge.log" 2>&1 </dev/null &
JUDGE_PID=$!
echo "$JUDGE_PID" >"$RUN_DIR/stream_judge/judge.pid"

nohup env CODE_ROOT="$BASE" RUN_DIR="$RUN_DIR" JUDGE_PY="$JUDGE_PY" \
  EXPECTED_ROWS=900 INTERVAL_SECONDS=60 \
  bash "$BASE/scripts/watch_mmmu_streaming_judge.sh" \
  >"$RUN_DIR/stream_judge/watchdog.stdout" 2>&1 </dev/null &
WATCHDOG_PID=$!
echo "$WATCHDOG_PID" >"$RUN_DIR/stream_judge/watchdog.pid"

nohup env WORK="$WORK" RUN_DIR="$RUN_DIR" ENV="$ENV" POLL_SECONDS=60 \
  bash "$METHOD/finalize_mmmu900_streaming.sh" \
  >"$RUN_DIR/finalizer.stdout" 2>&1 </dev/null &
FINALIZER_PID=$!
echo "$FINALIZER_PID" >"$RUN_DIR/finalizer.pid"

if ! A_SUBMIT=$("$SBATCH" --parsable \
  --export=ALL,ARIS_FORMAL_CONFIG_CONFIRMED=YES,RUN_DIR="$RUN_DIR" \
  "$METHOD/slurm/mmmu900_dynamic_a6000.sbatch" 2>&1); then
  cleanup_helpers
  touch "$RUN_DIR/SUBMISSION_FAILED"
  echo "$A_SUBMIT" >&2
  exit 1
fi
A_JOB_ID=${A_SUBMIT%%;*}

if ! L_SUBMIT=$("$SBATCH" --parsable \
  --export=ALL,ARIS_FORMAL_CONFIG_CONFIRMED=YES,RUN_DIR="$RUN_DIR" \
  "$METHOD/slurm/mmmu900_dynamic_l20.sbatch" 2>&1); then
  "$SCANCEL" "$A_JOB_ID" 2>/dev/null || true
  cleanup_helpers
  touch "$RUN_DIR/SUBMISSION_FAILED"
  echo "$L_SUBMIT" >&2
  exit 1
fi
L_JOB_ID=${L_SUBMIT%%;*}

printf 'execution_group\tpartition\tjob_id\tglobal_range\n' >"$RUN_DIR/submission.tsv"
printf 'a6000_half0\tYOUR_A6000_PARTITION\t%s\t[0,450)\n' "$A_JOB_ID" >>"$RUN_DIR/submission.tsv"
printf 'l20_half1\tYOUR_L20_PARTITION\t%s\t[450,900)\n' "$L_JOB_ID" >>"$RUN_DIR/submission.tsv"
printf '%s\n' "$A_JOB_ID" >"$RUN_DIR/job.a6000.id"
printf '%s\n' "$L_JOB_ID" >"$RUN_DIR/job.l20.id"

echo "RUN_DIR=$RUN_DIR"
echo "A6000_JOB_ID=$A_JOB_ID range=[0,450) GPUs=4"
echo "L20_JOB_ID=$L_JOB_ID range=[450,900) GPUs=4"
echo "JUDGE_PID=$JUDGE_PID"
echo "WATCHDOG_PID=$WATCHDOG_PID"
echo "FINALIZER_PID=$FINALIZER_PID"
