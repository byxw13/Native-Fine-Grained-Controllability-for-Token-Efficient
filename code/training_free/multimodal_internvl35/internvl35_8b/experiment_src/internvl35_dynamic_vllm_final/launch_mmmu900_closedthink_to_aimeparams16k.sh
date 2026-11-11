#!/bin/bash
set -euo pipefail

WORK=/path/to/aris_data/VLM/ARIS_qwen3vl_dynamic_vllm_final
RUN_DIR=/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/mmmu900_dynamic_closedthink_to_t06_topk20_rp100_init0p1_scale0p8_16384_20260717
JUDGE_PY=/path/to/conda/envs/aris_sup_fix/bin/python
BASE="$WORK/code/remote_scripts/mmmu_qwen3vl"
METHOD="$WORK/code/experiment_src/qwen3vl_vllm_dynamic_final"
SBATCH=sbatch

mkdir -p "$RUN_DIR"/{raw,logs,stream_judge,traces,stats}
if find "$RUN_DIR/raw" -type f -name 'predictions.shard*.jsonl' -size +0c | grep -q .; then
  echo "Refusing to reuse a run directory containing predictions: $RUN_DIR" >&2
  exit 1
fi

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

nohup env RUN_DIR="$RUN_DIR" EXPECTED_ROWS=900 POLL_SECONDS=60 \
  bash "$METHOD/finalize_mmmu900_single_streaming.sh" \
  >"$RUN_DIR/finalizer.stdout" 2>&1 </dev/null &
FINALIZER_PID=$!
echo "$FINALIZER_PID" >"$RUN_DIR/finalizer.pid"

if ! SUBMIT_OUT=$("$SBATCH" --parsable "$METHOD/slurm/mmmu900_dynamic_closedthink_to_aimeparams16k_a100.sbatch" 2>&1); then
  cleanup_helpers
  touch "$RUN_DIR/SUBMISSION_FAILED"
  echo "$SUBMIT_OUT" >&2
  exit 1
fi
JOB_ID=${SUBMIT_OUT%%;*}
printf '%s\n' "$JOB_ID" >"$RUN_DIR/job.id"
printf 'job_id\tpartition\tnode\tglobal_range\tgpus\n%s\tYOUR_A100_PARTITION\tYOUR_NODE\t[0,900)\t4\n' "$JOB_ID" >"$RUN_DIR/submission.tsv"

echo "RUN_DIR=$RUN_DIR"
echo "JOB_ID=$JOB_ID"
echo "JUDGE_PID=$JUDGE_PID"
echo "WATCHDOG_PID=$WATCHDOG_PID"
echo "FINALIZER_PID=$FINALIZER_PID"
