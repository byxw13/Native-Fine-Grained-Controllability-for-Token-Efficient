#!/bin/bash
set -euo pipefail

WORK=${WORK:-/path/to/aris_data/VLM/ARIS_qwen3vl_dynamic_vllm_final}
RUN_DIR=${RUN_DIR:-/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/mmmu900_final}
ENV=${ENV:-/path/to/conda/envs/aris_qwen3vl_dynamic_vllm011}
BASELINE_RUN=${BASELINE_RUN:-/path/to/aris_data/VLM/experiments/mmmuval_qwen3vl4b_full900_vllm_l20_vlmevalprompt_p0_temp07_seed0_rp105_32768}
POLL_SECONDS=${POLL_SECONDS:-60}
JUDGE_WAIT_SECONDS=${JUDGE_WAIT_SECONDS:-86400}

stop_streaming_helpers() {
  for name in judge watchdog; do
    pid_file="$RUN_DIR/stream_judge/${name}.pid"
    if [[ -f "$pid_file" ]]; then
      kill "$(cat "$pid_file")" 2>/dev/null || true
    fi
  done
}

fail() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$RUN_DIR/finalizer.log"
  touch "$RUN_DIR/FORMAL_EVALUATION_FAILED" "$RUN_DIR/INFERENCE_FAILED"
  stop_streaming_helpers
  exit 1
}

printf '%s finalizer started\n' "$(date -Is)" >> "$RUN_DIR/finalizer.log"
while true; do
  if [[ -f "$RUN_DIR/INFERENCE_FAILED.a6000_half0" || -f "$RUN_DIR/INFERENCE_FAILED.l20_half1" ]]; then
    fail "one or both inference half-jobs failed"
  fi
  if [[ -f "$RUN_DIR/INFERENCE_COMPLETE.a6000_half0" && -f "$RUN_DIR/INFERENCE_COMPLETE.l20_half1" ]]; then
    touch "$RUN_DIR/INFERENCE_COMPLETE"
    break
  fi
  sleep "$POLL_SECONDS"
done

deadline=$((SECONDS + JUDGE_WAIT_SECONDS))
while [[ ! -f "$RUN_DIR/stream_judge/JUDGE_COMPLETE" ]]; do
  if (( SECONDS >= deadline )); then
    fail "streaming judge did not finish within ${JUDGE_WAIT_SECONDS}s after inference"
  fi
  sleep "$POLL_SECONDS"
done

export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$WORK/code:${PYTHONPATH:-}"
if ! "$ENV/bin/python" -u -m experiment_src.qwen35_vllm_dynamic_final.analyze_mmmu900_results \
  --run-dir "$RUN_DIR" \
  --baseline-run-dir "$BASELINE_RUN" \
  --expected-rows 900 \
  --calibration-count 100 \
  --output-json "$RUN_DIR/formal_report.json" \
  --output-md "$RUN_DIR/formal_report.md" \
  >> "$RUN_DIR/finalizer.log" 2>&1; then
  fail "formal integrity analysis failed"
fi

touch "$RUN_DIR/FORMAL_EVALUATION_COMPLETE"
printf '%s formal evaluation complete\n' "$(date -Is)" >> "$RUN_DIR/finalizer.log"
