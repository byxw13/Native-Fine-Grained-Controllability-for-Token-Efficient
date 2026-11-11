#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/path/to/aris_data/VLM/ARIS_qwen35_dynamic_vllm_final}
PYTHON=${PYTHON:-/path/to/conda/envs/aris_qwen35_dynamic_vllm019/bin/python}
MODEL=${MODEL:-/path/to/aris_data/Qwen3.5-4B}
SIGNAL=${SIGNAL:-$ROOT/configs/bootstrap_signal_calibration_qwen3vl.json}
OUTPUT=${OUTPUT:-$ROOT/configs/runtime_init0p1_scale0p8_max32k.json}
CALIBRATION_ONLY=${CALIBRATION_ONLY:-0}

CALIBRATION_ARGS=(--signal-calibration "$SIGNAL")
if [[ "$CALIBRATION_ONLY" == "1" ]]; then
  CALIBRATION_ARGS=(--calibration-only)
fi

export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
"$PYTHON" -m experiment_src.qwen35_vllm_dynamic_final.build_runtime_config \
  --tokenizer "$MODEL" \
  --output "$OUTPUT" \
  "${CALIBRATION_ARGS[@]}" \
  --max-new-tokens 32768 \
  --initial-suppression 0.1 \
  --suppression-upper-bound 1.0 \
  --cue-threshold 0.1 \
  --cue-gate-threshold 0.1 \
  --entropy-upper-bound 1.0 \
  --suppression-curve exp_saturating \
  --exp-beta 2.0 \
  --suppression-scale 0.8 \
  --raw-history-size 10 \
  --k-smoothing-initial-rho 1.0 \
  --k-smoothing-late-rho 1.0 \
  --k-smoothing-initial-checkpoints 6 \
  --upgrade-newline-span 1 \
  --downgrade-newline-span 10 \
  --downgrade-confirm-windows 5 \
  --tail-soft-ratio 0.99999 \
  --tail-full-ratio 1.0 \
  --tail-threshold-max 1.0 \
  --tail-signal-trigger-ratio 0.8 \
  --tail-signal-wait-token-limit 300 \
  --tail-signal-text $'</think>\n' \
  --end-think-bridge-text '.' \
  --end-think-to-l1-lock \
  --end-think-mask-newlines 0 \
  --end-think-mask-tokens 1 \
  --final-answer-prefix-text 'the answer is ' \
  --final-answer-prefix-trigger-ratio 0.98
