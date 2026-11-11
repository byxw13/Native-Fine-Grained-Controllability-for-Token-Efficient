#!/bin/bash
set -euo pipefail

: "${ARIS_FORMAL_CONFIG_CONFIRMED:?formal run requires ARIS_FORMAL_CONFIG_CONFIRMED=YES}"
[[ "${ARIS_FORMAL_CONFIG_CONFIRMED}" == YES ]]
: "${EXECUTION_GROUP:?missing execution group}"
: "${GLOBAL_START_INDEX:?missing range start}"
: "${GLOBAL_END_INDEX:?missing range end}"
: "${RANK_OFFSET:?missing rank offset}"

case "${EXECUTION_GROUP}:${GLOBAL_START_INDEX}:${GLOBAL_END_INDEX}:${RANK_OFFSET}" in
  a6000_half0:0:450:0|l20_half1:450:900:4) ;;
  *) echo "Refusing unexpected formal split" >&2; exit 2 ;;
esac

WORK=/path/to/aris_data/VLM/ARIS_qwen3vl_dynamic_vllm_final
RUN_DIR=${RUN_DIR:-/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/mmmu900_final}
ENV=/path/to/conda/envs/aris_qwen3vl_dynamic_vllm011
MODEL=/path/to/aris_data/Qwen3.5-4B
OVERLAY="$WORK/tokenizer_overlays/qwen3vl_4b_method"
HEAD_RUN=/path/to/aris_data/VLM/experiments/aris_qwen3vl_dynamic_vllm/qwen3vl_head_summary_first100_nothink_t07_4k
MANIFEST=/path/to/aris_data/VLM/data/mmmuval/samples.jsonl
FROZEN_CONFIG="$WORK/configs/qwen3vl_dynamic_final_32768.json"
FREEZE_REPORT="$WORK/configs/qwen3vl_full900_freeze.json"
LOCAL_SHARDS=4
TOTAL_WORKERS=8

mkdir -p "$RUN_DIR"/{raw,logs,traces,stats,stream_judge}
COMPLETED=0
mark_failed_on_exit() {
  if [[ "$COMPLETED" != 1 ]]; then
    touch "$RUN_DIR/INFERENCE_FAILED.${EXECUTION_GROUP}" "$RUN_DIR/INFERENCE_FAILED"
  fi
}
trap mark_failed_on_exit EXIT
trap 'exit 143' TERM INT

export PATH="$ENV/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$WORK/code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

test -f "$HEAD_RUN/head_scores_summary.json"
test -f "$HEAD_RUN/signal_calibration.json"
test -f "$FROZEN_CONFIG"
test -f "$FREEZE_REPORT"
"$ENV/bin/python" -m experiment_src.qwen35_vllm_dynamic_final.verify_formal_freeze \
  --report "$FREEZE_REPORT" \
  --output "$RUN_DIR/formal_freeze_verification.${EXECUTION_GROUP}.json"

RAW_CUDA=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
IFS=',' read -r -a GPU_IDS <<< "$RAW_CUDA"
[[ ${#GPU_IDS[@]} -ge ${LOCAL_SHARDS} ]]

PIDS=()
for IDX in 0 1 2 3; do
  GPU=${GPU_IDS[$IDX]}
  GLOBAL_RANK=$((RANK_OFFSET + IDX))
  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export VLLM_PORT=$((55000 + IDX * 32))
    export VLLM_DP_MASTER_PORT=$((VLLM_PORT + 16))
    export ARIS_DYNAMIC_CONFIG_JSON="$FROZEN_CONFIG"
    export ARIS_DYNAMIC_TRACE_PATH="$RUN_DIR/traces/${EXECUTION_GROUP}_rank${GLOBAL_RANK}_tp{rank}.jsonl"
    export ARIS_DYNAMIC_TRACE_FSYNC=1
    export ARIS_DYNAMIC_STATS_PATH="$RUN_DIR/stats/dynamic_rank${GLOBAL_RANK}_tp{rank}.json"
    export ARIS_VLLM_V1_SUPPRESSION_ENABLED=1 ARIS_VLLM_V1_NATIVE_FUSED=1
    export ARIS_VLLM_V1_SUMMARY_JSON="$HEAD_RUN/head_scores_summary.json"
    export ARIS_VLLM_V1_BUDGET=0 ARIS_VLLM_V1_END_THINK_TOKEN_IDS=151668 ARIS_VLLM_V1_DEBUG=0
    export ARIS_VLLM_V1_STATS_STDOUT=0
    export ARIS_VLLM_V1_STATS_PATH="$RUN_DIR/stats/native_fused_rank${GLOBAL_RANK}.json"
    "$ENV/bin/python" -u -m experiment_src.qwen35_vllm_dynamic_final.mmmu_dynamic_vllm \
      --mode dynamic \
      --model-path "$MODEL" \
      --tokenizer "$OVERLAY" \
      --manifest "$MANIFEST" \
      --output "$RUN_DIR/raw/predictions.shard$(printf '%02d' "$GLOBAL_RANK")of${TOTAL_WORKERS}.jsonl" \
      --trace-path "$RUN_DIR/traces/${EXECUTION_GROUP}_rank${GLOBAL_RANK}_tp0.jsonl" \
      --shard-index "$IDX" \
      --shard-count "$LOCAL_SHARDS" \
      --rank-offset "$RANK_OFFSET" \
      --execution-group "$EXECUTION_GROUP" \
      --global-start-index "$GLOBAL_START_INDEX" \
      --global-end-index "$GLOBAL_END_INDEX" \
      --resume \
      --max-new-tokens 32768 \
      --temperature 0.7 \
      --top-p 0.95 \
      --top-k -1 \
      --seed 0 \
      --repetition-penalty 1.05 \
      --max-model-len 40960 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.90 \
      --batch-size 1
  ) >"$RUN_DIR/logs/${EXECUTION_GROUP}_rank${GLOBAL_RANK}.out" \
    2>"$RUN_DIR/logs/${EXECUTION_GROUP}_rank${GLOBAL_RANK}.err" &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAIL=1
done

if [[ "$FAIL" == 0 ]]; then
  touch "$RUN_DIR/INFERENCE_COMPLETE.${EXECUTION_GROUP}"
  COMPLETED=1
else
  touch "$RUN_DIR/INFERENCE_FAILED.${EXECUTION_GROUP}" "$RUN_DIR/INFERENCE_FAILED"
fi
exit "$FAIL"
