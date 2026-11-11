#!/bin/bash
set -euo pipefail

RUN_DIR=${RUN_DIR:?RUN_DIR is required}
EXPECTED_ROWS=${EXPECTED_ROWS:-900}
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
  printf '%s %s\n' "$(date -Is)" "$*" >>"$RUN_DIR/finalizer.log"
  touch "$RUN_DIR/EXPERIMENT_FAILED"
  stop_streaming_helpers
  exit 1
}

printf '%s single-job finalizer started\n' "$(date -Is)" >>"$RUN_DIR/finalizer.log"
while [[ ! -f "$RUN_DIR/INFERENCE_COMPLETE" ]]; do
  [[ -f "$RUN_DIR/INFERENCE_FAILED" ]] && fail "inference job failed"
  sleep "$POLL_SECONDS"
done

deadline=$((SECONDS + JUDGE_WAIT_SECONDS))
while [[ ! -f "$RUN_DIR/stream_judge/JUDGE_COMPLETE" ]]; do
  if ((SECONDS >= deadline)); then
    fail "streaming judge did not finish within ${JUDGE_WAIT_SECONDS}s"
  fi
  sleep "$POLL_SECONDS"
done

python3 - "$RUN_DIR" "$EXPECTED_ROWS" <<'PY' >>"$RUN_DIR/finalizer.log" 2>&1
import glob
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = []
for path in sorted(glob.glob(str(run_dir / "raw" / "predictions.shard*.jsonl"))):
    with open(path, encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
if len(rows) != expected:
    raise SystemExit(f"prediction count mismatch: {len(rows)} != {expected}")
ids = [str(row.get("id")) for row in rows]
if len(set(ids)) != expected:
    raise SystemExit(f"prediction IDs are not unique: {len(set(ids))} != {expected}")
summary = {
    "prediction_rows": len(rows),
    "unique_ids": len(set(ids)),
    "stream_judge_complete": True,
}
(run_dir / "completion_integrity.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

touch "$RUN_DIR/EXPERIMENT_COMPLETE"
stop_streaming_helpers
printf '%s single-job experiment complete\n' "$(date -Is)" >>"$RUN_DIR/finalizer.log"
