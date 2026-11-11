#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/path/to/conda/envs/aris_qwen35_dynamic_vllm019}
SRC_DIR=${SRC_DIR:-/path/to/aris_data/VLM/ARIS_qwen35_dynamic_vllm_final/third_party/vllm-flash-attn-aris-v019}
BUILD_DIR=${BUILD_DIR:-$SRC_DIR/build-aris-sm80}
STAGE_DIR=${STAGE_DIR:-$SRC_DIR/stage-aris-sm80}
PYTHON=$ENV_PREFIX/bin/python

test -x "$PYTHON"
test -f "$SRC_DIR/csrc/flash_attn/flash_api.cpp"
rm -rf "$BUILD_DIR" "$STAGE_DIR"
mkdir -p "$BUILD_DIR" "$STAGE_DIR"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-2}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NINJA_BIN=$($PYTHON -c 'import ninja; print(ninja.BIN_DIR)')
export PATH="$ENV_PREFIX/bin:$NINJA_BIN:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

"$ENV_PREFIX/bin/cmake" -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON" \
  -DVLLM_PYTHON_PATH="$($PYTHON -c 'import sys; print(":".join(sys.path))')" \
  -DNVCC_THREADS="$NVCC_THREADS"
"$ENV_PREFIX/bin/cmake" --build "$BUILD_DIR" --target _vllm_fa2_C -j "$MAX_JOBS"
SITE=$($PYTHON -c 'import site; print(site.getsitepackages()[0])')
# BUILD_DIR, STAGE_DIR and the environment live on the same shared filesystem.
# Hard-linking the 300 MB extension avoids two very slow NFS copies while still
# leaving the installed inode alive when a later rebuild removes BUILD_DIR.
BUILT_SO=$(find "$BUILD_DIR" -maxdepth 1 -type f -name '_vllm_fa2_C*.so' -print -quit)
test -n "$BUILT_SO"
mkdir -p "$STAGE_DIR/vllm_flash_attn"
SO="$STAGE_DIR/vllm_flash_attn/$(basename "$BUILT_SO")"
ln -f "$BUILT_SO" "$SO"
chmod 0755 "$SO"
ln -f "$SO" "$SITE/vllm/vllm_flash_attn/$(basename "$SO")"
install -m 0644 "$SRC_DIR/vllm_flash_attn/flash_attn_interface.py" \
  "$SITE/vllm/vllm_flash_attn/flash_attn_interface.py"

"$PYTHON" - <<'PY'
import hashlib
import pathlib
import torch
import vllm.vllm_flash_attn  # noqa: F401

schema = str(torch._C._dispatch_find_schema_or_throw("_vllm_fa2_C::varlen_fwd", "").schema())
required = ("aris_enabled", "aris_budget_ids", "aris_end_positions", "aris_head_mask_table")
assert all(name in schema for name in required), schema
root = pathlib.Path(torch.__file__).parents[0].parent / "vllm" / "vllm_flash_attn"
so = next(root.glob("_vllm_fa2_C*.so"))
print({"schema": schema, "so": str(so), "sha256": hashlib.sha256(so.read_bytes()).hexdigest()})
PY
