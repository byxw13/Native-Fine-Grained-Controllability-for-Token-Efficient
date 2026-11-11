#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(realpath -- "${1:?Pass the FlashAttention source directory}")"
revision="$(git -C "$source_dir" rev-parse HEAD)"
[[ "$revision" == 29210221863736a08f71a866459e368ad1ac4a95 ]] || {
    echo 'FlashAttention source revision does not match the Qwen3.5 patches.' >&2
    exit 2
}
window_patch="$root/multimodal_runtime/qwen35_fa2/0002-aris-fa2-configurable-boundary-window.patch"
if ! git -C "$source_dir" apply --reverse --check "$window_patch" 2>/dev/null; then
    for patch in "$root"/multimodal_runtime/qwen35_fa2/*.patch; do
        if git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
            continue
        fi
        git -C "$source_dir" apply --check "$patch"
        git -C "$source_dir" apply "$patch"
    done
fi
build_dir="$(mktemp -d "$source_dir/aris-build-XXXXXX")"
python_bin="$(python -c 'import sys; print(sys.executable)')"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9}"
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$python_bin" \
    -DVLLM_PYTHON_PATH="$(python -c 'import sys; print(":".join(sys.path))')" \
    -DNVCC_THREADS="${NVCC_THREADS:-2}"
cmake --build "$build_dir" --target _vllm_fa2_C -j "${MAX_JOBS:-8}"
python - "$build_dir" "$source_dir" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import shutil
import sys
target = Path(metadata.distribution('vllm').locate_file('vllm/vllm_flash_attn'))
binaries = list(Path(sys.argv[1]).glob('_vllm_fa2_C*.so'))
if len(binaries) != 1:
    raise RuntimeError(f'Expected one FA2 extension, got {binaries}')
for source in (binaries[0], Path(sys.argv[2]) / 'vllm_flash_attn/flash_attn_interface.py'):
    destination = target / source.name
    backup = destination.with_name(destination.name + '.pre_aris')
    if destination.exists() and not backup.exists():
        shutil.copy2(destination, backup)
    shutil.copy2(source, destination)
print('Installed native FA2 and its matching Python interface.')
PY
