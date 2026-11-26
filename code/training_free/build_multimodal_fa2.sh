#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

target="qwen35"
if [[ "${1:-}" == "--target" ]]; then
    target="${2:?Pass --target qwen35 or --target internvl}"
    shift 2
fi
source_dir="$(realpath -- "${1:?Pass the FlashAttention source directory}")"
revision="$(git -C "$source_dir" rev-parse HEAD)"

case "$target" in
    qwen35)
        expected_rev="29210221863736a08f71a866459e368ad1ac4a95"
        patch_dir="$root/multimodal_runtime/qwen35_fa2"
        sentinel_patch="$patch_dir/0002-aris-fa2-configurable-boundary-window.patch"
        ;;
    internvl)
        expected_rev="ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a"
        patch_dir="$root/multimodal_runtime/internvl_fa2"
        sentinel_patch="$patch_dir/0001-aris-vllm011-fa2-varlen-fwd-aris.patch"
        ;;
    *)
        echo "Unsupported target: $target (expected qwen35 or internvl)" >&2
        exit 2
        ;;
esac

if [[ "$revision" != "$expected_rev" ]]; then
    echo "FlashAttention source revision ($revision) does not match $target ($expected_rev)." >&2
    exit 2
fi

if [[ ! -d "$source_dir/csrc/cutlass/include" ]]; then
    echo "Missing csrc/cutlass/include in $source_dir. Run: git -C $source_dir submodule update --init csrc/cutlass" >&2
    exit 2
fi

if ! git -C "$source_dir" apply --reverse --check "$sentinel_patch" 2>/dev/null; then
    for patch in "$patch_dir"/*.patch; do
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
