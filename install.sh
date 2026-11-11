#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
    text)
        python -m pip install 'torch==2.6.0' 'transformers==4.57.0' openai numpy accelerate sentencepiece protobuf
        ;;
    rl)
        python -m pip install 'torch==2.8.0' 'vllm==0.10.2' 'transformers==4.56.2' 'ray==2.55.1'
        python "${root}/code/reinforcement_learning/install_patches.py"
        ;;
    internvl)
        python -m pip install 'torch==2.8.0' 'vllm==0.11.0' 'transformers==4.57.6' openai pillow qwen-vl-utils tqdm numpy sentencepiece protobuf einops timm cmake ninja
        ;;
    qwen35)
        python -m pip install 'torch==2.10.0' 'vllm==0.19.1' 'transformers==5.14.1' openai pillow qwen-vl-utils tqdm numpy sentencepiece protobuf einops cmake ninja
        python "${root}/code/training_free/install_multimodal.py" --model-id "${2:-qwen3.5-4b}"
        ;;
    qwen35-fa2)
        bash "${root}/code/training_free/build_multimodal_fa2.sh" "${2:?Pass the FlashAttention source directory}"
        ;;
    *)
        echo "Usage: bash install.sh {text|rl|internvl|qwen35|qwen35-fa2 SOURCE_DIR} (use separate environments)" >&2
        exit 2
        ;;
esac
