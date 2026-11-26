# ARIS

Our model weights are available on [Hugging Face](https://huggingface.co/huang13/FineGrained-Reasoning-Checkpoints).

## Environment

Use a separate environment for each framework (Python 3.12).

| Framework | PyTorch | Transformers | vLLM |
|---|---|---|---|
| Training-free · DS-Qwen / Qwen3 | 2.6.0 | 4.57.0 | Not used |
| Training-free · InternVL3.5 | 2.8.0 | 4.57.6 | 0.11.0 + FA2 patch |
| Training-free · Qwen3.5 | 2.10.0+cu128 | 5.14.1 | 0.19.1 + FA2 patches |
| Reinforcement learning | 2.8.0 | 4.56.2 | 0.10.2 + ARIS patches |

## Training-Free

```bash
conda create -n aris-text python=3.12 -y
conda activate aris-text
bash install.sh text

python code/training_free/run.py --model-id ds-qwen-7b --dataset math500 -- \
  --model /models/DeepSeek-R1-Distill-Qwen-7B \
  --input-jsonl /data/math500_test.jsonl \
  --summary-json /assets/heads.json \
  --signal-calibration-json /assets/signals.json \
  --question-lambda-calibration-json /assets/question_lambda.json \
  --run-name ds7b_math500 --output-dir /output/ds7b_math500
```

To also evaluate answer correctness, add `--judge` before `--` and set the API environment variables (do not hard-code API keys):

```bash
export ARIS_PRIMARY_API_KEY="$YOUR_JUDGE_API_KEY"
export ARIS_PRIMARY_BASE_URL="https://your-provider.example/v1"
export ARIS_PRIMARY_LLM_MODEL="your-primary-model"
export ARIS_REVIEW_MODEL="your-review-model"
```

`--judge` uses the existing two-model evaluator. HumanEval requires its official functional tests; API-based judging is not a substitute.

Calibration entry points (provide a JSONL file with a `question` or `problem` field in each record):

```bash
python code/training_free/calibrate.py heads --family ds-qwen -- \
  --model /models/DeepSeek-R1-Distill-Qwen-7B --input /data/calibration.jsonl \
  --output-dir /assets/head_calibration
python code/training_free/calibrate.py signals -- \
  --output-dir /output/calibration_rollout --output-json /assets/signals.json
```

For Qwen3, set `--family` to `qwen3`. The first step generates `head_scores_summary.json`. The second step reads the existing `rollouts/eval_rollout.jsonl` and `reports/eval_summary.json` from a calibration run without generating new trajectories. Use the calibration files for the corresponding model; do not mix files across models.

Install multimodal dependencies in separate environments:

```bash
bash install.sh internvl
git clone https://github.com/vllm-project/flash-attention.git /sources/vllm-flash-attn-internvl
git -C /sources/vllm-flash-attn-internvl checkout ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a
git -C /sources/vllm-flash-attn-internvl submodule update --init csrc/cutlass
bash install.sh internvl-fa2 /sources/vllm-flash-attn-internvl
python code/training_free/install_multimodal.py --model-id internvl3.5-8b --check
# Or
bash install.sh qwen35 qwen3.5-4b
bash install.sh qwen35-fa2 /sources/vllm-flash-attn
python code/training_free/install_multimodal.py --model-id qwen3.5-4b --check
```

The InternVL FA2 source must be at commit `ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a` (with `csrc/cutlass` initialized), and the Qwen3.5 FA2 source must be at commit `29210221863736a08f71a866459e368ad1ac4a95`. Both `*-fa2` commands apply the patches, build `_vllm_fa2_C`, and install the compiled extension and `flash_attn_interface.py` into the current environment. A CUDA build toolchain is required.

The Qwen3.5 installation includes model-specific rank3/conv Python patches. The FA2 installation applies the suppression and boundary-window patches in sequence. For 2B or 9B, replace the model ID above with `qwen3.5-2b` or `qwen3.5-9b` and reinstall the corresponding Python patches. Do not mix runtime versions across models.

```bash
python code/training_free/run.py --model-id qwen3.5-4b --dataset mmmu -- \
  --model-path /models/Qwen3.5-4B --manifest /data/samples.jsonl \
  --output /output/qwen35_4b/predictions.jsonl
```

The input JSONL must follow the MMMU manifest format, with image paths accessible on the machine running inference. Model-specific head statistics, controllers, and sampling settings are included. The 2B model automatically uses its corresponding JSON-answer prompt. For multiple GPUs, launch independent single-GPU processes with separate `CUDA_VISIBLE_DEVICES` values, output files, and `--shard-index i --shard-count N` arguments. Cluster resources are not allocated automatically. This command generates predictions and runtime statistics; completion of generation does not imply that open-ended answers have been evaluated.

InternVL3.5 uses the same entry point with model ID `internvl3.5-2b`, `internvl3.5-4b`, or `internvl3.5-8b`. `bash install.sh internvl-fa2 /sources/vllm-flash-attn-internvl` builds and installs the required ARIS native FA2 operator (`varlen_fwd_aris`) for vLLM 0.11.0 from `code/training_free/multimodal_runtime/internvl_fa2/`. Before launching, run `python code/training_free/install_multimodal.py --model-id internvl3.5-8b --check`. An incompatible kernel causes an explicit error rather than an automatic fallback to another backend.

## Reinforcement Learning

```bash
conda create -n aris-rl python=3.12 -y
conda activate aris-rl
bash install.sh rl

bash code/reinforcement_learning/run.sh \
  /output/smartthinker_run /models/DeepSeek-R1-Distill-Qwen-1.5B \
  /data/deepscaler_preview.parquet /data/aime25.parquet /assets/heads.json
```

`code/reinforcement_learning/run.sh` runs all 150 training steps with mixed sampling (4 native thinking + 4 effort-controlled rollouts at `k = 0.9`) and Eq. (16) token-level ratio filtering (`[0.8, 1.2]`), using VERL's unified `engine_workers` implementation.


