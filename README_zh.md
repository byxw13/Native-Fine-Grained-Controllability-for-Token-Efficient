# ARIS

我们的模型权重也将很快开源，敬请关注。

## 环境

不同框架分别使用独立环境（Python 3.12）。

| 框架 | PyTorch | Transformers | vLLM |
|---|---|---|---|
| Training-free · DS-Qwen / Qwen3 | 2.6.0 | 4.57.0 | 不使用 |
| Training-free · InternVL3.5 | 2.8.0 | 4.57.6 | 0.11.0 |
| Training-free · Qwen3.5 | 2.10.0+cu128 | 5.14.1 | 0.19.1 + FA2 补丁 |
| 强化学习 | 2.8.0 | 4.56.2 | 0.10.2 + ARIS 补丁 |

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

需要同时判分时，在 `--` 前增加 `--judge`，并设置 API 环境变量（密钥不写入代码）：

```bash
export ARIS_PRIMARY_API_KEY="$YOUR_JUDGE_API_KEY"
export ARIS_PRIMARY_BASE_URL="https://your-provider.example/v1"
export ARIS_PRIMARY_LLM_MODEL="your-primary-model"
export ARIS_REVIEW_MODEL="your-review-model"
```

`--judge` 使用现有双模型判分器；HumanEval 应执行官方功能测试，不能用 API 判分替代。

校准入口（输入为用户提供的 JSONL，每行包含 `question` 或 `problem`）：

```bash
python code/training_free/calibrate.py heads --family ds-qwen -- \
  --model /models/DeepSeek-R1-Distill-Qwen-7B --input /data/calibration.jsonl \
  --output-dir /assets/head_calibration
python code/training_free/calibrate.py signals -- \
  --output-dir /output/calibration_rollout --output-json /assets/signals.json
```

Qwen3 将 `--family` 改成 `qwen3`。第一步生成 `head_scores_summary.json`；第二步读取校准运行已有的 `rollouts/eval_rollout.jsonl` 与 `reports/eval_summary.json`，不重新生成轨迹。运行时使用对应模型的校准文件，不能混用。

多模态环境安装（各自在独立环境执行）：

```bash
bash install.sh internvl
# 或
bash install.sh qwen35 qwen3.5-4b
bash install.sh qwen35-fa2 /sources/vllm-flash-attn
python code/training_free/install_multimodal.py --model-id qwen3.5-4b --check
```

FA2 源码需为 commit `29210221863736a08f71a866459e368ad1ac4a95`；`qwen35-fa2` 命令应用补丁并编译安装到当前环境，需要 CUDA 编译工具链。

Qwen3.5 安装包含对应模型的 rank3/conv Python 补丁；FA2 安装依次应用 suppression 和 boundary-window 两个补丁。切换 2B/9B 时，将上述模型标识改成 `qwen3.5-2b` / `qwen3.5-9b`，重新安装对应 Python 补丁，不能混用不同模型的运行版本。

```bash
python code/training_free/run.py --model-id qwen3.5-4b --dataset mmmu -- \
  --model-path /models/Qwen3.5-4B --manifest /data/samples.jsonl \
  --output /output/qwen35_4b/predictions.jsonl
```

输入 JSONL 使用 MMMU manifest 格式，图片路径必须在运行机器上可访问；各模型的 head 统计、控制器和采样参数已随包提供。2B 自动采用对应的 JSON 答案提示词。多卡按独立单卡进程运行，分别设置 `CUDA_VISIBLE_DEVICES`、不同输出文件及 `--shard-index i --shard-count N`；不自动申请集群资源。此命令生成预测和运行统计，不将生成完成等同于开放题判分完成。

InternVL3.5 同样使用上面的入口，将模型标识改为 `internvl3.5-2b` / `internvl3.5-4b` / `internvl3.5-8b`。其 vLLM 0.11.0 环境必须已有匹配的 ARIS native FA2 内核，普通 pip 安装不包含该内核；启动前执行 `install_multimodal.py --model-id internvl3.5-8b --check`。不兼容的内核会直接报错，不自动改用其他后端。

## 强化学习

```bash
conda create -n aris-rl python=3.12 -y
conda activate aris-rl
bash install.sh rl

bash code/reinforcement_learning/run.sh \
  /output/smartthinker_run /models/DeepSeek-R1-Distill-Qwen-7B \
  /data/deepscaler_preview.parquet /data/aime25.parquet /assets/heads.json
```


