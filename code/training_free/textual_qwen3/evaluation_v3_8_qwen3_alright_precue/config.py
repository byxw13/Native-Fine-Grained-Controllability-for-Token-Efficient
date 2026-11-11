from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PACKET_DIR = REPO_ROOT / "project-context" / "experiments" / "evaluation-v2"
LOCAL_RUN_ROOT = REPO_ROOT / "experiment-src" / "evaluation_v3_8_qwen3_alright_precue" / "runs"
LOCAL_PULL_ROOT = REPO_ROOT / "pulled-results" / "evaluation-v3-8-qwen3-alright-precue"
LOCAL_MONITOR_ROOT = REPO_ROOT / "monitor-logs" / "evaluation-v3-8-qwen3-alright-precue"


@dataclass(frozen=True)
class RemoteSpec:
    server_alias: str
    ssh_target: str
    workdir: str
    conda_init: str
    conda_env: str
    code_root: str
    run_root: str
    model_path: str
    summary_json_path: str
    default_partition: str
    candidate_partitions: tuple[str, ...]
    gpus: int
    cpus_per_task: int
    python_bin: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    remote_input_path: str
    default_run_name: str
    total_examples: int | None


REMOTE = RemoteSpec(
    server_alias="L20",
    ssh_target="SET_ARIS_SSH_TARGET",
    workdir="/path/to/user_data/HWY/difficulty/RL",
    conda_init='eval "$(conda shell.bash hook)"',
    conda_env="llama",
    code_root="/path/to/user_data/HWY/difficulty/RL/aris_evaluation_v3_8_qwen3_alright_precue_code",
    run_root="/path/to/user_data/HWY/difficulty/RL/aris_evaluation_v3_8_qwen3_runs",
    model_path="/path/to/user_data/HWY/difficulty/qwen3-8b",
    summary_json_path="/path/to/user_data/HWY/difficulty/RL/aris_evaluation_v3_8_assets/qwen3_8b_nothink_head_scores_summary_20260420.json",
    default_partition="YOUR_L20_PARTITION",
    candidate_partitions=("YOUR_L20_PARTITION",),
    gpus=1,
    cpus_per_task=2,
    python_bin="python",
)


DATASETS: dict[str, DatasetSpec] = {
    "math500": DatasetSpec(
        name="math500",
        remote_input_path="/path/to/user_data/HWY/difficulty/math500_test.jsonl",
        default_run_name="evaluation_v3_8_math500_lambdaq",
        total_examples=500,
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        remote_input_path="/path/to/user_data/HWY/difficulty/gsm8k_test.jsonl",
        default_run_name="evaluation_v3_8_gsm8k_lambdaq",
        total_examples=1319,
    ),
    "amc": DatasetSpec(
        name="amc",
        remote_input_path="/path/to/user_data/HWY/difficulty/amc_test.jsonl",
        default_run_name="evaluation_v3_8_amc_lambdaq",
        total_examples=40,
    ),
    "aime2024": DatasetSpec(
        name="aime2024",
        remote_input_path="/path/to/user_data/HWY/difficulty/aime2024_test.jsonl",
        default_run_name="evaluation_v3_8_aime2024_lambdaq",
        total_examples=30,
    ),
    "aime2025": DatasetSpec(
        name="aime2025",
        remote_input_path="/path/to/user_data/HWY/difficulty/aime2025_test.jsonl",
        default_run_name="evaluation_v3_8_aime2025_lambdaq",
        total_examples=30,
    ),
}


# 连续控制沿用 v2 的 checkpoint 频率和种子，但把离散档位改成连续 suppression。
DEFAULT_ASSISTANT_PREFIX_TEXT = "Alright"
DEFAULT_CUE_TEXT = ""
DEFAULT_CUE_THRESHOLD = 0.1
DEFAULT_CUE_GATE_THRESHOLD = None
DEFAULT_MIN_TOKENS_BEFORE_CUE = 0
DEFAULT_FINAL_ANSWER_PREFIX_TEXT = "the answer is "
DEFAULT_FINAL_ANSWER_PREFIX_TRIGGER_RATIO = 0.98
DEFAULT_ENTROPY_UPPER_BOUND = 1.0
SUPPORTED_SUPPRESSION_CURVES = ("power", "exp_saturating")
DEFAULT_SUPPRESSION_CURVE = "power"
DEFAULT_POWER_EXPONENT = 0.65
DEFAULT_EXP_BETA = 2.0
DEFAULT_SUPPRESSION_SCALE = 1.0
DEFAULT_SUPPRESSION_UPPER_BOUND = 1.0
DEFAULT_RAW_HISTORY_SIZE = 10
DEFAULT_K_SMOOTHING_INITIAL_RHO = 1.0
DEFAULT_K_SMOOTHING_LATE_RHO = 0.75
DEFAULT_K_SMOOTHING_INITIAL_CHECKPOINTS = 2
DEFAULT_QUESTION_LAMBDA_MIN = 0.8
DEFAULT_QUESTION_LAMBDA_MAX = 1.05
DEFAULT_QUESTION_LAMBDA_EXP_BETA = 2.0
DEFAULT_QUESTION_LAMBDA_D_MEAN_WEIGHT = 0.7
DEFAULT_QUESTION_LAMBDA_D_MIN_WEIGHT = 0.3

DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.6
DEFAULT_SEED = 3407
DEFAULT_UPGRADE_NEWLINE_SPAN = 1
DEFAULT_DOWNGRADE_NEWLINE_SPAN = 10
DEFAULT_DOWNGRADE_CONFIRM_WINDOWS = 3
DEFAULT_POST_DOWNGRADE_UPGRADE_FREEZE_NEWLINES = 0

DEFAULT_TAIL_SOFT_RATIO = 0.75
DEFAULT_TAIL_FULL_RATIO = 0.90
DEFAULT_TAIL_EXP_BETA = 3.0
DEFAULT_TAIL_THRESHOLD_MAX = 0.6
DEFAULT_TAIL_SIGNAL_TRIGGER_RATIO = 0.90
DEFAULT_TAIL_SIGNAL_TEXT = "</think>\n"
DEFAULT_TAIL_SIGNAL_WAIT_TOKEN_LIMIT = 300
DEFAULT_TAIL_SIGNAL_NEXT_CHECKPOINT_ONLY = False
DEFAULT_DISABLE_CUSTOM_ATTENTION_PATCH = False


def local_run_dir(run_name: str) -> Path:
    return LOCAL_RUN_ROOT / run_name


def remote_run_dir(run_name: str) -> str:
    return f"{REMOTE.run_root}/{run_name}"
