from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_PACKET_DIR = REPO_ROOT / "project-context" / "experiments" / "sampling-v1"
LOCAL_STAGE_ROOT = REPO_ROOT / "experiment-src" / "sampling_v1_qwen3" / "runs"
LOCAL_PULL_ROOT = REPO_ROOT / "pulled-results" / "sampling-v1-qwen3"
LOCAL_MONITOR_ROOT = REPO_ROOT / "monitor-logs" / "sampling-v1-qwen3"
LOCAL_EVAL_ROOT = REPO_ROOT / "local-eval" / "sampling-v1-qwen3"


@dataclass(frozen=True)
class LevelSpec:
    level_id: int
    label: str
    assistant_suffix: str
    suppression_budget: Optional[float]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    remote_input_path: str
    total_examples: int
    stage_name: str
    shard_count: int
    gpus_per_job: int
    requires_confirmation_after: bool


@dataclass(frozen=True)
class RemoteSpec:
    server_alias: str
    ssh_target: str
    workdir: str
    code_root: str
    stage_root: str
    model_path: str
    summary_json_path: str
    conda_base: str
    conda_env: str
    candidate_partitions: tuple[str, ...]
    cpus_per_task: int


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(1, "plain_chat", "", None),
    LevelSpec(2, "alright_only", "Alright", None),
    LevelSpec(3, "alright_sup_0p2", "Alright", 0.2),
    LevelSpec(4, "alright_sup_0p4", "Alright", 0.4),
    LevelSpec(5, "alright_sup_0p6", "Alright", 0.6),
)
LEVEL_BY_ID = {item.level_id: item for item in LEVEL_SPECS}

DATASETS: dict[str, DatasetSpec] = {
    "smoke": DatasetSpec(
        name="smoke",
        remote_input_path="/path/to/user_data/online1/HWY/difficulity/RL/math/math_val_last500.jsonl",
        total_examples=1,
        stage_name="stage0_smoke",
        shard_count=1,
        gpus_per_job=1,
        requires_confirmation_after=True,
    ),
    "val": DatasetSpec(
        name="val",
        remote_input_path="/path/to/user_data/online1/HWY/difficulity/RL/math/math_val_last500.jsonl",
        total_examples=500,
        stage_name="stage1_val500",
        shard_count=4,
        gpus_per_job=2,
        requires_confirmation_after=True,
    ),
    "train": DatasetSpec(
        name="train",
        remote_input_path="/path/to/user_data/online1/HWY/difficulity/RL/math/math_train_first5000.jsonl",
        total_examples=5000,
        stage_name="stage2_train5000",
        shard_count=4,
        gpus_per_job=2,
        requires_confirmation_after=False,
    ),
}

REMOTE = RemoteSpec(
    server_alias="h20",
    ssh_target="SET_ARIS_SSH_TARGET",
    workdir="/path/to/user_data/online1/HWY/difficulity/RL",
    code_root="/path/to/user_data/online1/HWY/difficulity/RL/aris_sampling_v1_code",
    stage_root="/path/to/user_data/online1/HWY/difficulity/RL/aris_sampling_v1",
    model_path="/path/to/user_data/online1/HWY/difficulity/DeepSeek-R1-Distill-Qwen-7B",
    summary_json_path="/path/to/user_data/online1/HWY/difficulity/RL/Dsqwen_7B_head_scores_summary.json",
    conda_base="/path/to/conda",
    conda_env="llama",
    candidate_partitions=(
        "YOUR_H20_PARTITION",
        "YOUR_H20_PARTITION",
        "YOUR_GPU_PARTITION",
    ),
    cpus_per_task=12,
)

CHECKPOINT_NEWLINE_SPAN = 10
DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.6
LOCAL_EVAL_SCRIPT = EXPERIMENT_PACKET_DIR / "evaluate_budget_results_attn_alright.py"


def shard_bounds(total_examples: int, shard_count: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for shard_id in range(shard_count):
        start = total_examples * shard_id // shard_count
        end = total_examples * (shard_id + 1) // shard_count
        bounds.append((start, end))
    return bounds


def stage_dir(stage_name: str) -> Path:
    return LOCAL_STAGE_ROOT / stage_name


def pulled_stage_dir(stage_name: str) -> Path:
    return LOCAL_PULL_ROOT / stage_name


def monitor_log_path(stage_name: str) -> Path:
    return LOCAL_MONITOR_ROOT / f"{stage_name}.log"


def local_eval_dir(stage_name: str) -> Path:
    return LOCAL_EVAL_ROOT / stage_name


def remote_stage_dir(stage_name: str) -> str:
    return f"{REMOTE.stage_root}/{stage_name}"


def manifest_dict(dataset: DatasetSpec) -> dict:
    return {
        "dataset": dataset.name,
        "stage_name": dataset.stage_name,
        "remote_stage_dir": remote_stage_dir(dataset.stage_name),
        "remote_input_path": dataset.remote_input_path,
        "requires_confirmation_after": dataset.requires_confirmation_after,
        "shards": [
            {
                "shard_id": shard_id,
                "start": start,
                "end": end,
            }
            for shard_id, (start, end) in enumerate(shard_bounds(dataset.total_examples, dataset.shard_count))
        ],
        "levels": [asdict(level) for level in LEVEL_SPECS],
        "remote": {
            "server_alias": REMOTE.server_alias,
            "ssh_target": REMOTE.ssh_target,
            "workdir": REMOTE.workdir,
            "code_root": REMOTE.code_root,
            "stage_root": REMOTE.stage_root,
            "model_path": REMOTE.model_path,
            "summary_json_path": REMOTE.summary_json_path,
            "conda_base": REMOTE.conda_base,
            "conda_env": REMOTE.conda_env,
            "candidate_partitions": list(REMOTE.candidate_partitions),
            "cpus_per_task": REMOTE.cpus_per_task,
            "gpus_per_job": dataset.gpus_per_job,
        },
    }
