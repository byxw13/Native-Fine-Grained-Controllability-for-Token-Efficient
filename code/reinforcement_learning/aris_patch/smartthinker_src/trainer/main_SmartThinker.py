# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SmartThinker entry point compatible with pinned VERL model-engine workers."""

import importlib.util
import inspect
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_object

from trainer.ppo.ray_trainer_SmartThinker import RayPPOTrainer

try:
    from verl.utils.device import auto_set_ascend_device_name
except ImportError:

    def auto_set_ascend_device_name(config):
        return None


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _use_engine_workers(config) -> bool:
    """Resolve whether to use VERL's unified engine_workers instead of removed legacy fsdp_workers."""
    mode = config.trainer.get("use_legacy_worker_impl", "auto")
    legacy_available = _has_module("verl.workers.fsdp_workers")
    if mode == "disable":
        return True
    if mode == "auto":
        return not legacy_available
    if mode == "enable":
        if not legacy_available:
            raise RuntimeError(
                "Legacy verl.workers.fsdp_workers is not available in this VERL revision; "
                "set trainer.use_legacy_worker_impl=disable or auto."
            )
        return False
    raise ValueError(f"Invalid use_legacy_worker_impl: {mode}")


def _resolve_need_reference_policy(config, role_worker_mapping) -> bool:
    try:
        params = tuple(inspect.signature(need_reference_policy).parameters)
    except (TypeError, ValueError):
        params = ()
    if params and params[0] == "config":
        return bool(need_reference_policy(config))
    try:
        return bool(need_reference_policy(config))
    except (TypeError, AttributeError):
        return bool(need_reference_policy(role_worker_mapping))


def _ensure_verl_reward_config(config) -> None:
    """Populate `config.reward` from legacy top-level reward fields when needed by VERL."""
    with open_dict(config):
        if config.get("reward") is None:
            config.reward = {}
        reward_cfg = config.reward
        if reward_cfg.get("reward_model") is None and config.get("reward_model") is not None:
            reward_cfg.reward_model = config.reward_model
        if reward_cfg.get("reward_manager") is None and config.get("reward_manager") is not None:
            reward_cfg.reward_manager = config.reward_manager
        if reward_cfg.get("custom_reward_function") is None and config.get("custom_reward_function") is not None:
            reward_cfg.custom_reward_function = config.custom_reward_function


def _reward_model_cfg(config):
    reward_cfg = config.get("reward")
    if reward_cfg is not None and reward_cfg.get("reward_model") is not None:
        return reward_cfg.reward_model
    return config.reward_model


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management."""
    auto_set_ascend_device_name(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker using engine_workers when legacy fsdp_workers is unavailable."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        if _use_engine_workers(config):
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            if _resolve_need_reference_policy(config, self.role_worker_mapping) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        if config.actor_rollout_ref.rollout.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` to use the native server rollout."
            )

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        from verl.trainer.ppo.ray_trainer import Role

        if _use_engine_workers(config):
            from verl.workers.engine_workers import TrainingWorker

            critic_worker_cls = TrainingWorker
        elif config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker

            critic_worker_cls = CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

            critic_worker_cls = CriticWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.Critic] = ray.remote(critic_worker_cls)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        rm_cfg = _reward_model_cfg(config)
        if rm_cfg.enable_resource_pool:
            if rm_cfg.n_gpus_per_node <= 0:
                raise ValueError("reward_model.n_gpus_per_node must be greater than 0")
            if rm_cfg.nnodes <= 0:
                raise ValueError("reward_model.nnodes must be greater than 0")

            reward_pool = [rm_cfg.n_gpus_per_node] * rm_cfg.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_worker(self, config):
        """Add reward model worker/pool if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        rm_cfg = _reward_model_cfg(config)
        if not rm_cfg.enable:
            return

        if _use_engine_workers(config):
            self.mapping[Role.RewardModel] = "reward_pool" if rm_cfg.enable_resource_pool else "global_pool"
            return

        if rm_cfg.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif rm_cfg.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        self.mapping[Role.RewardModel] = "reward_pool" if rm_cfg.enable_resource_pool else "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used in legacy mode."""
        from verl.trainer.ppo.ray_trainer import Role

        if _use_engine_workers(config):
            return

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        """Execute the main PPO training workflow."""
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        _ensure_verl_reward_config(config)
        if _use_engine_workers(config) and config.trainer.get("use_legacy_worker_impl", "auto") == "auto":
            with open_dict(config):
                config.trainer.use_legacy_worker_impl = "disable"

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=_resolve_need_reference_policy(config, self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_kwargs = config.reward_model.get("reward_kwargs", {}).copy()
        train_reward_kwargs = {
            **reward_kwargs,
            "num_generation": config.actor_rollout_ref.rollout.n,
            "ref_model": config.actor_rollout_ref.model.path,
        }
        val_reward_kwargs = {
            **reward_kwargs,
            "num_generation": config.actor_rollout_ref.rollout.val_kwargs.n,
            "ref_model": config.actor_rollout_ref.model.path,
        }
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **train_reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **val_reward_kwargs)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset."""
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset."""
    import torch
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
