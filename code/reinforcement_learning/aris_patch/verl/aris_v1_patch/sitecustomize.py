"""ARIS vLLM V1 auto patch and VERL compatibility layer for SmartThinker."""

from __future__ import annotations

import builtins as _aris_builtins
import inspect as _aris_inspect
import os as _aris_os
import sys as _aris_sys


def _cmdline_text() -> str:
    try:
        with open("/proc/self/cmdline", "rb") as handle:
            return handle.read().replace(b"\x00", b" ").decode(errors="ignore")
    except OSError:
        return " ".join(_aris_sys.argv)


def _should_skip_auto_install() -> bool:
    cmdline = _cmdline_text()
    skip_markers = (
        "multiprocessing.resource_tracker",
        "ray/_private/workers/default_worker.py",
        "ray/_private/log_monitor.py",
        "ray/_private/runtime_env/agent/main.py",
        "ray/autoscaler/_private/monitor.py",
        "ray/dashboard/agent.py",
        "ray/dashboard/dashboard.py",
        "bin/ray",
        " ray start",
        " ray stop",
        "main_SmartThinker.py",
        "vllm.model_executor.models.registry",
    )
    return any(marker in cmdline for marker in skip_markers)


if (
    _aris_os.getenv("ARIS_VLLM_V1_AUTO_INSTALL", "").lower() in {"1", "true", "yes", "on"}
    and not _should_skip_auto_install()
):
    try:
        import aris_vllm_v1_suppression as _aris_v1

        _aris_v1.configure_from_env()
        _aris_v1.install_aris_vllm_v1_suppression()
        if _aris_os.getenv("ARIS_HIDDEN_TRACE_ENABLED", "").lower() in {"1", "true", "yes", "on"}:
            import aris_hidden_trace_autohook as _aris_hidden_trace

            _aris_hidden_trace.install_from_env()
    except Exception as exc:  # pragma: no cover
        print(f"ARIS_VLLM_V1_AUTOINSTALL_FAILED {exc}", flush=True)


_ARIS_ORIG_IMPORT = _aris_builtins.__import__
_ARIS_PATCHING = False


def _skip_internal_ray() -> bool:
    argv = " ".join(_aris_sys.argv)
    return any(
        token in argv
        for token in (
            "ray/dashboard/agent.py",
            "ray/_private/runtime_env/agent/main.py",
            "ray/_private/log_monitor.py",
            "ray.util.client.server",
        )
    )


def _patch_device() -> None:
    device_mod = _aris_sys.modules.get("verl.utils.device")
    if device_mod is not None and not hasattr(device_mod, "auto_set_ascend_device_name"):

        def auto_set_ascend_device_name(config):
            return None

        device_mod.auto_set_ascend_device_name = auto_set_ascend_device_name


def _patch_reward() -> None:
    reward_mod = _aris_sys.modules.get("verl.trainer.ppo.reward")
    if reward_mod is None:
        return
    if not hasattr(reward_mod, "compute_reward"):

        def compute_reward(data, reward_fn):
            try:
                result = reward_fn(data, return_dict=True)
            except TypeError:
                result = reward_fn(data)
            if isinstance(result, dict):
                return result.get("reward_tensor"), result.get("reward_extra_info", {})
            return result, {}

        reward_mod.compute_reward = compute_reward
    if not hasattr(reward_mod, "compute_reward_async"):

        class _LazyComputeRewardAsync:
            def remote(self, *args, **kwargs):
                import ray as _ray

                @_ray.remote
                def _compute_reward_async(data, config=None, tokenizer=None, reward_fn=None):
                    import verl.trainer.ppo.reward as _reward_mod

                    if reward_fn is None:
                        if config is None or tokenizer is None:
                            raise ValueError("compute_reward_async needs reward_fn or config+tokenizer")
                        reward_fn = _reward_mod.load_reward_manager(config, tokenizer, num_examine=0)
                    return _reward_mod.compute_reward(data, reward_fn)

                return _compute_reward_async.remote(*args, **kwargs)

        reward_mod.compute_reward_async = _LazyComputeRewardAsync()


def _patch_ppo_utils() -> None:
    utils_mod = _aris_sys.modules.get("verl.trainer.ppo.utils")
    if utils_mod is None or getattr(utils_mod, "_aris_mapping_compat", False):
        return
    orig_need_reference_policy = getattr(utils_mod, "need_reference_policy", None)
    orig_need_reward_model = getattr(utils_mod, "need_reward_model", None)
    if orig_need_reference_policy is None or orig_need_reward_model is None:
        return

    def need_reference_policy(config_or_mapping):
        if isinstance(config_or_mapping, dict):
            return any(getattr(role, "name", str(role)) in {"RefPolicy", "ActorRolloutRef"} for role in config_or_mapping)
        return orig_need_reference_policy(config_or_mapping)

    def need_reward_model(config_or_mapping):
        if isinstance(config_or_mapping, dict):
            return any(getattr(role, "name", str(role)) == "RewardModel" for role in config_or_mapping)
        reward_cfg = config_or_mapping.get("reward") if hasattr(config_or_mapping, "get") else None
        if reward_cfg is None or reward_cfg.get("reward_model") is None:
            rm_cfg = config_or_mapping.get("reward_model") if hasattr(config_or_mapping, "get") else None
            return bool(rm_cfg.get("enable", False)) if rm_cfg is not None else False
        return orig_need_reward_model(config_or_mapping)

    utils_mod.need_reference_policy = need_reference_policy
    utils_mod.need_reward_model = need_reward_model
    utils_mod._aris_mapping_compat = True


def _wrap_init_filter_unknown_kwargs(cls, extra_defaults=None) -> None:
    if cls is None or getattr(cls, "_aris_filters_unknown_kwargs", False):
        return
    orig_init = cls.__init__
    allowed_keys = {
        name
        for name, param in _aris_inspect.signature(orig_init).parameters.items()
        if name != "self"
        and param.kind
        in (
            _aris_inspect.Parameter.POSITIONAL_OR_KEYWORD,
            _aris_inspect.Parameter.KEYWORD_ONLY,
        )
    }

    def __init__(self, *args, **kwargs):
        if extra_defaults:
            for key, value in extra_defaults.items():
                kwargs.setdefault(key, value)
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_keys}
        orig_init(self, *args, **filtered_kwargs)

    cls.__init__ = __init__
    cls._aris_filters_unknown_kwargs = True


def _patch_profiler_config() -> None:
    profiler_config = _aris_sys.modules.get("verl.utils.profiler.config")
    if profiler_config is None:
        return
    _wrap_init_filter_unknown_kwargs(getattr(profiler_config, "TorchProfilerToolConfig", None))
    _wrap_init_filter_unknown_kwargs(getattr(profiler_config, "NsightToolConfig", None))
    _wrap_init_filter_unknown_kwargs(getattr(profiler_config, "TorchMemoryToolConfig", None))
    _wrap_init_filter_unknown_kwargs(
        getattr(profiler_config, "ProfilerConfig", None),
        extra_defaults={"tool_config": None},
    )


def _patch_rollout_config() -> None:
    rollout_config = _aris_sys.modules.get("verl.workers.config.rollout")
    if rollout_config is None:
        return

    multi_turn_cfg = getattr(rollout_config, "MultiTurnConfig", None)
    if multi_turn_cfg is not None and not getattr(multi_turn_cfg, "_aris_accepts_interaction_config_path", False):
        orig_multi_turn_init = multi_turn_cfg.__init__

        def __init__(self, *args, interaction_config_path=None, **kwargs):
            if interaction_config_path is not None and kwargs.get("tool_config_path") is None:
                kwargs["tool_config_path"] = interaction_config_path
            orig_multi_turn_init(self, *args, **kwargs)

        multi_turn_cfg.__init__ = __init__
        multi_turn_cfg._aris_accepts_interaction_config_path = True

    _wrap_init_filter_unknown_kwargs(getattr(rollout_config, "RolloutConfig", None))


def _patch_agent_loop_manager() -> None:
    for module_name in ("verl.experimental.agent_loop", "verl.experimental.agent_loop.agent_loop"):
        agent_loop_mod = _aris_sys.modules.get(module_name)
        if agent_loop_mod is None:
            continue
        manager_cls = getattr(agent_loop_mod, "AgentLoopManager", None)
        if manager_cls is None or getattr(manager_cls, "_aris_accepts_worker_group_v2", False):
            continue

        orig_init = manager_cls.__init__
        orig_generate_sequences = getattr(manager_cls, "generate_sequences", None)

        def __init__(self, *args, worker_group=None, rm_resource_pool=None, **kwargs):
            if worker_group is not None and "llm_client" not in kwargs:
                import asyncio as _asyncio

                from verl.checkpoint_engine import CheckpointEngineManager
                from verl.utils.config import omega_conf_to_dataclass
                from verl.workers.rollout.llm_server import LLMServerManager

                config = kwargs.get("config", args[0] if args else None)
                self._aris_worker_group_compat = True
                self._aris_worker_group = worker_group
                self.llm_server_manager = LLMServerManager.create(
                    config=config,
                    worker_group=worker_group,
                    rollout_resource_pool=None,
                )
                orig_init(
                    self,
                    config=config,
                    llm_client=self.llm_server_manager.get_client(),
                    teacher_client=None,
                    reward_loop_worker_handles=None,
                )
                init_result = self._init_agent_loop_workers()
                if _aris_inspect.isawaitable(init_result):
                    try:
                        _asyncio.get_running_loop()
                    except RuntimeError:
                        _asyncio.run(init_result)
                    else:
                        raise RuntimeError("Cannot initialize AgentLoopManager compat inside a running event loop")

                checkpoint_engine_config = omega_conf_to_dataclass(
                    config.actor_rollout_ref.rollout.checkpoint_engine
                )
                self.checkpoint_manager = CheckpointEngineManager(
                    config=checkpoint_engine_config,
                    trainer=worker_group,
                    replicas=self.llm_server_manager.get_replicas(),
                )
                self.checkpoint_manager.sleep_replicas()
                return
            orig_init(self, *args, **kwargs)

        def generate_sequences(self, prompts):
            if getattr(self, "_aris_worker_group_compat", False):
                global_steps = None
                try:
                    global_steps = prompts.meta_info.get("global_steps")
                except Exception:
                    pass
                self.checkpoint_manager.update_weights(global_steps=global_steps)
                try:
                    return orig_generate_sequences(self, prompts)
                finally:
                    self.checkpoint_manager.sleep_replicas()
            return orig_generate_sequences(self, prompts)

        manager_cls.__init__ = __init__
        if orig_generate_sequences is not None:
            manager_cls.generate_sequences = generate_sequences
        manager_cls._aris_accepts_worker_group_v2 = True


def _patch_rlhf_dataset() -> None:
    dataset_mod = _aris_sys.modules.get("verl.utils.dataset.rl_dataset")
    if dataset_mod is None:
        return
    dataset_cls = getattr(dataset_mod, "RLHFDataset", None)
    if dataset_cls is None or getattr(dataset_cls, "_aris_legacy_tokenized_getitem", False):
        return
    orig_getitem = dataset_cls.__getitem__

    def __getitem__(self, item):
        if getattr(self, "processor", None) is not None:
            return orig_getitem(self, item)

        import torch as _torch

        row = dict(self.dataframe[item])
        messages = self._build_messages(row, key=self.prompt_key)
        apply_kwargs = dict(getattr(self, "apply_chat_template_kwargs", {}) or {})
        apply_kwargs.pop("tokenize", None)
        apply_kwargs.pop("return_dict", None)
        apply_kwargs.pop("return_tensors", None)
        if getattr(self, "tool_schemas", None) is not None:
            apply_kwargs["tools"] = self.tool_schemas

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **apply_kwargs,
        )
        try:
            from verl.utils.tokenizer import normalize_token_ids

            ids = list(normalize_token_ids(tokenized))
        except Exception:
            ids = list(tokenized)

        max_len = int(getattr(self, "max_prompt_length", 1024))
        if len(ids) > max_len:
            truncation = getattr(self, "truncation", "error")
            if truncation == "left":
                ids = ids[-max_len:]
            elif truncation == "right":
                ids = ids[:max_len]
            else:
                raise ValueError(f"Prompt length {len(ids)} exceeds max_prompt_length={max_len}")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = _torch.tensor(ids, dtype=_torch.long)
        attention_mask = _torch.ones_like(input_ids, dtype=_torch.long)
        pad_len = max_len - input_ids.numel()
        if pad_len > 0:
            pad_ids = _torch.full((pad_len,), pad_id, dtype=_torch.long)
            pad_mask = _torch.zeros((pad_len,), dtype=_torch.long)
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                input_ids = _torch.cat((pad_ids, input_ids), dim=0)
                attention_mask = _torch.cat((pad_mask, attention_mask), dim=0)
            else:
                input_ids = _torch.cat((input_ids, pad_ids), dim=0)
                attention_mask = _torch.cat((attention_mask, pad_mask), dim=0)

        position_ids = _torch.clamp(_torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        row.pop(getattr(self, "image_key", "images"), None)
        row.pop(getattr(self, "video_key", "videos"), None)
        row["input_ids"] = input_ids
        row["attention_mask"] = attention_mask
        row["position_ids"] = position_ids

        if getattr(self, "return_raw_chat", False):
            row["raw_prompt"] = messages
        if getattr(self, "return_full_prompt", False):
            row["full_prompts"] = self.tokenizer.decode(
                input_ids[attention_mask.bool()],
                skip_special_tokens=False,
            )

        if "extra_info" not in row or row["extra_info"] is None:
            row["extra_info"] = {}
        extra_info = row.get("extra_info") or {}
        row["index"] = extra_info.get("index", 0)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        return row

    dataset_cls.__getitem__ = __getitem__
    dataset_cls._aris_legacy_tokenized_getitem = True


def _apply_patches() -> None:
    try:
        _patch_device()
        _patch_reward()
        _patch_ppo_utils()
        _patch_profiler_config()
        _patch_rollout_config()
        _patch_agent_loop_manager()
        _patch_rlhf_dataset()
    except Exception:
        pass


def _aris_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _ARIS_PATCHING
    module = _ARIS_ORIG_IMPORT(name, globals, locals, fromlist, level)
    if not _ARIS_PATCHING and (
        name == "verl"
        or name.startswith("verl.")
    ):
        _ARIS_PATCHING = True
        try:
            _apply_patches()
        finally:
            _ARIS_PATCHING = False
    return module


if not _skip_internal_ray():
    _aris_builtins.__import__ = _aris_import
    _apply_patches()
