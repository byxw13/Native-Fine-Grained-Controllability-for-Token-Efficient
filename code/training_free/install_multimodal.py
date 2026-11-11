"""Install the version-specific Qwen3.5 Python overlay; verify native FA2."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import shutil


def check_environment(model_id, check_overlay=False, check_kernel=True):
    qwen = model_id.startswith('qwen3.5-')
    required = {'torch': '2.10.0', 'transformers': '5.14.1', 'vllm': '0.19.1'} if qwen else {
        'torch': '2.8.0', 'transformers': '4.57.6', 'vllm': '0.11.0'}
    for name, expected in required.items():
        actual = metadata.version(name).split('+')[0]
        if actual != expected:
            raise RuntimeError(f'{name}: expected {expected}, installed {actual}. Use a separate environment.')
    root = Path(metadata.distribution('vllm').locate_file('vllm'))
    if check_overlay:
        sources = Path(__file__).parent / 'multimodal_runtime' / model_id / 'vllm_overlay'
        for source in sources.rglob('*.py'):
            target = root / source.relative_to(sources)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise RuntimeError(f'Install the {model_id} overlay first: python {Path(__file__).name} --model-id {model_id}')
    if check_kernel:
        import torch
        import vllm.vllm_flash_attn  # Register the compiled operators, without constructing a GPU engine.
        # vLLM 0.11 使用独立的 ARIS 算子，0.19 才扩展原有 varlen_fwd。
        if not qwen:
            namespace = getattr(torch.ops, '_vllm_fa2_C', None)
            if namespace is None or not hasattr(namespace, 'varlen_fwd_aris'):
                raise RuntimeError('Native ARIS varlen_fwd_aris is missing from the vLLM 0.11 environment.')
            return root
        schema = str(torch._C._dispatch_find_schema_or_throw('_vllm_fa2_C::varlen_fwd', '').schema())
        fields = ('aris_enabled', 'aris_budget_ids', 'aris_end_positions', 'aris_head_mask_table')
        if qwen:
            fields += ('aris_window_left', 'aris_window_right')
        if not all(field in schema for field in fields):
            raise RuntimeError('Native ARIS FlashAttention is missing. Build the matching FA2 patch before inference.')
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', required=True, choices=['qwen3.5-2b', 'qwen3.5-4b', 'qwen3.5-9b',
                                                           'internvl3.5-2b', 'internvl3.5-4b', 'internvl3.5-8b'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    qwen = args.model_id.startswith('qwen')
    root = check_environment(args.model_id, check_overlay=args.check and qwen, check_kernel=args.check)
    if args.check:
        print('Versions, overlay and native FA2 operator: OK')
        return
    if not qwen:
        parser.error('InternVL needs the FA2 kernel, not a Qwen3.5 Python overlay.')
    source_root = Path(__file__).parent / 'multimodal_runtime' / args.model_id / 'vllm_overlay'
    for source in source_root.rglob('*.py'):
        target = root / source.relative_to(source_root)
        if not target.is_file():
            raise FileNotFoundError(target)
        backup = target.with_suffix('.py.pre_aris')
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        print('Installed:', source.relative_to(source_root))


if __name__ == '__main__':
    main()
