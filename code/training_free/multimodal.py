"""Launch the model-specific multimodal runtime with its recorded assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def launch(model_id, profile, extra, dry_run=False, judge=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(extra)
    if judge:
        parser.error('MMMU uses its separate multiple-choice/open-answer evaluator; --judge is a text-evaluation option.')
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Require 0 <= shard-index < shard-count.')
    root = Path(__file__).resolve().parent / 'multimodal_runtime' / model_id
    controller = json.loads((root / 'controller.json').read_text())
    runtime = profile['runtime']
    qwen = model_id.startswith('qwen3.5-')
    sampling = runtime.get('sampling', runtime)
    module = ('experiment_src.run_official_json' if model_id == 'qwen3.5-2b' else
              'experiment_src.qwen35_linear_full_dynamic_final.mmmu_dynamic_async_vllm' if qwen else
              'experiment_src.internvl35_dynamic_vllm_final.internvl_mmmu_dynamic_vllm')
    command = [sys.executable, '-u', '-m', module, '--mode', 'dynamic',
               '--model-path', str(args.model_path.resolve()), '--manifest', str(args.manifest.resolve()),
               '--output', str(args.output.resolve()), '--shard-index', str(args.shard_index),
               '--shard-count', str(args.shard_count), '--max-model-len', '40960', '--dtype', 'bfloat16']
    for key in ('temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed', 'max_new_tokens'):
        command += ['--' + key.replace('_', '-'), str(sampling[key])]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    if qwen:
        for key in ('min_p', 'presence_penalty'):
            command += ['--' + key.replace('_', '-'), str(sampling[key])]
        concurrency = 8 if model_id == 'qwen3.5-4b' else 4
        command += ['--method-cue', 'The', '--resume', '--tensor-parallel-size', '1',
                    '--trace-path', str(args.output.resolve()) + '.trace.jsonl',
                    '--max-concurrency', str(concurrency), '--max-num-seqs', str(concurrency),
                    '--gpu-memory-utilization', '0.88']
    elif 'prefix_selector' in runtime:
        selector = runtime['prefix_selector']
        command += ['--selector-words', ','.join(selector['words']),
                    '--selector-second-word-threshold', str(selector['second_word_threshold'])]
    else:
        command += ['--method-cue', 'Okay']
    env = os.environ.copy()
    # Do not inherit unrelated diagnostic/intervention switches from another run.
    env = {k: v for k, v in env.items() if not k.startswith(('ARIS_', 'VLLM_'))}
    env.update({
        'PYTHONPATH': str(root) + os.pathsep + env.get('PYTHONPATH', ''),
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_FLASH_ATTN_VERSION': '2',
        'ARIS_DYNAMIC_CONFIG_JSON': str(root / 'controller.json'),
        'ARIS_VLLM_V1_SUMMARY_JSON': str(root / 'heads.json'),
        'ARIS_VLLM_V1_END_THINK_TOKEN_IDS': ','.join(map(str, controller['end_think_token_ids'])),
        'ARIS_VLLM_V1_SUPPRESSION_ENABLED': '1', 'ARIS_VLLM_V1_NATIVE_FUSED': '1',
        'ARIS_VLLM_V1_BUDGET': '0', 'ARIS_VLLM_V1_STATS_STDOUT': '0',
        'ARIS_DYNAMIC_TRACE_FSYNC': '1', 'ARIS_DYNAMIC_STATS_ON_FINALIZE': '1',
        'ARIS_FAIL_ON_FALLBACK': '1', 'TOKENIZERS_PARALLELISM': 'false',
    })
    if qwen:
        env.update({
            'ARIS_FULL_ENABLE': '1', 'ARIS_LINEAR_ENABLE': '1', 'ARIS_LINEAR_CONV_REPLACE': '1',
            'ARIS_FULL_BOUNDARY_LEFT_TOKENS': '1', 'ARIS_FULL_BOUNDARY_RIGHT_TOKENS': '1',
            'ARIS_LINEAR_BOUNDARY_LEFT_TOKENS': '1', 'ARIS_LINEAR_BOUNDARY_RIGHT_TOKENS': '1',
            'ARIS_LINEAR_SUMMARY_JSON': str(root / 'heads.json'),
            'ARIS_LINEAR_ALPHA': '1', 'ARIS_LINEAR_LAMBDA': '0',
            'ARIS_LINEAR_MAX_EFFECTIVE_K': str(runtime['expected_max_k']),
            'ARIS_LINEAR_SIDECAR_DTYPE': 'float32', 'ARIS_LINEAR_CANONICAL_FP64': '0',
            'ARIS_LINEAR_CANONICAL_MANTISSA_BITS': '0', 'ARIS_VLLM_ENFORCE_EAGER': '0',
            'ARIS_NATIVE_SUPPRESSION_IMPL': 'integrated', 'ARIS_DYNAMIC_TRACE_BUFFER_PER_REQUEST': '1',
        })
    else:
        env.update({'VLLM_USE_V1': '1', 'VLLM_ATTENTION_BACKEND': 'FLASH_ATTN',
                    'VLLM_USE_FLASHINFER_SAMPLER': '0', 'ARIS_REQUIRE_NATIVE_FUSED': '1',
                    'OMP_NUM_THREADS': '1'})
    for key, suffix in [('ARIS_DYNAMIC_TRACE_PATH', '.trace.jsonl'),
                        ('ARIS_DYNAMIC_STATS_PATH', '.dynamic.json'),
                        ('ARIS_VLLM_V1_STATS_PATH', '.native.json')]:
        env[key] = str(args.output.resolve()) + suffix
    print(shlex.join(command), flush=True)
    if dry_run:
        print(json.dumps({k: v for k, v in env.items() if k.startswith(('ARIS_', 'VLLM_'))}, indent=2))
        return
    if not args.model_path.is_dir() or not args.manifest.is_file():
        parser.error('Model directory and manifest must exist.')
    from install_multimodal import check_environment
    check_environment(model_id, check_overlay=qwen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.run(command, env=env).returncode)
