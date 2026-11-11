# DS_qwen v3-8 Adaptation

This package is the runnable DS_qwen adaptation of the newer Qwen3 v3-8 control logic.

Key choices:

- The runnable module path is `experiment_src.evaluation_v3_8_ds`.
- The legacy local folder `experiment_src/evaluation_v3_8 _ds` is kept as a synchronized copy for compatibility with the original workspace layout, but the space in its directory name makes it unsuitable for `python -m` execution.
- Model-structure code remains DS_qwen/Qwen2-compatible: runtime attention patching uses `experiment_src.sampling_v1.remote_sampling_runner.custom_qwen2_attention_forward`.
- Control logic follows the Qwen3 v3-8 branch: assistant prefix support, initial suppression, runtime checkpoint control, dynamic/high-low-high/linear-down-up schedules, rethink cue, presence penalty, final-answer prefix, and tail signal insertion.
- Default DS adaptation parameters are `assistant_prefix_text="Alright"`, `initial_suppression=0.1`, `suppression_scale=0.8`, and `tail_signal_trigger_ratio=0.8`.

Use `write_sbatch.py` to generate remote scripts:

```bash
python -m experiment_src.evaluation_v3_8_ds.write_sbatch \
  --dataset math500 \
  --run-name your_run_name \
  --output /tmp/your_run_name.sbatch \
  --reset-seed-per-sample
```
