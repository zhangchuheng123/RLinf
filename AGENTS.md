# AGENTS.md

## Read This First

This repository is currently being used to debug and improve embodied reinforcement learning on LIBERO. Before making changes, read `CODEX.md` for the latest task history, debugging constraints, and temporary workflows that are still in active use.

## Project Goal

The main goal of this workspace is to make reinforcement learning run reliably on the LIBERO environment and to reach both good sample efficiency and good compute efficiency.

The current milestone is narrower and should drive most near-term work:

- Get DSRL working end to end on LIBERO with SmolVLA.
- Focus on a single LIBERO task first.
- Reach 100% success rate on that single task before expanding scope.

## Directory Ownership

Use the repository with these boundaries in mind:

- `rlinf_noray/` is the primary implementation directory. New logic and bug fixes should go here unless there is a strong reason otherwise.
- `lerobot/` is a reference and alignment directory. It is useful for comparing preprocessing, postprocessing, rollout, and normalization behavior. Keep edits minimal.
- `rlinf/` is the original reference directory. Do not use it as the primary place for new work.
- `examples/embodiment/` contains the main runnable shell entrypoints and Hydra configs for this embodied workflow.
- `exp_bash/` stores managed experiment launch scripts, copied configs, lightweight launch entrypoints, and experiment-specific analysis tools such as WandB fetch scripts. Keep experiment workflows as self-contained as practical because `examples/` is expected to be removed later.
- `models/` stores downloaded model checkpoints, including the local SmolVLA LIBERO model used by the current scripts.
- `requirements/` contains environment setup scripts.

For this project, assume that future work should converge into `rlinf_noray/`, not into `lerobot/` or `rlinf/`.

## Canonical Setup

The actual environment setup path in this repository is:

```bash
bash requirements/install.sh embodied --model smolvla --env maniskill_libero
```

## Canonical Run Entry

The main DSRL entrypoint for this workspace is:

```bash
bash examples/embodiment/run_libero_dsrl_smolvla_noray.sh
```

Useful debugging variants:

```bash
DISABLE_WANDB=1 bash examples/embodiment/run_libero_dsrl_smolvla_noray.sh
DISABLE_WANDB=1 SAVE_ROLLOUT_VIDEO=1 bash examples/embodiment/run_libero_dsrl_smolvla_noray.sh
```

The shell script already configures the common runtime environment, including `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONPATH`, the local `.venv`, and Hydra overrides.

## Main Code Paths

These files define the current DSRL workflow and should be the first places to inspect when debugging or changing behavior:

- `examples/embodiment/run_libero_dsrl_smolvla_noray.sh`
  - Main shell entrypoint.
  - Sets the model path to `models/smolvla_libero`.
  - Launches distributed training with `torchrun`.
  - Applies runtime overrides such as train env count, eval env count, rollout video settings, and logger backend selection.

- `examples/embodiment/train_embodied_agent_noray.py`
  - Dispatches to the no-ray runners.
  - The DSRL path is selected through the `libero_10_dsrl_smolvla` config.

- `examples/embodiment/config/libero_10_dsrl_smolvla.yaml`
  - Canonical Hydra config for the current DSRL workflow.
  - Default precision is `bf16`.
  - Both train and eval are currently pinned to `specific_task_id: 0`.
  - Contains DSRL hyperparameters, replay settings, rollout settings, and logger configuration.

- `rlinf_noray/runners/libero_dsrl_ddp_runner.py`
  - Main DSRL training loop.
  - Contains replay buffer logic, rollout collection, PPO/value updates, and evaluation.
  - This is the primary file for debugging training behavior.

- `rlinf_noray/envs/libero/libero_env_lerobot_adapter.py`
  - LIBERO environment adapter used by the no-ray runner.
  - Handles observation conversion, task descriptions, chunk stepping, and alignment hooks.

## Current Status

The current project status is:

- DSRL is the main line of work.
- The current target is single-task success, not broad multi-task coverage.
- The current runner is the no-ray path.
- The current best run to focus on is `libero_10_dsrl_smolvla_scalar_ddp16x4_mc`.
- The real target is not a single evaluation spike to `100%`. Evaluation volume is small, so the goal is to reach `100%` repeatedly and stably across many evaluations.
- The practical acceptance target is that the most recent 20 evaluation rounds should have mean `eval/success_rate >= 0.98`.
- The current problem is that the run can reach high single-eval success occasionally, but it still does not stay at stable `100%` success over repeated evaluations.

When choosing what to work on next, prioritize DSRL training dynamics, rollout correctness, model inference alignment, and reward/value learning behavior over unrelated cleanup.

## Metric Guide

Use the currently best run `libero_10_dsrl_smolvla_scalar_ddp16x4_mc` as the main reference when interpreting metrics.

### Core Metrics

- `eval/success_rate`
  - The most important outcome metric.
  - Because evaluation volume is small, do not treat a single `1.0` point as success.
  - The real target is repeated and stable `1.0` across many evaluation rounds.
  - The practical acceptance rule is that the most recent 20 evaluations should average at least `0.98`.

- `rollout/success_rate`
  - Success rate measured on the trajectories collected during training rollout for the current epoch.
  - Useful for judging whether online data collection is improving, but it is not the final acceptance metric.

- `rollout/recent_success`
  - Smoothed recent trajectory success statistic from the replay-buffer-side success tracking.
  - Useful for judging whether improvement is sustained rather than coming from one noisy epoch.

- `train/value_loss`
  - Main value learning objective.
  - For `scalar` value head, this is the same as value regression loss.
  - For `distributional` value head, this is the distributional classification loss and should not be compared numerically to scalar MSE.

- `train/value_mse`
  - Scalar regression error between predicted value and target return.
  - For scalar head, this is effectively the same quantity as `train/value_loss`.
  - For distributional head, this is an auxiliary interpretation metric and is easier to compare across runs than cross-entropy.

- `train/ref_value_loss`
  - Reference baseline loss from a simple running-mean value predictor.
  - If the learned value model is not clearly better than this reference, the critic is probably not learning useful signal.

### Debug-Oriented Metrics

- `train/ratio_update0` and `train/ratio_update0_abs_delta_from_1`
  - Debug-only freshness checks for the first PPO update.
  - They should stay extremely close to `1` and `0` respectively when the sampled transitions are being updated for the first time.
  - Small deviations can appear when replay buffer reuse mixes in samples that were already updated in the previous round.
  - Do not use these as the main measure of policy improvement.

- `train/ratio`
  - Mean PPO importance ratio over sampled minibatches.
  - Useful when debugging whether policy updates are moving too much or too little, but it is not a direct task-performance metric.

- `train/ratio_std`
  - Standard deviation of PPO importance ratios inside the sampled minibatch.
  - Useful for checking whether the update is globally tiny or whether only a small subset of samples is moving.

- `train/clip_fraction` and `train/clip_fraction_update0`
  - Fraction of samples whose PPO ratio is outside the clipping range.
  - These help diagnose whether PPO updates are too aggressive or too weak.
  - Interpret them together with success metrics and optimization settings rather than in isolation.

- `train/approx_kl`
  - Approximate KL-style distance between old and new policy log-probabilities.
  - Useful for tracking actual policy movement during PPO updates.

- `train/old_logprob_mean`, `train/new_logprob_mean`, `train/log_ratio_mean`
  - Describe the old/new action log-probability scale and the average PPO log-ratio.
  - These are debugging signals for policy update magnitude, not outcome metrics.

- `train/advantage_raw_mean`, `train/advantage_raw_std`, `train/advantage_raw_abs_mean`, `train/advantage_raw_min`, `train/advantage_raw_max`, `train/advantage_pos_frac`
  - Describe the raw, pre-normalization advantage signal currently used to train the actor.
  - These are the main diagnostics for checking whether actor-side learning signal is strong, sparse, saturated, or sign-imbalanced.

- `train/actor_grad_norm` and `train/value_grad_norm`
  - Gradient norm diagnostics measured before clipping.
  - Useful for checking whether actor/value updates are imbalanced or near-zero.

- `train/value_target_mean`, `train/value_target_min`, `train/value_target_max`
  - Describe the value-learning target distribution currently being fed into the critic.
  - Useful for checking whether targets are collapsed, sparse, or drifting unexpectedly.

- `train/value_pred_mean`, `train/value_pred_min`, `train/value_pred_max`
  - Describe the current critic prediction range.
  - Compare them with the target statistics to judge calibration and saturation.

- `train/value_target_oob_frac`
  - Fraction of value targets falling outside the expected support range.
  - Mainly useful for debugging target/support mismatch, especially with distributional value heads.

- `train/entropy`
  - Policy entropy statistic.
  - Useful for debugging exploration collapse or overly diffuse policies, but not a final success indicator by itself.

- `rollout/noise_latent_mean` and `rollout/noise_latent_std`
  - Describe the sampled latent noise fed into SmolVLA.
  - Useful for checking whether the actor distribution is collapsing or drifting unexpectedly.

- `rollout/actor_mean_mean`, `rollout/actor_mean_std`, `rollout/actor_logstd_mean`, `rollout/actor_logstd_std`
  - Describe the Gaussian actor output statistics in latent-noise space.
  - These are debugging signals for policy distribution drift, scale, and exploration behavior.
  - Do not treat them as direct evidence that task performance improved.

- `train/ppo_loss`
  - PPO objective value.
  - Useful for optimization debugging, but its absolute magnitude is usually not a reliable task-level success metric.

## Working Rules For Agents

Follow these rules when making changes in this workspace:

- Prefer minimal, targeted edits.
- Fail fast. Do not add default fallbacks or silent recovery logic that hides unexpected errors.
- Keep primary implementation work inside `rlinf_noray/`.
- Only modify `lerobot/` when it is necessary for alignment or reference-based comparison.
- Do not move new logic into `rlinf/`.
- Preserve ongoing debugging workflows unless explicitly asked to remove them.
- Temporary alignment or dump code may still be in active use. Do not delete it casually.
- If you add temporary debugging code, make it easy to identify and remove later.

## Experiment Management

When adding new experiment launchers or experiment-specific code, follow these rules:

- Put reusable experiment bash launchers under `exp_bash/`.
- Do not make `exp_bash/` launchers call bash scripts under `examples/`.
- Do not make `exp_bash/` experiment configs depend on Hydra config files under `examples/`.
- Prefer self-contained experiment bundles in `exp_bash/`: bash launcher, copied config, and any tiny experiment-specific entrypoint needed for launching.
- If a WandB or result-analysis tool is tightly coupled to active experiment scripts, put it under `exp_bash/` rather than general tooling directories.
- Keep experiment launchers flat and traceable. Favor local copies over indirect references when that reduces cross-directory coupling.
- Name experiment bash files with `YYYYMMDD` plus concise keywords and important parameter hints.
- In each experiment bash file, record at least: background, goal, and what changed relative to the previous baseline.
- Keep experiment-management notes up to date so future agents can see why each run exists.
- When comparing experiments across different machines, keep algorithm-level settings fixed as global quantities whenever possible.
- In particular, keep `env.train.total_num_envs`, `env.eval.total_num_envs`, rollout horizon settings, PPO update counts, and DSRL PPO minibatch size aligned across machines.
- Treat `algorithm.dsrl_minibatch_size` as a global PPO minibatch size across all DDP ranks, not a per-rank batch size.
- Different machines may run at different speeds, but experiment scripts should avoid changing algorithm-level sample counts just because GPU count changes.
- If DDP world size changes across machines, derive per-rank quantities from the same global experiment settings instead of retuning the algorithm implicitly.
- The long-term direction is to reduce dependence on `examples/` for active DSRL experimentation. New experiment assets should be added under `exp_bash/` unless there is a strong reason not to.

## Practical Guidance

For this repository, a useful default workflow is:

1. Read `CODEX.md` to understand the active debugging context.
2. Reproduce behavior with `examples/embodiment/run_libero_dsrl_smolvla_noray.sh`.
3. Inspect or modify `rlinf_noray/runners/libero_dsrl_ddp_runner.py` first for DSRL logic changes.
4. Inspect or modify `rlinf_noray/envs/libero/libero_env_lerobot_adapter.py` for rollout or environment-interface issues.
5. Compare against `lerobot/` only when alignment with the reference pipeline is required.

Keep the work focused on getting the DSRL path stable and successful on a single LIBERO task.

## Additional Requirement For Agents

- Compress key information/findings and interaction history, and add them to AGENT_HIST.md. 
- In AGENT_HIST.md, you can modify the content within the same day. If it is another day, please append.