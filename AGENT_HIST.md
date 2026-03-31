# AGENT_HIST.md

## 2026-03-31

### Request

Analyze the current DSRL plateau without modifying the training code. Read AGENTS.md first, fetch WandB experiment results through a separate module, and then propose debugging and tuning directions.

### What Was Added

- Added a standalone WandB analysis module: `toolkits/wandb/fetch_dsrl_wandb_runs.py`
- Generated analysis outputs:
  - `logs/analysis/dsrl_wandb_summary.json`
  - `logs/analysis/dsrl_wandb_current_run.json`
  - `logs/analysis/dsrl_wandb_current_run_actor.json`

### WandB Access Facts

- WandB host/entity/project/key were read from `examples/embodiment/config/libero_10_dsrl_smolvla.yaml`.
- Remote WandB API access works for `https://microsoft-research.wandb.io`.
- Local DSRL run directories under `logs/*libero_10_dsrl_smolvla_noray/wandb/` also exist and were used to cross-check older summaries.

### Main Findings

- The current main run is `libero_10_dsrl_smolvla_scalar_ddp16x4_mc`.
- A single `eval/success_rate = 1.0` is not sufficient evidence of success because evaluation volume is small.
- For this run, the stable-success requirement is still not met:
  - Longest consecutive `eval/success_rate = 1.0` streak is only `1`.
  - Last-50-eval mean is about `0.365`.
  - Last-20-eval mean is about `0.375`.
  - Last-10-eval mean is about `0.325`.
- The run still shows some training progress:
  - `rollout/success_rate` improved from about `0.22` to about `0.45`.
  - `rollout/recent_success` improved from about `0.25` to about `0.38`.
- Critic learning is not the main bottleneck in the current run:
  - `train/value_mse` dropped from about `0.049` to about `0.035`.
  - `train/value_mse` is better than `train/ref_value_loss`.
  - `train/value_target_oob_frac` stayed at `0.0`.
- The earlier interpretation of `train/ratio_update0_abs_delta_from_1` was too aggressive.
  - This metric is a debug freshness check and is expected to stay near `0` because most samples are updated for the first time.
  - It should not be used as a primary indicator of actor stagnation.
- More relevant actor-side observations are:
  - `train/clip_fraction` stays exactly `0.0`.
  - `train/ppo_loss` remains numerically small and oscillatory.
  - `rollout` success metrics improve only modestly and do not converge to stable perfect success.
- Actor distribution statistics are drifting:
  - `rollout/actor_mean_std` grew from about `0.0036` to about `0.091`.
  - `rollout/actor_logstd_mean` moved slightly negative.
  - `rollout/noise_latent_std` changed only slightly.
- Interpretation: the policy distribution is moving somewhat, but the current setup still does not convert that drift into stable `100%` repeated evaluation success.

### Older Run Cross-Checks

- Local `wandb-summary.json` for `libero_10_dsrl_smolvla_enhance_value_scalar` showed near-zero value loss but zero success and non-zero `value_target_oob_frac`.
- This suggests that making the critic loss look good in isolation is not sufficient and can become misleading.

### Recommended Next Focus

- Focus on stable repeated evaluation success, not isolated evaluation spikes.
- Focus on actor-side learning strength and advantage signal quality, not on critic stabilization first.
- Inspect advantage magnitude/distribution next.
- Rebalance actor-vs-value optimization budget for the current best scalar MC run.
- Treat eval spikes cautiously; use repeated-eval stability or rolling averages when judging whether the plateau is real.

### Follow-up Changes On 2026-03-31

- Updated `AGENTS.md` to:
  - require recent-20-eval mean `eval/success_rate >= 0.98` as the practical acceptance target;
  - separate core metrics from debug-oriented metrics;
  - clarify that `ratio_update0` and `ratio_update0_abs_delta_from_1` are freshness/debug metrics rather than primary outcome metrics;
  - add cross-machine experiment-management rules and global-quantity requirements.
- Added actor-side training diagnostics to `rlinf_noray/runners/libero_dsrl_ddp_runner.py`:
  - `train/approx_kl`
  - `train/ratio_std`
  - `train/old_logprob_mean`, `train/new_logprob_mean`, `train/log_ratio_mean`
  - `train/advantage_raw_mean`, `train/advantage_raw_std`, `train/advantage_raw_abs_mean`, `train/advantage_raw_min`, `train/advantage_raw_max`, `train/advantage_pos_frac`
  - `train/actor_grad_norm`, `train/value_grad_norm`
- Changed `algorithm.dsrl_minibatch_size` semantics in the no-ray DSRL runner to mean global PPO minibatch size across all DDP ranks.
  - The runner now derives a per-rank local minibatch from the same global setting.
  - This keeps PPO sample count per update aligned across machines with different world sizes.
- Updated `examples/embodiment/run_libero_dsrl_smolvla_noray.sh` so experiment-critical knobs can be overridden from the shell:
  - global DSRL minibatch size
  - actor/value learning rates
  - rollout/update/pre-value-update epochs
  - train/eval rollout horizon settings
  - automatic `NPROC_PER_NODE` selection that respects divisibility of global env counts and global minibatch size
- Added managed experiment launchers under `exp_bash/`:
  - `20260331_baseline_scalar_mc_env16_exec4_mb2048.sh`
  - `20260331_actorlr2e5_scalar_mc_env16_exec4_mb2048.sh`
  - `20260331_actorlr2e5_valuelr15e4_scalar_mc_env16_exec4_mb2048.sh`
- Added experiment-management documentation:
  - `exp_bash/README.md`
  - `exp_bash/EXPERIMENT_LOG.md`

### Follow-up Refactor On 2026-03-31

- Refactored `exp_bash/` so experiment assets no longer depend on `examples/` bash scripts or Hydra configs.
- Added a self-contained DSRL launch entrypoint:
  - `exp_bash/train_libero_dsrl_smolvla_noray.py`
- Added copied standalone configs under `exp_bash/config/`:
  - `20260331_baseline_scalar_mc_env16_exec4_mb2048.yaml`
  - `20260331_actorlr2e5_scalar_mc_env16_exec4_mb2048.yaml`
  - `20260331_actorlr2e5_valuelr15e4_scalar_mc_env16_exec4_mb2048.yaml`
- Rewrote all current `exp_bash/*.sh` launchers to:
  - launch `torchrun` directly;
  - use only `exp_bash/` local config and local entrypoint assets;
  - keep cross-machine global env counts and global PPO minibatch size fixed.
- Updated `AGENTS.md`, `exp_bash/README.md`, and `exp_bash/EXPERIMENT_LOG.md` to encode the principle that new experiments should be flat, traceable, and as independent from `examples/` as practical because `examples/` is expected to be removed later.

### WandB Migration On 2026-03-31

- Migrated experiment-oriented WandB analysis into `exp_bash/` by adding:
  - `exp_bash/fetch_exp_wandb_runs.py`
- The new script is designed to stay co-located with managed launchers and configs.
  - By default it scans `exp_bash/config/*.yaml` to discover managed experiment names and WandB settings.
  - It supports explicit legacy-run lookup through `--run-name` or substring filtering through `--name-contains`.
- Fixed two validation issues in the new script:
  - avoided resolving the full Hydra config so missing environment variables like `EMBODIED_PATH` do not break metadata loading;
  - changed WandB history collection to scan one metric at a time so sparse metric logging does not collapse summaries to empty values.
- Validation result:
  - the default config-driven scan can still return no matches until the new `exp_bash/*.sh` experiments are actually launched to WandB;
  - explicit validation against legacy run `libero_10_dsrl_smolvla_scalar_ddp16x4_mc` now works and correctly reports recent eval stability summaries plus available rollout/value metrics.
- Important interpretation note:
  - newly added actor diagnostics such as `train/advantage_raw_abs_mean`, `train/actor_grad_norm`, and `train/approx_kl` can still be missing for old runs that started before those metrics were added to the runner.

### Algorithm Audit Clarification On 2026-03-31

- Re-audited `rlinf_noray/runners/libero_dsrl_ddp_runner.py` because current symptoms suggested actor/value might not be learning the right objective rather than merely needing LR tuning.
- Confirmed that actor log-probability is internally consistent:
  - `GaussianPolicy.sample()` logprob,
  - `GaussianPolicy.evaluate_actions()` logprob,
  - and `distribution.log_prob()` all match numerically.
- Reverted an intermediate chunk-discount patch after clarifying the intended design with the user.
- Current intended DSRL semantics are:
  - `algorithm.gamma` is chunk-level gamma and should be read as `chunk_gamma` in runner reasoning;
  - replay-transition rewards stay as chunk reward sums under that chunk-level semantics;
  - critic training remains fixed to Monte Carlo returns for the current stability-first phase;
  - `pre_value_update_epoch` intentionally means extra value-only updates every epoch before PPO updates, not a one-time startup warmup.
- Current next step from this audit is to try larger DSRL actor/value networks rather than further reinterpret the discount semantics.

### Larger DSRL Net Experiment On 2026-03-31

- Updated the no-ray DSRL runner so actor/value MLP depth is configurable through config instead of being hard-coded.
- Added explicit baseline config fields for:
  - `actor.model.dsrl_actor_num_layers = 3`
  - `actor.model.dsrl_value_num_layers = 2`
- Replaced the non-baseline `exp_bash/` 20260331 experiment variants with a single network-capacity experiment bundle:
  - `exp_bash/20260331_scalar_mc_env16_exec4_mb2048_increase_net.sh`
  - `exp_bash/config/20260331_scalar_mc_env16_exec4_mb2048_increase_net.yaml`
- The new managed experiment keeps the scalar MC baseline algorithm fixed and changes only DSRL network capacity:
  - `dsrl_hidden_dim = 512`
  - actor hidden layers: `3 -> 4`
  - value hidden layers: `2 -> 4`