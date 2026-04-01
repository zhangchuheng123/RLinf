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

### WandB Comparative Readout On 2026-03-31 (ddp16x4_mc vs increase_net)

- Read `AGENTS.md` first, then fetched WandB directly from `https://microsoft-research.wandb.io`.
- Pulled latest runs by substring and exported full metric histories to:
  - `logs/wandb_analysis/latest_ddp16x4_mc_and_increase_net.json`
  - `logs/wandb_analysis/latest_two_runs_metric_summary.json`
- Latest matched runs:
  - `ddp16x4_mc`: `libero_10_dsrl_smolvla_scalar_ddp16x4_mc` (created `2026-03-30T03:00:07Z`, running)
  - `increase_net`: `20260331_scalar_mc_env16_exec4_mb2048_increase_net` (created `2026-03-31T10:28:42Z`, running)
- Config delta check confirms `increase_net` only changes DSRL net capacity:
  - `dsrl_hidden_dim: 256 -> 512`
  - `dsrl_actor_num_layers: 3 -> 4`
  - `dsrl_value_num_layers: 2 -> 4`
- Outcome metrics:
  - `ddp16x4_mc`: `eval/success_rate` first=0.25, last=0.25, max=1.0, recent20_mean=0.425, full-success frac in recent20=0.0
  - `increase_net`: `eval/success_rate` first=0.3125, last=0.375, max=0.625, recent20_mean=0.378, full-success frac in recent20=0.0
  - `rollout/success_rate` improves in both runs but does not translate to stable eval success.
- Critic metrics remain healthy in both runs:
  - `train/value_mse` drops (`~0.048 -> ~0.040`), clearly below `train/ref_value_loss` at tail.
  - `train/value_target_oob_frac` stays `0.0`.
- Actor-side diagnostics (available in `increase_net`) show update magnitude is extremely small:
  - `train/ratio ~ 1.0`, `train/ratio_std ~ 3.4e-4`
  - `train/approx_kl ~ 1e-7`
  - `train/clip_fraction = 0.0`
  - Interpretation: PPO updates are too weak to produce robust policy improvement.
- Advantage/gradient in `increase_net`:
  - `train/advantage_raw_abs_mean` increases (`0.034 -> 0.091`) and `train/advantage_raw_std` increases.
  - `train/actor_grad_norm` remains low and nearly flat (`~0.019`).
  - `train/value_grad_norm` rises strongly (`~0.086 -> ~0.431`).
  - Interpretation: actor signal exists but effective actor movement is still underpowered versus critic-side adaptation.
- Main diagnosis now:
  - The bottleneck is not value fitting; it is actor update strength / PPO step size effectiveness.
  - Enlarging actor/value networks alone did not fix stability; eval remains far below acceptance (`recent-20 mean >= 0.98`).

### Trend Protocol + Runner Metrics Fix On 2026-03-31

- Recomputed run-trend analysis using a mandatory smoothing protocol:
  - moving average first,
  - then 5 evenly spaced sampled points from the smoothed curve (including start/end).
- Saved structured outputs:
  - `logs/wandb_analysis/latest_two_runs_moving_average_5pts.json`
- Updated `AGENTS.md` to include this protocol as a required default under Metric Guide.

- Updated `rlinf_noray/runners/libero_dsrl_ddp_runner.py`:
  - Added configurable advantage normalization behavior in replay GAE preparation:
    - `algorithm.do_adv_norm` (default `True`)
    - `algorithm.adv_norm_eps` (default `1e-8`)
    - `algorithm.advantage_clip` (default `3.0`)
  - Added normalized-advantage logging metrics in PPO update:
    - `train/advantage_mean`
    - `train/advantage_std`
    - `train/advantage_abs_mean`
  - Kept existing raw-advantage metrics for diagnostics.
  - Fixed rollout/eval length numerator to count only actually executed sub-steps per env inside each chunk,
    instead of counting skipped trailing steps after `done`.
  - Added rollout done-reason diagnostics:
    - `rollout/truncation_trajectories`
    - `rollout/truncation_rate`

- Updated active managed experiment configs under `exp_bash/config/` to include:
  - `algorithm.do_adv_norm: true`
  - `algorithm.adv_norm_eps: 1.0e-8`
  - `algorithm.advantage_clip: 3.0`

## 2026-04-01

### Final Effective Changes

- Restored past-date experiment configs to keep historical experiments unchanged:
  - `exp_bash/config/20260331_baseline_scalar_mc_env16_exec4_mb2048.yaml`
  - `exp_bash/config/20260331_scalar_mc_env16_exec4_mb2048_increase_net.yaml`

- Updated rollout metrics in `rlinf_noray/runners/libero_dsrl_ddp_runner.py`:
  - removed truncation-specific rollout metrics;
  - added done-only `rollout/average_length`;
  - kept `rollout/average_length_running` as running length statistic.

- Kept PPO actor training on normalized advantages (`transition.advantage`) and removed no-op refactor noise.

- Final managed experiment for today:
  - launcher: `exp_bash/20260401_scalar_mc_env16_exec4_mb2048_increase_actor_lr.sh`
  - config: `exp_bash/config/20260401_scalar_mc_env16_exec4_mb2048_increase_actor_lr.yaml`
  - experiment name: `20260401_scalar_mc_env16_exec4_mb2048_increase_actor_lr`
  - actor LR: `1.0e-4`
  - value LR: `1.0e-4`
  - enabled normalized-advantage settings:
    - `algorithm.do_adv_norm: true`
    - `algorithm.adv_norm_eps: 1.0e-8`
    - `algorithm.advantage_clip: 3.0`

- Updated experiment notes in `exp_bash/EXPERIMENT_LOG.md` to match the final experiment definition.

### Workspace Sync Operation On 2026-04-01

- Cloned target repository under home path:
  - `~/eai-sim-rl` (SSH remote: `git@github.com:zhangchuheng123/eai-sim-rl.git`)
- Copied current workspace content from `~/RLinf/` into `~/eai-sim-rl/` using `rsync` with explicit exclusions.
- Excluded from copy:
  - git history from source: `.git/`
  - top-level directories: `examples/`, `docs/`, `docker/`, `logs/`, `ray_utils/`, `rlinf/`
  - log files: `*.log`
- Verified target remains a valid git work tree after sync.