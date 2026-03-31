# Experiment Log

## 2026-03-31

### exp_bash/fetch_exp_wandb_runs.py

- Background: experiment launchers and configs were moved to `exp_bash/` to reduce dependence on `examples/`.
- Goal: keep WandB analysis coupled to the same self-contained experiment bundle rather than depending on general tooling paths.
- Changed vs previous setup: added an `exp_bash/`-local WandB fetch/summarize script that discovers runs from `exp_bash/config/*.yaml`.

### 20260331_baseline_scalar_mc_env16_exec4_mb2048.sh

- Background: current best-known run is `libero_10_dsrl_smolvla_scalar_ddp16x4_mc`.
- Goal: reproduce the current scalar MC baseline with explicit global algorithm settings so it can be rerun on different machines from `exp_bash/` alone.
- Changed vs previous baseline: no algorithm change; copied the launch path and config into `exp_bash/` so the experiment is self-contained.

### 20260331_actorlr2e5_scalar_mc_env16_exec4_mb2048.sh

- Background: current best run improves but does not reach stable repeated high eval success.
- Goal: strengthen actor learning while keeping value learning ahead and unchanged, using a self-contained experiment package.
- Changed vs baseline: increase actor LR from `1e-5` to `2e-5`; keep value LR at `1e-4` and keep the rest of the global algorithm settings fixed.

### 20260331_actorlr2e5_valuelr15e4_scalar_mc_env16_exec4_mb2048.sh

- Background: actor learning may be too weak, but value-first learning remains important.
- Goal: test whether increasing both actor and value learning rates can improve policy learning while preserving the value-first training pattern, using a self-contained experiment package.
- Changed vs baseline: increase actor LR to `2e-5` and value LR to `1.5e-4`; keep the rest of the global algorithm settings fixed.
