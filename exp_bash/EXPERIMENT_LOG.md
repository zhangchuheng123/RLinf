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

### 20260331_scalar_mc_env16_exec4_mb2048_increase_net.sh

- Background: current baseline may be limited more by DSRL actor/value network capacity than by learning-rate choice.
- Goal: test a larger DSRL MLP while keeping the scalar MC training setup unchanged.
- Changed vs baseline: increase `dsrl_hidden_dim` from `256` to `512`; increase actor hidden-layer count from `3` to `4`; increase value hidden-layer count from `2` to `4`; keep actor/value learning rates and the rest of the global algorithm settings fixed.

## 2026-04-01

### 20260401_scalar_mc_env16_exec4_mb2048_increase_actor_lr.sh

- Background: after increase_net, PPO actor updates still appear too conservative (`ratio` near 1 and clipping near 0).
- Goal: increase actor update strength while keeping the increase_net architecture unchanged, and keep normalized-advantage PPO enabled.
- Changed vs 20260331 increase_net: increase `actor.optim.dsrl_actor_lr` from `1.0e-5` to `1.0e-4`; keep `actor.optim.dsrl_value_lr=1.0e-4`; set `algorithm.do_adv_norm=true`, `algorithm.adv_norm_eps=1.0e-8`, and `algorithm.advantage_clip=3.0`.
