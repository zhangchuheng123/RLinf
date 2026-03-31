# exp_bash

This directory stores managed DSRL experiment launchers.

Rules:

- Keep experiment launchers self-contained. Do not make `exp_bash/` bash scripts call bash scripts under `examples/`.
- Keep experiment configs local to `exp_bash/config/`. Do not reference Hydra configs under `examples/`.
- If an experiment needs a tiny launch entrypoint, place it in `exp_bash/` rather than depending on `examples/`.
- If an experiment needs a WandB/result analysis tool, place it in `exp_bash/` as well so launch, config, and analysis stay together.
- Favor flattening and traceability over cross-directory reuse. The goal is that an experiment can be understood and rerun from `exp_bash/` alone.
- Name files with `YYYYMMDD` plus concise keywords and important parameter hints.
- Treat `TRAIN_ENVS`, `EVAL_ENVS`, rollout horizons, PPO update counts, and `DSRL_MINIBATCH_SIZE` as global algorithm settings.
- Do not silently retune algorithm-level sample counts just because the machine has a different GPU count.
- The main runner auto-selects a default `NPROC_PER_NODE` that divides the global env counts and global PPO minibatch size. You can still override `NPROC_PER_NODE` manually when needed.
- Keep the experiment delta small and explicit. Each bash file should state background, goal, and what changed.
- Update `EXPERIMENT_LOG.md` when adding or changing an experiment script.

Recommended workflow:

1. Start from the strongest known baseline script.
2. Copy the needed config into `exp_bash/config/` and modify it locally.
3. Change one or two algorithm knobs at a time.
4. Keep run naming explicit through the local config name and `wandb.run`.
5. Compare runs using stable evaluation behavior, not single evaluation spikes.

WandB analysis:

1. Summarize all managed `exp_bash/config/*.yaml` runs:
	`.venv/bin/python exp_bash/fetch_exp_wandb_runs.py`
2. Summarize one managed experiment with full history:
	`.venv/bin/python exp_bash/fetch_exp_wandb_runs.py --config-name 20260331_scalar_mc_env16_exec4_mb2048_increase_net --include-history --output logs/analysis/20260331_scalar_mc_env16_exec4_mb2048_increase_net.json`
3. Add extra run filters when needed:
	`.venv/bin/python exp_bash/fetch_exp_wandb_runs.py --name-contains scalar_mc`
4. If no run has been launched yet for the current `exp_bash/config/*.yaml` names, point the tool at a legacy reference run explicitly:
	`.venv/bin/python exp_bash/fetch_exp_wandb_runs.py --run-name libero_10_dsrl_smolvla_scalar_ddp16x4_mc`

Naming hints:

- `exec4` means `runner.num_execute_steps = 4`.
- `mb2048` means global `algorithm.dsrl_minibatch_size = 2048` across all DDP ranks.
