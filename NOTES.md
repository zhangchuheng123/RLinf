# Install

Each model needs its own install. Re-running install.sh with a different `--model` will reconfigure the environment.

## OpenPI (Pi-0 / Pi-0.5)

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

## SmolVLA

```bash
bash requirements/install.sh embodied --model smolvla --env maniskill_libero
```

# Run Pi-0.5

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run --no-sync hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./models/RLinf-Pi05-LIBERO-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi05.sh
```

# Run Pi-0 (not tested)

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run --no-sync hf download RLinf/RLinf-Pi0-LIBERO-Long-SFT --local-dir ./models/RLinf-Pi0-LIBERO-Long-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi0.sh
```

# Run SmolVLA

```bash
rm -rf .venv
bash requirements/install.sh embodied --model smolvla --env maniskill_libero
uv run --no-sync hf download HuggingFaceVLA/smolvla_libero --local-dir ./models/smolvla_libero
# Verify model config exists
test -f ./models/smolvla_libero/config.json
# Verify normalization stats exists (legacy stats.* or lerobot normalizer processor file)
bash -lc '[[ -f ./models/smolvla_libero/stats.safetensors || -f ./models/smolvla_libero/dataset_stats.safetensors || -f ./models/smolvla_libero/stats.json || -f ./models/smolvla_libero/dataset_stats.json || -n "$(compgen -G "./models/smolvla_libero/policy_preprocessor_step_*_normalizer_processor.safetensors")" ]]'
bash examples/embodiment/run_libero_ppo_smolvla.sh
```

If you see:

```text
AssertionError: `mean` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model.
```

The SmolVLA checkpoint is missing or failed to load normalization stats.
Use a complete pretrained checkpoint directory (with stats files) as `MODEL_PATH`.