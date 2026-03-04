# Install

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

# Run Pi-0.5

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./models/RLinf-Pi05-LIBERO-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi05.sh
```

# Run Pi-0 (not tested)

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run hf download RLinf/RLinf-Pi0-LIBERO-Long-SFT --local-dir ./models/RLinf-Pi0-LIBERO-Long-SFT
bash examples/embodiment/run_libero_ppo_openpi_pi0.sh
```

# Run SmolVLA

```bash
uv run hf download HuggingFaceVLA/smolvla_libero --local-dir ./models/smolvla_libero
bash examples/embodiment/run_libero_10_ppo_smolvla.sh
```