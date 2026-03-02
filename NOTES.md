# Install

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
```

# Run

```bash
# Download assets
hf download --repo-type dataset RLinf/maniskill_assets --local-dir ./rlinf/envs/maniskill/assets
uv run hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./models/RLinf-Pi05-LIBERO-SFT
OPENPI_MODEL_PATH=./models/RLinf-Pi05-LIBERO-SFT bash examples/embodiment/run_libero_ppo_openpi.sh
```