#!/usr/bin/env bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="LIBERO"
export PYTHONPATH="${REPO_PATH}"
export RAY_OVERRIDE_RUNTIME_ENV_DEFAULT_EXCLUDES=""
export RAY_RUNTIME_ENV_WORKING_DIR_EXCLUDES=".git,.venv"
# Tell uv to always use this venv's absolute path, so Ray workers spawned in
# a temp working directory don't create a fresh empty venv and lose all packages.
export UV_PROJECT_ENVIRONMENT="${REPO_PATH}/.venv"

# Stop any stale Ray cluster from a previous run so workers pick up the
# current venv's Python environment cleanly.
uv run --no-sync ray stop --force 2>/dev/null || true

# Hard-coded training config (SmolVLA + LIBERO-10 + PPO).
CONFIG_NAME="libero_10_ppo_smolvla"
MODEL_PATH="${MODEL_PATH:-${REPO_PATH}/models/smolvla_libero}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] SmolVLA model directory not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Missing config.json in SmolVLA model directory: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/stats.safetensors" \
      && ! -f "${MODEL_PATH}/dataset_stats.safetensors" \
      && ! -f "${MODEL_PATH}/stats.json" \
      && ! -f "${MODEL_PATH}/dataset_stats.json" \
      && -z "$(compgen -G "${MODEL_PATH}/policy_preprocessor_step_*_normalizer_processor.safetensors")" ]]; then
  echo "[ERROR] Missing normalization stats in SmolVLA model directory: ${MODEL_PATH}" >&2
  echo "[ERROR] Expected one of: stats.safetensors, dataset_stats.safetensors, stats.json, dataset_stats.json, policy_preprocessor_step_*_normalizer_processor.safetensors" >&2
  exit 1
fi

# Single A100 80G friendly defaults.
TRAIN_ENVS=16
EVAL_ENVS=10
MICRO_BATCH=16
GLOBAL_BATCH=128

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

uv run --no-sync python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  actor.model.model_path="${MODEL_PATH}" \
  rollout.model.model_path="${MODEL_PATH}" \
  env.train.total_num_envs="${TRAIN_ENVS}" \
  env.eval.total_num_envs="${EVAL_ENVS}" \
  actor.micro_batch_size="${MICRO_BATCH}" \
  actor.global_batch_size="${GLOBAL_BATCH}" \
  2>&1 | tee -a "${MEGA_LOG_FILE}"
