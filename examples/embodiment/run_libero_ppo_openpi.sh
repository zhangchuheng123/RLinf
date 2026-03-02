#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="LIBERO"
export PYTHONPATH="${REPO_PATH}"

# Hard-coded training config (Pi0.5 + LIBERO + PPO).
CONFIG_NAME="libero_10_ppo_openpi_pi05"
MODEL_PATH="${REPO_PATH}/models/RLinf-Pi05-LIBERO-SFT"

# Single A100 80G friendly defaults.
TRAIN_ENVS=16
EVAL_ENVS=50
MICRO_BATCH=32
GLOBAL_BATCH=256

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

uv run python "${SRC_FILE}" \
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
