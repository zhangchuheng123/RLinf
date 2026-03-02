#!/usr/bin/env bash
set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Ensure correct action normalization for LIBERO.
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}

# Optional: include a custom LIBERO checkout on PYTHONPATH if provided.
if [[ -n "${LIBERO_REPO_PATH:-}" ]]; then
  export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH"
else
  export PYTHONPATH="${REPO_PATH}:$PYTHONPATH"
fi

CONFIG_NAME=${1:-"libero_10_ppo_openpi_pi05"}

# Single-GPU friendly overrides (can be customized via env vars).
TRAIN_ENVS=${TRAIN_ENVS:-16}
EVAL_ENVS=${EVAL_ENVS:-50}
MICRO_BATCH=${MICRO_BATCH:-32}
GLOBAL_BATCH=${GLOBAL_BATCH:-256}

if [[ -z "${OPENPI_MODEL_PATH:-}" ]]; then
  echo "ERROR: OPENPI_MODEL_PATH is not set."
  echo "Please export the path to your Pi0/Pi0.5 SFT checkpoint, e.g.:"
  echo "  export OPENPI_MODEL_PATH=/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
  exit 1
fi

if [[ ! -e "${OPENPI_MODEL_PATH}" ]]; then
  echo "ERROR: OPENPI_MODEL_PATH does not exist: ${OPENPI_MODEL_PATH}"
  exit 1
fi

echo "Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "Using Python at $(which python)"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD=(
  python "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  runner.logger.log_path="${LOG_DIR}"
  actor.model.model_path="${OPENPI_MODEL_PATH}"
  rollout.model.model_path="${OPENPI_MODEL_PATH}"
  env.train.total_num_envs="${TRAIN_ENVS}"
  env.eval.total_num_envs="${EVAL_ENVS}"
  actor.micro_batch_size="${MICRO_BATCH}"
  actor.global_batch_size="${GLOBAL_BATCH}"
)

printf '%q ' "${CMD[@]}" | tee "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
