# Install:
# sudo bash requirements/install.sh embodied --model smolvla --env maniskill_libero
# Debug:
# SAVE_EVAL_VIDEO=false bash examples/embodiment/run_libero_dsrl_smolvla_noray.sh

#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="LIBERO"
export PYTHONPATH="${REPO_PATH}"
export UV_PROJECT_ENVIRONMENT="${REPO_PATH}/.venv"
export EMBODIED_PATH="${EMBODIED_PATH}"

if [[ "${WANDB_DISABLED:-}" == "true" || "${WANDB_DISABLED:-}" == "1" || "${DISABLE_WANDB:-}" == "true" || "${DISABLE_WANDB:-}" == "1" ]]; then
  LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard]'
else
  LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard,wandb]'
fi

CONFIG_NAME="libero_10_dsrl_smolvla"
MODEL_PATH="${REPO_PATH}/models/smolvla_libero"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"

TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-8}"
NUM_EXECUTE_STEPS="${NUM_EXECUTE_STEPS:-4}"
DSRL_VALUE_HEAD_TYPE="${DSRL_VALUE_HEAD_TYPE:-scalar}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-True}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_noray}"
EVAL_VIDEO_BASE_DIR="${EVAL_VIDEO_BASE_DIR:-${LOG_DIR}/video/eval}"
ROLLOUT_VIDEO_BASE_DIR="${ROLLOUT_VIDEO_BASE_DIR:-${LOG_DIR}/video/rollout}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
WANDB_RUN_BASE="${CONFIG_NAME}"
WANDB_RUNNAME_SUFFIX="${WANDB_RUNNAME_SUFFIX:-}"
if [[ -n "${WANDB_RUNNAME_SUFFIX}" ]]; then
  WANDB_RUN="${WANDB_RUN_BASE}_${WANDB_RUNNAME_SUFFIX}"
else
  WANDB_RUN="${WANDB_RUN_BASE}"
fi

if [[ "${DSRL_VALUE_HEAD_TYPE}" != "scalar" && "${DSRL_VALUE_HEAD_TYPE}" != "distributional" ]]; then
  echo "DSRL_VALUE_HEAD_TYPE must be either 'scalar' or 'distributional', got: ${DSRL_VALUE_HEAD_TYPE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

HYDRA_OVERRIDES=(
  "runner.logger.log_path=${LOG_DIR}"
  "wandb.run=${WANDB_RUN}"
  "actor.model.model_path=${MODEL_PATH}"
  "rollout.model.model_path=${MODEL_PATH}"
  "env.train.total_num_envs=${TRAIN_ENVS}"
  "env.eval.total_num_envs=${EVAL_ENVS}"
  "runner.save_eval_video=${SAVE_EVAL_VIDEO}"
  "runner.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
  "runner.num_execute_steps=${NUM_EXECUTE_STEPS}"
  "runner.eval_video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "runner.rollout_video_base_dir=${ROLLOUT_VIDEO_BASE_DIR}"
  "env.eval.video_cfg.save_eval_video=${SAVE_EVAL_VIDEO}"
  "env.eval.video_cfg.video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "actor.model.dsrl_value_head_type=${DSRL_VALUE_HEAD_TYPE}"
  "actor.training_backend=ddp"
  "actor.fsdp_config.disable=True"
  "${LOGGER_BACKENDS_OVERRIDE}"
)

uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_PATH}/examples/embodiment/train_embodied_agent_noray.py" \
  --config-path "${REPO_PATH}/examples/embodiment/config/" \
  --config-name "${CONFIG_NAME}" \
  "${HYDRA_OVERRIDES[@]}" \
  2>&1 | tee -a "${MEGA_LOG_FILE}"
