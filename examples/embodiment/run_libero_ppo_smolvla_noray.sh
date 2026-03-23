# Install:
# sudo bash requirements/install.sh embodied --model smolvla --env maniskill_libero
# Debug:
# DISABLE_WANDB=1 SAVE_EVAL_VIDEO=true bash examples/embodiment/run_libero_ppo_smolvla_noray.sh

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

# Keep wandb enabled by default; disable only when DISABLE_WANDB is requested.
if [[ "${DISABLE_WANDB:-0}" == "1" || "${DISABLE_WANDB:-}" == "true" || "${DISABLE_WANDB:-}" == "TRUE" ]]; then
  export WANDB_DISABLED="true"
  LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard]'
else
  unset WANDB_DISABLED
  LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard,wandb]'
fi

CONFIG_NAME="libero_10_ppo_smolvla"
MODEL_PATH="${REPO_PATH}/models/smolvla_libero"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"
MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
STATE_DIM="${STATE_DIM:-8}"
# User should set TRAIN_ENVS to be divisible by NPROC_PER_NODE.
TRAIN_ENVS="${TRAIN_ENVS:-2}"
# User should set EVAL_ENVS to be divisible by NPROC_PER_NODE.
EVAL_ENVS="${EVAL_ENVS:-2}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
NUM_EXECUTE_STEPS="${NUM_EXECUTE_STEPS:-16}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-True}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_noray"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
WANDB_RUN="${CONFIG_NAME}_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

HYDRA_OVERRIDES=(
  "runner.logger.log_path=${LOG_DIR}"
  "wandb.run=${WANDB_RUN}"
  "actor.model.model_path=${MODEL_PATH}"
  "rollout.model.model_path=${MODEL_PATH}"
  "actor.model.precision=${MODEL_PRECISION}"
  "rollout.model.precision=${MODEL_PRECISION}"
  "actor.model.state_dim=${STATE_DIM}"
  "env.train.total_num_envs=${TRAIN_ENVS}"
  "env.eval.total_num_envs=${EVAL_ENVS}"
  "runner.save_eval_video=${SAVE_EVAL_VIDEO}"
  "runner.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
  "runner.num_execute_steps=${NUM_EXECUTE_STEPS}"
  "runner.eval_video_base_dir=${LOG_DIR}/video/eval"
  "runner.rollout_video_base_dir=${LOG_DIR}/video/rollout"
  "env.eval.video_cfg.save_eval_video=${SAVE_EVAL_VIDEO}"
  "env.eval.video_cfg.video_base_dir=${LOG_DIR}/video/eval"
  "actor.micro_batch_size=${MICRO_BATCH}"
  "actor.global_batch_size=${GLOBAL_BATCH}"
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
