# Install:
# sudo bash requirements/install.sh embodied --model smolvla --env maniskill_libero

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

CONFIG_NAME="libero_10_ppo_smolvla"
MODEL_PATH="${MODEL_PATH:-${REPO_PATH}/models/smolvla_libero}"

NPROC_PER_NODE=1
MODEL_PRECISION="fp32"
STATE_DIM=8
TRAIN_ENVS=2
EVAL_ENVS=4
MICRO_BATCH=16
GLOBAL_BATCH=128
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-True}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_noray"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
WANDB_RUN="${CONFIG_NAME}_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_PATH}/examples/embodiment/train_embodied_agent_noray.py" \
  --config-path "${REPO_PATH}/examples/embodiment/config/" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  wandb.run="${WANDB_RUN}" \
  actor.model.model_path="${MODEL_PATH}" \
  rollout.model.model_path="${MODEL_PATH}" \
  actor.model.precision="${MODEL_PRECISION}" \
  rollout.model.precision="${MODEL_PRECISION}" \
  actor.model.state_dim="${STATE_DIM}" \
  env.train.total_num_envs="${TRAIN_ENVS}" \
  env.eval.total_num_envs="${EVAL_ENVS}" \
  env.eval.video_cfg.save_video="${SAVE_EVAL_VIDEO}" \
  env.eval.video_cfg.video_base_dir="${LOG_DIR}/video/eval" \
  actor.micro_batch_size="${MICRO_BATCH}" \
  actor.global_batch_size="${GLOBAL_BATCH}" \
  actor.training_backend="ddp" \
  actor.fsdp_config.disable=True \
  2>&1 | tee -a "${MEGA_LOG_FILE}"
