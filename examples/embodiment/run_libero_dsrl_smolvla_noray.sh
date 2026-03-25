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

LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard]'

CONFIG_NAME="libero_10_dsrl_smolvla"
MODEL_PATH="${REPO_PATH}/models/smolvla_libero"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPU_COUNT}}"

MODEL_PRECISION="${MODEL_PRECISION:-bf16}"
STATE_DIM="${STATE_DIM:-8}"
TRAIN_ENVS="${TRAIN_ENVS:-2}"
EVAL_ENVS="${EVAL_ENVS:-2}"
MICRO_BATCH="${MICRO_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
NUM_EXECUTE_STEPS="${NUM_EXECUTE_STEPS:-4}"
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-True}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"

DSRL_HIDDEN_DIM="${DSRL_HIDDEN_DIM:-256}"
DSRL_ACTOR_LR="${DSRL_ACTOR_LR:-1e-5}"
DSRL_VALUE_LR="${DSRL_VALUE_LR:-2e-5}"
DSRL_MINIBATCH_SIZE="${DSRL_MINIBATCH_SIZE:-1024}"
DSRL_REPLAY_CAPACITY="${DSRL_REPLAY_CAPACITY:-3000}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_noray"
EVAL_VIDEO_BASE_DIR="${EVAL_VIDEO_BASE_DIR:-${LOG_DIR}/video/eval}"
ROLLOUT_VIDEO_BASE_DIR="${ROLLOUT_VIDEO_BASE_DIR:-${LOG_DIR}/video/rollout}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

HYDRA_OVERRIDES=(
  "runner.logger.log_path=${LOG_DIR}"
  "actor.model.model_path=${MODEL_PATH}"
  "rollout.model.model_path=${MODEL_PATH}"
  "actor.model.precision=${MODEL_PRECISION}"
  "rollout.model.precision=${MODEL_PRECISION}"
  "actor.model.state_dim=${STATE_DIM}"
  "actor.model.dsrl_hidden_dim=${DSRL_HIDDEN_DIM}"
  "env.train.total_num_envs=${TRAIN_ENVS}"
  "env.eval.total_num_envs=${EVAL_ENVS}"
  "runner.save_eval_video=${SAVE_EVAL_VIDEO}"
  "runner.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
  "runner.num_execute_steps=${NUM_EXECUTE_STEPS}"
  "runner.eval_video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "runner.rollout_video_base_dir=${ROLLOUT_VIDEO_BASE_DIR}"
  "env.eval.video_cfg.save_eval_video=${SAVE_EVAL_VIDEO}"
  "env.eval.video_cfg.video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "actor.micro_batch_size=${MICRO_BATCH}"
  "actor.global_batch_size=${GLOBAL_BATCH}"
  "actor.training_backend=ddp"
  "actor.fsdp_config.disable=True"
  "actor.optim.dsrl_actor_lr=${DSRL_ACTOR_LR}"
  "actor.optim.dsrl_value_lr=${DSRL_VALUE_LR}"
  "algorithm.dsrl_minibatch_size=${DSRL_MINIBATCH_SIZE}"
  "algorithm.dsrl_replay_buffer_capacity=${DSRL_REPLAY_CAPACITY}"
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
