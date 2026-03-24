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

# DSRL debug script: force-disable wandb.
export WANDB_DISABLED="true"
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
DSRL_ACTOR_LR="${DSRL_ACTOR_LR:-1e-4}"
DSRL_VALUE_LR="${DSRL_VALUE_LR:-1e-4}"
DSRL_Q_LR="${DSRL_Q_LR:-1e-4}"
DSRL_REPLAY_BUFFER_SIZE="${DSRL_REPLAY_BUFFER_SIZE:-50000}"
DSRL_REPLAY_MIN_SIZE="${DSRL_REPLAY_MIN_SIZE:-1024}"
DSRL_REPLAY_BATCH_SIZE="${DSRL_REPLAY_BATCH_SIZE:-256}"
DSRL_Q_UPDATES_PER_EPOCH="${DSRL_Q_UPDATES_PER_EPOCH:-100}"
DSRL_Q_TAU="${DSRL_Q_TAU:-0.005}"

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
  "actor.optim.dsrl_q_lr=${DSRL_Q_LR}"
  "algorithm.dsrl_replay_buffer_size=${DSRL_REPLAY_BUFFER_SIZE}"
  "algorithm.dsrl_replay_min_size=${DSRL_REPLAY_MIN_SIZE}"
  "algorithm.dsrl_replay_batch_size=${DSRL_REPLAY_BATCH_SIZE}"
  "algorithm.dsrl_q_updates_per_epoch=${DSRL_Q_UPDATES_PER_EPOCH}"
  "algorithm.dsrl_q_tau=${DSRL_Q_TAU}"
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
