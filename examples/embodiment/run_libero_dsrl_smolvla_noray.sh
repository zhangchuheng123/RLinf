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
TRAIN_ENVS="${TRAIN_ENVS:-8}"
EVAL_ENVS="${EVAL_ENVS:-8}"
NUM_EXECUTE_STEPS="${NUM_EXECUTE_STEPS:-4}"
DSRL_VALUE_HEAD_TYPE="${DSRL_VALUE_HEAD_TYPE:-scalar}"
DSRL_MINIBATCH_SIZE="${DSRL_MINIBATCH_SIZE:-2048}"
DSRL_ACTOR_LR="${DSRL_ACTOR_LR:-1.0e-5}"
DSRL_VALUE_LR="${DSRL_VALUE_LR:-1.0e-4}"
DSRL_ROLLOUT_EPOCH="${DSRL_ROLLOUT_EPOCH:-2}"
DSRL_UPDATE_EPOCH="${DSRL_UPDATE_EPOCH:-10}"
DSRL_PRE_VALUE_UPDATE_EPOCH="${DSRL_PRE_VALUE_UPDATE_EPOCH:-20}"
TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH="${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH:-512}"
EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH="${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH:-512}"
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

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE=""
  for (( candidate=GPU_COUNT; candidate>=1; candidate-- )); do
    if (( TRAIN_ENVS % candidate == 0 && EVAL_ENVS % candidate == 0 && DSRL_MINIBATCH_SIZE % candidate == 0 )); then
      NPROC_PER_NODE="${candidate}"
      break
    fi
  done
  if [[ -z "${NPROC_PER_NODE}" ]]; then
    echo "Failed to choose NPROC_PER_NODE automatically. Set NPROC_PER_NODE explicitly so it divides TRAIN_ENVS=${TRAIN_ENVS}, EVAL_ENVS=${EVAL_ENVS}, and DSRL_MINIBATCH_SIZE=${DSRL_MINIBATCH_SIZE}." >&2
    exit 1
  fi
fi

if (( TRAIN_ENVS % NPROC_PER_NODE != 0 )); then
  echo "TRAIN_ENVS=${TRAIN_ENVS} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

if (( EVAL_ENVS % NPROC_PER_NODE != 0 )); then
  echo "EVAL_ENVS=${EVAL_ENVS} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

if (( DSRL_MINIBATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "DSRL_MINIBATCH_SIZE=${DSRL_MINIBATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

echo "[dsrl-run] nproc=${NPROC_PER_NODE} train_envs=${TRAIN_ENVS} eval_envs=${EVAL_ENVS} global_minibatch=${DSRL_MINIBATCH_SIZE} actor_lr=${DSRL_ACTOR_LR} value_lr=${DSRL_VALUE_LR}" >&2

mkdir -p "${LOG_DIR}"

HYDRA_OVERRIDES=(
  "runner.logger.log_path=${LOG_DIR}"
  "wandb.run=${WANDB_RUN}"
  "actor.model.model_path=${MODEL_PATH}"
  "rollout.model.model_path=${MODEL_PATH}"
  "env.train.total_num_envs=${TRAIN_ENVS}"
  "env.eval.total_num_envs=${EVAL_ENVS}"
  "env.train.max_steps_per_rollout_epoch=${TRAIN_MAX_STEPS_PER_ROLLOUT_EPOCH}"
  "env.eval.max_steps_per_rollout_epoch=${EVAL_MAX_STEPS_PER_ROLLOUT_EPOCH}"
  "env.train.specific_task_id=0"
  "env.eval.specific_task_id=0"
  "runner.save_eval_video=${SAVE_EVAL_VIDEO}"
  "runner.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
  "runner.num_execute_steps=${NUM_EXECUTE_STEPS}"
  "algorithm.rollout_epoch=${DSRL_ROLLOUT_EPOCH}"
  "algorithm.update_epoch=${DSRL_UPDATE_EPOCH}"
  "algorithm.pre_value_update_epoch=${DSRL_PRE_VALUE_UPDATE_EPOCH}"
  "algorithm.dsrl_minibatch_size=${DSRL_MINIBATCH_SIZE}"
  "runner.eval_video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "runner.rollout_video_base_dir=${ROLLOUT_VIDEO_BASE_DIR}"
  "env.eval.video_cfg.save_eval_video=${SAVE_EVAL_VIDEO}"
  "env.eval.video_cfg.video_base_dir=${EVAL_VIDEO_BASE_DIR}"
  "actor.model.dsrl_value_head_type=${DSRL_VALUE_HEAD_TYPE}"
  "actor.optim.dsrl_actor_lr=${DSRL_ACTOR_LR}"
  "actor.optim.dsrl_value_lr=${DSRL_VALUE_LR}"
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
