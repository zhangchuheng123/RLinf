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
unset RLINF_SELECT_ACTION_ALIGN_DUMP_PATH
unset RLINF_ROLLOUT_COMPARE_DUMP_PATH
unset RLINF_ALIGN_ACTION_LOG_PATH
unset LEROBOT_EVAL_DUMP_PATH
# ####### TEMP START #######
# export RLINF_TEMP_EXIT_AFTER_FIRST_ROLLOUT="1"
# export RLINF_TEMP_NUM_ROLLOUTS="3"
# export RLINF_TEMP_MAX_STEPS="520"
export WANDB_DISABLED="true"
# TEMP_LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard]'
# TEMP_MAX_STEPS_OVERRIDE='env.train.max_steps_per_rollout_epoch=520'
# TEMP_EVAL_MAX_STEPS_OVERRIDE='env.eval.max_steps_per_rollout_epoch=520'
# ####### TEMP END #######

TEMP_LOGGER_BACKENDS_OVERRIDE="${TEMP_LOGGER_BACKENDS_OVERRIDE:-runner.logger.logger_backends=[tensorboard]}"
TEMP_MAX_STEPS_OVERRIDE="${TEMP_MAX_STEPS_OVERRIDE:-}"
TEMP_EVAL_MAX_STEPS_OVERRIDE="${TEMP_EVAL_MAX_STEPS_OVERRIDE:-}"

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
SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-False}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}_noray"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
WANDB_RUN="${CONFIG_NAME}_$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

# ####### TEMP START #######
# Set ENABLE_COLLECT_PROFILE=1 to write per-chunk timing JSONL for collect_samples analysis.
ENABLE_COLLECT_PROFILE="${ENABLE_COLLECT_PROFILE:-0}"
if [[ "${ENABLE_COLLECT_PROFILE}" == "1" ]]; then
  export RLINF_COLLECT_PROFILE_PATH="${LOG_DIR}/collect_profile.jsonl"
  export RLINF_COLLECT_PROFILE_SYNC_CUDA="1"
fi
# ####### TEMP END #######

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
  "env.eval.video_cfg.save_video=${SAVE_EVAL_VIDEO}"
  "env.eval.video_cfg.video_base_dir=${LOG_DIR}/video/eval"
  "actor.micro_batch_size=${MICRO_BATCH}"
  "actor.global_batch_size=${GLOBAL_BATCH}"
  "actor.training_backend=ddp"
  "actor.fsdp_config.disable=True"
)

if [[ -n "${TEMP_LOGGER_BACKENDS_OVERRIDE}" ]]; then
  HYDRA_OVERRIDES+=("${TEMP_LOGGER_BACKENDS_OVERRIDE}")
fi
if [[ -n "${TEMP_MAX_STEPS_OVERRIDE}" ]]; then
  HYDRA_OVERRIDES+=("${TEMP_MAX_STEPS_OVERRIDE}")
fi
if [[ -n "${TEMP_EVAL_MAX_STEPS_OVERRIDE}" ]]; then
  HYDRA_OVERRIDES+=("${TEMP_EVAL_MAX_STEPS_OVERRIDE}")
fi

uv run --no-sync torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_PATH}/examples/embodiment/train_embodied_agent_noray.py" \
  --config-path "${REPO_PATH}/examples/embodiment/config/" \
  --config-name "${CONFIG_NAME}" \
  "${HYDRA_OVERRIDES[@]}" \
  2>&1 | tee -a "${MEGA_LOG_FILE}"
