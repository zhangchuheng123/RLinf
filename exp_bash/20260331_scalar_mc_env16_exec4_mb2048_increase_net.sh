#!/usr/bin/env bash
set -euo pipefail

# Background:
# - The current scalar MC baseline can improve, but actor/value capacity may still be too small.
# Goal:
# - Test a larger DSRL actor/value network while keeping the scalar MC training setup fixed.
# Changes vs baseline:
# - Increase `dsrl_hidden_dim` from 256 to 512.
# - Increase both actor and value MLP depth to 4 hidden layers.

EXP_BASH_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "${EXP_BASH_PATH}")"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export ROBOT_PLATFORM="LIBERO"
export PYTHONPATH="${REPO_PATH}"
export UV_PROJECT_ENVIRONMENT="${REPO_PATH}/.venv"
export EMBODIED_PATH="${EXP_BASH_PATH}"

CONFIG_NAME="20260331_scalar_mc_env16_exec4_mb2048_increase_net"
MODEL_PATH="${REPO_PATH}/models/smolvla_libero"
TRAIN_ENVS=16
EVAL_ENVS=16
GLOBAL_MINIBATCH_SIZE=2048

if [[ "${WANDB_DISABLED:-}" == "true" || "${WANDB_DISABLED:-}" == "1" || "${DISABLE_WANDB:-}" == "true" || "${DISABLE_WANDB:-}" == "1" ]]; then
	LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard]'
else
	LOGGER_BACKENDS_OVERRIDE='runner.logger.logger_backends=[tensorboard,wandb]'
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
	NPROC_PER_NODE=""
	for (( candidate=GPU_COUNT; candidate>=1; candidate-- )); do
		if (( TRAIN_ENVS % candidate == 0 && EVAL_ENVS % candidate == 0 && GLOBAL_MINIBATCH_SIZE % candidate == 0 )); then
			NPROC_PER_NODE="${candidate}"
			break
		fi
	done
	if [[ -z "${NPROC_PER_NODE}" ]]; then
		echo "Failed to choose NPROC_PER_NODE automatically for ${CONFIG_NAME}." >&2
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
if (( GLOBAL_MINIBATCH_SIZE % NPROC_PER_NODE != 0 )); then
	echo "GLOBAL_MINIBATCH_SIZE=${GLOBAL_MINIBATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
	exit 1
fi

SAVE_EVAL_VIDEO="${SAVE_EVAL_VIDEO:-False}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}}"
EVAL_VIDEO_BASE_DIR="${EVAL_VIDEO_BASE_DIR:-${LOG_DIR}/video/eval}"
ROLLOUT_VIDEO_BASE_DIR="${ROLLOUT_VIDEO_BASE_DIR:-${LOG_DIR}/video/rollout}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
WANDB_RUN="${WANDB_RUN:-${CONFIG_NAME}}"

mkdir -p "${LOG_DIR}"

HYDRA_OVERRIDES=(
	"runner.logger.log_path=${LOG_DIR}"
	"wandb.run=${WANDB_RUN}"
	"actor.model.model_path=${MODEL_PATH}"
	"rollout.model.model_path=${MODEL_PATH}"
	"runner.save_eval_video=${SAVE_EVAL_VIDEO}"
	"runner.save_rollout_video=${SAVE_ROLLOUT_VIDEO}"
	"runner.eval_video_base_dir=${EVAL_VIDEO_BASE_DIR}"
	"runner.rollout_video_base_dir=${ROLLOUT_VIDEO_BASE_DIR}"
	"env.eval.video_cfg.save_eval_video=${SAVE_EVAL_VIDEO}"
	"env.eval.video_cfg.video_base_dir=${EVAL_VIDEO_BASE_DIR}"
	"${LOGGER_BACKENDS_OVERRIDE}"
)

uv run --no-sync torchrun \
	--standalone \
	--nproc_per_node="${NPROC_PER_NODE}" \
	"${REPO_PATH}/exp_bash/train_libero_dsrl_smolvla_noray.py" \
	--config-path "${REPO_PATH}/exp_bash/config/" \
	--config-name "${CONFIG_NAME}" \
	"${HYDRA_OVERRIDES[@]}" \
	2>&1 | tee -a "${MEGA_LOG_FILE}"