from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rlinf_noray.integrations.lerobot_local_import import ensure_local_lerobot

ensure_local_lerobot()

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_numpy(v) for v in obj)
    return obj


def build_lerobot_pre_post_processors(policy, model_path: str):
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_cfg = SimpleNamespace(type="libero")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy.config,
    )
    return preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _state_to_robot_state(states: np.ndarray) -> dict[str, Any]:
    states = np.asarray(states)
    eef_pos = states[:, :3]
    axis_angle = states[:, 3:6]
    gripper = states[:, 6:8]

    angles = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    axis = np.divide(axis_angle, np.maximum(angles, 1e-10))
    half = angles / 2.0
    sin_half = np.sin(half)
    quat_xyz = axis * sin_half
    quat_w = np.cos(half)
    eef_quat = np.concatenate([quat_xyz, quat_w], axis=1)

    zeros_gripper = np.zeros_like(gripper)
    zeros_joint = np.zeros((states.shape[0], 7), dtype=states.dtype)
    identity_mat = np.tile(np.eye(3, dtype=states.dtype)[None, :, :], (states.shape[0], 1, 1))

    return {
        "eef": {
            "pos": eef_pos,
            "quat": eef_quat,
            "mat": identity_mat,
        },
        "gripper": {
            "qpos": gripper,
            "qvel": zeros_gripper,
        },
        "joints": {
            "pos": zeros_joint,
            "vel": zeros_joint,
        },
    }


def build_lerobot_observation_from_noray_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
    if "pixels" in env_obs and "robot_state" in env_obs:
        observation = {
            "pixels": _to_numpy(env_obs["pixels"]),
            "robot_state": _to_numpy(env_obs["robot_state"]),
        }
    else:
        main = _to_numpy(env_obs["main_images"])
        pixels = {"image": main}
        wrist = env_obs.get("wrist_images", None)
        if wrist is not None:
            pixels["image2"] = _to_numpy(wrist)

        states = _to_numpy(env_obs["states"])
        observation = {
            "pixels": pixels,
            "robot_state": _state_to_robot_state(states),
        }

    observation["task"] = list(env_obs.get("task_descriptions", []))
    return observation


def run_lerobot_inference_preprocess(
    env_obs: dict[str, Any],
    env_preprocessor,
    preprocessor,
) -> dict[str, Any]:
    raw_observation = build_lerobot_observation_from_noray_obs(env_obs)
    task_descriptions = list(raw_observation.get("task", []))

    observation = preprocess_observation(raw_observation)

    if not task_descriptions:
        batch_size = 0
        if "states" in env_obs and hasattr(env_obs["states"], "shape"):
            batch_size = int(env_obs["states"].shape[0])
        elif "main_images" in env_obs and hasattr(env_obs["main_images"], "shape"):
            batch_size = int(env_obs["main_images"].shape[0])
        task_descriptions = [""] * batch_size
    observation["task"] = task_descriptions

    observation = env_preprocessor(observation)
    observation = preprocessor(observation)
    return observation


def run_lerobot_action_postprocess(
    action: torch.Tensor,
    postprocessor,
    env_postprocessor,
) -> torch.Tensor:
    action = postprocessor(action)
    action_transition = {ACTION: action}
    action_transition = env_postprocessor(action_transition)
    return action_transition[ACTION]
