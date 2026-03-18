#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from rlinf_noray.envs.action_utils import prepare_actions
from rlinf_noray.envs.libero.utils import quat2axisangle


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _get_first_vec_env(envs: dict[str, dict[int, Any]]):
    suite = next(iter(envs.values()))
    return next(iter(suite.values()))


def _build_noray_env_obs(raw_obs: dict[str, Any], task_list: list[str]) -> dict[str, Any]:
    if "pixels" not in raw_obs or not isinstance(raw_obs["pixels"], dict):
        raise KeyError("Expected raw_obs['pixels'] as dict with LIBERO camera keys")
    pixel_dict = raw_obs["pixels"]

    main_key = None
    for candidate in ("agentview_image", "image"):
        if candidate in pixel_dict:
            main_key = candidate
            break
    if main_key is None:
        raise KeyError("Missing main camera key in raw_obs['pixels']; expected one of ['agentview_image', 'image']")

    wrist_key = None
    for candidate in ("robot0_eye_in_hand_image", "image2"):
        if candidate in pixel_dict:
            wrist_key = candidate
            break
    if wrist_key is None:
        raise KeyError(
            "Missing wrist camera key in raw_obs['pixels']; expected one of ['robot0_eye_in_hand_image', 'image2']"
        )
    if "robot_state" not in raw_obs:
        raise KeyError("Missing raw_obs['robot_state']")

    robot_state = raw_obs["robot_state"]
    required_robot_keys = ["eef", "gripper"]
    for key in required_robot_keys:
        if key not in robot_state:
            raise KeyError(f"Missing raw_obs['robot_state']['{key}']")

    eef_pos = robot_state["eef"]["pos"]
    eef_quat = robot_state["eef"]["quat"]
    gripper_qpos = robot_state["gripper"]["qpos"]

    if isinstance(eef_pos, torch.Tensor):
        eef_pos = eef_pos.detach().cpu().numpy()
    if isinstance(eef_quat, torch.Tensor):
        eef_quat = eef_quat.detach().cpu().numpy()
    if isinstance(gripper_qpos, torch.Tensor):
        gripper_qpos = gripper_qpos.detach().cpu().numpy()

    if eef_pos.ndim != 2 or eef_pos.shape[1] != 3:
        raise ValueError(f"Unexpected eef_pos shape: {eef_pos.shape}")
    if eef_quat.ndim != 2 or eef_quat.shape[1] != 4:
        raise ValueError(f"Unexpected eef_quat shape: {eef_quat.shape}")
    if gripper_qpos.ndim != 2 or gripper_qpos.shape[1] != 2:
        raise ValueError(f"Unexpected gripper_qpos shape: {gripper_qpos.shape}")

    axis_angles = np.stack([quat2axisangle(q.copy()) for q in eef_quat], axis=0)
    states = np.concatenate([eef_pos, axis_angles, gripper_qpos], axis=-1).astype(np.float32)

    main_images = pixel_dict[main_key]
    wrist_images = pixel_dict[wrist_key]

    if isinstance(main_images, torch.Tensor):
        main_images = main_images.detach().cpu().numpy()
    if isinstance(wrist_images, torch.Tensor):
        wrist_images = wrist_images.detach().cpu().numpy()

    if main_images.dtype != np.uint8 or wrist_images.dtype != np.uint8:
        raise TypeError("Expected uint8 HWC images from raw LIBERO observation")

    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "states": states,
        "task_descriptions": list(task_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one-step lerobot rollout payload for cross-stack alignment")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--task", type=str, default="libero_10")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    env_cfg = LiberoEnv(task=args.task)
    envs = make_env(env_cfg, n_envs=args.n_envs, use_async_envs=False)
    env = _get_first_vec_env(envs)

    policy = SmolVLAPolicy.from_pretrained(str(model_path))
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)

    obs_raw, _ = env.reset()
    obs_t = preprocess_observation(obs_raw)
    obs_with_task = add_envs_task(env, obs_t)

    env_payload = env_preprocessor(deepcopy(obs_with_task))
    policy_payload = preprocessor(deepcopy(env_payload))

    if "task" not in policy_payload:
        raise KeyError("policy_payload missing task")
    if "observation.state" not in policy_payload:
        raise KeyError("policy_payload missing observation.state")

    batch_size = policy_payload["observation.state"].shape[0]
    if batch_size <= 0:
        raise ValueError("Empty batch from policy payload")

    noise = torch.randn(
        (batch_size, policy.config.chunk_size, policy.config.max_action_dim),
        dtype=torch.float32,
        device=policy.config.device,
    )

    policy.reset()
    with torch.no_grad():
        action_norm = policy.select_action(policy_payload, noise=noise)

    action_post = postprocessor(action_norm)
    action_transition = env_postprocessor({ACTION: action_post})
    action_env = action_transition[ACTION]

    action_env_np = action_env.detach().cpu().numpy()
    if action_env_np.ndim != 2:
        raise ValueError(f"Expected env action [B, action_dim], got {action_env_np.shape}")

    noray_action_env = prepare_actions(
        raw_chunk_actions=action_env_np[:, None, :],
        env_type="libero",
        model_type="smolvla",
        num_action_chunks=1,
        action_dim=action_env_np.shape[-1],
    )

    # ####### TEMP START #######
    payload = {
        "meta": {
            "task": args.task,
            "model_path": str(model_path),
            "seed": args.seed,
            "batch_size": batch_size,
        },
        "obs_raw": _to_cpu(obs_raw),
        "obs_tensor": _to_cpu(obs_t),
        "obs_with_task": _to_cpu(obs_with_task),
        "env_payload": _to_cpu(env_payload),
        "policy_payload": _to_cpu(policy_payload),
        "policy_noise": _to_cpu(noise),
        "action_norm": _to_cpu(action_norm),
        "action_post": _to_cpu(action_post),
        "action_env": _to_cpu(action_env),
        "noray_env_obs": _to_cpu(_build_noray_env_obs(obs_raw, obs_with_task["task"])),
        "noray_action_env": _to_cpu(noray_action_env),
    }
    # ####### TEMP END #######

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"[dump_lerobot_single_step] saved payload: {output_path}")


if __name__ == "__main__":
    main()
