#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf_noray.envs.action_utils import prepare_actions
from rlinf_noray.envs.libero.utils import quat2axisangle
from rlinf_noray.models.embodiment.smolvla.smolvla_policy import SmolVLAForRLActionPrediction


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    raise TypeError(f"Unsupported type for conversion to numpy: {type(x)}")


def _to_torch_cpu(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(f"Unsupported type for conversion to torch: {type(x)}")


def _max_abs_diff(a: Any, b: Any) -> float:
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"Shape mismatch: {a_np.shape} vs {b_np.shape}")
    return float(np.max(np.abs(a_np - b_np)))


def _trim_to_min_batch(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray, int]:
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    min_batch = min(a_np.shape[0], b_np.shape[0])
    if min_batch <= 0:
        raise ValueError("Empty batch for comparison")
    return a_np[:min_batch], b_np[:min_batch], min_batch


def _build_noray_env_obs_from_lerobot_env_obs(step_env_obs: dict[str, Any], task_list: list[str]) -> dict[str, Any]:
    if "pixels" not in step_env_obs or "robot_state" not in step_env_obs:
        raise KeyError("step_0_env_obs must contain 'pixels' and 'robot_state'")

    pixels = step_env_obs["pixels"]
    if "image" not in pixels or "image2" not in pixels:
        raise KeyError("step_0_env_obs['pixels'] must contain image/image2")

    robot_state = step_env_obs["robot_state"]
    eef = robot_state["eef"]
    gripper = robot_state["gripper"]

    eef_pos = _to_numpy(eef["pos"])
    eef_quat = _to_numpy(eef["quat"])
    gripper_qpos = _to_numpy(gripper["qpos"])

    axis_angles = np.stack([quat2axisangle(q.copy()) for q in eef_quat], axis=0)
    states = np.concatenate([eef_pos, axis_angles, gripper_qpos], axis=-1).astype(np.float32)

    return {
        "main_images": _to_torch_cpu(pixels["image"]).to(torch.uint8),
        "wrist_images": _to_torch_cpu(pixels["image2"]).to(torch.uint8),
        "states": torch.from_numpy(states),
        "task_descriptions": [str(x) for x in task_list],
    }


def _print_stage_diff(stage: str, a: Any, b: Any) -> dict[str, Any]:
    a_np, b_np, min_batch = _trim_to_min_batch(a, b)
    diff = float(np.max(np.abs(a_np.astype(np.float64) - b_np.astype(np.float64))))
    equal = bool(np.array_equal(a_np, b_np))
    allclose = bool(np.allclose(a_np, b_np, atol=1e-6, rtol=0.0))
    print(
        f"[align][{stage}] min_batch={min_batch} max_abs_diff={diff:.6e} "
        f"equal={equal} allclose_atol1e-6={allclose}",
        flush=True,
    )
    return {
        "min_batch": min_batch,
        "max_abs_diff": diff,
        "equal": equal,
        "allclose_atol1e-6": allclose,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one-step payload in no-ray and compare step_0_* intermediates"
    )
    parser.add_argument("--payload", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="/home/chuheng/RLinf/logs/rollout_alignment/step_compare.json",
    )
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    if not payload_path.exists():
        raise FileNotFoundError(f"Payload not found: {payload_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)

    step_prefix = f"step_{args.step}_"
    required = [
        f"{step_prefix}env_obs",
        f"{step_prefix}policy_payload",
        f"{step_prefix}model_raw_action",
        f"{step_prefix}postprocessor_action",
        f"{step_prefix}env_action",
        f"{step_prefix}policy_noise",
    ]
    for key in required:
        if key not in payload:
            raise KeyError(f"Payload missing required key: {key}")

    lerobot_env_obs = payload[f"{step_prefix}env_obs"]
    lerobot_policy_payload = payload[f"{step_prefix}policy_payload"]
    lerobot_model_raw_action = payload[f"{step_prefix}model_raw_action"]
    lerobot_postprocessor_action = payload[f"{step_prefix}postprocessor_action"]
    lerobot_env_action = payload[f"{step_prefix}env_action"]
    lerobot_policy_noise = _to_torch_cpu(payload[f"{step_prefix}policy_noise"])

    if "task" not in lerobot_policy_payload:
        raise KeyError(f"{step_prefix}policy_payload missing 'task'")

    noray_env_obs = _build_noray_env_obs_from_lerobot_env_obs(
        lerobot_env_obs,
        task_list=list(lerobot_policy_payload["task"]),
    )

    cfg = OmegaConf.create(
        {
            "model_path": str(model_path),
            "action_dim": int(_to_torch_cpu(lerobot_env_action).shape[-1]),
            "num_action_chunks": 1,
            "add_value_head": False,
            "image_keys": ["image", "image2"],
            "main_image_env_key": "main_images",
            "wrist_image_env_key": "wrist_images",
            "flip_libero_images": True,
        }
    )

    model = SmolVLAForRLActionPrediction(cfg=cfg)
    model.policy = model.policy.to(dtype=torch.float32)
    model.eval()

    device = next(model.policy.parameters()).device

    with torch.no_grad():
        noray_policy_payload = model._preprocess_obs_batch(noray_env_obs, device=device)
        raw_actions, rollout_result = model.predict_action_batch(
            env_obs=noray_env_obs,
            mode="eval",
            external_policy_noise=lerobot_policy_noise,
        )

    noray_env_action = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type="libero",
        model_type="smolvla",
        num_action_chunks=1,
        action_dim=int(raw_actions.shape[-1]),
    )

    # Stages: env_obs -> policy_payload -> model_raw_action -> postprocessor_action -> env_action
    print("[align] step-wise comparison starts", flush=True)

    env_obs_state_diff = _print_stage_diff(
        f"{step_prefix}env_obs.states",
        _to_numpy(noray_env_obs["states"]),
        np.asarray(
            np.concatenate(
                [
                    _to_numpy(lerobot_env_obs["robot_state"]["eef"]["pos"]),
                    np.stack(
                        [
                            quat2axisangle(q.copy())
                            for q in _to_numpy(lerobot_env_obs["robot_state"]["eef"]["quat"])
                        ],
                        axis=0,
                    ),
                    _to_numpy(lerobot_env_obs["robot_state"]["gripper"]["qpos"]),
                ],
                axis=1,
            )
        ),
    )

    policy_payload_diffs = {}
    compare_payload_keys = [
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "observation.language.tokens",
        "observation.language.attention_mask",
    ]
    for key in compare_payload_keys:
        if key not in lerobot_policy_payload:
            raise KeyError(f"Lerobot payload missing key: {key}")
        if key not in noray_policy_payload:
            raise KeyError(f"No-ray payload missing key: {key}")
        policy_payload_diffs[key] = _print_stage_diff(
            f"{step_prefix}policy_payload.{key}",
            lerobot_policy_payload[key],
            noray_policy_payload[key],
        )

    noray_model_raw_action = _to_torch_cpu(rollout_result["norm_actions"])[:, 0, :]
    model_raw_action_diff = _print_stage_diff(
        f"{step_prefix}model_raw_action",
        lerobot_model_raw_action,
        noray_model_raw_action,
    )

    noray_postprocessor_action = torch.as_tensor(raw_actions)[:, 0, :]
    postprocessor_action_diff = _print_stage_diff(
        f"{step_prefix}postprocessor_action",
        lerobot_postprocessor_action,
        noray_postprocessor_action,
    )

    noray_env_action_step = _to_torch_cpu(noray_env_action)[:, 0, :]
    env_action_diff = _print_stage_diff(
        f"{step_prefix}env_action",
        lerobot_env_action,
        noray_env_action_step,
    )

    output = {
        "meta": {
            "step": args.step,
            "payload": str(payload_path),
            "model_path": str(model_path),
        },
        "diffs": {
            f"{step_prefix}env_obs.states": env_obs_state_diff,
            f"{step_prefix}policy_payload": policy_payload_diffs,
            f"{step_prefix}model_raw_action": model_raw_action_diff,
            f"{step_prefix}postprocessor_action": postprocessor_action_diff,
            f"{step_prefix}env_action": env_action_diff,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[align] wrote compare report: {out_path}", flush=True)


if __name__ == "__main__":
    main()
