#!/usr/bin/env python3
"""Minimal benchmark for LiberoPPODDPNoRayRunner collect_samples hot path.

This script is intentionally standalone and non-invasive:
- It does not modify runner/model/env source code.
- It hardcodes key values inferred from
  examples/embodiment/run_libero_ppo_smolvla_noray.sh.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf_noray.envs import get_env_cls
from rlinf_noray.envs.action_utils import prepare_actions
from rlinf_noray.models import get_model


# Hardcoded from bash + config defaults used by the same entry.
MODEL_PATH = "models/smolvla_libero"
MODEL_TYPE = "smolvla"
MODEL_PRECISION = "bf16"
STATE_DIM = 8
ACTION_DIM = 7
NUM_ACTION_CHUNKS = 16
TASK_SUITE = "libero_10"
MAX_STEPS_PER_ROLLOUT_EPOCH = 480
MODE = "train"

# Keep runtime short but representative.
WARMUP_CHUNKS = 2
MEASURED_CHUNKS = 8
ENV_PARALLEL_CANDIDATES = [1, 2, 4]


@dataclass
class ChunkRecord:
    predict_action_batch: float
    prepare_actions: float
    env_chunk_step: float
    collect_samples_pack: float
    chunk_total: float


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed() -> Any:
    _cuda_sync()
    start = time.perf_counter()
    try:
        yield lambda: time.perf_counter() - start
    finally:
        _cuda_sync()


def _reduce_to_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.float()
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1).float()


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


def _build_model() -> torch.nn.Module:
    model_cfg = OmegaConf.create(
        {
            "model_type": MODEL_TYPE,
            "model_path": MODEL_PATH,
            "precision": MODEL_PRECISION,
            "is_lora": False,
            "add_value_head": True,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "num_action_chunks": NUM_ACTION_CHUNKS,
            "main_image_env_key": "main_images",
            "wrist_image_env_key": "wrist_images",
            "image_keys": ["image", "image2"],
        }
    )
    model = get_model(model_cfg)
    model.eval()
    return model


def _build_env(num_envs: int):
    env_cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": TASK_SUITE,
            "total_num_envs": num_envs,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": MAX_STEPS_PER_ROLLOUT_EPOCH,
            "max_episode_steps": 512,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": False,
            "specific_task_id": None,
            "use_rel_reward": True,
            "reward_coef": 5.0,
            "reset_gripper_open": True,
            "is_eval": False,
            "seed": 0,
            "group_size": 1,
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
            },
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "logs/bench_tmp",
            },
        }
    )
    env_cls = get_env_cls("libero", env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    return env


def _run_one_case(num_envs: int, model: torch.nn.Module) -> dict[str, float]:
    env = _build_env(num_envs=num_envs)
    obs, _ = env.reset()

    measured_records: list[ChunkRecord] = []
    total_chunks = WARMUP_CHUNKS + MEASURED_CHUNKS

    with torch.no_grad():
        for chunk_idx in range(total_chunks):
            with timed() as chunk_elapsed:
                with timed() as predict_elapsed:
                    raw_actions, rollout_result = model.predict_action_batch(
                        env_obs=obs,
                        mode=MODE,
                    )
                t_predict = predict_elapsed()

                with timed() as prepare_elapsed:
                    chunk_actions = torch.as_tensor(raw_actions)
                    chunk_actions = prepare_actions(
                        raw_chunk_actions=chunk_actions,
                        env_type="libero",
                        model_type=MODEL_TYPE,
                        num_action_chunks=NUM_ACTION_CHUNKS,
                        action_dim=ACTION_DIM,
                        policy=None,
                        wm_env_type=None,
                    )
                    if isinstance(chunk_actions, np.ndarray):
                        chunk_actions = torch.from_numpy(chunk_actions)
                t_prepare = prepare_elapsed()

                with timed() as env_elapsed:
                    obs_list, chunk_rewards, chunk_terminations, chunk_truncations, _ = env.chunk_step(
                        chunk_actions
                    )
                    obs = obs_list[-1]
                t_env = env_elapsed()

                with timed() as collect_elapsed:
                    chunk_returns = chunk_rewards.sum(dim=1).float().cpu()
                    dones = torch.logical_or(
                        chunk_terminations[:, -1], chunk_truncations[:, -1]
                    ).float().cpu()
                    _old_logprobs = _reduce_to_batch(_to_cpu(rollout_result["prev_logprobs"]))
                    _prev_values = _reduce_to_batch(_to_cpu(rollout_result["prev_values"]))
                    _returns = chunk_returns * (1.0 - dones)
                    _ = (_old_logprobs, _prev_values, _returns)
                t_collect = collect_elapsed()

                t_chunk = chunk_elapsed()

            if chunk_idx >= WARMUP_CHUNKS:
                measured_records.append(
                    ChunkRecord(
                        predict_action_batch=t_predict,
                        prepare_actions=t_prepare,
                        env_chunk_step=t_env,
                        collect_samples_pack=t_collect,
                        chunk_total=t_chunk,
                    )
                )

    env.close()

    def mean_ms(key: str) -> float:
        return statistics.mean(getattr(r, key) for r in measured_records) * 1000.0

    result = {
        "num_envs": float(num_envs),
        "predict_action_batch_ms": mean_ms("predict_action_batch"),
        "prepare_actions_ms": mean_ms("prepare_actions"),
        "env_chunk_step_ms": mean_ms("env_chunk_step"),
        "collect_samples_pack_ms": mean_ms("collect_samples_pack"),
        "chunk_total_ms": mean_ms("chunk_total"),
    }

    chunk_total_s = result["chunk_total_ms"] / 1000.0
    result["throughput_env_steps_per_s"] = (
        (num_envs * NUM_ACTION_CHUNKS) / chunk_total_s if chunk_total_s > 0 else 0.0
    )
    result["predict_pct"] = (
        result["predict_action_batch_ms"] / result["chunk_total_ms"] * 100.0
    )
    result["env_pct"] = result["env_chunk_step_ms"] / result["chunk_total_ms"] * 100.0
    result["collect_pct"] = (
        result["collect_samples_pack_ms"] / result["chunk_total_ms"] * 100.0
    )
    return result


def main() -> None:
    torch.set_grad_enabled(False)

    print("=== collect_samples path benchmark (non-invasive) ===")
    print(f"model_path={MODEL_PATH}")
    print(
        "hardcoded: "
        f"precision={MODEL_PRECISION}, state_dim={STATE_DIM}, action_dim={ACTION_DIM}, "
        f"num_action_chunks={NUM_ACTION_CHUNKS}, max_steps_per_rollout_epoch={MAX_STEPS_PER_ROLLOUT_EPOCH}"
    )
    print(
        f"warmup_chunks={WARMUP_CHUNKS}, measured_chunks={MEASURED_CHUNKS}, mode={MODE}, task_suite={TASK_SUITE}"
    )

    model = _build_model()

    results: list[dict[str, float]] = []
    for num_envs in ENV_PARALLEL_CANDIDATES:
        print(f"\n[run] num_envs={num_envs}")
        res = _run_one_case(num_envs=num_envs, model=model)
        results.append(res)
        print(
            "  times_ms: "
            f"predict={res['predict_action_batch_ms']:.2f}, "
            f"prepare={res['prepare_actions_ms']:.2f}, "
            f"env={res['env_chunk_step_ms']:.2f}, "
            f"collect={res['collect_samples_pack_ms']:.2f}, "
            f"total={res['chunk_total_ms']:.2f}"
        )
        print(
            "  share_pct: "
            f"predict={res['predict_pct']:.1f}%, "
            f"env={res['env_pct']:.1f}%, "
            f"collect={res['collect_pct']:.1f}%"
        )
        print(f"  throughput_env_steps_per_s={res['throughput_env_steps_per_s']:.2f}")

    base = results[0]
    print("\n=== scaling summary (vs num_envs=1) ===")
    for res in results:
        speedup = (
            res["throughput_env_steps_per_s"] / base["throughput_env_steps_per_s"]
            if base["throughput_env_steps_per_s"] > 0
            else 0.0
        )
        print(
            f"num_envs={int(res['num_envs'])}: "
            f"throughput={res['throughput_env_steps_per_s']:.2f} steps/s, "
            f"speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
