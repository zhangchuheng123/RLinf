#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    def _cast_floating_tensors(obj: Any, dtype: torch.dtype) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype=dtype) if obj.is_floating_point() else obj
        if isinstance(obj, dict):
            return {k: _cast_floating_tensors(v, dtype) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_cast_floating_tensors(v, dtype) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_cast_floating_tensors(v, dtype) for v in obj)
        return obj

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)

    # ####### TEMP START #######
    temp_dump_path = os.environ.get("LEROBOT_EVAL_DUMP_PATH", "").strip()
    align_pickle_path = os.environ.get("RLINF_ROLLOUT_ALIGN_PICKLE_PATH", "").strip()
    align_records: dict[str, Any] | None = None

    def _to_cpu_debug(obj: Any):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _to_cpu_debug(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu_debug(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cpu_debug(v) for v in obj)
        return obj

    def _tensor_fingerprint(tensor: torch.Tensor | None) -> dict[str, Any] | None:
        if tensor is None:
            return None
        t_cpu = tensor.detach().cpu().contiguous()
        t_float = t_cpu.float()
        t_hash = t_float if t_cpu.dtype == torch.bfloat16 else t_cpu
        return {
            "shape": list(t_cpu.shape),
            "dtype": str(t_cpu.dtype),
            "sha256": hashlib.sha256(t_hash.numpy().tobytes()).hexdigest(),
            "min": float(t_float.min().item()) if t_float.numel() else 0.0,
            "max": float(t_float.max().item()) if t_float.numel() else 0.0,
            "mean": float(t_float.mean().item()) if t_float.numel() else 0.0,
        }

    def _batch_fingerprint(batch: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = _tensor_fingerprint(value)
            elif isinstance(value, list):
                out[key] = {"type": "list", "len": len(value)}
            else:
                out[key] = {"type": type(value).__name__}
        return out

    def _model_fingerprint(model: nn.Module) -> dict[str, Any]:
        hasher = hashlib.sha256()
        for _, param in model.named_parameters():
            p_cpu = param.detach().cpu().contiguous()
            p_hash = p_cpu.float() if p_cpu.dtype == torch.bfloat16 else p_cpu
            hasher.update(p_hash.numpy().tobytes())
        first_param = next(model.parameters())
        return {
            "class": model.__class__.__name__,
            "module": model.__class__.__module__,
            "model_sha256": hasher.hexdigest(),
            "param_dtype": str(first_param.dtype),
            "param_device": str(first_param.device),
        }

    def _backend_snapshot() -> dict[str, Any]:
        return {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

    def _resolve_policy_noise_shape() -> tuple[int, int]:
        chunk_size = getattr(policy, "num_actions", None)
        if chunk_size is None and hasattr(policy, "config"):
            chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise AttributeError("Policy missing num_actions/chunk_size for deterministic noise sampling")

        max_action_dim = getattr(policy, "max_action_dim", None)
        if max_action_dim is None and hasattr(policy, "config"):
            max_action_dim = getattr(policy.config, "max_action_dim", None)
        if max_action_dim is None:
            raise AttributeError("Policy missing max_action_dim for deterministic noise sampling")

        return int(chunk_size), int(max_action_dim)
    # ####### TEMP END #######

    if align_pickle_path:
        align_records = {
            "meta": {
                "source": "lerobot_eval.rollout",
                "num_envs": int(env.num_envs),
                "max_steps": int(max_steps),
            },
            "reset_obs": _to_cpu_debug(observation),
            "steps": [],
        }

    while not np.all(done) and step < max_steps:
        # ####### TEMP START #######
        observation_raw = deepcopy(observation)
        # ####### TEMP END #######

        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        policy_dtype = next(policy.parameters()).dtype
        state_key = "observation.state"
        if state_key in observation and isinstance(observation[state_key], torch.Tensor):
            if observation[state_key].dtype != policy_dtype:
                observation[state_key] = observation[state_key].to(dtype=policy_dtype)
        chunk_size, max_action_dim = _resolve_policy_noise_shape()
        batch_size = int(env.num_envs)
        policy_device = next(policy.parameters()).device
        temp_policy_noise = torch.randn(
            batch_size,
            chunk_size,
            max_action_dim,
            device=policy_device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            if select_action_align_dump_path := os.environ.get("RLINF_SELECT_ACTION_ALIGN_DUMP_PATH", "").strip():
                policy.eval()
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                action = policy._get_action_chunk(observation, noise=temp_policy_noise)
            else:
                action = policy.select_action(observation, noise=temp_policy_noise)

        import pdb; pdb.set_trace()

        select_action_align_dump_path = os.environ.get("RLINF_SELECT_ACTION_ALIGN_DUMP_PATH", "").strip()
        if select_action_align_dump_path:
            align_payload = {
                "meta": {
                    "source": "lerobot_eval.select_action",
                    "step": step,
                },
                "runtime": {
                    "backend": _backend_snapshot(),
                    "model": _model_fingerprint(policy),
                },
                "batch_obs": _to_cpu_debug(observation),
                "step_noise": _to_cpu_debug(temp_policy_noise),
                "norm_actions": _to_cpu_debug(action),
                "batch_obs_fingerprint": _batch_fingerprint(observation),
                "step_noise_fingerprint": _tensor_fingerprint(temp_policy_noise),
                "norm_actions_fingerprint": _tensor_fingerprint(action),
            }
            align_dump_file = Path(select_action_align_dump_path)
            align_dump_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(align_payload, align_dump_file)
            logging.warning("[TEMP] select_action align dump written: %s", align_dump_file)
            raise SystemExit(0)
        
        action_model_raw = action
        action = postprocessor(action)
        action_after_postprocessor = action

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]
        action_numpy: np.ndarray = action.to("cpu").numpy()

        if align_records is not None:
            align_records["steps"].append(
                {
                    "step": int(step),
                    "obs_before_step": _to_cpu_debug(observation_raw),
                    "policy_payload": _to_cpu_debug(observation),
                    "policy_noise": _to_cpu_debug(temp_policy_noise),
                    "model_raw_action": _to_cpu_debug(action_model_raw),
                    "postprocessor_action": _to_cpu_debug(action_after_postprocessor),
                    "env_action": _to_cpu_debug(action),
                    "action_numpy": _to_cpu_debug(action_numpy),
                }
            )

        # ####### TEMP START #######
        if temp_dump_path:
            step_prefix = f"step_{step}_"
            dump_payload = {
                "meta": {
                    "source": "lerobot_eval",
                    "step": step,
                    "max_steps": max_steps,
                    "num_envs": env.num_envs,
                },
                f"{step_prefix}env_obs": _to_cpu_debug(observation_raw),
                f"{step_prefix}policy_payload": _to_cpu_debug(observation),
                f"{step_prefix}model_raw_action": _to_cpu_debug(action_model_raw),
                f"{step_prefix}postprocessor_action": _to_cpu_debug(action_after_postprocessor),
                f"{step_prefix}env_action": _to_cpu_debug(action),
                f"{step_prefix}action_numpy": _to_cpu_debug(action_numpy),
                f"{step_prefix}policy_noise": _to_cpu_debug(temp_policy_noise),
                "obs_raw": _to_cpu_debug(observation_raw),
                "policy_debug": {
                    "policy_payload": _to_cpu_debug(observation),
                    "policy_action_model_raw": _to_cpu_debug(action_model_raw),
                    "policy_action_after_postprocessor": _to_cpu_debug(action_after_postprocessor),
                    "policy_action_post": _to_cpu_debug(action),
                    "policy_noise": _to_cpu_debug(temp_policy_noise),
                },
                "robot_action_to_send": _to_cpu_debug(action),
                "action_numpy": _to_cpu_debug(action_numpy),
            }
            dump_file = Path(temp_dump_path)
            dump_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dump_payload, dump_file)
            logging.warning("[TEMP] lerobot-eval dump written: %s", dump_file)
            raise SystemExit(0)
        # ####### TEMP END #######

        # Convert to CPU / numpy.
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if align_records is not None:
            align_records["steps"][-1]["obs_after_step"] = _to_cpu_debug(observation)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    if align_records is not None:
        align_file = Path(align_pickle_path)
        align_file.parent.mkdir(parents=True, exist_ok=True)
        with open(align_file, "wb") as file:
            pickle.dump(align_records, file)
        logging.warning("[ALIGN] rollout alignment pickle written: %s", align_file)

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    requested_precision = (cfg.inference_precision or "auto").lower()
    if requested_precision not in {"auto", "fp32", "bf16"}:
        raise ValueError(
            f"Unsupported inference_precision='{cfg.inference_precision}'. "
            "Expected one of: auto, fp32, bf16"
        )

    if requested_precision == "fp32":
        policy = policy.to(dtype=torch.float32)
    elif requested_precision == "bf16":
        if device.type != "cuda":
            raise RuntimeError("bf16 inference_precision requires CUDA device")
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 8:
            logging.warning(
                "Running bf16 on GPU capability sm_%s%s; kernels may fall back and performance may degrade.",
                capability[0],
                capability[1],
            )
        policy = policy.to(dtype=torch.bfloat16)

    logging.info(
        "Inference precision mode: %s, first_param_dtype=%s",
        requested_precision,
        str(next(policy.parameters()).dtype),
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)

        # Print per-task success detail in requested format.
        if "per_task" in info:
            print("\nPer-task success details:")
            per_task_sorted = sorted(info["per_task"], key=lambda item: (item["task_group"], item["task_id"]))
            for item in per_task_sorted:
                task_id = item["task_id"]
                instruction = item.get("instruction", "")
                successes = item["metrics"].get("successes", [])
                success_flags = ["T" if bool(v) else "F" for v in successes]
                print(f'task_{task_id}, instruction="{instruction}", success={success_flags}')
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    instruction = ""
    try:
        task_result = env.call("task_description")
        if isinstance(task_result, list) and task_result and all(isinstance(v, str) for v in task_result):
            instruction = task_result[0]
    except Exception:
        instruction = ""

    return task_group, task_id, metrics, instruction


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics, instruction = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append(
                {
                    "task_group": tg,
                    "task_id": tid,
                    "instruction": instruction,
                    "metrics": metrics,
                }
            )
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics, instruction = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append(
                    {
                        "task_group": tg,
                        "task_id": tid,
                        "instruction": instruction,
                        "metrics": metrics,
                    }
                )

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
