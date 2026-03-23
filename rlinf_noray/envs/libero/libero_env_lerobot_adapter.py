from __future__ import annotations

from typing import Any, Optional, Union
from pathlib import Path
import pickle
import os

import gymnasium as gym
import numpy as np
import torch

from rlinf_noray.integrations.lerobot_local_import import ensure_local_lerobot
from rlinf_noray.envs.utils import list_of_dict_to_dict_of_list, to_tensor

ensure_local_lerobot()

from lerobot.envs.libero import create_libero_envs


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    den = np.sqrt(np.clip(1.0 - quat[..., 3] * quat[..., 3], a_min=0.0, a_max=None))
    result = np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
    mask = den > 1e-10
    if np.any(mask):
        angle = 2.0 * np.arccos(np.clip(quat[..., 3][mask], -1.0, 1.0))
        axis = quat[..., :3][mask] / den[mask][..., None]
        result[mask] = axis * angle[..., None]
    return result


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        self.seed = int(cfg.seed) + self.seed_offset
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.max_episode_steps = int(cfg.max_episode_steps)

        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        task_ids = cfg.get("task_ids", None)
        if task_ids is None:
            suite_task_num = {
                "libero_spatial": 10,
                "libero_object": 10,
                "libero_goal": 10,
                "libero_10": 10,
                "libero_90": 90,
            }
            total_tasks = suite_task_num.get(str(cfg.task_suite_name), 10)
            task_ids = list(range(total_tasks))
        task_ids = [int(task_id) for task_id in task_ids]

        if cfg.get("specific_task_id", None) is not None:
            selected_task_id = int(cfg.specific_task_id)
        else:
            selected_task_id = task_ids[self.seed_offset % len(task_ids)]

        init_params = dict(cfg.get("init_params", {}))
        gym_kwargs = {}
        if "camera_heights" in init_params:
            gym_kwargs["observation_height"] = int(init_params["camera_heights"])
        if "camera_widths" in init_params:
            gym_kwargs["observation_width"] = int(init_params["camera_widths"])
        if "observation_height" in init_params:
            gym_kwargs["observation_height"] = int(init_params["observation_height"])
        if "observation_width" in init_params:
            gym_kwargs["observation_width"] = int(init_params["observation_width"])
        gym_kwargs["obs_type"] = str(cfg.get("obs_type", "pixels_agent_pos"))
        gym_kwargs["render_mode"] = str(cfg.get("render_mode", "rgb_array"))
        env_map = create_libero_envs(
            task=str(cfg.task_suite_name),
            n_envs=self.num_envs,
            gym_kwargs={**gym_kwargs, "task_ids": [selected_task_id]},
            camera_name=cfg.get("camera_name", "agentview_image,robot0_eye_in_hand_image"),
            init_states=True,
            env_cls=gym.vector.SyncVectorEnv,
            control_mode=cfg.get("control_mode", "relative"),
            episode_length=int(cfg.max_episode_steps),
        )
        self.env = env_map[str(cfg.task_suite_name)][selected_task_id]
        self.task_ids = np.full((self.num_envs,), selected_task_id, dtype=np.int32)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    def _convert_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        pixels = observation["pixels"]
        main_images = torch.from_numpy(pixels["image"]).contiguous()
        wrist_images = None
        if "image2" in pixels:
            wrist_images = torch.from_numpy(pixels["image2"]).contiguous()

        robot_state = observation["robot_state"]
        eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
        eef_quat = np.asarray(robot_state["eef"]["quat"], dtype=np.float32)
        gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)

        states = np.concatenate([eef_pos, _quat2axisangle(eef_quat), gripper_qpos], axis=-1)
        states_tensor = torch.from_numpy(states).contiguous()

        task_descriptions = self.env.call("task_description")
        if isinstance(task_descriptions, tuple):
            task_descriptions = list(task_descriptions)

        return {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states_tensor,
            "task_descriptions": [str(task_desc) for task_desc in task_descriptions],
            "pixels": pixels,
            "robot_state": robot_state,
        }

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        del env_idx, reset_state_ids
        seeds = [self.seed + i for i in range(self.num_envs)]
        observation, _ = self.env.reset(seed=seeds)
        self._elapsed_steps[:] = 0

        align_pickle_path = os.environ.get("RLINF_ROLLOUT_ALIGN_PICKLE_PATH", "")
        if align_pickle_path:
            align_file = Path(align_pickle_path)
            with open(align_file, "rb") as file:
                align_data = pickle.load(file)
            observation = align_data["observation_from_env"]

        return self._convert_observation(observation), {}

    def step(self, actions=None, auto_reset=True):
        del auto_reset
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().float().numpy()

        observation, reward, terminations, truncations, infos = self.env.step(actions)
        self._elapsed_steps += 1

        infos_dict = dict(infos) if isinstance(infos, dict) else {}
        if self.ignore_terminations:
            infos_dict.setdefault("episode", {})
            infos_dict["episode"]["success_at_end"] = to_tensor(terminations)
            terminations = np.zeros_like(terminations, dtype=bool)

        dones = np.logical_or(terminations, truncations)
        self._elapsed_steps[dones] = 0

        return (
            self._convert_observation(observation),
            to_tensor(reward),
            to_tensor(terminations),
            to_tensor(truncations),
            list_of_dict_to_dict_of_list([infos_dict] * self.num_envs) if infos_dict else {},
        )

    def chunk_step(self, chunk_actions):
        chunk_size = int(chunk_actions.shape[1])
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        done_in_chunk = torch.zeros(self.num_envs, dtype=torch.bool)

        align_pickle_path = os.environ.get("RLINF_ROLLOUT_ALIGN_PICKLE_PATH", "")
        if align_pickle_path:
            align_file = Path(align_pickle_path)
            with open(align_file, "rb") as file:
                align_data = pickle.load(file)

            assert torch.allclose(chunk_actions[:, 0], torch.tensor(align_data['action_after_postprocessor']))
            print("Passed consistency check for first action in chunk with align data")

        for step_index in range(chunk_size):
            actions = chunk_actions[:, step_index]
            obs, step_reward, terminations, truncations, infos = self.step(actions, auto_reset=False)

            if done_in_chunk.any():
                step_reward = step_reward.clone()
                terminations = terminations.clone()
                truncations = truncations.clone()
                step_reward[done_in_chunk] = 0
                terminations[done_in_chunk] = False
                truncations[done_in_chunk] = False

            newly_done = torch.logical_and(torch.logical_or(terminations, truncations), ~done_in_chunk)
            if newly_done.any():
                done_in_chunk = torch.logical_or(done_in_chunk, newly_done)

            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_terminations[:, -1] = raw_chunk_terminations.any(dim=1)
            chunk_truncations[:, -1] = raw_chunk_truncations.any(dim=1)
        else:
            chunk_terminations = raw_chunk_terminations
            chunk_truncations = raw_chunk_truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )
