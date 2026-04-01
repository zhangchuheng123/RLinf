import copy
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from rlinf_noray.envs import get_env_cls
from rlinf_noray.models import get_model
from rlinf_noray.models.embodiment.modules.gaussian_policy import GaussianPolicy
from rlinf_noray.utils.metric_logger import MetricLogger


@dataclass(eq=False)
class Transition:
    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    env_id: int = -1
    effective: bool = False
    returns: float = 0.0
    advantage: float = 0.0
    advantage_raw: float = 0.0
    value_target: float = 0.0
    debug_frames: list[np.ndarray] | None = None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        verbose: bool,
        debug_video_enabled: bool = False,
    ):
        assert capacity % num_envs == 0, (
            f"replay capacity={capacity} must be divisible by num_envs={num_envs}"
        )
        self.capacity = capacity
        self.num_envs = num_envs
        self.verbose = verbose

        self.data: list[Transition] = []
        self.env_queues: list[list[Transition]] = [[] for _ in range(num_envs)]
        self._pending_by_env: list[list[Transition]] = [[] for _ in range(num_envs)]
        self._recent_success = deque(maxlen=100)
        self.debug_video_enabled = debug_video_enabled
        self._debug_completed_trajectories: list[list[Transition]] = []
        self._debug_video_counter = 0

    def __len__(self) -> int:
        return len(self.data)

    def _remove_entry(self, entry: Transition) -> None:
        env_id = entry.env_id
        if entry in self.env_queues[env_id]:
            self.env_queues[env_id].remove(entry)
        if entry in self._pending_by_env[env_id]:
            self._pending_by_env[env_id].remove(entry)

    def _append_entry(self, entry: Transition) -> None:
        self.data.append(entry)
        self.env_queues[entry.env_id].append(entry)
        while len(self.data) > self.capacity:
            dropped = self.data.pop(0)
            self._remove_entry(dropped)

    def _mark_completed_trajectory(self, trajectory: list[Transition], gamma: float) -> None:
        running_return = 0.0
        for transition in reversed(trajectory):
            running_return = float(transition.reward) + gamma * running_return
            transition.returns = running_return
            transition.effective = True

        if self.debug_video_enabled:
            self._debug_completed_trajectories.append(list(trajectory))

        success = float(trajectory[-1].reward) > 0.0
        self._recent_success.append(1 if success else 0)

    def recent_success_rate(self) -> float:
        total = len(self._recent_success)
        if total == 0:
            return 0.0
        return float(sum(self._recent_success)) / float(total)

    @staticmethod
    def _to_uint8_hwc(img: Any) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    @staticmethod
    def _annotate_debug_frame(
        frame: np.ndarray,
        traj_idx: int,
        frame_idx: int,
        success: bool,
        reward: float,
        ret: float,
    ) -> np.ndarray:
        image = Image.fromarray(np.flipud(ReplayBuffer._to_uint8_hwc(frame)).copy())
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = (
            f"traj={traj_idx} frame={frame_idx} "
            f"success={int(success)} reward={reward:.4f} return={ret:.4f}"
        )
        draw.text((8, 8), text, fill=(255, 255, 255), font=font)
        return np.asarray(image)

    def save_video(self) -> None:
        if not self.debug_video_enabled or not self._debug_completed_trajectories:
            return

        out_frames: list[np.ndarray] = []
        for traj_idx, trajectory in enumerate(self._debug_completed_trajectories):
            if not trajectory:
                continue

            success = float(trajectory[-1].reward) > 0.0
            for transition in trajectory:
                frames = transition.debug_frames or []
                if not frames:
                    continue
                reward = float(transition.reward)
                ret = float(transition.returns)
                for frame_idx, frame in enumerate(frames):
                    out_frames.append(
                        self._annotate_debug_frame(
                            frame=frame,
                            traj_idx=traj_idx,
                            frame_idx=frame_idx,
                            success=success,
                            reward=reward,
                            ret=ret,
                        )
                    )

        if out_frames:
            self._debug_video_counter += 1
            out_file = f"debug_{self._debug_video_counter:04d}.mp4"
            imageio.mimsave(out_file, out_frames, fps=1)

        self._debug_completed_trajectories = []

    def add_rollout(self, transitions: list[Transition], gamma: float) -> None:
        for transition_idx, transition in enumerate(transitions):
            env_id = transition_idx % self.num_envs
            transition.env_id = env_id
            transition.effective = False
            transition.returns = 0.0
            transition.advantage = 0.0
            transition.advantage_raw = 0.0
            transition.value_target = 0.0

            self._append_entry(transition)
            self._pending_by_env[env_id].append(transition)

            if bool(transition.done):
                finished = self._pending_by_env[env_id]
                assert finished, "pending trajectory should not be empty when done=True"
                self._mark_completed_trajectory(finished, gamma)
                self._pending_by_env[env_id] = []

    def _effective_entries(self) -> list[Transition]:
        return [transition for transition in self.data if transition.effective]

    def prepare_gae_targets(
        self,
        value_model: DistributedDataParallel,
        device: torch.device,
        gamma: float,
        gae_lambda: float,
        do_adv_norm: bool,
        adv_norm_eps: float,
        advantage_clip: float,
    ) -> int:
        effective_entries = self._effective_entries()
        if not effective_entries:
            return 0

        states = torch.stack([entry.state for entry in effective_entries], dim=0).to(device)
        next_states = torch.stack([entry.next_state for entry in effective_entries], dim=0).to(device)

        with torch.no_grad():
            values = _predict_values(value_model, states).detach().cpu()
            next_values = _predict_values(value_model, next_states).detach().cpu()

        next_value_by_id: dict[int, float] = {}
        for index, entry in enumerate(effective_entries):
            entry_value = float(values[index].item())
            entry_next_value = float(next_values[index].item())
            entry.value_target = entry_value
            entry.advantage = 0.0
            entry.returns = entry.returns
            next_value_by_id[id(entry)] = entry_next_value

        for env_id in range(self.num_envs):
            env_entries = [entry for entry in self.env_queues[env_id] if entry.effective]
            if not env_entries:
                continue

            gae = 0.0
            for entry in reversed(env_entries):
                done_mask = 0.0 if bool(entry.done) else 1.0
                delta = (
                    float(entry.reward)
                    + gamma * next_value_by_id[id(entry)] * done_mask
                    - entry.value_target
                )
                gae = delta + gamma * gae_lambda * done_mask * gae
                entry.advantage_raw = gae
                entry.advantage = gae
                entry.value_target = gae + entry.value_target

        if do_adv_norm:
            advantages = torch.tensor(
                [entry.advantage for entry in effective_entries], dtype=torch.float32
            )
            mean = float(advantages.mean().item())
            std = float(advantages.std(unbiased=False).item())
            denom = std + adv_norm_eps
            for entry in effective_entries:
                normalized_adv = (entry.advantage - mean) / denom
                entry.advantage = float(
                    np.clip(normalized_adv, -advantage_clip, advantage_clip)
                )

        return len(effective_entries)

    def sample(self, batch_size: int) -> list[Transition]:
        effective_entries = self._effective_entries()
        assert effective_entries, "No effective transitions available in replay buffer"
        indices = np.random.choice(len(effective_entries), size=batch_size, replace=True)
        return [effective_entries[int(index)] for index in indices]


class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float32)
        self.m2 = torch.zeros(dim, dtype=torch.float32)
        self.eps = eps

    def update(self, x: torch.Tensor) -> None:
        x = x.float().reshape(-1, x.shape[-1]).cpu()
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / float(self.count)
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def normalize(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self.count > 0, "RunningNorm must be updated before normalization"
        if self.count == 1:
            mean = self.mean.to(device)
            return x.float() - mean
        var = self.m2 / float(self.count - 1)
        std = torch.sqrt(var + self.eps).to(device)
        mean = self.mean.to(device)
        return (x.float() - mean) / std


def _build_value_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> torch.nn.Sequential:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    layers: list[torch.nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(current_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        current_dim = hidden_dim
    layers.append(torch.nn.Linear(current_dim, output_dim))
    return torch.nn.Sequential(*layers)


class DSRLScalarValueNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.net = _build_value_mlp(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)

    def predict_value_from_output(self, output: torch.Tensor) -> torch.Tensor:
        return output.float()

    def predict_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.predict_value_from_output(self.forward(states))

    def compute_loss_from_output(
        self,
        output: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(output.float(), targets.float())

    def compute_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_loss_from_output(self.forward(states), targets)


class DSRLDistributionalValueNet(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        v_min: float = 0.0,
        v_max: float = 1.0,
        n_bins: int = 16,
    ):
        super().__init__()
        if n_bins <= 1:
            raise ValueError(f"n_bins must be > 1, got {n_bins}")
        if v_max <= v_min:
            raise ValueError(f"v_max must be > v_min, got v_min={v_min}, v_max={v_max}")

        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.n_bins = int(n_bins)
        self.net = _build_value_mlp(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=n_bins,
            num_layers=num_layers,
        )
        self.register_buffer(
            "bin_centers",
            torch.linspace(self.v_min, self.v_max, self.n_bins, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def clamp_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets.float().clamp(min=self.v_min, max=self.v_max)

    def target_to_bin_indices(self, targets: torch.Tensor) -> torch.Tensor:
        clamped = self.clamp_targets(targets)
        distances = torch.abs(clamped.unsqueeze(-1) - self.bin_centers.unsqueeze(0))
        return torch.argmin(distances, dim=-1)

    def predict_value_from_output(self, output: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(output.float(), dim=-1)
        return torch.sum(probs * self.bin_centers.unsqueeze(0), dim=-1)

    def predict_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.predict_value_from_output(self.forward(states))

    def compute_loss_from_output(
        self,
        output: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        target_bins = self.target_to_bin_indices(targets)
        return torch.nn.functional.cross_entropy(output.float(), target_bins)

    def compute_loss(self, states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_loss_from_output(self.forward(states), targets)


def _to_tensor_states(states: Any) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().float()
    return torch.as_tensor(states, dtype=torch.float32)


def _unwrap_value_model(value_model: torch.nn.Module) -> torch.nn.Module:
    return value_model.module if hasattr(value_model, "module") else value_model


def _predict_values(value_model: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    raw_output = value_model(states)
    model = _unwrap_value_model(value_model)
    if hasattr(model, "predict_value_from_output"):
        return model.predict_value_from_output(raw_output).float()
    if hasattr(model, "predict_value"):
        return model.predict_value(states).float()
    return raw_output.float()


def _compute_value_loss(
    value_model: torch.nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    raw_output = value_model(states)
    model = _unwrap_value_model(value_model)
    if hasattr(model, "compute_loss_from_output"):
        return model.compute_loss_from_output(raw_output, targets)
    if hasattr(model, "compute_loss"):
        return model.compute_loss(states, targets)
    return torch.nn.functional.mse_loss(raw_output.float(), targets.float())


def _reduce_mean_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(metrics.keys())
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}


def _to_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    return bool(value)


class LiberoDSRLDDPNoRayRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        model_type = str(cfg.actor.model.model_type).lower()
        env_type = str(cfg.env.train.env_type).lower()
        assert env_type == "libero", f"Only libero env is supported, got {env_type}"
        assert model_type == "smolvla", (
            f"Only smolvla is supported in dsrl runner, got {model_type}"
        )

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            backend = "nccl"
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            backend = "gloo"
            self.device = torch.device("cpu")

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        total_num_envs = int(cfg.env.train.total_num_envs)
        assert total_num_envs % self.world_size == 0, (
            f"env.train.total_num_envs={total_num_envs} must be divisible by world_size={self.world_size}"
        )
        self.local_num_envs = total_num_envs // self.world_size

        env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
        env_cfg.total_num_envs = self.local_num_envs
        env_cls = get_env_cls(env_cfg.env_type, env_cfg)
        self.env = env_cls(
            cfg=env_cfg,
            num_envs=self.local_num_envs,
            seed_offset=self.rank,
            total_num_processes=self.world_size,
            worker_info=None,
        )

        self.val_check_interval = int(cfg.runner.val_check_interval)
        self.only_eval = bool(cfg.runner.only_eval)
        self.eval_env = None
        self.eval_rollout_epoch = int(cfg.algorithm.eval_rollout_epoch)
        self.eval_chunk_steps = 0

        if self.val_check_interval > 0 or self.only_eval:
            eval_total_num_envs = int(cfg.env.eval.total_num_envs)
            assert eval_total_num_envs % self.world_size == 0, (
                f"env.eval.total_num_envs={eval_total_num_envs} must be divisible by world_size={self.world_size}"
            )
            eval_local_num_envs = eval_total_num_envs // self.world_size
            eval_env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
            eval_env_cfg.total_num_envs = eval_local_num_envs
            eval_env_cls = get_env_cls(eval_env_cfg.env_type, eval_env_cfg)
            self.eval_env = eval_env_cls(
                cfg=eval_env_cfg,
                num_envs=eval_local_num_envs,
                seed_offset=self.rank,
                total_num_processes=self.world_size,
                worker_info=None,
            )
            self.eval_chunk_steps = int(cfg.env.eval.max_steps_per_rollout_epoch) // int(cfg.runner.num_execute_steps)

        model_cfg = copy.deepcopy(cfg.actor.model)
        self.generator = get_model(model_cfg)
        self.generator.eval()
        for parameter in self.generator.parameters():
            parameter.requires_grad = False

        self.max_epochs = int(cfg.runner.max_epochs)
        self.rollout_epoch = int(cfg.algorithm.rollout_epoch)
        self.update_epoch = int(cfg.algorithm.update_epoch)
        self.pre_value_update_epoch = int(cfg.algorithm.get("pre_value_update_epoch", 20))
        if self.pre_value_update_epoch < 0:
            raise ValueError(
                "algorithm.pre_value_update_epoch must be >= 0, "
                f"got {self.pre_value_update_epoch}"
            )
        self.num_execute_steps = int(cfg.runner.num_execute_steps)
        if self.num_execute_steps <= 0:
            raise ValueError(
                f"runner.num_execute_steps must be > 0, got {self.num_execute_steps}"
            )
        self.chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // self.num_execute_steps

        self.chunk_size = int(self.generator.policy.config.chunk_size)
        self.max_action_dim = int(self.generator.policy.config.max_action_dim)
        self.dsrl_noise_dim = self.chunk_size * self.max_action_dim
        self.dsrl_hidden_dim = int(cfg.actor.model.dsrl_hidden_dim)
        self.dsrl_actor_num_layers = int(cfg.actor.model.dsrl_actor_num_layers)
        self.dsrl_value_num_layers = int(cfg.actor.model.dsrl_value_num_layers)
        if self.dsrl_actor_num_layers <= 0:
            raise ValueError(
                f"actor.model.dsrl_actor_num_layers must be > 0, got {self.dsrl_actor_num_layers}"
            )
        if self.dsrl_value_num_layers <= 0:
            raise ValueError(
                f"actor.model.dsrl_value_num_layers must be > 0, got {self.dsrl_value_num_layers}"
            )
        self.dsrl_value_head_type = str(cfg.actor.model.get("dsrl_value_head_type", "scalar")).lower()
        self.dsrl_value_v_min = float(cfg.actor.model.get("dsrl_value_v_min", 0.0))
        self.dsrl_value_v_max = float(cfg.actor.model.get("dsrl_value_v_max", 1.0))
        self.dsrl_value_n_bins = int(cfg.actor.model.get("dsrl_value_n_bins", 16))
        if not hasattr(self.generator, "get_dsrl_state_dim"):
            raise AttributeError("SmolVLA aligned policy must implement get_dsrl_state_dim() for DSRL")
        self.state_dim = int(self.generator.get_dsrl_state_dim())

        configured_noise_dim = int(cfg.actor.model.dsrl_noise_dim)
        if configured_noise_dim != self.dsrl_noise_dim:
            raise ValueError(
                "actor.model.dsrl_noise_dim must match SmolVLA policy noise size "
                f"(chunk_size * max_action_dim = {self.dsrl_noise_dim}), got {configured_noise_dim}"
            )

        self.policy_noise_eps = float(cfg.actor.model.policy_noise_eps)
        self.policy_noise_scale = float(cfg.actor.model.policy_noise_scale)
        self.policy_noise_bias = float(cfg.actor.model.policy_noise_bias)

        self.actor = GaussianPolicy(
            input_dim=self.state_dim,
            output_dim=self.dsrl_noise_dim,
            hidden_dims=tuple(self.dsrl_hidden_dim for _ in range(self.dsrl_actor_num_layers)),
            action_horizon=1,
        )
        if self.dsrl_value_head_type == "scalar":
            self.value_net = DSRLScalarValueNet(
                self.state_dim,
                hidden_dim=self.dsrl_hidden_dim,
                num_layers=self.dsrl_value_num_layers,
            )
        elif self.dsrl_value_head_type == "distributional":
            self.value_net = DSRLDistributionalValueNet(
                self.state_dim,
                hidden_dim=self.dsrl_hidden_dim,
                num_layers=self.dsrl_value_num_layers,
                v_min=self.dsrl_value_v_min,
                v_max=self.dsrl_value_v_max,
                n_bins=self.dsrl_value_n_bins,
            )
        else:
            raise ValueError(
                "actor.model.dsrl_value_head_type must be either 'scalar' or 'distributional', "
                f"got {self.dsrl_value_head_type}"
            )

        self.actor.to(self.device)
        self.value_net.to(self.device)

        ddp_kwargs = {}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [self.local_rank]
            ddp_kwargs["output_device"] = self.local_rank
        self.actor = DistributedDataParallel(self.actor, **ddp_kwargs)
        self.value_net = DistributedDataParallel(self.value_net, **ddp_kwargs)

        actor_lr = float(cfg.actor.optim.dsrl_actor_lr)
        value_lr = float(cfg.actor.optim.dsrl_value_lr)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            betas=(float(cfg.actor.optim.adam_beta1), float(cfg.actor.optim.adam_beta2)),
            eps=float(cfg.actor.optim.adam_eps),
            weight_decay=float(cfg.actor.optim.weight_decay),
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=value_lr,
            betas=(float(cfg.actor.optim.adam_beta1), float(cfg.actor.optim.adam_beta2)),
            eps=float(cfg.actor.optim.adam_eps),
            weight_decay=float(cfg.actor.optim.weight_decay),
        )

        self.chunk_gamma = float(cfg.algorithm.gamma)
        self.gae_lambda = float(cfg.algorithm.gae_lambda)
        self.ppo_clip = float(cfg.algorithm.dsrl_ppo_clip)
        self.max_log_ratio = float(cfg.algorithm.dsrl_max_log_ratio)
        self.entropy_coef = float(cfg.algorithm.dsrl_entropy_coef)
        self.grad_clip = float(cfg.actor.optim.clip_grad)
        self.global_minibatch_size = int(cfg.algorithm.dsrl_minibatch_size)
        self.do_adv_norm = bool(cfg.algorithm.get("do_adv_norm", True))
        self.adv_norm_eps = float(cfg.algorithm.get("adv_norm_eps", 1.0e-8))
        self.advantage_clip = float(cfg.algorithm.get("advantage_clip", 3.0))
        if self.adv_norm_eps <= 0.0:
            raise ValueError(
                f"algorithm.adv_norm_eps must be > 0, got {self.adv_norm_eps}"
            )
        if self.advantage_clip <= 0.0:
            raise ValueError(
                f"algorithm.advantage_clip must be > 0, got {self.advantage_clip}"
            )
        replay_capacity_in_epoch = float(cfg.algorithm.dsrl_replay_buffer_capacity_in_epoch)
        assert replay_capacity_in_epoch > 0.0, (
            "algorithm.dsrl_replay_buffer_capacity_in_epoch must be > 0, "
            f"got {replay_capacity_in_epoch}"
        )

        assert self.global_minibatch_size > 0, (
            "algorithm.dsrl_minibatch_size must be > 0, "
            f"got {self.global_minibatch_size}"
        )
        assert self.global_minibatch_size % self.world_size == 0, (
            "algorithm.dsrl_minibatch_size is interpreted as a global PPO minibatch size and "
            f"must be divisible by world_size={self.world_size}, got {self.global_minibatch_size}"
        )
        self.local_minibatch_size = self.global_minibatch_size // self.world_size

        local_transitions_per_epoch = self.local_num_envs * self.rollout_epoch * self.chunk_steps
        global_transitions_per_epoch = local_transitions_per_epoch * self.world_size
        replay_capacity = int(math.ceil(local_transitions_per_epoch * replay_capacity_in_epoch))
        # ReplayBuffer requires capacity divisible by num_envs; round up to preserve requested horizon.
        if replay_capacity % self.local_num_envs != 0:
            replay_capacity = (
                (replay_capacity + self.local_num_envs - 1) // self.local_num_envs
            ) * self.local_num_envs
        self.debug_replay_buffer_returns = os.getenv("DEBUG_REPLAY_BUFFER_RETURNS") is not None
        self.debug_value_only = os.getenv("DEBUG_VALUE_ONLY") is not None
        self._ref_value_running_mean = 0.0
        self._ref_value_count = 0
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            num_envs=self.local_num_envs,
            verbose=self.rank == 0,
            debug_video_enabled=self.debug_replay_buffer_returns and self.rank == 0,
        )
        # self.state_norm = RunningNorm(self.state_dim)

        self.metric_logger = MetricLogger(cfg) if self.rank == 0 else None

        self.save_eval_video = bool(cfg.runner.save_eval_video)
        self.save_rollout_video = bool(cfg.runner.save_rollout_video)
        self.eval_video_base_dir = str(cfg.runner.eval_video_base_dir)
        self.rollout_video_base_dir = str(cfg.runner.rollout_video_base_dir)
        self._video_traj_counter = self.rank * 1_000_000
        self._train_obs: dict[str, Any] | None = None
        self._train_dsrl_states: torch.Tensor | None = None

        if self.rank == 0:
            init_summary = {
                "rollout_epoch": self.rollout_epoch,
                "update_epoch": self.update_epoch,
                "pre_value_update_epoch": self.pre_value_update_epoch,
                "num_execute_steps": self.num_execute_steps,
                "chunk_steps": self.chunk_steps,
                "state_dim": self.state_dim,
                "dsrl_noise_dim": self.dsrl_noise_dim,
                "dsrl_hidden_dim": self.dsrl_hidden_dim,
                "dsrl_actor_num_layers": self.dsrl_actor_num_layers,
                "dsrl_value_num_layers": self.dsrl_value_num_layers,
                "dsrl_value_head_type": self.dsrl_value_head_type,
                "dsrl_value_v_min": self.dsrl_value_v_min,
                "dsrl_value_v_max": self.dsrl_value_v_max,
                "dsrl_value_n_bins": self.dsrl_value_n_bins,
                "actor_lr": actor_lr,
                "value_lr": value_lr,
                "chunk_gamma": self.chunk_gamma,
                "gae_lambda": self.gae_lambda,
                "ppo_clip": self.ppo_clip,
                "max_log_ratio": self.max_log_ratio,
                "entropy_coef": self.entropy_coef,
                "grad_clip": self.grad_clip,
                "global_minibatch_size": self.global_minibatch_size,
                "local_minibatch_size": self.local_minibatch_size,
                "do_adv_norm": self.do_adv_norm,
                "adv_norm_eps": self.adv_norm_eps,
                "advantage_clip": self.advantage_clip,
                "replay_capacity_in_epoch": replay_capacity_in_epoch,
                "global_transitions_per_epoch": global_transitions_per_epoch,
                "local_transitions_per_epoch": local_transitions_per_epoch,
                "replay_capacity": replay_capacity,
                "world_size": self.world_size,
                "policy_noise_eps": self.policy_noise_eps,
                "policy_noise_scale": self.policy_noise_scale,
                "policy_noise_bias": self.policy_noise_bias,
                "save_eval_video": self.save_eval_video,
                "save_rollout_video": self.save_rollout_video,
                "eval_video_base_dir": self.eval_video_base_dir,
                "rollout_video_base_dir": self.rollout_video_base_dir,
                "debug_replay_buffer_returns": self.debug_replay_buffer_returns,
                "debug_value_only": self.debug_value_only,
            }
            print(f"[noray][dsrl][init] {json.dumps(init_summary, sort_keys=True)}", flush=True)

    @staticmethod
    def _to_uint8_hwc(img: Any) -> np.ndarray:
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)

        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    @staticmethod
    def _render_overlay_frame(main_image: Any, action: Any) -> np.ndarray:
        frame = np.flipud(LiberoDSRLDDPNoRayRunner._to_uint8_hwc(main_image)).copy()
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()

        if isinstance(action, torch.Tensor):
            action_np = action.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
        else:
            action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        action_text = "[" + ", ".join(f"{v:+.2f}" for v in action_np.tolist()) + "]"
        text = f"action: {action_text}"
        _, _, _, text_h = draw.textbbox((0, 0), text, font=font)
        y = max(0, frame.shape[0] - text_h - 8)
        draw.text((8, y), text, fill=(255, 255, 255), font=font)
        return np.asarray(pil)

    @staticmethod
    def _instruction_to_slug(instruction: str) -> str:
        slug = "".join(ch if ch.isalnum() else "_" for ch in str(instruction).strip())
        slug = "_".join(part for part in slug.split("_") if part)
        return (slug or "task")[:80]

    def _init_video_buffers(self, num_envs: int) -> dict[str, Any]:
        return {
            "frames_by_env": [[] for _ in range(num_envs)],
            "instructions_by_env": ["" for _ in range(num_envs)],
        }

    def _next_video_traj_id(self) -> int:
        traj_id = int(self._video_traj_counter)
        self._video_traj_counter += 1
        return traj_id

    def _save_single_video(
        self,
        frames: list[np.ndarray],
        instruction: str,
        success: bool,
        mode: str,
    ) -> None:
        base_dir_str = (
            self.eval_video_base_dir if mode == "eval" else self.rollout_video_base_dir
        )
        if not base_dir_str or not frames:
            return

        base_dir = Path(base_dir_str)
        base_dir.mkdir(parents=True, exist_ok=True)

        traj_id = self._next_video_traj_id()
        instruction_slug = self._instruction_to_slug(instruction)
        status = "success" if success else "fail"
        out_path = base_dir / f"traj_{traj_id:04d}_{instruction_slug}_{status}.mp4"
        imageio.mimsave(str(out_path), frames, fps=15)

    def _append_and_maybe_flush_videos(
        self,
        video_buffers: dict[str, Any],
        obs_list: list[dict[str, Any]],
        chunk_actions: torch.Tensor,
        chunk_terminations: torch.Tensor,
        chunk_truncations: torch.Tensor,
        mode: str,
    ) -> None:
        frames_by_env: list[list[np.ndarray]] = video_buffers["frames_by_env"]
        instructions_by_env: list[str] = video_buffers["instructions_by_env"]

        for step_idx, step_obs in enumerate(obs_list):
            main_images = step_obs.get("main_images")
            task_descs = step_obs.get("task_descriptions")
            if main_images is None or task_descs is None:
                continue

            step_actions = chunk_actions[:, step_idx]
            for env_idx in range(len(frames_by_env)):
                if not instructions_by_env[env_idx]:
                    instructions_by_env[env_idx] = str(task_descs[env_idx])

                frame = self._render_overlay_frame(
                    main_image=main_images[env_idx],
                    action=step_actions[env_idx],
                )
                frames_by_env[env_idx].append(frame)

                terminated = _to_bool(chunk_terminations[env_idx, step_idx])
                truncated = _to_bool(chunk_truncations[env_idx, step_idx])
                if terminated or truncated:
                    is_success = terminated and (not truncated)
                    print(
                        f"[noray][ddp] Flushing video for env_idx={env_idx} success={is_success}",
                    )
                    self._save_single_video(
                        frames=frames_by_env[env_idx],
                        instruction=instructions_by_env[env_idx],
                        success=is_success,
                        mode=mode,
                    )
                    frames_by_env[env_idx].clear()
                    instructions_by_env[env_idx] = ""

    def _sample_noise_policy(
        self, states: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # norm_states = self.state_norm.normalize(states, self.device)
        norm_states = states.to(self.device)

        noise, logprob, actor_mean, actor_logstd = self.actor.module.sample(
            norm_states,
            deterministic=deterministic,
            return_stats=True,
        )
        # noise: [B, 1, noise_dim=1600=50*32]
        # logprob: [B,]
        noise = noise[:, 0, :].float()
        entropy = torch.zeros_like(logprob)
        if not deterministic:
            _, entropy = self.actor.module.evaluate_actions(
                norm_states,
                noise,
                average_entropy=True,
            )
        return noise, logprob.float(), entropy.float(), actor_mean.float(), actor_logstd.float()

    def _evaluate_actions_ddp(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        average_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions via DDP wrapper forward so actor gradients synchronize correctly."""
        if actions.dim() == 3:
            actions = actions[:, 0, :]

        distribution = self.actor(features)
        action_dim = float(max(int(actions.shape[-1]), 1))

        log_prob = distribution.log_prob(actions) / action_dim
        entropy = distribution.base_dist.entropy()
        if average_entropy:
            entropy = entropy / action_dim
        return log_prob.float(), entropy.float()

    def _noise_to_policy_noise(self, noise_latent: torch.Tensor) -> torch.Tensor:
        latent = torch.clamp(noise_latent, -1.0 + self.policy_noise_eps, 1.0 - self.policy_noise_eps)
        gaussian_like = torch.atanh(latent)
        scaled = gaussian_like * self.policy_noise_scale + self.policy_noise_bias
        return scaled.reshape(-1, self.chunk_size, self.max_action_dim)

    @staticmethod
    def _flat_to_env_time(tensor: torch.Tensor, num_envs: int) -> torch.Tensor:
        if tensor.ndim < 1:
            raise ValueError(f"Expected tensor with ndim >= 1, got {tensor.ndim}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        if tensor.shape[0] % num_envs != 0:
            raise ValueError(
                f"Leading dimension {tensor.shape[0]} is not divisible by num_envs={num_envs}"
            )

        num_steps = tensor.shape[0] // num_envs
        reshaped = tensor.reshape(num_steps, num_envs, *tensor.shape[1:])
        return reshaped.permute(1, 0, *range(2, reshaped.ndim)).contiguous()

    @staticmethod
    def _env_time_to_flat(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(f"Expected tensor with ndim >= 2, got {tensor.ndim}")
        permuted = tensor.permute(1, 0, *range(2, tensor.ndim)).contiguous()
        return permuted.reshape(-1, *tensor.shape[2:])

    def _collect_rollouts(
        self,
        env,
        rollout_epoch: int,
        chunk_steps: int,
        deterministic: bool,
        save_video: bool,
        mode: str,
    ) -> tuple[list[Transition], dict[str, float]]:
        if mode == "train" and self._train_obs is not None and self._train_dsrl_states is not None:
            obs = self._train_obs
            current_dsrl_states = self._train_dsrl_states
        else:
            obs, _ = env.reset()
            current_dsrl_states = self.generator.extract_dsrl_state_features(obs).float()

        transitions: list[Transition] = []
        video_buffers: dict[str, Any] | None = None
        if save_video:
            video_buffers = self._init_video_buffers(env.num_envs)

        stats = torch.zeros(5, dtype=torch.float64)
        # [return_sum, step_count_running, done_count, success_count, done_length_sum]
        episode_steps_by_env = torch.zeros(env.num_envs, dtype=torch.int64)
        noise_sum = 0.0
        noise_sq_sum = 0.0
        noise_count = 0
        actor_mean_sum = 0.0
        actor_mean_sq_sum = 0.0
        actor_logstd_sum = 0.0
        actor_logstd_sq_sum = 0.0
        actor_stat_count = 0

        total_chunks = rollout_epoch * chunk_steps
        progress_bar = tqdm(
            total=total_chunks,
            desc="rollout[dsrl]",
            disable=self.rank != 0,
            leave=False,
        )

        for _ in range(total_chunks):
            with torch.no_grad():
                dsrl_states = current_dsrl_states
                # dsrl_states: [B, state_dim=960 * 2] (concat last token and mean-pooled)
                # self.state_norm.update(dsrl_states)

                noise_latent, old_logprob, _, actor_mean, actor_logstd = self._sample_noise_policy(
                    dsrl_states.to(self.device),
                    deterministic=deterministic,
                )
                policy_noise = self._noise_to_policy_noise(noise_latent)
                chunk_actions, _ = self.generator.predict_action_batch(
                    obs,
                    external_policy_noise=policy_noise,
                )

            chunk_actions = chunk_actions[:, : self.num_execute_steps]
            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, _ = env.chunk_step(chunk_actions)
            next_obs = obs_list[-1]
            next_dsrl_states = self.generator.extract_dsrl_state_features(next_obs).float()

            chunk_debug_frames: list[list[np.ndarray]] = [[] for _ in range(dsrl_states.shape[0])]
            if self.debug_replay_buffer_returns:
                for step_obs in obs_list:
                    main_images = step_obs.get("main_images")
                    if main_images is None:
                        continue
                    for env_idx in range(dsrl_states.shape[0]):
                        chunk_debug_frames[env_idx].append(self._to_uint8_hwc(main_images[env_idx]))
            # next_dsrl_states: [B, state_dim=960 * 2] (concat last token and mean-pooled)
            # self.state_norm.update(next_dsrl_states)

            if save_video and video_buffers is not None:
                self._append_and_maybe_flush_videos(
                    video_buffers=video_buffers,
                    obs_list=obs_list,
                    chunk_actions=chunk_actions,
                    chunk_terminations=chunk_terminations,
                    chunk_truncations=chunk_truncations,
                    mode=mode,
                )

            chunk_sum_rewards = chunk_rewards.sum(dim=1).float().cpu()
            done_events = torch.logical_or(chunk_terminations, chunk_truncations)
            dones = done_events.any(dim=1).float().cpu()
            success = torch.logical_and(
                chunk_terminations.any(dim=1),
                torch.logical_not(chunk_truncations.any(dim=1)),
            ).float().cpu()

            # Count only actually executed sub-steps in this chunk.
            # With done-in-chunk skipping, trailing skipped steps should not contribute to length stats.
            done_int = done_events.to(dtype=torch.int64)
            first_done_idx = torch.argmax(done_int, dim=1)
            executed_steps = torch.where(
                done_events.any(dim=1),
                first_done_idx + 1,
                torch.full_like(first_done_idx, fill_value=done_events.shape[1]),
            )
            episode_steps_by_env += executed_steps

            done_mask = done_events.any(dim=1)
            done_length_sum = int(episode_steps_by_env[done_mask].sum().item())
            episode_steps_by_env[done_mask] = 0

            stats[0] += float(chunk_sum_rewards.sum().item())
            stats[1] += float(executed_steps.sum().item())
            stats[2] += float(dones.sum().item())
            stats[3] += float(success.sum().item())
            stats[4] += float(done_length_sum)

            noise_sum += float(noise_latent.sum().item())
            noise_sq_sum += float((noise_latent * noise_latent).sum().item())
            noise_count += int(noise_latent.numel())

            actor_mean_sum += float(actor_mean.sum().item())
            actor_mean_sq_sum += float((actor_mean * actor_mean).sum().item())
            actor_logstd_sum += float(actor_logstd.sum().item())
            actor_logstd_sq_sum += float((actor_logstd * actor_logstd).sum().item())
            actor_stat_count += int(actor_mean.numel())

            old_logprob_cpu = old_logprob.detach().cpu().float()
            action_cpu = noise_latent.detach().cpu().float()
            for idx in range(dsrl_states.shape[0]):
                transition = Transition(
                    state=dsrl_states[idx].clone(),
                    next_state=next_dsrl_states[idx].clone(),
                    action=action_cpu[idx].clone(),
                    old_logprob=old_logprob_cpu[idx].clone(),
                    reward=chunk_sum_rewards[idx].clone(),
                    done=dones[idx].clone(),
                    debug_frames=chunk_debug_frames[idx] if self.debug_replay_buffer_returns else None,
                )
                transitions.append(transition)

            current_dsrl_states = next_dsrl_states

            obs = next_obs
            progress_bar.update(1)

        if mode == "train":
            self._train_obs = obs
            self._train_dsrl_states = current_dsrl_states

        stats = stats.to(self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        rollout_aux_stats = torch.tensor(
            [
                noise_sum,
                noise_sq_sum,
                float(noise_count),
                actor_mean_sum,
                actor_mean_sq_sum,
                actor_logstd_sum,
                actor_logstd_sq_sum,
                float(actor_stat_count),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(rollout_aux_stats, op=dist.ReduceOp.SUM)

        noise_sum = float(rollout_aux_stats[0].item())
        noise_sq_sum = float(rollout_aux_stats[1].item())
        noise_count = int(rollout_aux_stats[2].item())
        actor_mean_sum = float(rollout_aux_stats[3].item())
        actor_mean_sq_sum = float(rollout_aux_stats[4].item())
        actor_logstd_sum = float(rollout_aux_stats[5].item())
        actor_logstd_sq_sum = float(rollout_aux_stats[6].item())
        actor_stat_count = int(rollout_aux_stats[7].item())

        done_count = max(float(stats[2].item()), 1.0)
        step_count = max(float(stats[1].item()), 1.0)
        noise_denom = float(max(noise_count, 1))
        actor_stat_denom = float(max(actor_stat_count, 1))
        noise_mean = noise_sum / noise_denom
        noise_var = max(noise_sq_sum / noise_denom - noise_mean * noise_mean, 0.0)
        actor_mean_mean = actor_mean_sum / actor_stat_denom
        actor_mean_var = max(
            actor_mean_sq_sum / actor_stat_denom - actor_mean_mean * actor_mean_mean,
            0.0,
        )
        actor_logstd_mean = actor_logstd_sum / actor_stat_denom
        actor_logstd_var = max(
            actor_logstd_sq_sum / actor_stat_denom
            - actor_logstd_mean * actor_logstd_mean,
            0.0,
        )
        metrics = {
            "return_per_step": float(stats[0].item() / step_count),
            "return_per_traj_running": float(stats[0].item() / done_count),
            "average_length_running": float(stats[1].item() / done_count),
            "average_length": float(stats[4].item() / done_count),
            "success_rate": float(stats[3].item() / done_count),
            "total_trajectories": float(stats[2].item()),
            "success_trajectories": float(stats[3].item()),
            "noise_latent_mean": float(noise_mean),
            "noise_latent_std": float(np.sqrt(noise_var)),
            "actor_mean_mean": float(actor_mean_mean),
            "actor_mean_std": float(np.sqrt(actor_mean_var)),
            "actor_logstd_mean": float(actor_logstd_mean),
            "actor_logstd_std": float(np.sqrt(actor_logstd_var)),
        }
        return transitions, metrics

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rewards.ndim != 1:
            raise ValueError(f"Expected flat rewards shape [N*T], got {tuple(rewards.shape)}")
        if values.shape != rewards.shape or next_values.shape != rewards.shape or dones.shape != rewards.shape:
            raise ValueError(
                "values/next_values/dones must have same shape as rewards, "
                f"got rewards={tuple(rewards.shape)}, values={tuple(values.shape)}, "
                f"next_values={tuple(next_values.shape)}, dones={tuple(dones.shape)}"
            )

        rewards_env_time = self._flat_to_env_time(rewards, self.local_num_envs)
        values_env_time = self._flat_to_env_time(values, self.local_num_envs)
        next_values_env_time = self._flat_to_env_time(next_values, self.local_num_envs)
        dones_env_time = self._flat_to_env_time(dones, self.local_num_envs)

        advantages_env_time = torch.zeros_like(rewards_env_time)
        gae = torch.zeros(
            rewards_env_time.shape[0], device=rewards.device, dtype=rewards.dtype
        )
        for step_idx in reversed(range(rewards_env_time.shape[1])):
            not_done = 1.0 - dones_env_time[:, step_idx]
            delta = (
                rewards_env_time[:, step_idx]
                + self.chunk_gamma * next_values_env_time[:, step_idx] * not_done
                - values_env_time[:, step_idx]
            )
            gae = delta + self.chunk_gamma * self.gae_lambda * not_done * gae
            advantages_env_time[:, step_idx] = gae

        returns_env_time = advantages_env_time + values_env_time
        advantages = self._env_time_to_flat(advantages_env_time)
        returns = self._env_time_to_flat(returns_env_time)
        return advantages, returns

    def _ppo_update(self) -> dict[str, float]:

        # Replay buffer is intentionally rank-local to keep data collection and PPO updates independent per rank.
        effective_count = self.replay_buffer.prepare_gae_targets(
            value_model=self.value_net,
            device=self.device,
            gamma=self.chunk_gamma,
            gae_lambda=self.gae_lambda,
            do_adv_norm=self.do_adv_norm,
            adv_norm_eps=self.adv_norm_eps,
            advantage_clip=self.advantage_clip,
        )
        if effective_count == 0:
            return {}

        metrics_acc = {
            "ppo_loss": 0.0,
            "approx_kl": 0.0,
            "value_loss": 0.0,
            "value_mse": 0.0,
            "ref_value_loss": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "ratio_std": 0.0,
            "log_ratio_mean": 0.0,
            "old_logprob_mean": 0.0,
            "new_logprob_mean": 0.0,
            "clip_fraction": 0.0,
            "advantage_raw_mean": 0.0,
            "advantage_raw_std": 0.0,
            "advantage_raw_abs_mean": 0.0,
            "advantage_raw_min": 0.0,
            "advantage_raw_max": 0.0,
            "advantage_pos_frac": 0.0,
            "advantage_mean": 0.0,
            "advantage_std": 0.0,
            "advantage_abs_mean": 0.0,
            "actor_grad_norm": 0.0,
            "value_grad_norm": 0.0,
            "value_pred_mean": 0.0,
            "value_pred_min": 0.0,
            "value_pred_max": 0.0,
            "value_target_mean": 0.0,
            "value_target_min": 0.0,
            "value_target_max": 0.0,
            "value_target_oob_frac": 0.0,
        }
        first_update_metrics = {
            "ratio_update0": 0.0,
            "ratio_update0_abs_delta_from_1": 0.0,
            "clip_fraction_update0": 0.0,
        }
        if self.dsrl_value_head_type == "distributional":
            metrics_acc["value_ce_loss"] = 0.0
        updates = 0

        # Optional value-only pre-updates before each standard PPO update block.
        for _ in range(self.pre_value_update_epoch):
            batch = self.replay_buffer.sample(self.local_minibatch_size)
            states = torch.stack([transition.state for transition in batch], dim=0).to(self.device)
            returns = torch.tensor(
                [transition.returns for transition in batch],
                dtype=torch.float32,
                device=self.device,
            )

            value_loss = _compute_value_loss(self.value_net, states, returns)
            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
            self.value_optimizer.step()

        for update_idx in range(self.update_epoch):
            batch = self.replay_buffer.sample(self.local_minibatch_size)

            states = torch.stack([transition.state for transition in batch], dim=0).to(self.device)
            actions = torch.stack([transition.action for transition in batch], dim=0).to(self.device)
            old_logprobs = torch.tensor(
                [float(transition.old_logprob.item()) for transition in batch],
                dtype=torch.float32,
                device=self.device,
            )
            advantages = torch.tensor(
                [transition.advantage for transition in batch],
                dtype=torch.float32,
                device=self.device,
            )
            raw_advantages = torch.tensor(
                [transition.advantage_raw for transition in batch],
                dtype=torch.float32,
                device=self.device,
            )
            returns = torch.tensor(
                [transition.returns for transition in batch],
                dtype=torch.float32,
                device=self.device,
            )

            batch_count = int(returns.numel())
            batch_sum = float(returns.detach().sum().item())
            total_count = self._ref_value_count + batch_count
            if total_count > 0:
                self._ref_value_running_mean = (
                    self._ref_value_running_mean * float(self._ref_value_count) + batch_sum
                ) / float(total_count)
                self._ref_value_count = total_count

            norm_states = states.to(self.device)
            new_logprobs, entropy = self._evaluate_actions_ddp(
                norm_states,
                actions,
                average_entropy=True,
            )
            log_ratio = (new_logprobs.float() - old_logprobs.float()).clamp(
                min=-self.max_log_ratio,
                max=self.max_log_ratio,
            )
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                torch.logical_or(ratio > (1.0 + self.ppo_clip), ratio < (1.0 - self.ppo_clip))
                .float()
                .mean()
            )
            if update_idx == 0:
                ratio_update0 = float(ratio.detach().mean().item())
                first_update_metrics["ratio_update0"] = ratio_update0
                first_update_metrics["ratio_update0_abs_delta_from_1"] = abs(ratio_update0 - 1.0)
                first_update_metrics["clip_fraction_update0"] = float(clip_fraction.detach().item())
            policy_loss = - torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            entropy_loss = - self.entropy_coef * entropy.mean()

            value_pred = _predict_values(self.value_net, norm_states)
            value_loss = _compute_value_loss(self.value_net, norm_states, returns)
            value_mse = torch.nn.functional.mse_loss(value_pred.float(), returns.float())
            value_target_oob_frac = (
                torch.logical_or(returns < self.dsrl_value_v_min, returns > self.dsrl_value_v_max)
                .float()
                .mean()
            )
            ref_value_pred = torch.full_like(returns, fill_value=float(self._ref_value_running_mean))
            ref_value_loss = torch.nn.functional.mse_loss(ref_value_pred, returns)

            total_actor_loss = policy_loss + entropy_loss

            actor_grad_norm = 0.0
            if not self.debug_value_only:
                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                actor_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip).item()
                )
                self.actor_optimizer.step()

            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip).item()
            )
            self.value_optimizer.step()

            metrics_acc["ppo_loss"] += float(policy_loss.detach().item())
            metrics_acc["approx_kl"] += float(approx_kl.detach().item())
            metrics_acc["value_loss"] += float(value_loss.detach().item())
            metrics_acc["value_mse"] += float(value_mse.detach().item())
            metrics_acc["ref_value_loss"] += float(ref_value_loss.detach().item())
            metrics_acc["entropy"] += float(entropy.detach().mean().item())
            metrics_acc["ratio"] += float(ratio.detach().mean().item())
            metrics_acc["ratio_std"] += float(ratio.detach().std(unbiased=False).item())
            metrics_acc["log_ratio_mean"] += float(log_ratio.detach().mean().item())
            metrics_acc["old_logprob_mean"] += float(old_logprobs.detach().mean().item())
            metrics_acc["new_logprob_mean"] += float(new_logprobs.detach().mean().item())
            metrics_acc["clip_fraction"] += float(clip_fraction.detach().item())
            metrics_acc["advantage_raw_mean"] += float(raw_advantages.detach().mean().item())
            metrics_acc["advantage_raw_std"] += float(raw_advantages.detach().std(unbiased=False).item())
            metrics_acc["advantage_raw_abs_mean"] += float(raw_advantages.detach().abs().mean().item())
            metrics_acc["advantage_raw_min"] += float(raw_advantages.detach().min().item())
            metrics_acc["advantage_raw_max"] += float(raw_advantages.detach().max().item())
            metrics_acc["advantage_pos_frac"] += float(
                (raw_advantages.detach() > 0.0).float().mean().item()
            )
            metrics_acc["advantage_mean"] += float(advantages.detach().mean().item())
            metrics_acc["advantage_std"] += float(advantages.detach().std(unbiased=False).item())
            metrics_acc["advantage_abs_mean"] += float(advantages.detach().abs().mean().item())
            metrics_acc["actor_grad_norm"] += actor_grad_norm
            metrics_acc["value_grad_norm"] += value_grad_norm
            metrics_acc["value_pred_mean"] += float(value_pred.detach().mean().item())
            metrics_acc["value_pred_min"] += float(value_pred.detach().min().item())
            metrics_acc["value_pred_max"] += float(value_pred.detach().max().item())
            metrics_acc["value_target_mean"] += float(returns.detach().mean().item())
            metrics_acc["value_target_min"] += float(returns.detach().min().item())
            metrics_acc["value_target_max"] += float(returns.detach().max().item())
            metrics_acc["value_target_oob_frac"] += float(value_target_oob_frac.detach().item())
            if self.dsrl_value_head_type == "distributional":
                metrics_acc["value_ce_loss"] += float(value_loss.detach().item())
            updates += 1

        assert updates > 0, "PPO update produced zero optimization steps"
        for key in metrics_acc:
            metrics_acc[key] /= float(updates)
        metrics_acc.update(first_update_metrics)
        return _reduce_mean_dict(metrics_acc, self.device)

    def _evaluate(self) -> dict[str, float]:
        assert self.eval_env is not None
        transitions, rollout_metrics = self._collect_rollouts(
            env=self.eval_env,
            rollout_epoch=self.eval_rollout_epoch,
            chunk_steps=self.eval_chunk_steps,
            deterministic=True,
            save_video=self.save_eval_video,
            mode="eval",
        )
        del transitions
        return {f"eval/{key}": value for key, value in rollout_metrics.items()}

    def run(self) -> None:
        if self.rank == 0 and self.save_rollout_video:
            print(
                (
                    "[noray][dsrl] save_rollout_video is enabled; "
                    f"training rollout videos will be saved to {self.rollout_video_base_dir}"
                ),
                flush=True,
            )

        if self.only_eval:
            if self.eval_env is None:
                raise ValueError("runner.only_eval=True requires eval env to be enabled")
            eval_metrics = self._evaluate()
            if self.rank == 0:
                self.metric_logger.log(eval_metrics, step=0)
                self.metric_logger.finish()
            dist.barrier()
            dist.destroy_process_group()
            return

        for epoch in range(self.max_epochs):
            transitions, rollout_metrics = self._collect_rollouts(
                env=self.env,
                rollout_epoch=self.rollout_epoch,
                chunk_steps=self.chunk_steps,
                deterministic=False,
                save_video=self.save_rollout_video,
                mode="train",
            )

            self.replay_buffer.add_rollout(transitions, gamma=self.chunk_gamma)
            rollout_metrics["recent_success"] = self.replay_buffer.recent_success_rate()
            if self.debug_replay_buffer_returns:
                self.replay_buffer.save_video()

            ppo_metrics = self._ppo_update()

            eval_metrics = {}
            if self.eval_env is not None and self.val_check_interval > 0:
                if (epoch + 1) % self.val_check_interval == 0:
                    eval_metrics = self._evaluate()

            if self.rank == 0:
                metrics_to_log = {
                    "rollout/return_per_step": rollout_metrics["return_per_step"],
                    "rollout/return_per_traj_running": rollout_metrics["return_per_traj_running"],
                    "rollout/average_length_running": rollout_metrics["average_length_running"],
                    "rollout/average_length": rollout_metrics["average_length"],
                    "rollout/success_rate": rollout_metrics["success_rate"],
                    "rollout/total_trajectories": rollout_metrics["total_trajectories"],
                    "rollout/success_trajectories": rollout_metrics["success_trajectories"],
                    "rollout/recent_success": rollout_metrics["recent_success"],
                    "rollout/noise_latent_mean": rollout_metrics["noise_latent_mean"],
                    "rollout/noise_latent_std": rollout_metrics["noise_latent_std"],
                    "rollout/actor_mean_mean": rollout_metrics["actor_mean_mean"],
                    "rollout/actor_mean_std": rollout_metrics["actor_mean_std"],
                    "rollout/actor_logstd_mean": rollout_metrics["actor_logstd_mean"],
                    "rollout/actor_logstd_std": rollout_metrics["actor_logstd_std"],
                    "train/transition_count": float(len(transitions)),
                    "train/replay_size": float(len(self.replay_buffer)),
                }
                metrics_to_log.update({f"train/{key}": value for key, value in ppo_metrics.items()})
                metrics_to_log.update(eval_metrics)
                self.metric_logger.log(metrics_to_log, step=epoch)
                value_metric_label = (
                    "value_ce" if self.dsrl_value_head_type == "distributional" else "value_mse"
                )
                value_metric_value = ppo_metrics.get("value_ce_loss", ppo_metrics["value_loss"])
                print(
                    (
                        f"[noray][dsrl] epoch={epoch} trans={len(transitions)} "
                        f"ppo={ppo_metrics['ppo_loss']:.6f} "
                        f"{value_metric_label}={value_metric_value:.6f} "
                        f"value_mse={ppo_metrics['value_mse']:.6f} "
                        f"target=[{ppo_metrics['value_target_min']:.4f},{ppo_metrics['value_target_max']:.4f}] "
                        f"pred=[{ppo_metrics['value_pred_min']:.4f},{ppo_metrics['value_pred_max']:.4f}] "
                        f"target_oob={ppo_metrics['value_target_oob_frac']:.4f} "
                        f"adv_abs={ppo_metrics['advantage_raw_abs_mean']:.4f} "
                        f"approx_kl={ppo_metrics['approx_kl']:.6f} "
                        f"clip_frac={ppo_metrics['clip_fraction']:.4f} "
                        f"actor_gn={ppo_metrics['actor_grad_norm']:.4f} "
                        f"value_gn={ppo_metrics['value_grad_norm']:.4f} "
                        f"noise_mean={rollout_metrics['noise_latent_mean']:.4f} "
                        f"noise_std={rollout_metrics['noise_latent_std']:.4f} "
                        f"actor_mean={rollout_metrics['actor_mean_mean']:.4f} "
                        f"actor_logstd={rollout_metrics['actor_logstd_mean']:.4f} "
                        f"total_traj={rollout_metrics['total_trajectories']:.0f} "
                        f"success_traj={rollout_metrics['success_trajectories']:.0f} "
                        f"success={rollout_metrics['success_rate']:.4f}"
                    ),
                    flush=True,
                )

        dist.barrier()
        if self.rank == 0:
            self.metric_logger.finish()
        dist.destroy_process_group()
