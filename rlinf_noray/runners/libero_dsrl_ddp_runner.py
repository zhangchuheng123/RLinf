import copy
import json
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


@dataclass
class Transition:
    state: torch.Tensor
    next_state: torch.Tensor
    noise: torch.Tensor
    old_logprob: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int) -> list[Transition]:
        assert len(self._buffer) >= batch_size, (
            f"Replay buffer too small: {len(self._buffer)} < {batch_size}"
        )
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(index)] for index in indices]


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


class DSRLValueNet(torch.nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class DSRLQNet(torch.nn.Module):
    def __init__(self, state_dim: int, noise_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + noise_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, noises], dim=-1)
        return self.net(x).squeeze(-1)


def _to_tensor_states(states: Any) -> torch.Tensor:
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().float()
    return torch.as_tensor(states, dtype=torch.float32)


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
        self.num_execute_steps = int(cfg.runner.num_execute_steps)
        if self.num_execute_steps <= 0:
            raise ValueError(
                f"runner.num_execute_steps must be > 0, got {self.num_execute_steps}"
            )
        self.chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // self.num_execute_steps

        self.debug_dist = os.environ.get("DEBUG_DIST", "") not in {"", "0", "false", "False", "FALSE"}
        self.debug_normal = os.environ.get("DEBUG_NORMAL", "") not in {"", "0", "false", "False", "FALSE"}
        self._debug_noise_test_done = False

        self.chunk_size = int(self.generator.policy.config.chunk_size)
        self.max_action_dim = int(self.generator.policy.config.max_action_dim)
        self.dsrl_noise_dim = self.chunk_size * self.max_action_dim
        self.dsrl_hidden_dim = int(cfg.actor.model.dsrl_hidden_dim)
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
            hidden_dims=(self.dsrl_hidden_dim, self.dsrl_hidden_dim, self.dsrl_hidden_dim),
            action_horizon=1,
        )
        self.value_net = DSRLValueNet(self.state_dim, hidden_dim=self.dsrl_hidden_dim)
        self.q_net = DSRLQNet(self.state_dim, self.dsrl_noise_dim, hidden_dim=self.dsrl_hidden_dim)
        self.q_target = DSRLQNet(self.state_dim, self.dsrl_noise_dim, hidden_dim=self.dsrl_hidden_dim)
        self.q_target.load_state_dict(self.q_net.state_dict())

        self.actor.to(self.device)
        self.value_net.to(self.device)
        self.q_net.to(self.device)
        self.q_target.to(self.device)
        self.q_target.eval()

        ddp_kwargs = {}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [self.local_rank]
            ddp_kwargs["output_device"] = self.local_rank
        self.actor = DistributedDataParallel(self.actor, **ddp_kwargs)
        self.value_net = DistributedDataParallel(self.value_net, **ddp_kwargs)
        self.q_net = DistributedDataParallel(self.q_net, **ddp_kwargs)

        actor_lr = float(cfg.actor.optim.dsrl_actor_lr)
        value_lr = float(cfg.actor.optim.dsrl_value_lr)
        q_lr = float(cfg.actor.optim.dsrl_q_lr)

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
        self.q_optimizer = torch.optim.AdamW(
            self.q_net.parameters(),
            lr=q_lr,
            betas=(float(cfg.actor.optim.adam_beta1), float(cfg.actor.optim.adam_beta2)),
            eps=float(cfg.actor.optim.adam_eps),
            weight_decay=float(cfg.actor.optim.weight_decay),
        )

        self.gamma = float(cfg.algorithm.gamma)
        self.gae_lambda = float(cfg.algorithm.gae_lambda)
        self.ppo_clip = float(cfg.algorithm.dsrl_ppo_clip)
        self.max_log_ratio = float(cfg.algorithm.dsrl_max_log_ratio)
        self.entropy_coef = float(cfg.algorithm.dsrl_entropy_coef)
        self.q_tau = float(cfg.algorithm.dsrl_q_tau)
        self.grad_clip = float(cfg.actor.optim.clip_grad)
        self.replay_batch_size = int(cfg.algorithm.dsrl_replay_batch_size)
        self.replay_updates = int(cfg.algorithm.dsrl_q_updates_per_epoch)
        self.replay_min_size = int(cfg.algorithm.dsrl_replay_min_size)
        replay_capacity = int(cfg.algorithm.dsrl_replay_buffer_size)

        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.state_norm = RunningNorm(self.state_dim)

        self.metric_logger = MetricLogger(cfg) if self.rank == 0 else None

        self.save_eval_video = bool(cfg.runner.save_eval_video)
        self.save_rollout_video = bool(cfg.runner.save_rollout_video)
        self.eval_video_base_dir = str(cfg.runner.eval_video_base_dir)
        self.rollout_video_base_dir = str(cfg.runner.rollout_video_base_dir)
        self._video_traj_counter = self.rank * 1_000_000

        if self.rank == 0:
            init_summary = {
                "rollout_epoch": self.rollout_epoch,
                "update_epoch": self.update_epoch,
                "num_execute_steps": self.num_execute_steps,
                "chunk_steps": self.chunk_steps,
                "state_dim": self.state_dim,
                "dsrl_noise_dim": self.dsrl_noise_dim,
                "dsrl_hidden_dim": self.dsrl_hidden_dim,
                "actor_lr": actor_lr,
                "value_lr": value_lr,
                "q_lr": q_lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "ppo_clip": self.ppo_clip,
                "max_log_ratio": self.max_log_ratio,
                "entropy_coef": self.entropy_coef,
                "q_tau": self.q_tau,
                "grad_clip": self.grad_clip,
                "replay_batch_size": self.replay_batch_size,
                "replay_updates": self.replay_updates,
                "replay_min_size": self.replay_min_size,
                "replay_capacity": replay_capacity,
                "policy_noise_eps": self.policy_noise_eps,
                "policy_noise_scale": self.policy_noise_scale,
                "policy_noise_bias": self.policy_noise_bias,
                "debug_dist": self.debug_dist,
                "debug_normal": self.debug_normal,
                "save_eval_video": self.save_eval_video,
                "save_rollout_video": self.save_rollout_video,
                "eval_video_base_dir": self.eval_video_base_dir,
                "rollout_video_base_dir": self.rollout_video_base_dir,
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
                        f"[noray][ddp] Flushing video for env_idx={env_idx} terminated={terminated} truncated={truncated}",
                    )
                    self._save_single_video(
                        frames=frames_by_env[env_idx],
                        instruction=instructions_by_env[env_idx],
                        success=is_success,
                        mode=mode,
                    )
                    frames_by_env[env_idx].clear()
                    instructions_by_env[env_idx] = ""

    #### TEST: DEBUG-only test for initial policy-noise distribution closeness.
    def _maybe_debug_test_noise_distribution(self, policy_noise: torch.Tensor) -> None:
        if not self.debug_dist or self._debug_noise_test_done:
            return

        with torch.no_grad():
            ref_noise = torch.randn_like(policy_noise)
            current = policy_noise.float().reshape(-1)
            ref = ref_noise.float().reshape(-1)

            current_mean = float(current.mean().item())
            current_std = float(current.std(unbiased=False).item())
            current_p01 = float(torch.quantile(current, 0.01).item())
            current_p99 = float(torch.quantile(current, 0.99).item())

            ref_mean = float(ref.mean().item())
            ref_std = float(ref.std(unbiased=False).item())
            ref_p01 = float(torch.quantile(ref, 0.01).item())
            ref_p99 = float(torch.quantile(ref, 0.99).item())

            diff_mean = abs(current_mean - ref_mean)
            diff_std = abs(current_std - ref_std)
            diff_p01 = abs(current_p01 - ref_p01)
            diff_p99 = abs(current_p99 - ref_p99)

            print(
                (
                    "[noray][dsrl][DEBUG][noise_test] "
                    f"current(mean={current_mean:.6f},std={current_std:.6f},p01={current_p01:.6f},p99={current_p99:.6f}) "
                    f"ref(mean={ref_mean:.6f},std={ref_std:.6f},p01={ref_p01:.6f},p99={ref_p99:.6f}) "
                    f"abs_diff(mean={diff_mean:.6f},std={diff_std:.6f},p01={diff_p01:.6f},p99={diff_p99:.6f})"
                ),
                flush=True,
            )

        self._debug_noise_test_done = True
    #### TEST END

    def _sample_noise_policy(
        self, states: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_states = self.state_norm.normalize(states, self.device)
        noise, logprob = self.actor.module.sample(norm_states, deterministic=deterministic)
        noise = noise[:, 0, :].float()
        entropy = torch.zeros_like(logprob)
        if not deterministic:
            _, entropy = self.actor.module.evaluate_actions(norm_states, noise)
        return noise, logprob.float(), entropy.float()

    def _noise_to_policy_noise(self, noise_latent: torch.Tensor) -> torch.Tensor:
        latent = torch.clamp(noise_latent, -1.0 + self.policy_noise_eps, 1.0 - self.policy_noise_eps)
        gaussian_like = torch.atanh(latent)
        scaled = gaussian_like * self.policy_noise_scale + self.policy_noise_bias
        return scaled.reshape(-1, self.chunk_size, self.max_action_dim)

    def _collect_rollouts(
        self,
        env,
        rollout_epoch: int,
        chunk_steps: int,
        deterministic: bool,
        collect_replay: bool,
        save_video: bool,
        mode: str,
    ) -> tuple[list[Transition], dict[str, float]]:
        obs, _ = env.reset()
        transitions: list[Transition] = []
        video_buffers: dict[str, Any] | None = None
        if save_video:
            video_buffers = self._init_video_buffers(env.num_envs)

        sums = torch.zeros(4, dtype=torch.float64)
        # [return_sum, step_count, done_count, success_count]
        noise_sums = torch.zeros(2, dtype=torch.float64)
        noise_count = 0.0

        total_chunks = rollout_epoch * chunk_steps
        progress_bar = tqdm(
            total=total_chunks,
            desc="rollout[dsrl]",
            disable=self.rank != 0,
            leave=False,
        )

        for _ in range(total_chunks):
            with torch.no_grad():
                dsrl_states = self.generator.extract_dsrl_state_features(obs).float()
                if dsrl_states.ndim != 2:
                    raise ValueError(f"Expected dsrl_state_features [B, D], got shape {tuple(dsrl_states.shape)}")
                if dsrl_states.shape[-1] != self.state_dim:
                    raise ValueError(
                        f"DSRL state dim mismatch: got {dsrl_states.shape[-1]}, expected {self.state_dim}"
                    )
                self.state_norm.update(dsrl_states)
                states_device = dsrl_states.to(self.device)

                noise_latent, old_logprob, _ = self._sample_noise_policy(
                    states_device,
                    deterministic=deterministic,
                )
                policy_noise = self._noise_to_policy_noise(noise_latent)

                if self.debug_normal and not deterministic:
                    policy_noise = torch.randn_like(policy_noise)

                #### TEST: compare initial policy-generated noise vs standard Gaussian noise.
                if self.debug_dist and not deterministic:
                    self._maybe_debug_test_noise_distribution(policy_noise)
                #### TEST END

                if not deterministic:
                    noise_sums[0] += float(policy_noise.mean().item())
                    noise_sums[1] += float(policy_noise.std(unbiased=False).item())
                    noise_count += 1.0

                chunk_actions, _ = self.generator.predict_action_batch(
                    obs,
                    external_policy_noise=policy_noise,
                )

            chunk_actions = chunk_actions[:, : self.num_execute_steps]
            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, _ = env.chunk_step(chunk_actions)
            next_obs = obs_list[-1]

            if save_video and video_buffers is not None:
                self._append_and_maybe_flush_videos(
                    video_buffers=video_buffers,
                    obs_list=obs_list,
                    chunk_actions=chunk_actions,
                    chunk_terminations=chunk_terminations,
                    chunk_truncations=chunk_truncations,
                    mode=mode,
                )

            chunk_returns = chunk_rewards.sum(dim=1).float().cpu()
            dones = torch.logical_or(chunk_terminations, chunk_truncations).any(dim=1).float().cpu()
            success = torch.logical_and(
                chunk_terminations.any(dim=1),
                torch.logical_not(chunk_truncations.any(dim=1)),
            ).float().cpu()

            sums[0] += float(chunk_returns.sum().item())
            sums[1] += float(chunk_rewards.numel())
            sums[2] += float(dones.sum().item())
            sums[3] += float(success.sum().item())

            if collect_replay:
                next_dsrl_states = self.generator.extract_dsrl_state_features(next_obs).float()
                if next_dsrl_states.ndim != 2:
                    raise ValueError(
                        f"Expected next dsrl_state_features [B, D], got shape {tuple(next_dsrl_states.shape)}"
                    )
                if next_dsrl_states.shape[-1] != self.state_dim:
                    raise ValueError(
                        f"Next DSRL state dim mismatch: got {next_dsrl_states.shape[-1]}, expected {self.state_dim}"
                    )
                self.state_norm.update(next_dsrl_states)
                old_logprob_cpu = old_logprob.detach().cpu().float()
                noise_cpu = noise_latent.detach().cpu().float()
                for idx in range(dsrl_states.shape[0]):
                    transition = Transition(
                        state=dsrl_states[idx].clone(),
                        next_state=next_dsrl_states[idx].clone(),
                        noise=noise_cpu[idx].clone(),
                        old_logprob=old_logprob_cpu[idx].clone(),
                        reward=chunk_returns[idx].clone(),
                        done=dones[idx].clone(),
                    )
                    transitions.append(transition)
                    self.replay_buffer.add(transition)

            obs = next_obs
            progress_bar.update(1)

        sums = sums.to(self.device)
        noise_sums = noise_sums.to(self.device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(noise_sums, op=dist.ReduceOp.SUM)
        noise_count_tensor = torch.tensor([noise_count], dtype=torch.float64, device=self.device)
        dist.all_reduce(noise_count_tensor, op=dist.ReduceOp.SUM)

        done_count = max(float(sums[2].item()), 1.0)
        step_count = max(float(sums[1].item()), 1.0)
        global_noise_count = max(float(noise_count_tensor.item()), 1.0)
        metrics = {
            "return_per_step": float(sums[0].item() / step_count),
            "return_per_traj_running": float(sums[0].item() / done_count),
            "average_length_running": float(sums[1].item() / done_count),
            "success_rate": float(sums[3].item() / done_count),
            "done_count": float(sums[2].item()),
            "total_trajectories": float(sums[2].item()),
            "success_trajectories": float(sums[3].item()),
            "policy_noise_mean": float(noise_sums[0].item() / global_noise_count),
            "policy_noise_std": float(noise_sums[1].item() / global_noise_count),
        }
        return transitions, metrics

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device)
        for idx in reversed(range(rewards.shape[0])):
            not_done = 1.0 - dones[idx]
            delta = rewards[idx] + self.gamma * next_values[idx] * not_done - values[idx]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[idx] = gae
        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, transitions: list[Transition]) -> dict[str, float]:
        if not transitions:
            return {}

        states = torch.stack([transition.state for transition in transitions], dim=0).to(self.device)
        noises = torch.stack([transition.noise for transition in transitions], dim=0).to(self.device)
        rewards = torch.stack([transition.reward for transition in transitions], dim=0).to(self.device)
        dones = torch.stack([transition.done for transition in transitions], dim=0).to(self.device)
        next_states = torch.stack([transition.next_state for transition in transitions], dim=0).to(self.device)

        with torch.no_grad():
            norm_states = self.state_norm.normalize(states, self.device)
            norm_next_states = self.state_norm.normalize(next_states, self.device)
            old_logprobs, _ = self.actor.module.evaluate_actions(norm_states, noises)
            values = self.value_net(norm_states).float()
            next_values = self.value_net(norm_next_states).float()
            advantages, returns = self._compute_gae(rewards, values, next_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        metrics_acc = {
            "ppo_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
        }
        updates = 0

        for _ in range(self.update_epoch):
            norm_states = self.state_norm.normalize(states, self.device)
            new_logprobs, entropy = self.actor.module.evaluate_actions(norm_states, noises)
            log_ratio = (new_logprobs.float() - old_logprobs.float()).clamp(
                min=-self.max_log_ratio,
                max=self.max_log_ratio,
            )
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            entropy_loss = -self.entropy_coef * entropy.mean()

            value_pred = self.value_net(norm_states).float()
            value_loss = torch.nn.functional.mse_loss(value_pred, returns)

            total_actor_loss = policy_loss + entropy_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.grad_clip,
            )
            self.actor_optimizer.step()

            self.value_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
            self.value_optimizer.step()

            metrics_acc["ppo_loss"] += float(policy_loss.detach().item())
            metrics_acc["value_loss"] += float(value_loss.detach().item())
            metrics_acc["entropy"] += float(entropy.detach().mean().item())
            metrics_acc["ratio"] += float(ratio.detach().mean().item())
            updates += 1

        assert updates > 0, "PPO update produced zero optimization steps"
        for key in metrics_acc:
            metrics_acc[key] /= float(updates)
        return _reduce_mean_dict(metrics_acc, self.device)

    def _q_update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.replay_min_size:
            return {"q_loss": 0.0, "q_updates": 0.0}

        q_loss_total = 0.0
        updates = 0
        for _ in range(self.replay_updates):
            batch = self.replay_buffer.sample(self.replay_batch_size)
            states = torch.stack([transition.state for transition in batch], dim=0).to(self.device)
            noises = torch.stack([transition.noise for transition in batch], dim=0).to(self.device)
            rewards = torch.stack([transition.reward for transition in batch], dim=0).to(self.device)
            dones = torch.stack([transition.done for transition in batch], dim=0).to(self.device)
            next_states = torch.stack([transition.next_state for transition in batch], dim=0).to(self.device)

            norm_states = self.state_norm.normalize(states, self.device)
            norm_next_states = self.state_norm.normalize(next_states, self.device)

            q_pred = self.q_net(norm_states, noises).float()
            with torch.no_grad():
                next_noise, _, _ = self._sample_noise_policy(
                    next_states,
                    deterministic=False,
                )
                q_target_next = self.q_target(norm_next_states, next_noise).float()
                q_target = rewards + self.gamma * (1.0 - dones) * q_target_next

            q_loss = torch.nn.functional.mse_loss(q_pred, q_target)
            self.q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
            self.q_optimizer.step()

            with torch.no_grad():
                for target_param, param in zip(self.q_target.parameters(), self.q_net.module.parameters(), strict=True):
                    target_param.data.mul_(1.0 - self.q_tau).add_(self.q_tau * param.data)

            q_loss_total += float(q_loss.detach().item())
            updates += 1

        metrics = {
            "q_loss": q_loss_total / float(max(updates, 1)),
            "q_updates": float(updates),
        }
        return _reduce_mean_dict(metrics, self.device)

    def _evaluate(self) -> dict[str, float]:
        assert self.eval_env is not None
        transitions, rollout_metrics = self._collect_rollouts(
            env=self.eval_env,
            rollout_epoch=self.eval_rollout_epoch,
            chunk_steps=self.eval_chunk_steps,
            deterministic=True,
            collect_replay=False,
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
                collect_replay=True,
                save_video=self.save_rollout_video,
                mode="train",
            )

            ppo_metrics = self._ppo_update(transitions)
            q_metrics = self._q_update()

            eval_metrics = {}
            if self.eval_env is not None and self.val_check_interval > 0:
                if (epoch + 1) % self.val_check_interval == 0:
                    eval_metrics = self._evaluate()

            if self.rank == 0:
                metrics_to_log = {
                    "rollout/return_per_step": rollout_metrics["return_per_step"],
                    "rollout/return_per_traj_running": rollout_metrics["return_per_traj_running"],
                    "rollout/average_length_running": rollout_metrics["average_length_running"],
                    "rollout/success_rate": rollout_metrics["success_rate"],
                    "rollout/total_trajectories": rollout_metrics["total_trajectories"],
                    "rollout/success_trajectories": rollout_metrics["success_trajectories"],
                    "rollout/policy_noise_mean": rollout_metrics["policy_noise_mean"],
                    "rollout/policy_noise_std": rollout_metrics["policy_noise_std"],
                    "train/transition_count": float(len(transitions)),
                    "train/replay_size": float(len(self.replay_buffer)),
                }
                metrics_to_log.update({f"train/{key}": value for key, value in ppo_metrics.items()})
                metrics_to_log.update({f"train/{key}": value for key, value in q_metrics.items()})
                metrics_to_log.update(eval_metrics)
                self.metric_logger.log(metrics_to_log, step=epoch)
                print(
                    (
                        f"[noray][dsrl] epoch={epoch} trans={len(transitions)} "
                        f"ppo={ppo_metrics['ppo_loss']:.6f} "
                        f"q={q_metrics['q_loss']:.6f} "
                        f"noise_mean={rollout_metrics['policy_noise_mean']:.4f} "
                        f"noise_std={rollout_metrics['policy_noise_std']:.4f} "
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
