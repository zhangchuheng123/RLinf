import copy
import os
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import rlinf_noray.algorithms  # noqa: F401
from rlinf_noray.algorithms.registry import policy_loss
from rlinf_noray.envs import get_env_cls
from rlinf_noray.models import get_model
from rlinf_noray.utils.metric_logger import MetricLogger


@dataclass
class RolloutSample:
    forward_inputs: dict[str, Any]
    old_logprobs: torch.Tensor
    prev_values: torch.Tensor
    returns: torch.Tensor


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(value) for value in obj)
    return obj


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(value, device) for value in obj)
    return obj


def _reduce_to_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.float()
    return tensor.reshape(tensor.shape[0], -1).mean(dim=1).float()


class LiberoPPODDPNoRayRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model_type = str(cfg.actor.model.model_type).lower()
        env_type = str(cfg.env.train.env_type).lower()

        assert env_type == "libero", f"Only libero env is supported, got {env_type}"
        assert model_type in {"openpi", "smolvla"}, (
            f"Only openpi/smolvla are supported in noray runner, got {model_type}"
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

        self.val_check_interval = int(cfg.runner.get("val_check_interval", -1))
        self.only_eval = bool(cfg.runner.get("only_eval", False))
        self.eval_env = None
        self.eval_rollout_epoch = int(cfg.algorithm.get("eval_rollout_epoch", 1))
        self.eval_chunk_steps = 0

        if self.val_check_interval > 0 or self.only_eval:
            eval_total_num_envs = int(cfg.env.eval.total_num_envs)
            assert eval_total_num_envs % self.world_size == 0, (
                f"env.eval.total_num_envs={eval_total_num_envs} must be divisible by world_size={self.world_size}"
            )
            eval_local_num_envs = eval_total_num_envs // self.world_size
            eval_env_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.env.eval, resolve=True)
            )
            eval_env_cfg.total_num_envs = eval_local_num_envs
            eval_env_cls = get_env_cls(eval_env_cfg.env_type, eval_env_cfg)
            self.eval_env = eval_env_cls(
                cfg=eval_env_cfg,
                num_envs=eval_local_num_envs,
                seed_offset=self.rank,
                total_num_processes=self.world_size,
                worker_info=None,
            )
            self.eval_chunk_steps = int(
                cfg.env.eval.max_steps_per_rollout_epoch
            ) // int(cfg.actor.model.num_action_chunks)

        model_cfg = copy.deepcopy(cfg.actor.model)
        self.model = get_model(model_cfg)
        self.model.train()

        ddp_kwargs = {}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [self.local_rank]
            ddp_kwargs["output_device"] = self.local_rank
        self.ddp_model = DistributedDataParallel(self.model, **ddp_kwargs)

        trainable_params = [
            parameter for parameter in self.ddp_model.parameters() if parameter.requires_grad
        ]
        assert trainable_params, "No trainable parameters found for DDP model"

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(cfg.actor.optim.lr),
            betas=(float(cfg.actor.optim.adam_beta1), float(cfg.actor.optim.adam_beta2)),
            eps=float(cfg.actor.optim.adam_eps),
            weight_decay=float(cfg.actor.optim.weight_decay),
        )

        self.max_epochs = int(cfg.runner.max_epochs)
        self.rollout_epoch = int(cfg.algorithm.rollout_epoch)
        self.update_epoch = int(cfg.algorithm.update_epoch)
        self.num_execute_steps = int(
            cfg.runner.get("num_execute_steps", cfg.actor.model.num_action_chunks)
        )
        if self.num_execute_steps <= 0:
            raise ValueError(
                f"runner.num_execute_steps must be > 0, got {self.num_execute_steps}"
            )
        if self.num_execute_steps > int(self.model.policy.config.chunk_size):
            raise ValueError(
                "runner.num_execute_steps must be <= self.model.policy.config.chunk_size, "
                f"got {self.num_execute_steps} > {int(self.model.policy.config.chunk_size)}"
            )
        self.chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // self.num_execute_steps
        self.metric_logger = MetricLogger(cfg) if self.rank == 0 else None

        eval_video_cfg = cfg.env.eval.get("video_cfg", {})
        self.save_eval_video = bool(
            cfg.runner.get("save_eval_video", eval_video_cfg.get("save_eval_video", True))
        )
        self.save_rollout_video = bool(cfg.runner.get("save_rollout_video", False))
        self.eval_video_base_dir = str(
            cfg.runner.get("eval_video_base_dir", eval_video_cfg.get("video_base_dir", ""))
        )
        default_rollout_video_base_dir = os.path.join(
            str(cfg.runner.logger.get("log_path", "logs")),
            "video",
            "rollout",
        )
        self.rollout_video_base_dir = str(
            cfg.runner.get("rollout_video_base_dir", default_rollout_video_base_dir)
        )

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
        frame = LiberoPPODDPNoRayRunner._to_uint8_hwc(main_image)
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

    def _save_eval_videos(
        self,
        frames_by_env: list[list[np.ndarray]],
        instructions_by_env: list[str],
        epoch: int,
        mode: str = "eval",
    ) -> None:
        base_dir_str = (
            self.eval_video_base_dir if mode == "eval" else self.rollout_video_base_dir
        )
        if not base_dir_str:
            return
        base_dir = Path(base_dir_str)
        base_dir.mkdir(parents=True, exist_ok=True)

        traj_offset = self.rank * len(frames_by_env)
        for local_idx, frames in enumerate(frames_by_env):
            if not frames:
                continue
            traj_idx = traj_offset + local_idx
            instruction_slug = self._instruction_to_slug(instructions_by_env[local_idx])
            out_path = base_dir / f"{mode}_epoch_{epoch}_traj_{traj_idx:02d}_{instruction_slug}.mp4"
            imageio.mimsave(str(out_path), frames, fps=15)

    def _reduce_sums(self, sums: torch.Tensor) -> torch.Tensor:
        sums = sums.to(self.device, dtype=torch.float64)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        return sums

    def _pack_forward_inputs(self, rollout_result: dict[str, Any]) -> dict[str, Any]:
        if "forward_inputs" in rollout_result:
            return _to_cpu(rollout_result["forward_inputs"])

        required = [
            "states",
            "task_descriptions",
            "main_images",
            "noise",
            "timestep",
            "norm_actions",
        ]
        packed: dict[str, Any] = {}
        for key in required:
            assert key in rollout_result, f"Missing key in rollout result: {key}"
            packed[key] = _to_cpu(rollout_result[key])

        if "wrist_images" in rollout_result:
            packed["wrist_images"] = _to_cpu(rollout_result["wrist_images"])
        if "prev_logprobs" in rollout_result:
            packed["prev_logprobs"] = _to_cpu(rollout_result["prev_logprobs"])
        return packed

    def _collect_rollouts(
        self,
        env,
        rollout_epoch: int,
        chunk_steps: int,
        mode: str,
        collect_samples: bool,
        save_video: bool = False,
        video_epoch: int = 0,
    ) -> tuple[list[RolloutSample], dict[str, float]]:

        align_pickle_path = os.environ.get("RLINF_ROLLOUT_ALIGN_PICKLE_PATH", "")
        if align_pickle_path:
            align_file = Path(align_pickle_path)
            with open(align_file, "rb") as file:
                align_data = pickle.load(file)

        obs, _ = env.reset()
        # obs: dict of 
        #   "main_images": (num_envs, H, W, C) uint8
        #   "wrist_images": (num_envs, H, W, C) uint8
        #   "task_descriptions": list of str
        #   "states": (num_envs, state_dim) float32

        samples: list[RolloutSample] = []
        frames_by_env: list[list[np.ndarray]] = []
        instructions_by_env: list[str] = []
        if save_video:
            frames_by_env = [[] for _ in range(env.num_envs)]
            instructions_by_env = ["" for _ in range(env.num_envs)]

        sums = torch.zeros(6, dtype=torch.float64)
        # [sum_chunk_return, sum_step_reward, done_count, success_count, num_chunks, num_steps]

        total_chunks = rollout_epoch * chunk_steps
        progress_bar = tqdm(
            total=total_chunks,
            desc=f"rollout[{mode}]",
            disable=self.rank != 0,
            leave=False,
        )

        for _ in range(total_chunks):
            if mode == "eval":
                self.ddp_model.module.eval()

            if align_data is not None:
                predict_kwargs = {
                    "external_policy_noise": align_data["policy_noise"],
                    "observation_before_policy": align_data["observation_before_policy"],
                    "action_after_policy": align_data["action_after_policy"],
                    "action_chunk": align_data["action_chunk"],
                    "action_after_postprocessor": align_data["action_after_postprocessor"],
                }
            else:
                predict_kwargs = {}

            with torch.no_grad():
                chunk_actions, rollout_result = \
                    self.ddp_model.module.predict_action_batch(obs, **predict_kwargs)
            chunk_actions = chunk_actions[:, : self.num_execute_steps]

            obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = \
                env.chunk_step(chunk_actions)
            obs = obs_list[-1]

            if save_video:
                for step_idx, step_obs in enumerate(obs_list):
                    main_images = step_obs["main_images"]
                    task_descs = step_obs["task_descriptions"]
                    if main_images is None or task_descs is None:
                        continue
                    step_actions = chunk_actions[:, step_idx]
                    for env_idx in range(env.num_envs):
                        if not instructions_by_env[env_idx]:
                            instructions_by_env[env_idx] = str(task_descs[env_idx])
                        frame = self._render_overlay_frame(
                            main_image=main_images[env_idx],
                            action=step_actions[env_idx],
                        )
                        frames_by_env[env_idx].append(frame)

            chunk_returns = chunk_rewards.sum(dim=1).float().cpu()
            dones = torch.logical_or(
                chunk_terminations[:, -1], chunk_truncations[:, -1]
            ).float().cpu()

            success = chunk_terminations[:, -1].float().cpu()
            if infos_list and isinstance(infos_list[-1], dict):
                episode_info = infos_list[-1].get("episode")
                if isinstance(episode_info, dict) and "success_at_end" in episode_info:
                    success_at_end = episode_info["success_at_end"]
                    if isinstance(success_at_end, torch.Tensor):
                        success = success_at_end.float().cpu()

            sums[0] += float(chunk_returns.sum().item())
            sums[1] += float(chunk_rewards.sum().item())
            sums[2] += float(dones.sum().item())
            sums[3] += float(success.sum().item())
            sums[4] += float(chunk_returns.numel())
            sums[5] += float(chunk_rewards.numel())

            if collect_samples:
                old_logprobs = _reduce_to_batch(_to_cpu(rollout_result["prev_logprobs"]))
                prev_values = _reduce_to_batch(_to_cpu(rollout_result["prev_values"]))
                returns = chunk_returns * (1.0 - dones)

                sample = RolloutSample(
                    forward_inputs=self._pack_forward_inputs(rollout_result),
                    old_logprobs=old_logprobs,
                    prev_values=prev_values,
                    returns=returns,
                )
                samples.append(sample)

            progress_bar.update(1)

        sums = self._reduce_sums(sums)
        num_chunks = max(float(sums[4].item()), 1.0)
        num_steps = max(float(sums[5].item()), 1.0)
        metrics = {
            "chunk_return_mean": float(sums[0].item() / num_chunks),
            "step_reward_mean": float(sums[1].item() / num_steps),
            "done_rate": float(sums[2].item() / num_chunks),
            "success_rate": float(sums[3].item() / num_chunks),
            "num_chunks": float(sums[4].item()),
        }
        if save_video:
            self._save_eval_videos(
                frames_by_env,
                instructions_by_env,
                video_epoch,
                mode=mode,
            )
        return samples, metrics

    def _train_one_epoch(self, samples: list[RolloutSample]) -> dict[str, float]:
        loss_total = 0.0
        value_total = 0.0
        return_total = 0.0
        adv_total = 0.0
        update_count = 0

        for _ in range(self.update_epoch):
            for sample in samples:
                forward_inputs = _to_device(sample.forward_inputs, self.device)
                outputs = self.ddp_model.module.default_forward(
                    forward_inputs=forward_inputs,
                    compute_values=True,
                )

                logprobs = _reduce_to_batch(outputs["logprobs"])
                values = _reduce_to_batch(outputs["values"])
                old_logprobs = sample.old_logprobs.to(self.device).float()
                prev_values = sample.prev_values.to(self.device).float()
                returns = sample.returns.to(self.device).float()
                advantages = returns - prev_values
                loss_mask = torch.ones_like(advantages, dtype=torch.bool)

                loss, _ = policy_loss(
                    loss_type=str(self.cfg.algorithm.loss_type),
                    task_type="embodied",
                    logprobs=logprobs.float(),
                    old_logprobs=old_logprobs.float(),
                    advantages=advantages.float(),
                    values=values.float(),
                    returns=returns.float(),
                    prev_values=prev_values.float(),
                    clip_ratio_low=float(self.cfg.algorithm.clip_ratio_low),
                    clip_ratio_high=float(self.cfg.algorithm.clip_ratio_high),
                    clip_ratio_c=float(self.cfg.algorithm.get("clip_ratio_c", 3.0)),
                    value_clip=self.cfg.algorithm.get("value_clip", None),
                    huber_delta=self.cfg.algorithm.get("huber_delta", None),
                    loss_mask=loss_mask,
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_model.parameters(), float(self.cfg.actor.optim.clip_grad)
                )
                self.optimizer.step()

                loss_total += float(loss.detach().item())
                value_total += float(values.detach().mean().item())
                return_total += float(returns.detach().mean().item())
                adv_total += float(advantages.detach().mean().item())
                update_count += 1

        assert update_count > 0, "No optimization steps were executed"
        sums = torch.tensor(
            [loss_total, value_total, return_total, adv_total, float(update_count)],
            dtype=torch.float64,
        )
        sums = self._reduce_sums(sums)
        global_updates = max(float(sums[4].item()), 1.0)
        return {
            "avg_loss": float(sums[0].item() / global_updates),
            "value_mean": float(sums[1].item() / global_updates),
            "return_mean": float(sums[2].item() / global_updates),
            "adv_mean": float(sums[3].item() / global_updates),
        }

    def _evaluate(self, epoch: int = 0) -> dict[str, float]:
        assert self.eval_env is not None
        self.ddp_model.module.eval()
        with torch.no_grad():
            _, eval_metrics = self._collect_rollouts(
                env=self.eval_env,
                rollout_epoch=self.eval_rollout_epoch,
                chunk_steps=self.eval_chunk_steps,
                mode="eval",
                collect_samples=False,
                save_video=self.save_eval_video,
                video_epoch=epoch,
            )
        self.ddp_model.module.train()
        return {f"eval/{k}": v for k, v in eval_metrics.items()}

    def run(self) -> None:
        if self.rank == 0 and self.save_rollout_video:
            print(
                (
                    "[noray][ddp] save_rollout_video is enabled; "
                    f"training rollout videos will be saved to {self.rollout_video_base_dir}"
                ),
                flush=True,
            )

        if self.rank == 0 and self.val_check_interval <= 0 and not self.only_eval:
            print(
                "[noray][ddp] eval disabled because runner.val_check_interval <= 0",
                flush=True,
            )

        if self.only_eval:
            if self.eval_env is None:
                raise ValueError("runner.only_eval=True requires eval env to be enabled")
            eval_metrics = self._evaluate(epoch=0)
            if self.rank == 0:
                self.metric_logger.log(eval_metrics, step=0)
                self.metric_logger.finish()
            dist.barrier()
            dist.destroy_process_group()
            return

        for epoch in range(self.max_epochs):
            samples, rollout_metrics = self._collect_rollouts(
                env=self.env,
                rollout_epoch=self.rollout_epoch,
                chunk_steps=self.chunk_steps,
                mode="train",
                collect_samples=True,
                save_video=self.save_rollout_video,
                video_epoch=epoch,
            )

            print(rollout_metrics)
            raise SystemExit("Debug exit after first rollout collection")

            train_metrics = self._train_one_epoch(samples)

            eval_metrics = {}
            if self.eval_env is not None and self.val_check_interval > 0:
                if (epoch + 1) % self.val_check_interval == 0:
                    eval_metrics = self._evaluate(epoch=epoch)

            if self.rank == 0:
                metrics_to_log = {
                    "train/avg_loss": train_metrics["avg_loss"],
                    "train/value_mean": train_metrics["value_mean"],
                    "train/return_mean": train_metrics["return_mean"],
                    "train/adv_mean": train_metrics["adv_mean"],
                    "rollout/chunk_return_mean": rollout_metrics["chunk_return_mean"],
                    "rollout/step_reward_mean": rollout_metrics["step_reward_mean"],
                    "rollout/done_rate": rollout_metrics["done_rate"],
                    "rollout/success_rate": rollout_metrics["success_rate"],
                    "rollout/num_chunks": rollout_metrics["num_chunks"],
                    "train/sample_count": len(samples),
                }
                metrics_to_log.update(eval_metrics)
                self.metric_logger.log(
                    metrics_to_log,
                    step=epoch,
                )
                print(
                    (
                        f"[noray][ddp] epoch={epoch} samples={len(samples)} "
                        f"avg_loss={train_metrics['avg_loss']:.6f} "
                        f"rollout_success={rollout_metrics['success_rate']:.4f}"
                    ),
                    flush=True,
                )

        dist.barrier()
        if self.rank == 0:
            self.metric_logger.finish()
        dist.destroy_process_group()
