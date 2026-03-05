import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import policy_loss
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import get_model


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
        self.chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // int(
            cfg.actor.model.num_action_chunks
        )

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

    def _collect_rollouts(self) -> list[RolloutSample]:
        obs, _ = self.env.reset()
        samples: list[RolloutSample] = []

        for _ in range(self.rollout_epoch):
            for _ in range(self.chunk_steps):
                with torch.no_grad():
                    raw_actions, rollout_result = self.ddp_model.module.predict_action_batch(
                        env_obs=obs,
                        mode="train",
                    )

                chunk_actions = torch.as_tensor(raw_actions)
                chunk_actions = prepare_actions(
                    raw_chunk_actions=chunk_actions,
                    env_type=self.cfg.env.train.env_type,
                    model_type=self.cfg.actor.model.model_type,
                    num_action_chunks=self.cfg.actor.model.num_action_chunks,
                    action_dim=self.cfg.actor.model.action_dim,
                    policy=self.cfg.actor.model.get("policy_setup", None),
                    wm_env_type=self.cfg.env.train.get("wm_env_type", None),
                )

                if isinstance(chunk_actions, np.ndarray):
                    chunk_actions = torch.from_numpy(chunk_actions)

                obs_list, chunk_rewards, chunk_terminations, chunk_truncations, _ = self.env.chunk_step(
                    chunk_actions
                )
                obs = obs_list[-1]

                rewards = chunk_rewards.sum(dim=1).float().cpu()
                dones = torch.logical_or(
                    chunk_terminations[:, -1], chunk_truncations[:, -1]
                ).float().cpu()

                old_logprobs = _reduce_to_batch(_to_cpu(rollout_result["prev_logprobs"]))
                prev_values = _reduce_to_batch(_to_cpu(rollout_result["prev_values"]))
                returns = rewards * (1.0 - dones)

                sample = RolloutSample(
                    forward_inputs=self._pack_forward_inputs(rollout_result),
                    old_logprobs=old_logprobs,
                    prev_values=prev_values,
                    returns=returns,
                )
                samples.append(sample)

        return samples

    def _train_one_epoch(self, samples: list[RolloutSample]) -> float:
        loss_total = 0.0
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
                update_count += 1

        assert update_count > 0, "No optimization steps were executed"
        avg_loss = loss_total / update_count
        loss_tensor = torch.tensor([avg_loss], dtype=torch.float32, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_all = float((loss_tensor / self.world_size).item())
        return avg_loss_all

    def run(self) -> None:
        for epoch in range(self.max_epochs):
            samples = self._collect_rollouts()
            avg_loss_all = self._train_one_epoch(samples)
            if self.rank == 0:
                print(
                    f"[noray][ddp] epoch={epoch} samples={len(samples)} avg_loss={avg_loss_all:.6f}",
                    flush=True,
                )

        dist.barrier()
        dist.destroy_process_group()
