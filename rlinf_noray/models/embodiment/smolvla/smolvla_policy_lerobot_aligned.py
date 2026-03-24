from __future__ import annotations

from typing import Any

import torch

from rlinf_noray.integrations.lerobot_local_import import ensure_local_lerobot
from rlinf_noray.integrations.lerobot_pipeline_bridge import (
    build_lerobot_pre_post_processors,
    run_lerobot_action_postprocess,
    run_lerobot_inference_preprocess,
)
from rlinf_noray.models.embodiment.smolvla.smolvla_policy import (
    SmolVLAForRLActionPrediction as BaseSmolVLAForRLActionPrediction,
)

ensure_local_lerobot()


class SmolVLAForRLActionPrediction(BaseSmolVLAForRLActionPrediction):
    def __init__(self, cfg, policy=None) -> None:
        super().__init__(cfg=cfg, policy=policy)
        (
            self.policy_preprocessor,
            self.policy_postprocessor,
            self.env_preprocessor,
            self.env_postprocessor,
        ) = build_lerobot_pre_post_processors(self.policy, cfg.model_path)

    def _preprocess_obs_batch(self, env_obs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        batch_obs = run_lerobot_inference_preprocess(
            env_obs=env_obs,
            env_preprocessor=self.env_preprocessor,
            preprocessor=self.policy_preprocessor,
        )
        policy_dtype = next(self.policy.parameters()).dtype
        state_key = "observation.state"
        if state_key in batch_obs and isinstance(batch_obs[state_key], torch.Tensor):
            if batch_obs[state_key].dtype != policy_dtype:
                batch_obs[state_key] = batch_obs[state_key].to(dtype=policy_dtype)
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch_obs.items()
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        **kwargs,
    ):
        device = next(self.policy.parameters()).device
        batch_obs = self._preprocess_obs_batch(env_obs, device=device)

        self.policy.reset()

        batch_size = batch_obs["observation.state"].shape[0]
        policy_dtype = next(self.policy.parameters()).dtype

        chunk_size = int(self.policy.config.chunk_size)
        max_action_dim = int(self.policy.config.max_action_dim)
        policy_noise = torch.randn(
            batch_size,
            chunk_size,
            max_action_dim,
            device=device,
            dtype=policy_dtype,
        )

        if "external_policy_noise" in kwargs:
            policy_noise = kwargs["external_policy_noise"]

        action_norm, action_chunk = self.policy.select_action(
            batch_obs,
            noise=policy_noise,
            return_chunk=True,
        )

        action_raw = run_lerobot_action_postprocess(
            action=action_chunk,
            postprocessor=self.policy_postprocessor,
            env_postprocessor=self.env_postprocessor,
        )

        noise = policy_noise[..., : action_norm.shape[-1]]
        timestep = torch.rand(batch_size, device=device)
        prev_logprobs = self._compute_flow_logprob(action_chunk, noise, timestep)

        states = env_obs["states"]
        prev_values = self._compute_value(states, device)

        result: dict[str, Any] = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "noise": noise.detach().cpu(),
            "policy_noise": policy_noise.detach().cpu(),
            "timestep": timestep.detach().cpu(),
            "norm_actions": action_chunk.detach().cpu(),
            "states": states.cpu() if isinstance(states, torch.Tensor) else torch.as_tensor(states),
            "main_images": env_obs[self.main_image_env_key].cpu(),
            "wrist_images": (
                env_obs[self.wrist_image_env_key].cpu()
                if (
                    self.wrist_image_env_key
                    and self.wrist_image_env_key in env_obs
                    and env_obs[self.wrist_image_env_key] is not None
                )
                else None
            ),
            "task_descriptions": list(env_obs["task_descriptions"]),
        }

        return action_raw.detach().cpu(), result
