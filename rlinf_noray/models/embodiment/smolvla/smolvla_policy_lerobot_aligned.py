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

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


class SmolVLAForRLActionPrediction(BaseSmolVLAForRLActionPrediction):
    def __init__(self, cfg, policy=None) -> None:
        super().__init__(cfg=cfg, policy=policy)
        (
            self.policy_preprocessor,
            self.policy_postprocessor,
            self.env_preprocessor,
            self.env_postprocessor,
        ) = build_lerobot_pre_post_processors(self.policy, cfg.model_path)

    def get_dsrl_state_dim(self) -> int:
        hidden_size = int(self.policy.model.vlm_with_expert.config.text_config.hidden_size)
        return hidden_size * 2

    @torch.no_grad()
    def _extract_dsrl_state_features_from_batch(self, batch_obs: dict[str, Any]) -> torch.Tensor:
        
        images, img_masks = self.policy.prepare_images(batch_obs)
        state = self.policy.prepare_state(batch_obs)
        lang_tokens = batch_obs["observation.language.tokens"]
        lang_masks = batch_obs["observation.language.attention_mask"]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state=state,
        )
        # prefix_embs: [B, L=142, D=960]
        # prefix_pad_masks: [B, L=142]
        # prefix_att_masks: [B, L=142]

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # prefix_att_2d_masks: [B, L, L]
        # prefix_position_ids: [B, L]

        outputs_embeds, past_key_values = self.policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.policy.config.use_cache,
            fill_kv_cache=True,
        )

        # outputs_embeds[0]: [B, L, D=960]
        # past_key_values: list of 32 {'key_states'/'value_states': [B, L, 5, 64]}

        prefix_last_layer = outputs_embeds[0]
        last_token = prefix_last_layer[:, -1, :]
        last_layer_mean = prefix_last_layer.mean(dim=1)
        dsrl_features = torch.cat([last_token, last_layer_mean], dim=-1)

        return dsrl_features

    @torch.no_grad()
    def extract_dsrl_state_features(self, env_obs: dict[str, Any]) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        batch_obs = self._preprocess_obs_batch(env_obs, device=device)
        features = self._extract_dsrl_state_features_from_batch(batch_obs)
        return features.detach().cpu()

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

        # dsrl_state_features = self._extract_dsrl_state_features_from_batch(batch_obs)

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
            # "dsrl_state_features": dsrl_state_features.detach().cpu(),
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
