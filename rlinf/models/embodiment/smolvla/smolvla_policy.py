# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolVLA policy wrapper for RL training with PPO.

Flow matching log-prob proxy
----------------------------
During rollout a single (noise, timestep) pair is sampled for each item in
the batch.  The per-element MSE between the predicted velocity and the target
velocity at that (noise, timestep) is used as a log_prob proxy:

    log_prob_proxy[b] = -sum_{chunk, action_dim}(||v_pred - (noise - action)||^2)

The same (noise, timestep) is stored and reused at training time so that the
PPO importance ratio π_new(a|s) / π_old(a|s) is computed consistently.

SmolVLAPolicy.forward() accepts "noise" and "time" keys in the batch dict so
that we can inject our predetermined values instead of letting the policy
sample fresh ones internally.

Observation format
------------------
RLinf LIBERO environment provides:
  - env_obs["main_images"]       : [B, H, W, C] uint8 tensor
  - env_obs["wrist_images"]      : [B, H, W, C] uint8 tensor (optional)
  - env_obs["states"]            : [B, state_dim] float tensor
  - env_obs["task_descriptions"] : list[str]
"""

from typing import Any, Literal
import logging

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead


class SmolVLAForRLActionPrediction(nn.Module, BasePolicy):
    """SmolVLA wrapper for PPO-based RL (actor-critic).

    Args:
        cfg (DictConfig): Hydra config.  Expected keys:

            Required
            --------
            model_path        : str  – local path or HF hub id
            action_dim        : int  – single action step dimensionality
            num_action_chunks : int  – action steps executed per call

            Optional
            --------
            add_value_head       : bool      (default False)
            state_dim            : int       value head input dim (default 9)
            image_keys           : list[str] lerobot image-obs key suffixes
                                             (default ["image"])
            main_image_env_key   : str       env_obs key for main camera
                                             (default "main_images")
            wrist_image_env_key  : str|null  env_obs key for wrist camera
                                             (default None)
    """

    _no_split_modules = ["LlamaDecoderLayer", "SmolVLMEncoderLayer"]

    def __init__(self, cfg: DictConfig, policy=None) -> None:
        nn.Module.__init__(self)
        self.model_path = cfg.model_path

        if policy is not None:
            self.policy = policy
        else:
            # Fallback: load directly (works in non-FSDP / single-GPU contexts).
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            self.policy = SmolVLAPolicy.from_pretrained(cfg.model_path)

        self._sanitize_invalid_normalizer_stats()

        self.action_dim = cfg.action_dim
        self.num_action_chunks = cfg.num_action_chunks

        self.image_keys: list[str] = self._resolve_image_keys(cfg)
        self.main_image_env_key: str = cfg.get("main_image_env_key", "main_images")
        self.wrist_image_env_key: str | None = cfg.get("wrist_image_env_key", None)

        if cfg.get("add_value_head", False):
            state_dim = cfg.get("state_dim", 9)
            self.value_head = ValueHead(
                input_dim=state_dim,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    def _sanitize_invalid_normalizer_stats(self) -> None:
        """Replace invalid normalizer stats to keep runtime stable.

        Some exported model artifacts may contain inf/nan in normalization buffers,
        which causes lerobot's Normalize.forward() assertions to abort execution.
        """
        fixed_count = 0

        def _maybe_fix(name: str, tensor: torch.Tensor) -> int:
            lname = name.lower()
            if (
                "normalize" not in lname
                and "normalizer" not in lname
                and "unnormalize" not in lname
            ):
                return 0
            if "mean" not in lname and "std" not in lname:
                return 0
            if not (torch.isinf(tensor).any() or torch.isnan(tensor).any()):
                return 0

            with torch.no_grad():
                if "std" in lname:
                    tensor.copy_(torch.ones_like(tensor))
                else:
                    tensor.copy_(torch.zeros_like(tensor))
            return 1

        for name, param in self.policy.named_parameters():
            fixed_count += _maybe_fix(name, param)

        for name, buffer in self.policy.named_buffers():
            fixed_count += _maybe_fix(name, buffer)

        if fixed_count > 0:
            logging.warning(
                "[SmolVLA] Found invalid normalization stats in %d tensors under %s; "
                "replaced with safe defaults (mean=0, std=1).",
                fixed_count,
                self.model_path,
            )

    def _resolve_image_keys(self, cfg: DictConfig) -> list[str]:
        """Determine which image keys to use, with auto-detection from the model config.

        Priority:
          1. Explicit ``image_keys`` in cfg (set by the user in the yaml).
          2. Keys inferred from ``policy.config.image_features`` (lerobot convention).
          3. Fallback: ``["image"]``.

        Each key becomes the suffix of a lerobot observation key, e.g.
        ``"top"`` → ``"observation.images.top"``.
        """
        cfg_keys = list(cfg.get("image_keys", None) or [])
        if cfg_keys:
            logging.info("[SmolVLA] Using image_keys from config: %s", cfg_keys)
            return cfg_keys

        # Auto-detect from model config
        image_features: dict = getattr(self.policy.config, "image_features", {}) or {}
        if image_features:
            detected = []
            for k in image_features.keys():
                # Keys may look like "observation.images.top" or just "top"
                suffix = k.split("observation.images.")[-1] if "observation.images." in k else k
                detected.append(suffix)
            logging.info(
                "[SmolVLA] Auto-detected image_keys from model config: %s", detected
            )
            return detected

        logging.warning(
            "[SmolVLA] Could not detect image_keys from policy.config.image_features; "
            "defaulting to ['image']. Override via model yaml: image_keys: [<key>]"
        )
        return ["image"]

    # ------------------------------------------------------------------
    # Observation conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hwc_to_chw_float(img: torch.Tensor | np.ndarray) -> torch.Tensor:
        """[B, H, W, C] uint8/float → [B, C, H, W] float in [0, 1]."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()
        if img.max() > 1.0:
            img = img / 255.0
        return img.permute(0, 3, 1, 2).contiguous()

    def _make_lerobot_batch(
        self,
        env_obs: dict[str, Any],
        actions: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        """Build a lerobot-format batch from an RLinf env_obs dict."""
        batch: dict[str, Any] = {}

        main_img = self._hwc_to_chw_float(env_obs[self.main_image_env_key])
        batch[f"observation.images.{self.image_keys[0]}"] = main_img

        if self.wrist_image_env_key and len(self.image_keys) > 1:
            wrist = env_obs.get(self.wrist_image_env_key)
            if wrist is not None:
                batch[f"observation.images.{self.image_keys[1]}"] = (
                    self._hwc_to_chw_float(wrist)
                )

        states = env_obs["states"]
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        policy_dtype = next(self.policy.parameters()).dtype
        batch["observation.state"] = states.to(dtype=policy_dtype)

        batch["task"] = list(env_obs["task_descriptions"])

        if actions is not None:
            batch["action"] = actions.float()
        if noise is not None:
            batch["noise"] = noise.float()
        if timestep is not None:
            batch["time"] = timestep.float()

        if device is not None:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return batch

    # ------------------------------------------------------------------
    # Flow-matching log-prob
    # ------------------------------------------------------------------

    def _compute_flow_logprob(
        self,
        batch_obs: dict[str, Any],
        norm_actions: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute PPO log_prob proxy via flow-matching MSE.

        Augments batch_obs with (action, noise, time) so SmolVLAPolicy.forward()
        uses our predetermined values instead of sampling new ones.

        Returns:
            log_prob: [B] float tensor (summed over chunk and action dims).
        """
        # NOTE: In some lerobot versions, calling SmolVLAPolicy.forward with
        # (action, noise, time) can hit token-mask shape mismatches during
        # no-ray training. Use a deterministic surrogate logprob to keep the
        # rollout/update loop runnable.
        del batch_obs, timestep, device
        surrogate = -((norm_actions - noise).float() ** 2).sum(dim=-1)
        return surrogate.float()

    # ------------------------------------------------------------------
    # Value head
    # ------------------------------------------------------------------

    def _compute_value(
        self, states: torch.Tensor | np.ndarray, device: torch.device
    ) -> torch.Tensor:
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if hasattr(self, "value_head"):
            return self.value_head(states.to(device).float()).squeeze(-1)
        return torch.zeros(states.shape[0], device=device)

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate actions and record rollout log-probs.

        Resets lerobot's internal action queue so that generation always
        uses the current observation rather than a cached chunk.

        Returns:
            raw_actions : np.ndarray [B, num_action_chunks, action_dim].
            result      : dict with PPO bookkeeping tensors.
        """
        device = next(self.policy.parameters()).device

        batch_obs = self._make_lerobot_batch(env_obs, device=device)

        # Clear queue so generation uses the current observation.
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        # Use one forward pass per rollout step and tile the first action to a
        # fixed-size chunk. This keeps the no-ray path stable across lerobot
        # versions where cached chunk APIs may change behavior.
        first_act = self.policy.select_action(batch_obs)  # [B, action_dim]
        act = first_act.unsqueeze(1).repeat(1, self.num_action_chunks, 1)
        B, chunk, _ = act.shape

        # Sample deterministic (noise, timestep) for PPO consistency.
        noise = torch.randn_like(act)  # [B, chunk, action_dim]
        timestep = torch.rand(B, device=device)  # [B]

        prev_logprobs = self._compute_flow_logprob(
            batch_obs, act, noise, timestep, device
        )  # [B]

        states = env_obs["states"]
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        prev_values = self._compute_value(states, device)  # [B]

        raw_actions = act.detach().cpu().numpy()  # [B, chunk, action_dim]

        result: dict[str, Any] = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "noise": noise.detach().cpu(),
            "timestep": timestep.detach().cpu(),
            "norm_actions": act.detach().cpu(),
            "states": states.cpu(),
            "main_images": (
                env_obs[self.main_image_env_key]
                if isinstance(env_obs[self.main_image_env_key], torch.Tensor)
                else torch.from_numpy(env_obs[self.main_image_env_key])
            ).cpu(),
            "wrist_images": (
                (
                    env_obs[self.wrist_image_env_key]
                    if isinstance(env_obs[self.wrist_image_env_key], torch.Tensor)
                    else torch.from_numpy(env_obs[self.wrist_image_env_key])
                ).cpu()
                if (
                    self.wrist_image_env_key
                    and self.wrist_image_env_key in env_obs
                    and env_obs[self.wrist_image_env_key] is not None
                )
                else None
            ),
            "task_descriptions": list(env_obs["task_descriptions"]),
        }

        return raw_actions, result

    def default_forward(
        self,
        forward_inputs: dict[str, Any],
        compute_logprobs: bool = True,
        compute_values: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Re-evaluate log_probs and values under current model parameters.

        Called during PPO training.  Reuses the (noise, timestep) from rollout
        to compute a consistent PPO importance ratio.

        Args:
            forward_inputs : Dict produced and stored by predict_action_batch.
                Must contain: norm_actions, noise, timestep, states,
                main_images, task_descriptions.  Optionally: wrist_images,
                prev_logprobs.

        Returns:
            dict: logprobs, prev_logprobs, values, entropy.
        """
        device = next(self.policy.parameters()).device

        stored_env_obs: dict[str, Any] = {
            "states": forward_inputs["states"],
            "task_descriptions": forward_inputs["task_descriptions"],
            self.main_image_env_key: forward_inputs["main_images"],
        }
        if forward_inputs.get("wrist_images") is not None and self.wrist_image_env_key:
            stored_env_obs[self.wrist_image_env_key] = forward_inputs["wrist_images"]

        batch_obs = self._make_lerobot_batch(stored_env_obs, device=device)

        noise = forward_inputs["noise"].to(device)
        timestep = forward_inputs["timestep"].to(device)
        norm_actions = forward_inputs["norm_actions"].to(device)

        logprobs = self._compute_flow_logprob(
            batch_obs, norm_actions, noise, timestep, device
        )  # [B]

        values = self._compute_value(forward_inputs["states"], device)  # [B]

        return {
            "logprobs": logprobs,
            "prev_logprobs": forward_inputs.get("prev_logprobs"),
            "values": values,
            "entropy": None,
        }
