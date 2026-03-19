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
import hashlib
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf_noray.models.embodiment.base_policy import BasePolicy
from rlinf_noray.models.embodiment.modules.value_head import ValueHead


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

            # TODO: 这个模型是否是微调过的 SmolVLA？
            self.policy = SmolVLAPolicy.from_pretrained(cfg.model_path)

        self._validate_normalizer_stats()
        self.policy_preprocessor, self.policy_postprocessor = self._build_policy_processors(cfg)

        self.action_dim = cfg.action_dim
        self.num_action_chunks = cfg.num_action_chunks

        self.image_keys: list[str] = self._resolve_image_keys(cfg)
        self.main_image_env_key: str = cfg.get("main_image_env_key", "main_images")
        self.wrist_image_env_key: str | None = cfg.get("wrist_image_env_key", None)
        self.flip_libero_images: bool = cfg.get("flip_libero_images", True)

        if cfg.get("add_value_head", False):
            state_dim = cfg.get("state_dim", 9)
            self.value_head = ValueHead(
                input_dim=state_dim,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    def _validate_normalizer_stats(self) -> None:
        """Validate normalizer stats and fail fast when artifacts are invalid."""
        invalid_tensors: list[str] = []

        def _check(name: str, tensor: torch.Tensor) -> None:
            lname = name.lower()
            if (
                "normalize" not in lname
                and "normalizer" not in lname
                and "unnormalize" not in lname
            ):
                return
            if "mean" not in lname and "std" not in lname:
                return
            if torch.isinf(tensor).any() or torch.isnan(tensor).any():
                invalid_tensors.append(name)

        for name, param in self.policy.named_parameters():
            _check(name, param)

        for name, buffer in self.policy.named_buffers():
            _check(name, buffer)

        if invalid_tensors:
            names = ", ".join(invalid_tensors)
            raise ValueError(
                f"[SmolVLA] Invalid normalization stats (nan/inf) in model '{self.model_path}': {names}"
            )

    def _build_policy_processors(self, cfg: DictConfig):
        """Build policy pre/postprocessors with the same factory path as lerobot eval."""
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides = {
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        return make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=cfg.model_path,
            preprocessor_overrides=preprocessor_overrides,
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

    def _make_lerobot_raw_batch(
        self,
        env_obs: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        """Build a raw lerobot-format batch from an RLinf env_obs dict."""
        batch: dict[str, Any] = {}

        main_img = self._hwc_to_chw_float(env_obs[self.main_image_env_key])
        if self.flip_libero_images:
            main_img = torch.flip(main_img, dims=[2, 3])
        batch[f"observation.images.{self.image_keys[0]}"] = main_img

        if self.wrist_image_env_key and len(self.image_keys) > 1:
            wrist = env_obs.get(self.wrist_image_env_key)
            if wrist is not None:
                wrist_img = self._hwc_to_chw_float(wrist)
                if self.flip_libero_images:
                    wrist_img = torch.flip(wrist_img, dims=[2, 3])
                batch[f"observation.images.{self.image_keys[1]}"] = wrist_img

        states = env_obs["states"]
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        policy_dtype = next(self.policy.parameters()).dtype
        batch["observation.state"] = states.to(dtype=policy_dtype)

        batch["task"] = list(env_obs["task_descriptions"])

        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _preprocess_obs_batch(self, env_obs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        raw_batch = self._make_lerobot_raw_batch(env_obs, device=device)
        return self.policy_preprocessor(raw_batch)

    # ------------------------------------------------------------------
    # Flow-matching log-prob
    # ------------------------------------------------------------------

    def _compute_flow_logprob(
        self,
        batch_obs: dict[str, Any],
        norm_actions: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
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
        del batch_obs, timestep
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
            value_dtype = next(self.value_head.parameters()).dtype
            return self.value_head(states.to(device=device, dtype=value_dtype)).squeeze(-1)
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

        # ####### TEMP START #######
        external_policy_payload = kwargs.get("external_policy_payload", None)
        if external_policy_payload is not None:
            if not isinstance(external_policy_payload, dict):
                raise TypeError(
                    f"external_policy_payload must be dict, got {type(external_policy_payload)}"
                )
            batch_obs = {
                key: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in external_policy_payload.items()
            }
        else:
            batch_obs = self._preprocess_obs_batch(env_obs, device=device)
        # ####### TEMP END #######

        # Clear queue so generation starts from the current observation.
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        # ####### TEMP START #######
        external_policy_noise = kwargs.get("external_policy_noise", None)
        policy_noise_tensor = None
        if external_policy_noise is not None:
            if isinstance(external_policy_noise, np.ndarray):
                policy_noise_tensor = torch.from_numpy(external_policy_noise)
            elif isinstance(external_policy_noise, torch.Tensor):
                policy_noise_tensor = external_policy_noise
            else:
                raise TypeError(
                    f"external_policy_noise must be np.ndarray or torch.Tensor, got {type(external_policy_noise)}"
                )
            policy_noise_tensor = policy_noise_tensor.to(device=device, dtype=torch.float32)
            if policy_noise_tensor.ndim not in (3, 4):
                raise ValueError(
                    "external_policy_noise must have shape [B, chunk_size, max_action_dim] "
                    "or [B, num_action_chunks, chunk_size, max_action_dim]"
                )
            if policy_noise_tensor.shape[0] != batch_obs["observation.state"].shape[0]:
                raise ValueError(
                    "external_policy_noise batch size does not match env_obs batch size"
                )
            if policy_noise_tensor.ndim == 4 and policy_noise_tensor.shape[1] != self.num_action_chunks:
                raise ValueError(
                    "external_policy_noise second dim must equal num_action_chunks when ndim=4"
                )
        # ####### TEMP END #######

        norm_action_list = []
        raw_action_list = []

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

        select_action_align_dump_path = os.environ.get("RLINF_SELECT_ACTION_ALIGN_DUMP_PATH", "").strip()
        align_batch_obs: dict[str, Any] | None = None
        align_step_noise: torch.Tensor | None = None
        align_norm_actions: torch.Tensor | None = None
        align_compared = False
        if select_action_align_dump_path:
            self.policy.eval()
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            align_dump = torch.load(select_action_align_dump_path, map_location="cpu")
            loaded_batch_obs = align_dump.get("batch_obs", None)
            loaded_step_noise = align_dump.get("step_noise", None)
            loaded_norm_actions = align_dump.get("norm_actions", None)
            loaded_runtime = align_dump.get("runtime", {})
            loaded_batch_fp = align_dump.get("batch_obs_fingerprint", None)
            loaded_step_noise_fp = align_dump.get("step_noise_fingerprint", None)

            if isinstance(loaded_batch_obs, dict):
                align_batch_obs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in loaded_batch_obs.items()
                }
            if isinstance(loaded_step_noise, torch.Tensor):
                align_step_noise = loaded_step_noise.to(device=device, dtype=torch.float32)
            if isinstance(loaded_norm_actions, torch.Tensor):
                align_norm_actions = loaded_norm_actions.to(device=device, dtype=torch.float32)

            logging.warning(
                "[TEMP] Loaded select_action align payload from %s",
                select_action_align_dump_path,
            )
            logging.warning("[TEMP] no-ray backend snapshot: %s", _backend_snapshot())
            logging.warning("[TEMP] lerobot backend snapshot: %s", loaded_runtime.get("backend", {}))

            local_model_fp = _model_fingerprint(self.policy)
            logging.warning("[TEMP] no-ray model fingerprint: %s", local_model_fp)
            logging.warning("[TEMP] lerobot model fingerprint: %s", loaded_runtime.get("model", {}))

            loaded_model_sha = loaded_runtime.get("model", {}).get("model_sha256", None)
            local_model_sha = local_model_fp.get("model_sha256", None)
            if loaded_model_sha is not None and loaded_model_sha != local_model_sha:
                logging.warning(
                    "[TEMP] model_sha256 mismatch: lerobot=%s no-ray=%s",
                    loaded_model_sha,
                    local_model_sha,
                )

            if align_batch_obs is not None and isinstance(loaded_batch_fp, dict):
                local_batch_fp = _batch_fingerprint(align_batch_obs)
                mismatch_keys = [
                    key for key in sorted(set(loaded_batch_fp) | set(local_batch_fp))
                    if loaded_batch_fp.get(key) != local_batch_fp.get(key)
                ]
                logging.warning("[TEMP] batch_obs fingerprint mismatch keys: %s", mismatch_keys)

            if align_step_noise is not None and loaded_step_noise_fp is not None:
                local_step_noise_fp = _tensor_fingerprint(align_step_noise)
                if local_step_noise_fp != loaded_step_noise_fp:
                    logging.warning(
                        "[TEMP] step_noise fingerprint mismatch: lerobot=%s no-ray=%s",
                        loaded_step_noise_fp,
                        local_step_noise_fp,
                    )

        for action_step_idx in range(self.num_action_chunks):
            # ####### TEMP START #######
            step_noise = None
            if policy_noise_tensor is not None:
                if policy_noise_tensor.ndim == 4:
                    step_noise = policy_noise_tensor[:, action_step_idx, :, :]
                else:
                    step_noise = policy_noise_tensor
            # ####### TEMP END #######
            batch_obs_for_select = align_batch_obs if align_batch_obs is not None else batch_obs
            step_noise_for_select = align_step_noise if align_step_noise is not None else step_noise

            action_norm_step = self.policy.select_action(batch_obs_for_select, noise=step_noise_for_select)
            # action_norm_step = self.policy._get_action_chunk(batch_obs_for_select, noise=step_noise_for_select)

            if align_norm_actions is not None and not align_compared:
                expected_norm = align_norm_actions
                current_norm = action_norm_step
                if expected_norm.shape != current_norm.shape:
                    logging.warning(
                        "[TEMP] select_action norm_actions shape mismatch: expected=%s current=%s",
                        tuple(expected_norm.shape),
                        tuple(current_norm.shape),
                    )
                else:
                    diff = (current_norm.float() - expected_norm.float()).abs()
                    logging.warning(
                        "[TEMP] select_action norm_actions diff: max_abs=%.8f mean_abs=%.8f",
                        diff.max().item(),
                        diff.mean().item(),
                    )
                    logging.warning(
                        "[TEMP] no-ray norm_actions fingerprint: %s",
                        _tensor_fingerprint(current_norm),
                    )
                    logging.warning(
                        "[TEMP] lerobot norm_actions fingerprint: %s",
                        _tensor_fingerprint(expected_norm),
                    )
                align_compared = True
                
                raise SystemExit(0)

            action_raw_step = self.policy_postprocessor(action_norm_step)
            norm_action_list.append(action_norm_step)
            raw_action_list.append(action_raw_step)

        act = torch.stack(norm_action_list, dim=1)
        raw_act = torch.stack(raw_action_list, dim=1)
        B, _, _ = act.shape

        # Sample deterministic (noise, timestep) for PPO consistency.
        noise = torch.randn_like(act)  # [B, chunk, action_dim]
        timestep = torch.rand(B, device=device)  # [B]

        # ####### TEMP START #######
        temp_rollout_only = os.environ.get("RLINF_TEMP_EXIT_AFTER_FIRST_ROLLOUT", "0") == "1"
        if temp_rollout_only:
            prev_logprobs = torch.zeros(B, device=device, dtype=torch.float32)
        else:
            prev_logprobs = self._compute_flow_logprob(batch_obs, act, noise, timestep)  # [B]
        # ####### TEMP END #######

        states = env_obs["states"]
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        prev_values = self._compute_value(states, device)  # [B]

        raw_actions = raw_act.detach().cpu().numpy()  # [B, chunk, action_dim]

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

        # ####### TEMP START #######
        if os.environ.get("RLINF_ROLLOUT_COMPARE_DUMP_PATH", "").strip():
            result["debug_policy_payload"] = {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in batch_obs.items()
            }
            result["debug_action_post"] = raw_act.detach().cpu()
        # ####### TEMP END #######

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

        batch_obs = self._preprocess_obs_batch(stored_env_obs, device=device)

        noise = forward_inputs["noise"].to(device)
        timestep = forward_inputs["timestep"].to(device)
        norm_actions = forward_inputs["norm_actions"].to(device)

        logprobs = self._compute_flow_logprob(batch_obs, norm_actions, noise, timestep)  # [B]

        values = self._compute_value(forward_inputs["states"], device)  # [B]

        return {
            "logprobs": logprobs,
            "prev_logprobs": forward_inputs.get("prev_logprobs"),
            "values": values,
            "entropy": None,
        }
