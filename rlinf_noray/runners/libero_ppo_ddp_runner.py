import copy
import contextlib
import json
import os
import time
from pathlib import Path
from textwrap import wrap
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
from rlinf_noray.envs.action_utils import prepare_actions
from rlinf_noray.envs.libero.utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
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


def _to_numpy(obj: Any) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    raise TypeError(f"Unsupported type for numpy conversion: {type(obj)}")


def _max_abs_diff(a: Any, b: Any) -> float:
    a_np = _to_numpy(a).astype(np.float64)
    b_np = _to_numpy(b).astype(np.float64)
    if a_np.shape != b_np.shape:
        raise ValueError(f"Shape mismatch: {a_np.shape} vs {b_np.shape}")
    return float(np.max(np.abs(a_np - b_np)))


class _ChunkJsonlProfiler:
    def __init__(
        self,
        *,
        output_path: str,
        rank: int,
        world_size: int,
        sync_cuda: bool,
    ) -> None:
        if not output_path:
            raise ValueError("Profiler output path must be non-empty")

        target_path = Path(output_path)
        if world_size > 1:
            target_path = target_path.with_name(
                f"{target_path.stem}.rank{rank}{target_path.suffix}"
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = target_path.open("a", encoding="utf-8")
        self._path = target_path
        self._rank = rank
        self._sync_cuda = bool(sync_cuda)
        self._timings: dict[str, float] = {}

    @property
    def path(self) -> str:
        return str(self._path)

    @contextlib.contextmanager
    def profile(self, key: str):
        if self._sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self._sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            self._timings[key] = self._timings.get(key, 0.0) + float(elapsed)

    def flush_chunk(self, *, payload: dict[str, Any]) -> None:
        record = {
            "time_unix": time.time(),
            "rank": self._rank,
            **payload,
            "timings_s": self._timings,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._timings = {}

    def close(self) -> None:
        self._file.close()


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
        self.chunk_steps = int(cfg.env.train.max_steps_per_rollout_epoch) // int(
            cfg.actor.model.num_action_chunks
        )
        self.metric_logger = MetricLogger(cfg) if self.rank == 0 else None

        self._chunk_profiler: _ChunkJsonlProfiler | None = None
        self._chunk_profile_output_path = os.environ.get("RLINF_COLLECT_PROFILE_PATH", "").strip()
        if self._chunk_profile_output_path:
            profile_sync_cuda = os.environ.get("RLINF_COLLECT_PROFILE_SYNC_CUDA", "1") == "1"
            self._chunk_profiler = _ChunkJsonlProfiler(
                output_path=self._chunk_profile_output_path,
                rank=self.rank,
                world_size=self.world_size,
                sync_cuda=profile_sync_cuda,
            )
            if self.rank == 0:
                print(
                    f"[profile] collect profiler enabled: {self._chunk_profiler.path}",
                    flush=True,
                )

        eval_video_cfg = cfg.env.eval.get("video_cfg", {})
        self.save_eval_video = bool(eval_video_cfg.get("save_video", False))
        self.eval_video_base_dir = str(eval_video_cfg.get("video_base_dir", ""))

        # Optional stage-level debug prints for collect rollout hang diagnosis.
        self._collect_debug_verbose = os.environ.get("RLINF_COLLECT_DEBUG_VERBOSE", "0") == "1"
        self._collect_debug_max_chunks = int(os.environ.get("RLINF_COLLECT_DEBUG_MAX_CHUNKS", "3"))

        # ####### TEMP START #######
        self._temp_alignment_sample_path = os.environ.get("RLINF_ROLLOUT_COMPARE_DUMP_PATH", "").strip()
        self._temp_alignment_sample = None
        self._temp_alignment_compared = False
        self._temp_alignment_log_path = os.environ.get("RLINF_ALIGN_ACTION_LOG_PATH", "").strip()
        if self._temp_alignment_sample_path:
            sample_path = Path(self._temp_alignment_sample_path)
            if not sample_path.exists():
                raise FileNotFoundError(f"RLINF_ROLLOUT_COMPARE_DUMP_PATH not found: {sample_path}")
            self._temp_alignment_sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            if not self._temp_alignment_log_path:
                self._temp_alignment_log_path = str(
                    Path("/home/chuheng/RLinf/logs/rollout_alignment/noray_lerobot_action_compare.json")
                )
        # ####### TEMP END #######

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

        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        action_text = "[" + ", ".join(f"{v:+.2f}" for v in action_np.tolist()) + "]"
        max_chars = max(12, frame.shape[1] // 8)
        lines = ["action:"] + (wrap(action_text, width=max_chars) or [""])
        text = "\n".join(lines)
        _, _, _, text_h = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
        y = max(0, frame.shape[0] - text_h - 8)
        draw.multiline_text((8, y), text, fill=(255, 255, 255), font=font, spacing=2)
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
    ) -> None:
        if not self.eval_video_base_dir:
            return
        base_dir = Path(self.eval_video_base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        traj_offset = self.rank * len(frames_by_env)
        for local_idx, frames in enumerate(frames_by_env):
            if not frames:
                continue
            traj_idx = traj_offset + local_idx
            instruction_slug = self._instruction_to_slug(instructions_by_env[local_idx])
            out_path = base_dir / f"val_epoch_{epoch}_traj_{traj_idx:02d}_{instruction_slug}.mp4"
            imageio.mimsave(str(out_path), frames, fps=15)

    def _reduce_sums(self, sums: torch.Tensor) -> torch.Tensor:
        sums = sums.to(self.device, dtype=torch.float64)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        return sums

    def _collect_debug_print(self, message: str) -> None:
        if self._collect_debug_verbose and self.rank == 0:
            print(f"[collect-debug] {message}", flush=True)

    # ####### TEMP START #######
    @staticmethod
    def _temp_to_hwc_uint8(images: Any) -> torch.Tensor:
        arr = _to_numpy(images)
        if arr.ndim != 4:
            raise ValueError(f"Expected image batch rank=4, got shape={arr.shape}")
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return torch.from_numpy(arr)

    @staticmethod
    def _temp_compute_state_from_raw(obs_raw: dict[str, Any]) -> torch.Tensor:
        pos = _to_numpy(obs_raw["robot0_eef_pos"])
        quat = _to_numpy(obs_raw["robot0_eef_quat"])
        gripper = _to_numpy(obs_raw["robot0_gripper_qpos"])
        if pos.ndim != 2 or quat.ndim != 2 or gripper.ndim != 2:
            raise ValueError(
                "robot0_eef_pos/quat/gripper must be rank-2 batched arrays"
            )
        axis_angles = np.stack([quat2axisangle(quat[i].copy()) for i in range(quat.shape[0])], axis=0)
        states = np.concatenate([pos, axis_angles, gripper], axis=1).astype(np.float32)
        return torch.from_numpy(states)

    def _temp_build_env_obs_from_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if "noray_obs_reference" in sample:
            ref = _to_cpu(sample["noray_obs_reference"])
            return {
                "main_images": self._temp_to_hwc_uint8(ref["main_images"]),
                "wrist_images": self._temp_to_hwc_uint8(ref["wrist_images"]),
                "states": _to_cpu(ref["states"]).float(),
                "task_descriptions": list(ref["task_descriptions"]),
            }

        if "obs_raw" not in sample:
            raise KeyError("Alignment sample missing 'obs_raw'")
        obs_raw = sample["obs_raw"]

        if "main_images" in obs_raw and "states" in obs_raw and "task_descriptions" in obs_raw:
            env_obs = {
                "main_images": self._temp_to_hwc_uint8(obs_raw["main_images"]),
                "states": _to_cpu(obs_raw["states"]).float(),
                "task_descriptions": list(obs_raw["task_descriptions"]),
            }
            if "wrist_images" in obs_raw:
                env_obs["wrist_images"] = self._temp_to_hwc_uint8(obs_raw["wrist_images"])
            return env_obs

        if "agentview_image" in obs_raw and "robot0_eef_pos" in obs_raw:
            batch_size = int(_to_numpy(obs_raw["agentview_image"]).shape[0])
            main_images = np.stack(
                [get_libero_image({"agentview_image": _to_numpy(obs_raw["agentview_image"])[i]}) for i in range(batch_size)],
                axis=0,
            )
            if "robot0_eye_in_hand_image" in obs_raw:
                wrist_images = np.stack(
                    [
                        get_libero_wrist_image(
                            {"robot0_eye_in_hand_image": _to_numpy(obs_raw["robot0_eye_in_hand_image"])[i]}
                        )
                        for i in range(batch_size)
                    ],
                    axis=0,
                )
            else:
                wrist_images = main_images.copy()

            task_key = "task" if "task" in obs_raw else "task_descriptions"
            if task_key not in obs_raw:
                raise KeyError("obs_raw missing task/task_descriptions for TEMP alignment")

            task_values = obs_raw[task_key]
            task_descriptions = [str(x) for x in list(task_values)]
            if len(task_descriptions) != batch_size:
                raise ValueError(
                    f"task length mismatch: {len(task_descriptions)} vs batch_size {batch_size}"
                )

            return {
                "main_images": self._temp_to_hwc_uint8(main_images),
                "wrist_images": self._temp_to_hwc_uint8(wrist_images),
                "states": self._temp_compute_state_from_raw(obs_raw),
                "task_descriptions": task_descriptions,
            }

        if "pixels" in obs_raw and "robot_state" in obs_raw:
            pixels = obs_raw["pixels"]
            robot_state = obs_raw["robot_state"]

            if "image" not in pixels:
                raise KeyError("obs_raw['pixels'] missing 'image'")
            if "image2" not in pixels:
                raise KeyError("obs_raw['pixels'] missing 'image2'")

            main_images = self._temp_to_hwc_uint8(pixels["image"])
            wrist_images = self._temp_to_hwc_uint8(pixels["image2"])

            if "eef" not in robot_state or "gripper" not in robot_state:
                raise KeyError("obs_raw['robot_state'] missing eef/gripper")
            eef = robot_state["eef"]
            gripper = robot_state["gripper"]
            if "pos" not in eef or "quat" not in eef or "qpos" not in gripper:
                raise KeyError("obs_raw['robot_state'] missing eef.pos/eef.quat/gripper.qpos")

            axis_angles = np.stack(
                [quat2axisangle(_to_numpy(eef["quat"])[i].copy()) for i in range(_to_numpy(eef["quat"]).shape[0])],
                axis=0,
            )
            states = np.concatenate(
                [_to_numpy(eef["pos"]), axis_angles, _to_numpy(gripper["qpos"])], axis=1
            ).astype(np.float32)

            task_descriptions = None
            if "policy_debug" in sample:
                payload = sample["policy_debug"].get("policy_payload", {})
                task = payload.get("task", None)
                if task is not None:
                    task_descriptions = [str(x) for x in list(task)]
            if task_descriptions is None:
                raise KeyError(
                    "Alignment sample missing task descriptions (expected policy_debug.policy_payload['task'])"
                )

            return {
                "main_images": main_images,
                "wrist_images": wrist_images,
                "states": torch.from_numpy(states),
                "task_descriptions": task_descriptions,
            }

        raise KeyError("Unsupported alignment sample format for obs replacement")

    @staticmethod
    def _temp_get_external_noise(sample: dict[str, Any]) -> torch.Tensor:
        if "policy_debug" not in sample:
            raise KeyError("Alignment sample missing 'policy_debug'")
        sample_debug = sample["policy_debug"]
        if "policy_noise" not in sample_debug:
            raise KeyError("Alignment sample missing policy_debug['policy_noise']")
        policy_noise = sample_debug["policy_noise"]
        if policy_noise is None:
            raise ValueError("policy_debug['policy_noise'] is None")
        return _to_cpu(policy_noise).float()

    @staticmethod
    def _temp_compare_first_action(
        sample: dict[str, Any],
        predict_env_obs: dict[str, Any],
        rollout_payload: dict[str, Any] | None,
        rollout_norm_actions: torch.Tensor,
        rollout_post_actions: torch.Tensor | None,
        chunk_actions: torch.Tensor,
        alignment_log_path: str | None,
    ) -> None:
        step_prefix = "step_0_"

        required = [
            f"{step_prefix}env_obs",
            f"{step_prefix}policy_payload",
            f"{step_prefix}model_raw_action",
            f"{step_prefix}postprocessor_action",
            f"{step_prefix}env_action",
        ]
        for key in required:
            if key not in sample:
                raise KeyError(f"Alignment sample missing {key}")

        def _cmp(stage_name: str, expected: Any, actual: Any) -> dict[str, Any]:
            expected_np = _to_numpy(expected).astype(np.float64)
            actual_np = _to_numpy(actual).astype(np.float64)
            if expected_np.ndim == 3:
                expected_np = expected_np[:, 0, :]
            if actual_np.ndim == 3:
                actual_np = actual_np[:, 0, :]

            min_batch = min(expected_np.shape[0], actual_np.shape[0])
            if min_batch <= 0:
                raise ValueError(f"Empty batch in {stage_name} comparison")

            expected_trim = expected_np[:min_batch]
            actual_trim = actual_np[:min_batch]
            if expected_trim.shape != actual_trim.shape:
                raise ValueError(
                    f"Shape mismatch in {stage_name}: {expected_trim.shape} vs {actual_trim.shape}"
                )

            diff = float(np.max(np.abs(expected_trim - actual_trim)))
            equal = bool(np.array_equal(expected_trim, actual_trim))
            allclose = bool(np.allclose(expected_trim, actual_trim, atol=1e-6, rtol=0.0))
            print(
                (
                    f"[TEMP][align] {stage_name}: "
                    f"max_abs_diff={diff:.6e}, equal={equal}, allclose_atol1e-6={allclose}"
                ),
                flush=True,
            )
            return {
                "max_abs_diff": diff,
                "equal": equal,
                "allclose_atol1e-6": allclose,
            }

        sample_env_obs = sample[f"{step_prefix}env_obs"]
        if "pixels" not in sample_env_obs or "robot_state" not in sample_env_obs:
            raise KeyError(f"{step_prefix}env_obs missing pixels/robot_state")

        sample_state = np.concatenate(
            [
                _to_numpy(sample_env_obs["robot_state"]["eef"]["pos"]),
                np.stack(
                    [
                        quat2axisangle(q.copy())
                        for q in _to_numpy(sample_env_obs["robot_state"]["eef"]["quat"])
                    ],
                    axis=0,
                ),
                _to_numpy(sample_env_obs["robot_state"]["gripper"]["qpos"]),
            ],
            axis=1,
        ).astype(np.float64)

        env_obs_diffs = {
            "main_images": _cmp(
                f"{step_prefix}env_obs.main_images",
                sample_env_obs["pixels"]["image"],
                predict_env_obs["main_images"],
            ),
            "wrist_images": _cmp(
                f"{step_prefix}env_obs.wrist_images",
                sample_env_obs["pixels"]["image2"],
                predict_env_obs["wrist_images"],
            ),
            "states": _cmp(
                f"{step_prefix}env_obs.states",
                sample_state,
                predict_env_obs["states"],
            ),
        }

        if rollout_payload is None:
            raise KeyError("Rollout result missing debug_policy_payload for TEMP alignment")

        sample_payload = sample[f"{step_prefix}policy_payload"]
        payload_compare_keys = [
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "observation.language.tokens",
            "observation.language.attention_mask",
        ]
        payload_diffs: dict[str, Any] = {}
        for key in payload_compare_keys:
            if key not in sample_payload:
                raise KeyError(f"Alignment sample missing payload key: {key}")
            if key not in rollout_payload:
                raise KeyError(f"Rollout payload missing key: {key}")
            payload_diffs[key] = _cmp(f"{step_prefix}policy_payload.{key}", sample_payload[key], rollout_payload[key])

        # TODO(alignment): Validation point for step_0_model_raw_action mismatch.
        # This compares lerobot dump `step_0_model_raw_action` against no-ray
        # `norm_actions[:, 0, :]` produced in the same first-step rollout.
        model_raw_diff = _cmp(
            f"{step_prefix}model_raw_action",
            sample[f"{step_prefix}model_raw_action"],
            rollout_norm_actions[:, 0, :],
        )

        if rollout_post_actions is None:
            raise KeyError("Rollout result missing debug_action_post for TEMP alignment")
        post_action_diff = _cmp(
            f"{step_prefix}postprocessor_action",
            sample[f"{step_prefix}postprocessor_action"],
            rollout_post_actions[:, 0, :],
        )

        env_action_diff = _cmp(
            f"{step_prefix}env_action",
            sample[f"{step_prefix}env_action"],
            chunk_actions[:, 0, :],
        )

        if alignment_log_path:
            out_path = Path(alignment_log_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {
                    "source": "noray_alignment_compare",
                    "step": 0,
                },
                "diffs": {
                    f"{step_prefix}env_obs": env_obs_diffs,
                    f"{step_prefix}policy_payload": payload_diffs,
                    f"{step_prefix}model_raw_action": model_raw_diff,
                    f"{step_prefix}postprocessor_action": post_action_diff,
                    f"{step_prefix}env_action": env_action_diff,
                },
            }
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            print(f"[TEMP][align] wrote action compare file: {out_path}", flush=True)
    # ####### TEMP END #######

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
        self._collect_debug_print(
            (
                f"rollout start: mode={mode}, collect_samples={collect_samples}, "
                f"rollout_epoch={rollout_epoch}, chunk_steps={chunk_steps}, num_envs={env.num_envs}"
            )
        )
        profile_reset = (
            self._chunk_profiler.profile("env_reset")
            if self._chunk_profiler is not None
            else contextlib.nullcontext()
        )
        t0_reset = time.perf_counter()
        with profile_reset:
            obs, _ = env.reset()
        self._collect_debug_print(
            f"env.reset done in {(time.perf_counter() - t0_reset):.3f}s"
        )

        if self._chunk_profiler is not None:
            self._chunk_profiler.flush_chunk(
                payload={
                    "event": "rollout_init",
                    "mode": mode,
                    "collect_samples": collect_samples,
                    "rollout_epoch": int(rollout_epoch),
                    "chunk_steps": int(chunk_steps),
                    "num_envs": int(env.num_envs),
                }
            )
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
        try:
            for rollout_idx in range(rollout_epoch):
                for chunk_idx in range(chunk_steps):
                    debug_chunk = (
                        self._collect_debug_verbose
                        and self.rank == 0
                        and rollout_idx == 0
                        and chunk_idx < self._collect_debug_max_chunks
                    )
                    chunk_start = time.perf_counter()
                    if debug_chunk:
                        self._collect_debug_print(
                            f"chunk[{rollout_idx},{chunk_idx}] start"
                        )
                # ####### TEMP START #######
                    predict_env_obs = obs
                    predict_kwargs: dict[str, Any] = {}
                    if self._temp_alignment_sample is not None and not self._temp_alignment_compared:
                        predict_env_obs = self._temp_build_env_obs_from_sample(self._temp_alignment_sample)
                        predict_kwargs["external_policy_noise"] = self._temp_get_external_noise(
                            self._temp_alignment_sample
                        )
                        if self.rank == 0:
                            print(
                                "[TEMP][align] replaced env obs with dump obs_raw and injected dump policy_noise",
                                flush=True,
                            )
                # ####### TEMP END #######

                    profile_predict = (
                        self._chunk_profiler.profile("predict_action_batch")
                        if self._chunk_profiler is not None
                        else contextlib.nullcontext()
                    )
                    if debug_chunk:
                        self._collect_debug_print(
                            f"chunk[{rollout_idx},{chunk_idx}] entering predict_action_batch"
                        )
                    t0_predict = time.perf_counter()
                    with profile_predict:
                        module_was_training = self.ddp_model.module.training
                        if mode == "eval":
                            self.ddp_model.module.eval()
                        try:
                            with torch.no_grad():
                                raw_actions, rollout_result = self.ddp_model.module.predict_action_batch(
                                    env_obs=predict_env_obs,
                                    mode=mode,
                                    **predict_kwargs,
                                )
                                # raw_actions: (num_envs, num_action_chunks, action_dim)
                        finally:
                            if mode == "eval" and module_was_training:
                                self.ddp_model.module.train()
                    if debug_chunk:
                        self._collect_debug_print(
                            (
                                f"chunk[{rollout_idx},{chunk_idx}] predict_action_batch done "
                                f"in {(time.perf_counter() - t0_predict):.3f}s"
                            )
                        )

                    profile_prepare = (
                        self._chunk_profiler.profile("prepare_actions")
                        if self._chunk_profiler is not None
                        else contextlib.nullcontext()
                    )
                    if debug_chunk:
                        self._collect_debug_print(
                            f"chunk[{rollout_idx},{chunk_idx}] entering prepare_actions"
                        )
                    t0_prepare = time.perf_counter()
                    with profile_prepare:
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
                    if debug_chunk:
                        self._collect_debug_print(
                            (
                                f"chunk[{rollout_idx},{chunk_idx}] prepare_actions done "
                                f"in {(time.perf_counter() - t0_prepare):.3f}s"
                            )
                        )

                # ####### TEMP START #######
                if (
                    self._temp_alignment_sample is not None
                    and not self._temp_alignment_compared
                    and self.rank == 0
                ):
                    # TODO(alignment): Mainflow trigger of step_0_* validation.
                    # `_temp_compare_first_action` performs stage-by-stage checks,
                    # including `step_0_model_raw_action` mismatch validation.
                    rollout_payload = rollout_result.get("debug_policy_payload", None)
                    rollout_post_actions = rollout_result.get("debug_action_post", None)
                    rollout_norm_actions = rollout_result.get("norm_actions", None)
                    if rollout_norm_actions is None:
                        raise KeyError("Rollout result missing norm_actions for TEMP alignment")
                    self._temp_compare_first_action(
                        self._temp_alignment_sample,
                        predict_env_obs,
                        rollout_payload,
                        rollout_norm_actions,
                        rollout_post_actions,
                        chunk_actions,
                        self._temp_alignment_log_path,
                    )
                    self._temp_alignment_compared = True
                # ####### TEMP END #######

                profile_env = (
                    self._chunk_profiler.profile("env_chunk_step")
                    if self._chunk_profiler is not None
                    else contextlib.nullcontext()
                )
                if debug_chunk:
                    self._collect_debug_print(
                        f"chunk[{rollout_idx},{chunk_idx}] entering env.chunk_step"
                    )
                t0_env = time.perf_counter()
                with profile_env:
                    obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = \
                        env.chunk_step(chunk_actions)
                    obs = obs_list[-1]
                if debug_chunk:
                    self._collect_debug_print(
                        (
                            f"chunk[{rollout_idx},{chunk_idx}] env.chunk_step done "
                            f"in {(time.perf_counter() - t0_env):.3f}s"
                        )
                    )

                if save_video:
                    for step_idx, step_obs in enumerate(obs_list):
                        main_images = step_obs.get("main_images", None)
                        task_descs = step_obs.get("task_descriptions", None)
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

                profile_metrics = (
                    self._chunk_profiler.profile("compute_metrics")
                    if self._chunk_profiler is not None
                    else contextlib.nullcontext()
                )
                with profile_metrics:
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
                    profile_collect = (
                        self._chunk_profiler.profile("collect_samples_pack")
                        if self._chunk_profiler is not None
                        else contextlib.nullcontext()
                    )
                    if debug_chunk:
                        self._collect_debug_print(
                            f"chunk[{rollout_idx},{chunk_idx}] entering collect_samples_pack"
                        )
                    t0_collect = time.perf_counter()
                    with profile_collect:
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
                    if debug_chunk:
                        self._collect_debug_print(
                            (
                                f"chunk[{rollout_idx},{chunk_idx}] collect_samples_pack done "
                                f"in {(time.perf_counter() - t0_collect):.3f}s"
                            )
                        )

                progress_bar.update(1)

                if self._chunk_profiler is not None:
                    self._chunk_profiler.flush_chunk(
                        payload={
                            "mode": mode,
                            "collect_samples": collect_samples,
                            "rollout_idx": rollout_idx,
                            "chunk_idx": chunk_idx,
                            "chunk_total_s": float(time.perf_counter() - chunk_start),
                            "num_envs": int(env.num_envs),
                            "num_action_chunks": int(chunk_actions.shape[1]),
                        }
                    )
                if debug_chunk:
                    self._collect_debug_print(
                        (
                            f"chunk[{rollout_idx},{chunk_idx}] total "
                            f"{(time.perf_counter() - chunk_start):.3f}s"
                        )
                    )
        finally:
            progress_bar.close()

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
            self._save_eval_videos(frames_by_env, instructions_by_env, video_epoch)
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

        # ####### TEMP START #######
        temp_rollout_only = os.environ.get("RLINF_TEMP_EXIT_AFTER_FIRST_ROLLOUT", "0") == "1"
        if temp_rollout_only:
            temp_num_rollouts = int(os.environ.get("RLINF_TEMP_NUM_ROLLOUTS", "10"))
            temp_max_steps = int(os.environ.get("RLINF_TEMP_MAX_STEPS", "520"))
            temp_chunk_steps = max(1, temp_max_steps // int(self.cfg.actor.model.num_action_chunks))

            rollout_env = self.eval_env if self.eval_env is not None else self.env
            rollout_mode = "eval" if self.eval_env is not None else "train"

            success_rates = []
            done_rates = []
            returns = []
            for rollout_idx in range(temp_num_rollouts):
                _, rollout_metrics = self._collect_rollouts(
                    env=rollout_env,
                    rollout_epoch=1,
                    chunk_steps=temp_chunk_steps,
                    mode=rollout_mode,
                    collect_samples=False,
                    save_video=True,
                    video_epoch=rollout_idx,
                )
                success_rates.append(float(rollout_metrics["success_rate"]))
                done_rates.append(float(rollout_metrics["done_rate"]))
                returns.append(float(rollout_metrics["chunk_return_mean"]))

                if self.rank == 0:
                    print(
                        (
                            f"[TEMP][rollout-only] rollout={rollout_idx + 1}/{temp_num_rollouts} "
                            f"max_steps={temp_max_steps} "
                            f"success_rate={rollout_metrics['success_rate']:.4f} "
                            f"done_rate={rollout_metrics['done_rate']:.4f} "
                            f"chunk_return_mean={rollout_metrics['chunk_return_mean']:.4f} "
                            f"num_chunks={rollout_metrics['num_chunks']:.0f}"
                        ),
                        flush=True,
                    )

            if self.rank == 0:
                overall_success = float(np.mean(success_rates)) if success_rates else 0.0
                overall_done = float(np.mean(done_rates)) if done_rates else 0.0
                overall_return = float(np.mean(returns)) if returns else 0.0
                print(
                    (
                        "[TEMP][rollout-only][summary] "
                        f"num_rollouts={temp_num_rollouts} "
                        f"max_steps={temp_max_steps} "
                        f"avg_success_rate={overall_success:.4f} "
                        f"avg_done_rate={overall_done:.4f} "
                        f"avg_chunk_return_mean={overall_return:.4f}"
                    ),
                    flush=True,
                )
                if self.metric_logger is not None:
                    self.metric_logger.log(
                        {
                            "temp/avg_success_rate": overall_success,
                            "temp/avg_done_rate": overall_done,
                            "temp/avg_chunk_return_mean": overall_return,
                            "temp/num_rollouts": float(temp_num_rollouts),
                            "temp/max_steps": float(temp_max_steps),
                        },
                        step=0,
                    )
                    self.metric_logger.finish()

            dist.barrier()
            dist.destroy_process_group()
            return
        # ####### TEMP END #######

        for epoch in range(self.max_epochs):
            samples, rollout_metrics = self._collect_rollouts(
                env=self.env,
                rollout_epoch=self.rollout_epoch,
                chunk_steps=self.chunk_steps,
                mode="train",
                collect_samples=True,
            )

            # ####### TEMP START #######
            temp_exit_after_first_rollout = os.environ.get(
                "RLINF_TEMP_EXIT_AFTER_FIRST_ROLLOUT", "0"
            ) == "1"
            if temp_exit_after_first_rollout and epoch == 0:
                if self.rank == 0:
                    print(
                        (
                            "[TEMP][rollout-only] first rollout done: "
                            f"success_rate={rollout_metrics['success_rate']:.4f}, "
                            f"done_rate={rollout_metrics['done_rate']:.4f}, "
                            f"chunk_return_mean={rollout_metrics['chunk_return_mean']:.4f}, "
                            f"num_chunks={rollout_metrics['num_chunks']:.0f}"
                        ),
                        flush=True,
                    )
                    if self.metric_logger is not None:
                        self.metric_logger.log(
                            {
                                "temp/rollout_success_rate": rollout_metrics["success_rate"],
                                "temp/rollout_done_rate": rollout_metrics["done_rate"],
                                "temp/rollout_chunk_return_mean": rollout_metrics[
                                    "chunk_return_mean"
                                ],
                                "temp/rollout_num_chunks": rollout_metrics["num_chunks"],
                                "temp/rollout_sample_count": len(samples),
                            },
                            step=0,
                        )
                        self.metric_logger.finish()
                dist.barrier()
                dist.destroy_process_group()
                return
            # ####### TEMP END #######

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
