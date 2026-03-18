import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf_noray.models.embodiment.smolvla.smolvla_policy import SmolVLAForRLActionPrediction


class IdentityPreprocessor:
    def __call__(self, batch):
        return batch


class ShiftPostprocessor:
    def __init__(self, shift: float = 10.0):
        self.shift = shift

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action + self.shift


class FakeSmolVLAPolicy(torch.nn.Module):
    def __init__(self, invalid_normalizer: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        if invalid_normalizer:
            self.register_buffer("normalizer_std", torch.tensor([float("inf")]))
        else:
            self.register_buffer("normalizer_std", torch.tensor([1.0]))

        self.config = types.SimpleNamespace(
            device="cpu",
            image_features={
                "observation.images.image": None,
                "observation.images.image2": None,
            },
        )
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action(self, batch):
        self.select_calls += 1
        batch_size = batch["observation.state"].shape[0]
        action_dim = 7
        return torch.full((batch_size, action_dim), float(self.select_calls), device=self.weight.device)

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        del time
        action = batch["action"]
        if noise is None:
            noise = batch["noise"]
        losses = ((action - noise) ** 2).mean(dim=(1, 2))
        if reduction == "none":
            return losses, {"loss": losses.mean().item()}
        return losses.mean(), {"loss": losses.mean().item()}


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "action_dim": 7,
            "num_action_chunks": 3,
            "add_value_head": False,
            "image_keys": ["image", "image2"],
            "main_image_env_key": "main_images",
            "wrist_image_env_key": "wrist_images",
        }
    )


@pytest.fixture
def env_obs():
    batch_size = 2
    h, w = 4, 4
    return {
        "main_images": np.random.randint(0, 255, size=(batch_size, h, w, 3), dtype=np.uint8),
        "wrist_images": np.random.randint(0, 255, size=(batch_size, h, w, 3), dtype=np.uint8),
        "states": np.random.randn(batch_size, 8).astype(np.float32),
        "task_descriptions": ["open drawer", "close drawer"],
    }


def test_predict_action_batch_uses_chunkwise_select_action_and_postprocess(monkeypatch, cfg, env_obs):
    monkeypatch.setattr(
        SmolVLAForRLActionPrediction,
        "_build_policy_processors",
        lambda self, cfg: (IdentityPreprocessor(), ShiftPostprocessor(shift=10.0)),
    )

    fake_policy = FakeSmolVLAPolicy()
    model = SmolVLAForRLActionPrediction(cfg=cfg, policy=fake_policy)

    raw_actions, result = model.predict_action_batch(env_obs)

    assert fake_policy.reset_calls == 1
    assert fake_policy.select_calls == cfg.num_action_chunks
    assert raw_actions.shape == (2, cfg.num_action_chunks, cfg.action_dim)
    assert np.allclose(raw_actions[:, 0, :], 11.0)
    assert np.allclose(raw_actions[:, 1, :], 12.0)
    assert np.allclose(raw_actions[:, 2, :], 13.0)
    assert result["norm_actions"].shape == (2, cfg.num_action_chunks, cfg.action_dim)


def test_default_forward_returns_model_based_logprobs(monkeypatch, cfg, env_obs):
    monkeypatch.setattr(
        SmolVLAForRLActionPrediction,
        "_build_policy_processors",
        lambda self, cfg: (IdentityPreprocessor(), ShiftPostprocessor(shift=0.0)),
    )

    fake_policy = FakeSmolVLAPolicy()
    model = SmolVLAForRLActionPrediction(cfg=cfg, policy=fake_policy)

    _, rollout_info = model.predict_action_batch(env_obs)
    forward_out = model.default_forward(rollout_info)

    expected = -((rollout_info["norm_actions"] - rollout_info["noise"]) ** 2).mean(dim=(1, 2))
    assert torch.allclose(forward_out["logprobs"].cpu(), expected, atol=1e-6)


def test_invalid_normalizer_stats_raise(monkeypatch, cfg):
    monkeypatch.setattr(
        SmolVLAForRLActionPrediction,
        "_build_policy_processors",
        lambda self, cfg: (IdentityPreprocessor(), ShiftPostprocessor(shift=0.0)),
    )

    with pytest.raises(ValueError, match="Invalid normalization stats"):
        SmolVLAForRLActionPrediction(cfg=cfg, policy=FakeSmolVLAPolicy(invalid_normalizer=True))
