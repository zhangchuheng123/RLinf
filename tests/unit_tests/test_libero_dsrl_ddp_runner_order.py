from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf_noray.runners.libero_dsrl_ddp_runner import LiberoDSRLDDPNoRayRunner


def test_flat_env_time_roundtrip_preserves_order() -> None:
    num_envs = 3
    num_steps = 4
    feat_dim = 2

    flat = torch.arange(num_envs * num_steps * feat_dim, dtype=torch.float32).reshape(
        num_envs * num_steps, feat_dim
    )

    env_time = LiberoDSRLDDPNoRayRunner._flat_to_env_time(flat, num_envs)
    recovered = LiberoDSRLDDPNoRayRunner._env_time_to_flat(env_time)

    assert torch.equal(recovered, flat)

    expected = flat.reshape(num_steps, num_envs, feat_dim).permute(1, 0, 2).contiguous()
    assert torch.equal(env_time, expected)


def test_gae_env_time_then_flat_aligns_with_flat_transition_order() -> None:
    runner = LiberoDSRLDDPNoRayRunner.__new__(LiberoDSRLDDPNoRayRunner)
    runner.gamma = 0.99
    runner.gae_lambda = 0.95
    runner.local_num_envs = 2

    num_envs = runner.local_num_envs
    num_steps = 3

    rewards_flat = torch.tensor([1.0, 10.0, 2.0, 20.0, 3.0, 30.0], dtype=torch.float32)
    values_flat = torch.tensor([0.1, 1.0, 0.2, 2.0, 0.3, 3.0], dtype=torch.float32)
    next_values_flat = torch.tensor([0.2, 1.1, 0.3, 2.1, 0.4, 3.1], dtype=torch.float32)
    dones_flat = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    rewards_nt = runner._flat_to_env_time(rewards_flat, num_envs)
    values_nt = runner._flat_to_env_time(values_flat, num_envs)
    next_values_nt = runner._flat_to_env_time(next_values_flat, num_envs)
    dones_nt = runner._flat_to_env_time(dones_flat, num_envs)

    advantages_flat, returns_flat = runner._compute_gae(
        rewards_flat,
        values_flat,
        next_values_flat,
        dones_flat,
    )

    manual_adv = torch.zeros_like(rewards_nt)
    for env_idx in range(num_envs):
        gae = torch.tensor(0.0, dtype=torch.float32)
        for step_idx in reversed(range(num_steps)):
            not_done = 1.0 - dones_nt[env_idx, step_idx]
            delta = (
                rewards_nt[env_idx, step_idx]
                + runner.gamma * next_values_nt[env_idx, step_idx] * not_done
                - values_nt[env_idx, step_idx]
            )
            gae = delta + runner.gamma * runner.gae_lambda * not_done * gae
            manual_adv[env_idx, step_idx] = gae

    manual_ret = manual_adv + values_nt

    advantages_nt = runner._flat_to_env_time(advantages_flat, num_envs)
    returns_nt = runner._flat_to_env_time(returns_flat, num_envs)

    assert torch.allclose(advantages_nt, manual_adv, atol=1e-6)
    assert torch.allclose(returns_nt, manual_ret, atol=1e-6)

    expected_adv_flat = manual_adv.permute(1, 0).reshape(-1)
    expected_ret_flat = manual_ret.permute(1, 0).reshape(-1)

    assert torch.allclose(advantages_flat, expected_adv_flat, atol=1e-6)
    assert torch.allclose(returns_flat, expected_ret_flat, atol=1e-6)
