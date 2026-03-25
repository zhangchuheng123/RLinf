from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf_noray.runners.libero_dsrl_ddp_runner import ReplayBuffer, Transition


class DummyValueModel(torch.nn.Module):
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states[:, 0]


def _make_transition(
    state: float,
    next_state: float,
    action: float,
    old_logprob: float,
    reward: float,
    done: bool,
) -> Transition:
    return Transition(
        state=torch.tensor([state], dtype=torch.float32),
        next_state=torch.tensor([next_state], dtype=torch.float32),
        action=torch.tensor([action], dtype=torch.float32),
        old_logprob=torch.tensor(old_logprob, dtype=torch.float32),
        reward=torch.tensor(reward, dtype=torch.float32),
        done=torch.tensor(done, dtype=torch.bool),
    )


def test_replay_buffer_marks_completed_trajectories_and_mc_returns() -> None:
    buffer = ReplayBuffer(capacity=6, num_envs=2, verbose=False)

    transitions = [
        _make_transition(0.0, 1.0, 0.0, -0.1, 1.0, False),
        _make_transition(10.0, 11.0, 0.0, -0.2, 10.0, False),
        _make_transition(1.0, 2.0, 0.0, -0.3, 2.0, True),
        _make_transition(11.0, 12.0, 0.0, -0.4, 20.0, True),
    ]
    buffer.add_rollout(transitions, gamma=1.0)

    assert len(buffer.data) == 4
    assert all(transition.effective for transition in buffer.data)

    env0 = buffer.env_queues[0]
    env1 = buffer.env_queues[1]
    assert [round(t.returns, 6) for t in env0] == [3.0, 2.0]
    assert [round(t.returns, 6) for t in env1] == [30.0, 20.0]


def test_replay_buffer_gae_targets_and_sampling_only_use_effective_entries() -> None:
    buffer = ReplayBuffer(capacity=8, num_envs=2, verbose=False)

    complete = [
        _make_transition(0.5, 1.5, 0.0, -0.1, 1.0, False),
        _make_transition(2.0, 3.0, 0.0, -0.1, 4.0, False),
        _make_transition(1.5, 2.5, 0.0, -0.1, 2.0, True),
        _make_transition(3.0, 4.0, 0.0, -0.1, 5.0, True),
    ]
    incomplete = [
        _make_transition(7.0, 8.0, 0.0, -0.1, 0.5, False),
        _make_transition(9.0, 10.0, 0.0, -0.1, 0.7, False),
    ]

    buffer.add_rollout(complete, gamma=0.99)
    buffer.add_rollout(incomplete, gamma=0.99)

    prepared = buffer.prepare_gae_targets(
        value_model=DummyValueModel(),
        device=torch.device("cpu"),
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert prepared == 4

    effective = [transition for transition in buffer.data if transition.effective]
    ineffective = [transition for transition in buffer.data if not transition.effective]
    assert len(effective) == 4
    assert len(ineffective) == 2

    advantages = torch.tensor([transition.advantage for transition in effective], dtype=torch.float32)
    assert abs(float(advantages.mean().item())) < 1e-5
    assert float(advantages.std(unbiased=False).item()) > 0.9

    batch = buffer.sample(batch_size=16)
    assert len(batch) == 16
    assert all(transition.effective for transition in batch)
