from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf_noray.models.embodiment.modules.gaussian_policy import GaussianPolicy


def test_sample_and_evaluate_actions_logprob_are_consistent() -> None:
    torch.manual_seed(0)

    batch_size = 8
    input_dim = 32
    output_dim = 64

    policy = GaussianPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=(64, 64),
        action_horizon=1,
    )

    features = torch.randn(batch_size, input_dim, dtype=torch.float32)
    sampled_actions, sampled_logprob = policy.sample(features, deterministic=False)

    eval_logprob, _ = policy.evaluate_actions(features, sampled_actions)

    assert torch.isfinite(sampled_logprob).all()
    assert torch.isfinite(eval_logprob).all()
    assert torch.equal(eval_logprob, sampled_logprob)
