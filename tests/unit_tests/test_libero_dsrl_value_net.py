from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf_noray.runners.libero_dsrl_ddp_runner import (  # noqa: E402
    DSRLDistributionalValueNet,
    DSRLScalarValueNet,
)


def test_scalar_value_net_predict_matches_forward() -> None:
    model = DSRLScalarValueNet(state_dim=3, hidden_dim=8)
    states = torch.randn(5, 3)

    output = model(states)
    predicted = model.predict_value(states)

    assert output.shape == (5,)
    assert torch.allclose(output, predicted)


def test_scalar_value_net_compute_loss_matches_mse() -> None:
    model = DSRLScalarValueNet(state_dim=2, hidden_dim=4)
    states = torch.randn(6, 2)
    targets = torch.randn(6)

    output = model(states)
    loss = model.compute_loss(states, targets)
    expected = torch.nn.functional.mse_loss(output, targets)

    assert torch.allclose(loss, expected)


def test_distributional_value_net_bin_centers_match_config() -> None:
    model = DSRLDistributionalValueNet(
        state_dim=2,
        hidden_dim=4,
        v_min=0.0,
        v_max=1.0,
        n_bins=16,
    )

    expected = torch.linspace(0.0, 1.0, 16)
    assert torch.allclose(model.bin_centers, expected)


def test_distributional_value_net_target_bins_clamp_to_support() -> None:
    model = DSRLDistributionalValueNet(
        state_dim=2,
        hidden_dim=4,
        v_min=0.0,
        v_max=1.0,
        n_bins=16,
    )
    targets = torch.tensor([-1.0, 0.0, 0.49, 1.0, 2.0], dtype=torch.float32)

    indices = model.target_to_bin_indices(targets)

    assert int(indices[0].item()) == 0
    assert int(indices[1].item()) == 0
    assert 0 <= int(indices[2].item()) < 16
    assert int(indices[3].item()) == 15
    assert int(indices[4].item()) == 15


def test_distributional_value_net_predicts_expected_value_from_logits() -> None:
    model = DSRLDistributionalValueNet(
        state_dim=2,
        hidden_dim=4,
        v_min=0.0,
        v_max=1.0,
        n_bins=16,
    )
    logits = torch.full((2, 16), fill_value=-100.0)
    logits[0, 3] = 100.0
    logits[1] = 0.0

    predicted = model.predict_value_from_output(logits)

    assert torch.allclose(predicted[0], model.bin_centers[3], atol=1e-6)
    assert torch.allclose(predicted[1], torch.tensor(0.5), atol=1e-6)


def test_distributional_value_net_compute_loss_runs_on_scalar_targets() -> None:
    model = DSRLDistributionalValueNet(
        state_dim=2,
        hidden_dim=4,
        v_min=0.0,
        v_max=1.0,
        n_bins=16,
    )
    states = torch.randn(8, 2)
    targets = torch.linspace(-0.5, 1.5, 8)

    loss = model.compute_loss(states, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)