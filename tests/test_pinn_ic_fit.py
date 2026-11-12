"""Tests for src.pinn.ic_fit (--hard-ic-fit): the fitted u0 is accurate,
smooth, frozen, saved with the PINN and restored by load_model."""

import os

import pytest
import torch

from src.pinn.ic_fit import FittedICField, fit_ic_field
from src.pinn.ic_torch import InitialConditionField, hard_ic_kwargs
from src.pinn.model import PINN


def test_fresh_field_is_zero_and_has_no_parameters():
    field = FittedICField(torch.zeros(2), torch.ones(2), hidden_layers=2, hidden_width=8)
    assert list(field.parameters()) == []
    assert torch.all(field(torch.rand(5, 2)) == 0)
    # weights are saved, the bounds come from the run args
    assert set(field.state_dict()) == {"weight0", "bias0", "weight1", "bias1", "weight2", "bias2"}


def test_fit_matches_a_smooth_1d_ic_and_leaves_the_global_rng_alone():
    target = InitialConditionField(1, "step", 0.05)  # O(1) amplitude, unlike "cosine"
    field = FittedICField(torch.zeros(1), torch.ones(1), hidden_layers=2, hidden_width=20)
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    stats = fit_ic_field(field, target, epsilon=0.05, n_points=512, adam_epochs=300, lbfgs_steps=30, log_every=0)
    assert torch.rand(1) == before
    assert stats["rel_l2"] < 1e-2
    x = torch.rand(100, 1)
    torch.testing.assert_close(field(x), target(x), atol=5e-2, rtol=0)


def test_fit_is_smooth_where_the_cross_is_not():
    """The cross u0 has eps^2 Lap u0 ~ 1/r at its convex corners; the fit stays bounded."""
    target = InitialConditionField(2, "cross", 0.05)
    field = FittedICField(torch.zeros(2), torch.ones(2), hidden_layers=2, hidden_width=16)
    stats = fit_ic_field(field, target, epsilon=0.05, n_points=1024, adam_epochs=200, lbfgs_steps=10, log_every=0)
    assert stats["eps2_lap_fit_max"] < stats["eps2_lap_exact_max"]


def test_hard_ic_kwargs_builds_a_fitted_field_of_the_run_size():
    args = {"hard_ic": True, "hard_ic_fit": True, "hard_ic_fit_layers": 3, "hard_ic_fit_width": 12,
            "dim": 2, "x_max": 1.0, "y_max": 2.0, "ic": "cross", "epsilon": 0.05}
    field = hard_ic_kwargs(args)["hard_ic"]
    assert isinstance(field, FittedICField)
    assert field.hidden_layers == 3 and field.hidden_width == 12
    torch.testing.assert_close(field.upper, torch.tensor([1.0, 2.0]))


def test_pinn_with_a_fitted_field_keeps_its_own_parameters_and_saves_the_fit():
    field = FittedICField(torch.zeros(2), torch.ones(2), hidden_layers=2, hidden_width=8)
    plain = PINN(lower_bound=(0.0, 0.0, 0.0), upper_bound=(1.0, 1.0, 0.01), output_dim=2, hidden_layers=2, hidden_width=8)
    model = PINN(lower_bound=(0.0, 0.0, 0.0), upper_bound=(1.0, 1.0, 0.01), output_dim=2,
                 hidden_layers=2, hidden_width=8, hard_ic=field)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in plain.parameters())
    assert "hard_ic.weight0" in model.state_dict()


def test_smoke_run_with_hard_ic_fit_round_trips_through_load_model(monkeypatch, tmp_path):
    from src.pinn import train as train_module
    from src.pinn.evaluate import load_model

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--smoke-test", "--device", "cpu", "--run-name", "fit", "--hard-ic", "--hard-ic-fit",
         "--hard-ic-fit-layers", "2", "--hard-ic-fit-width", "8", "--heldout-every", "5",
         "--heldout-n", "50", "--heldout-n-ic", "20", "--heldout-n-bc", "5"],
    )
    train_module.train(train_module.parse_args())
    path = os.path.join("results", "pinn_models", "fit", "final.pt")
    checkpoint = torch.load(path, weights_only=False)
    assert "rel_l2" in checkpoint["hard_ic_fit"]
    model = load_model(path)
    assert isinstance(model.hard_ic, FittedICField)
    assert torch.any(model.hard_ic.weight0 != 0)
    x = torch.rand(10, 2)
    t0 = torch.cat([x, torch.zeros(10, 1)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(model(t0)[:, 0:1], model.hard_ic(x))


def test_hard_ic_fit_needs_hard_ic(monkeypatch):
    from src.pinn.train import parse_args

    monkeypatch.setattr("sys.argv", ["train", "--hard-ic-fit"])
    with pytest.raises(SystemExit):
        parse_args()
