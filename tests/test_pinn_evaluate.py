"""Tests for src.pinn.evaluate: the saved grids hold the network's values at
the grid points, in the layout every downstream plot assumes."""

import warnings

import numpy as np
import pytest
import torch

from src.pinn.evaluate import evaluate_on_grid, evaluate_over_time, load_model, save_evaluation
from src.pinn.model import PINN


class _Analytic(torch.nn.Module):
    """Stands in for a trained PINN: known u and mu at every (x, y, t)."""

    def __init__(self, upper=(1.0, 2.0, 0.01)):
        super().__init__()
        self.register_buffer("lower_bound", torch.zeros(3))
        self.register_buffer("upper_bound", torch.tensor(upper))

    def forward(self, p):
        x, y, t = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        return torch.cat([x + 10 * y + 100 * t, x * y], dim=1)


def test_evaluate_on_grid_is_ij_indexed_with_u_then_mu():
    X, Y, U, MU = evaluate_on_grid(_Analytic(), t=0.004, nx=5, ny=3)

    assert X.shape == Y.shape == U.shape == MU.shape == (5, 3)
    # indexing="ij": x varies along axis 0, y along axis 1, over the model's own bounds
    np.testing.assert_allclose(X, np.repeat(np.linspace(0.0, 1.0, 5)[:, None], 3, axis=1))
    np.testing.assert_allclose(Y, np.repeat(np.linspace(0.0, 2.0, 3)[None, :], 5, axis=0))
    np.testing.assert_allclose(U, X + 10 * Y + 100 * 0.004, rtol=1e-6)  # output column 0
    np.testing.assert_allclose(MU, X * Y, rtol=1e-6, atol=1e-7)  # output column 1


def test_evaluate_over_time_stacks_snapshots_in_the_given_order():
    t_values = np.array([0.0, 0.01, 0.005])
    X, Y, U, MU = evaluate_over_time(_Analytic(), t_values, nx=4, ny=4)
    assert U.shape == MU.shape == (3, 4, 4)
    for k, t in enumerate(t_values):
        np.testing.assert_allclose(U[k], X + 10 * Y + 100 * t, rtol=1e-6)


def test_evaluate_warns_outside_the_trained_time_window():
    model = _Analytic()
    with pytest.warns(UserWarning, match="extrapolation"):
        evaluate_on_grid(model, t=0.02, nx=3, ny=3)
    with pytest.warns(UserWarning, match="extrapolation"):
        evaluate_on_grid(model, t=-0.001, nx=3, ny=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluate_on_grid(model, t=0.0, nx=3, ny=3)
        evaluate_on_grid(model, t=0.01, nx=3, ny=3)  # t_max itself, stored in float32


def test_load_model_restores_architecture_bounds_and_weights(tmp_path):
    torch.manual_seed(0)
    model = PINN(
        lower_bound=(0.0, 0.0, 0.0), upper_bound=(1.0, 1.0, 0.01), input_dim=3, output_dim=2,
        hidden_layers=2, hidden_width=8,
    )
    args = {"x_max": 1.0, "y_max": 1.0, "t_max": 0.01, "hidden_layers": 2, "hidden_width": 8}
    torch.save({"model_state": model.state_dict(), "args": args, "history": {}}, tmp_path / "final.pt")

    loaded = load_model(str(tmp_path / "final.pt"))
    points = torch.rand(20, 3) * torch.tensor([1.0, 1.0, 0.01])
    with torch.no_grad():
        torch.testing.assert_close(loaded(points), model(points))
    assert loaded(points).shape == (20, 2)


def test_save_evaluation_round_trip(tmp_path):
    X, Y, U, MU = evaluate_over_time(_Analytic(), np.array([0.0, 0.005]), nx=4, ny=3)
    path = tmp_path / "sub" / "evaluation.npz"
    save_evaluation(str(path), X, Y, np.array([0.0, 0.005]), U, MU)
    with np.load(path) as data:
        assert sorted(data.files) == ["mu", "t", "u", "x", "y"]
        np.testing.assert_array_equal(data["u"], U)
        np.testing.assert_array_equal(data["mu"], MU)
        np.testing.assert_array_equal(data["x"], X)
