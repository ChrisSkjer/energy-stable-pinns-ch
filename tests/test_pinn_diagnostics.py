"""Tests for src.pinn.diagnostics."""

import numpy as np
import pytest
import torch


def test_import():
    from src.pinn import diagnostics  # noqa: F401


def _small_model():
    from src.pinn.model import PINN

    return PINN(
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(1.0, 1.0, 1.0),
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_width=8,
    )


def test_field_and_gradient_chunking_is_exact():
    from src.pinn.diagnostics import field_and_gradient

    model = _small_model()
    u_a, grad_a = field_and_gradient(model, t=0.3, nx=13, ny=11, chunk_size=7)
    u_b, grad_b = field_and_gradient(model, t=0.3, nx=13, ny=11, chunk_size=10**9)

    assert u_a.shape == (13, 11)
    assert grad_a.shape == (13, 11, 2)
    # float32 matmul isn't strictly associative across different batch
    # groupings, so chunked vs. unchunked differ at float32 precision --
    # "exact" here means that noise floor, not bit-identical.
    assert np.allclose(u_a, u_b, rtol=1e-5, atol=1e-6)
    assert np.allclose(grad_a, grad_b, rtol=1e-5, atol=1e-6)
    assert np.isfinite(u_a).all()
    assert np.isfinite(grad_a).all()


def test_field_and_gradient_matches_finite_differences():
    from src.pinn.diagnostics import field_and_gradient

    model = _small_model()
    nx = ny = 201
    u, grad_u = field_and_gradient(model, t=0.5, nx=nx, ny=ny)

    h = 1.0 / (nx - 1)
    fd_dudx = np.gradient(u, h, axis=0)
    interior = np.s_[5:-5, 5:-5]
    assert np.allclose(grad_u[..., 0][interior], fd_dudx[interior], rtol=0.05, atol=1e-2)


class _Front(torch.nn.Module):
    """A stand-in PINN with a known u: the equilibrium tanh front
    u = tanh((x - x0(t)) / (sqrt(2) eps)), x0 = 0.5 + 10 t, moving in x."""

    def __init__(self, epsilon):
        super().__init__()
        self.width = np.sqrt(2.0) * epsilon
        self.register_buffer("lower_bound", torch.zeros(3))
        self.register_buffer("upper_bound", torch.tensor([1.0, 1.0, 0.01]))

    def forward(self, p):
        u = torch.tanh((p[:, 0:1] - 0.5 - 10 * p[:, 2:3]) / self.width)
        return torch.cat([u, torch.zeros_like(u)], dim=1)


def test_field_and_gradient_columns_are_du_dx_then_du_dy():
    from src.pinn.diagnostics import field_and_gradient

    class Plane(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("lower_bound", torch.zeros(3))
            self.register_buffer("upper_bound", torch.ones(3))

        def forward(self, p):
            u = 2 * p[:, 0:1] + 3 * p[:, 1:2] + 5 * p[:, 2:3]
            return torch.cat([u, 7 * u], dim=1)  # mu must not leak into grad u

    u, grad_u = field_and_gradient(Plane(), t=0.5, nx=4, ny=3)
    X, Y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3), indexing="ij")
    np.testing.assert_allclose(u, 2 * X + 3 * Y + 2.5, rtol=1e-6)
    np.testing.assert_allclose(grad_u[..., 0], 2.0, rtol=1e-6)
    np.testing.assert_allclose(grad_u[..., 1], 3.0, rtol=1e-6)


def test_diagnostics_over_time_match_the_analytic_front():
    """Energy of the equilibrium front is its surface tension times its length,
    2*sqrt(2)*eps/3 * 1; mass is the integral of the tanh, w*ln(cosh((1-x0)/w)/cosh(x0/w))."""
    from src.pinn.diagnostics import diagnostics_over_time

    eps = 0.05
    model = _Front(eps)
    t, energy, mass = diagnostics_over_time(model, np.array([0.0, 0.01]), nx=201, ny=201, epsilon=eps)

    np.testing.assert_allclose(energy, 2 * np.sqrt(2) * eps / 3, rtol=2e-3)
    w = model.width
    x0 = 0.5 + 10 * t
    np.testing.assert_allclose(mass, w * np.log(np.cosh((1 - x0) / w) / np.cosh(x0 / w)), atol=2e-6)
    assert mass[1] == pytest.approx(-0.2, abs=1e-3)  # front moved right: more of the -1 phase


def test_diagnostics_over_time_shapes():
    from src.pinn.diagnostics import diagnostics_over_time

    model = _small_model()
    t_values = np.linspace(0.0, 1.0, 5)
    t, energy, mass = diagnostics_over_time(model, t_values, nx=21, ny=21, epsilon=0.05)

    assert t.shape == energy.shape == mass.shape == (5,)
    assert np.isfinite(energy).all()
    assert np.isfinite(mass).all()


def test_save_and_load_diagnostics_round_trip(tmp_path):
    from src.common.plotting import load_diagnostics
    from src.pinn.diagnostics import save_diagnostics

    path = tmp_path / "diagnostics.csv"
    t = np.array([0.0, 0.5, 1.0])
    energy = np.array([0.1, 0.09, 0.08])
    mass = np.array([-0.6, -0.6001, -0.5999])
    save_diagnostics(str(path), t, energy, mass)

    with open(path, "rb") as fh:
        raw = fh.read()
    assert raw.startswith(b"t,free_energy,total_mass\n")
    assert b"\r" not in raw

    t_out, energy_out, mass_out = load_diagnostics(str(path))
    assert np.allclose(t_out, t)
    assert np.allclose(energy_out, energy)
    assert np.allclose(mass_out, mass)


def test_load_diagnostics_single_row(tmp_path):
    from src.common.plotting import load_diagnostics
    from src.pinn.diagnostics import save_diagnostics

    path = tmp_path / "diagnostics.csv"
    save_diagnostics(str(path), [0.0], [0.1], [-0.6])

    t_out, energy_out, mass_out = load_diagnostics(str(path))
    assert t_out.shape == energy_out.shape == mass_out.shape == (1,)
