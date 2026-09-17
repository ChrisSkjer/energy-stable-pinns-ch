"""Tests for src.pinn.diagnostics."""

import numpy as np


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
