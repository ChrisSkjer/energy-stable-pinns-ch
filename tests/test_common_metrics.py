"""Tests for src.common.metrics."""

import numpy as np
import pytest


def test_import():
    from src.common import metrics  # noqa: F401

    assert hasattr(metrics, "relative_l2_error")
    assert hasattr(metrics, "free_energy")
    assert hasattr(metrics, "mass_conservation_error")


def test_trapezoid_weights_2d_sums_to_area():
    from src.common.metrics import trapezoid_weights_2d

    weight = trapezoid_weights_2d(5, 7)
    assert weight.shape == (5, 7)
    assert weight.sum() == pytest.approx(1.0, abs=1e-12)
    assert weight[0, 0] == pytest.approx(0.25 / (4 * 6))


def test_trapezoid_weights_2d_rejects_boundary_double_counting():
    """A uniform 1/n**2 weight on an endpoint-inclusive grid (as in
    notebooks/pinn_CH_imp1.ipynb's ch_diagnostics) double-counts the
    boundary and is only first-order accurate -- this regression guard
    fails for that weight and passes for the correct trapezoid one."""
    from src.common.metrics import trapezoid_weights_2d

    n = 101
    x = np.linspace(0.0, 1.0, n)
    X, _ = np.meshgrid(x, x, indexing="ij")
    integral = np.sum(X**2 * trapezoid_weights_2d(n, n))
    assert integral == pytest.approx(1.0 / 3.0, abs=2e-5)


def test_free_energy_constant_fields():
    from src.common.metrics import free_energy, trapezoid_weights_2d

    weight = trapezoid_weights_2d(8, 8)
    zero_grad = np.zeros((8, 8, 2))

    assert free_energy(np.ones((8, 8)), zero_grad, 0.01, weight) == pytest.approx(0.0, abs=1e-12)
    assert free_energy(np.zeros((8, 8)), zero_grad, 0.01, weight) == pytest.approx(0.25, abs=1e-12)


def test_free_energy_surface_tension():
    """Integrated interfacial energy of a straight tanh front should match
    the analytic 1D surface tension sigma = 2*sqrt(2)*epsilon/3."""
    from src.common.metrics import free_energy, trapezoid_weights_2d

    epsilon = 0.05
    nx = ny = 201
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.tanh((X - 0.5) / (np.sqrt(2.0) * epsilon))
    du_dx = (1.0 - u**2) / (np.sqrt(2.0) * epsilon)
    grad_u = np.stack([du_dx, np.zeros_like(u)], axis=-1)

    weight = trapezoid_weights_2d(nx, ny)
    energy = free_energy(u, grad_u, epsilon, weight)
    assert energy == pytest.approx(2 * np.sqrt(2) * epsilon / 3, rel=0.01)


def test_free_energy_rejects_shape_mismatch():
    from src.common.metrics import free_energy

    with pytest.raises(ValueError):
        free_energy(np.zeros((4, 4)), np.zeros((5, 4, 2)), 0.01, 1.0)


def test_total_mass():
    from src.common.metrics import total_mass, trapezoid_weights_2d

    weight = trapezoid_weights_2d(10, 10)
    ones = np.ones((10, 10))
    assert total_mass(ones, weight) == pytest.approx(1.0)
