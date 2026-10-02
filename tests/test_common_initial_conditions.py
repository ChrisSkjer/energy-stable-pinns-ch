"""Tests for src.common.initial_conditions (numpy only, no torch or DOLFINx)."""

import numpy as np
import pytest

from src.common.initial_conditions import INITIAL_CONDITIONS, initial_condition

EPS = 0.05


def _grid(n=101):
    return np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="ij")


@pytest.mark.parametrize("name", sorted(INITIAL_CONDITIONS))
def test_shape_and_range(name):
    xx, yy = _grid()
    u = initial_condition(name, xx, yy, epsilon=EPS)
    assert u.shape == xx.shape
    assert np.all(np.abs(u) <= 1.0)


# The tanh interface (width ~sqrt(2)*eps ~ 0.07) is comparable to the arm
# half-width and circle radii, so centres sit near, not at, +1.


def test_cross_inside_and_outside():
    u = initial_condition("cross", np.array([0.5, 0.0]), np.array([0.5, 0.0]), epsilon=EPS)
    assert u[0] > 0.85
    assert u[1] == pytest.approx(-1.0, abs=1e-3)


def test_cross_matches_previous_formula():
    # The cross used to be hardcoded in sampling.py and cahn_hilliard.py;
    # keep old runs reproducible.
    xx, yy = _grid()
    px, py = xx - 0.5, yy - 0.5

    def rect(hw, hh):
        qx, qy = np.abs(px) - hw, np.abs(py) - hh
        return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)

    half_w, half_l = (6 / 32) / 2, (20 / 32) / 2
    expected = -np.tanh(np.minimum(rect(half_l, half_w), rect(half_w, half_l)) / (np.sqrt(2) * EPS))
    np.testing.assert_allclose(initial_condition("cross", xx, yy, epsilon=EPS), expected)


def test_two_circles_inside_and_between():
    x = np.array([0.28, 0.74, 0.53])
    y = np.array([0.5, 0.5, 0.5])
    u = initial_condition("two_circles", x, y, epsilon=EPS)
    assert u[0] > 0.85
    assert u[1] > 0.85
    assert u[2] < -0.85


def test_noise_is_deterministic_and_seeded():
    xx, yy = _grid()
    a = initial_condition("noise", xx, yy, epsilon=EPS, seed=1)
    b = initial_condition("noise", xx, yy, epsilon=EPS, seed=1)
    c = initial_condition("noise", xx, yy, epsilon=EPS, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_noise_is_small_and_mean_free():
    xx, yy = _grid(401)
    u = initial_condition("noise", xx, yy, epsilon=EPS)
    assert np.max(np.abs(u)) <= 0.05 + 1e-12
    assert np.abs(u).max() > 0.0
    assert np.trapezoid(np.trapezoid(u, dx=1 / 400), dx=1 / 400) == pytest.approx(0.0, abs=1e-6)


def test_noise_is_point_set_independent():
    # The PINN grid and the FEM mesh nodes are different point sets; a value
    # at a given (x, y) must not depend on what else is evaluated with it.
    full = initial_condition("noise", np.array([0.1, 0.37, 0.9]), np.array([0.2, 0.5, 0.8]), epsilon=EPS)
    single = initial_condition("noise", np.array([0.37]), np.array([0.5]), epsilon=EPS)
    assert full[1] == pytest.approx(single[0])


def test_noise_has_zero_normal_derivative_at_walls():
    h = 1e-6
    s = np.linspace(0.0, 1.0, 11)
    for wall in (0.0, 1.0):
        inward = h if wall == 0.0 else -h
        w = np.full_like(s, wall)
        du_dx = (initial_condition("noise", w + inward, s, epsilon=EPS) - initial_condition("noise", w, s, epsilon=EPS)) / h
        du_dy = (initial_condition("noise", s, w + inward, epsilon=EPS) - initial_condition("noise", s, w, epsilon=EPS)) / h
        assert np.max(np.abs(du_dx)) < 1e-3
        assert np.max(np.abs(du_dy)) < 1e-3


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="cross"):
        initial_condition("bogus", np.zeros(1), np.zeros(1), epsilon=EPS)
