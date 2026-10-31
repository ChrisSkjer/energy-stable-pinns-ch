"""Tests for src.pinn.collocation."""

import pytest
import torch

from src.pinn.collocation import (
    COLLOCATION_STRATEGIES,
    default_grid_shape,
    heldout_points,
    make_sampler,
    with_time_window,
)

LOWER = (0.0, 0.0, 0.0)
UPPER = (1.0, 2.0, 1e-3)


@pytest.mark.parametrize("name", sorted(COLLOCATION_STRATEGIES))
def test_points_shape_dtype_and_bounds(name):
    pts = make_sampler(name, LOWER, UPPER, 1000).sample()
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.dtype == torch.float32
    lo, hi = torch.tensor(LOWER), torch.tensor(UPPER)
    assert torch.all(pts >= lo) and torch.all(pts <= hi)


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_random_strategies_return_n_points(name):
    assert make_sampler(name, LOWER, UPPER, 1000).sample().shape == (1000, 3)


def test_grid_shape_and_cell_centres():
    pts = make_sampler("grid", LOWER, UPPER, 0, grid_shape=(4, 5, 6)).sample()
    assert pts.shape == (4 * 5 * 6, 3)
    assert pts[:, 0].unique().numel() == 4
    assert pts[:, 1].unique().numel() == 5
    assert pts[:, 2].unique().numel() == 6
    # Cell centres: nothing on t=0 or on a wall, where the IC/BC points are.
    assert pts[:, 2].min() > 0
    assert pts[:, 0].min() > 0 and pts[:, 0].max() < 1


def test_default_grid_shape_is_a_cube_near_n():
    assert default_grid_shape(20000) == (27, 27, 27)
    assert make_sampler("grid", LOWER, UPPER, 1000).sample().shape == (1000, 3)


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_same_seed_same_points_different_seed_different_points(name):
    a = make_sampler(name, LOWER, UPPER, 200, seed=1).sample()
    b = make_sampler(name, LOWER, UPPER, 200, seed=1).sample()
    c = make_sampler(name, LOWER, UPPER, 200, seed=2).sample()
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_resample_only_on_schedule(name):
    sampler = make_sampler(name, LOWER, UPPER, 200, resample_every=5)
    first = sampler.sample()
    assert sampler.update(None, 0) is None
    assert sampler.update(None, 3) is None
    new = sampler.update(None, 5)
    assert new is not None and new.shape == first.shape
    assert not torch.allclose(new, first)


@pytest.mark.parametrize("name", sorted(COLLOCATION_STRATEGIES))
def test_no_resampling_by_default(name):
    sampler = make_sampler(name, LOWER, UPPER, 200)
    sampler.sample()
    assert all(sampler.update(None, step) is None for step in range(1, 20))


def test_grid_rejects_resampling():
    with pytest.raises(ValueError):
        make_sampler("grid", LOWER, UPPER, 200, resample_every=10)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        make_sampler("halton", LOWER, UPPER, 200)


def _max_cell_count(pts: torch.Tensor, bins: int = 8) -> int:
    unit = pts / torch.tensor(UPPER)
    idx = (unit * bins).long().clamp(max=bins - 1)
    flat = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    return int(torch.bincount(flat, minlength=bins**3).max())


def test_sobol_is_more_even_than_random():
    # 8^3 = 512 cells, 4 points per cell on average: Sobol keeps the fullest
    # cell close to that, independent random draws pile up much higher.
    n = 2048
    sobol = _max_cell_count(make_sampler("sobol", LOWER, UPPER, n, seed=0).sample())
    random = _max_cell_count(make_sampler("random", LOWER, UPPER, n, seed=0).sample())
    assert sobol < random


def test_heldout_points_differ_from_training_draw_and_need_grad():
    train = make_sampler("random", LOWER, UPPER, 200, seed=0).sample()
    held = heldout_points(LOWER, UPPER, 200, seed=1)
    assert held.requires_grad
    assert not torch.allclose(train, held.detach())


LOWER_1D = (0.0, 0.0)
UPPER_1D = (2.0, 1e-3)


@pytest.mark.parametrize("name", sorted(COLLOCATION_STRATEGIES))
def test_1d_points_have_two_columns_inside_the_box(name):
    pts = make_sampler(name, LOWER_1D, UPPER_1D, 400).sample()
    assert pts.shape == (400, 2)
    assert torch.all(pts >= torch.tensor(LOWER_1D)) and torch.all(pts <= torch.tensor(UPPER_1D))
    assert heldout_points(LOWER_1D, UPPER_1D, 50, seed=0).shape == (50, 2)


def test_1d_grid_shape():
    assert default_grid_shape(20000, dim=2) == (141, 141)
    pts = make_sampler("grid", LOWER_1D, UPPER_1D, 0, grid_shape=(4, 6)).sample()
    assert pts.shape == (24, 2)
    with pytest.raises(ValueError):
        make_sampler("grid", LOWER_1D, UPPER_1D, 0, grid_shape=(4, 5, 6))


@pytest.mark.parametrize("name", ["random", "sobol"])
def test_time_window_adds_points_inside_the_window(name):
    base = make_sampler(name, LOWER, UPPER, 1000, seed=0)
    pts = with_time_window(base, name, LOWER, UPPER, (0.0, 1e-5), 500, seed=4).sample()
    assert pts.shape == (1500, 3)
    window = pts[1000:]
    assert torch.all(window[:, 2] > 0.0) and torch.all(window[:, 2] <= 1e-5)
    # Spread over the whole spatial domain, not squeezed with t.
    assert window[:, 1].max() > 1.5


def test_time_window_is_redrawn_with_the_base_set():
    base = make_sampler("sobol", LOWER, UPPER, 100, seed=0, resample_every=2)
    sampler = with_time_window(base, "sobol", LOWER, UPPER, (0.0, 1e-5), 50, seed=4)
    first = sampler.sample()
    assert sampler.update(None, 1) is None
    second = sampler.update(None, 2)
    assert second.shape == (150, 3)
    assert not torch.equal(first[100:], second[100:])
    assert torch.all(second[100:, 2] <= 1e-5)


def test_no_time_window_returns_base_unchanged():
    base = make_sampler("sobol", LOWER, UPPER, 100)
    assert with_time_window(base, "sobol", LOWER, UPPER, None, 0) is base
