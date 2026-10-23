"""Tests for src.pinn.sampling."""

import pytest
import torch

from src.common.initial_conditions import (
    INITIAL_CONDITIONS,
    INITIAL_CONDITIONS_1D,
    initial_condition,
    initial_condition_1d,
)
from src.pinn.sampling import heldout_ic_bc_points, sample_points, with_pde_at_t0


@pytest.mark.parametrize("ic", sorted(INITIAL_CONDITIONS))
def test_ic_values_shape_and_dtype(ic):
    points = sample_points(
        lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1e-3), collocation=torch.rand(50, 3), n_ic=8, n_bc=4, ic=ic
    )
    assert points.ic_values.shape == (64, 1)
    assert points.ic_values.dtype == torch.float32
    assert points.ic_values.device == points.ic_points.device
    assert not points.ic_values.requires_grad


def test_ic_choice_changes_ic_values():
    kwargs = dict(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1e-3), collocation=torch.rand(10, 3), n_ic=8, n_bc=4)
    cross = sample_points(**kwargs).ic_values
    circles = sample_points(**kwargs, ic="two_circles").ic_values
    assert not torch.allclose(cross, circles)


def test_pde_at_t0_points_are_appended_and_kept_on_resample():
    points = sample_points(
        lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1e-3), collocation=torch.rand(50, 3), n_ic=8, n_bc=4, pde_at_t0=5
    )
    assert points.collocation.shape == (55, 3)
    assert points.collocation.requires_grad
    assert torch.all(points.t0_points[:, 2] == 0)
    assert torch.equal(points.collocation[50:].detach(), points.t0_points)

    resampled = with_pde_at_t0(torch.rand(30, 3), points.t0_points)
    assert resampled.shape == (35, 3)
    assert resampled.requires_grad and resampled.is_leaf
    assert torch.equal(resampled[30:].detach(), points.t0_points)


_LO, _HI = (0.0, 0.0, 0.0), (1.0, 2.0, 1e-3)


def test_heldout_ic_points_lie_on_t0_with_matching_targets():
    ic_points, ic_values, _ = heldout_ic_bc_points(_LO, _HI, n_ic=50, n_bc_per_edge=10, seed=3, ic="two_circles")
    assert ic_points.shape == (50, 3) and ic_values.shape == (50, 1)
    assert torch.all(ic_points[:, 2] == _LO[2])
    assert torch.all((ic_points[:, 1] >= _LO[1]) & (ic_points[:, 1] <= _HI[1]))
    expected = initial_condition("two_circles", ic_points[:, 0].numpy(), ic_points[:, 1].numpy(), epsilon=0.05)
    torch.testing.assert_close(ic_values[:, 0], torch.as_tensor(expected, dtype=torch.float32))


def test_heldout_bc_points_follow_bc_loss_edge_order():
    _, _, bc = heldout_ic_bc_points(_LO, _HI, n_ic=5, n_bc_per_edge=20, seed=3)
    assert bc.shape == (80, 3) and bc.requires_grad
    left, right, bottom, top = bc.detach().split(20)
    assert torch.all(left[:, 0] == _LO[0]) and torch.all(right[:, 0] == _HI[0])
    assert torch.all(bottom[:, 1] == _LO[1]) and torch.all(top[:, 1] == _HI[1])
    t = bc.detach()[:, 2]
    assert torch.all((t >= _LO[2]) & (t <= _HI[2])) and t.unique().numel() == 80


def test_heldout_ic_bc_points_are_reproducible_and_off_the_training_grid():
    a = heldout_ic_bc_points(_LO, _HI, n_ic=30, n_bc_per_edge=10, seed=7)
    b = heldout_ic_bc_points(_LO, _HI, n_ic=30, n_bc_per_edge=10, seed=7)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.detach(), y.detach())
    c = heldout_ic_bc_points(_LO, _HI, n_ic=30, n_bc_per_edge=10, seed=8)
    assert not torch.allclose(a[0], c[0])

    train = sample_points(_LO, _HI, collocation=torch.rand(10, 3), n_ic=8, n_bc=8)
    dist = torch.cdist(a[0], train.ic_points)
    assert dist.min() > 0


_LO_1D, _HI_1D = (0.0, 0.0), (1.0, 1e-3)


@pytest.mark.parametrize("ic", sorted(INITIAL_CONDITIONS_1D))
def test_1d_sample_points_shapes_and_walls(ic):
    points = sample_points(_LO_1D, _HI_1D, collocation=torch.rand(50, 2), n_ic=16, n_bc=8, ic=ic, pde_at_t0=4)
    assert points.ic_points.shape == (16, 2) and torch.all(points.ic_points[:, 1] == 0)
    expected = initial_condition_1d(ic, points.ic_points[:, 0].numpy(), epsilon=0.05)
    torch.testing.assert_close(points.ic_values[:, 0], torch.as_tensor(expected, dtype=torch.float32))
    assert points.collocation.shape == (54, 2)
    # Two walls, x = 0 then x = 1, each at the same n_bc times.
    left, right = points.bc_points.detach().split(8)
    assert torch.all(left[:, 0] == 0) and torch.all(right[:, 0] == 1)
    torch.testing.assert_close(left[:, 1], torch.linspace(0, 1e-3, 8))


def test_1d_heldout_ic_bc_points():
    ic_points, ic_values, bc = heldout_ic_bc_points(_LO_1D, _HI_1D, n_ic=30, n_bc_per_edge=10, seed=3, ic="step")
    assert ic_points.shape == (30, 2) and ic_values.shape == (30, 1)
    assert bc.shape == (20, 2) and bc.requires_grad
    left, right = bc.detach().split(10)
    assert torch.all(left[:, 0] == 0) and torch.all(right[:, 0] == 1)
    assert torch.all((bc.detach()[:, 1] >= 0) & (bc.detach()[:, 1] <= 1e-3))
