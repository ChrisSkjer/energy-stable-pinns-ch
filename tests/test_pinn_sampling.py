"""Tests for src.pinn.sampling."""

import pytest
import torch

from src.common.initial_conditions import INITIAL_CONDITIONS
from src.pinn.sampling import sample_points, with_pde_at_t0


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
