"""Tests for src.pinn.sampling."""

import pytest
import torch

from src.common.initial_conditions import INITIAL_CONDITIONS
from src.pinn.sampling import sample_points


@pytest.mark.parametrize("ic", sorted(INITIAL_CONDITIONS))
def test_ic_values_shape_and_dtype(ic):
    points = sample_points(
        lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1e-3), n_collocation=50, n_ic=8, n_bc=4, ic=ic
    )
    assert points.ic_values.shape == (64, 1)
    assert points.ic_values.dtype == torch.float32
    assert points.ic_values.device == points.ic_points.device
    assert not points.ic_values.requires_grad


def test_ic_choice_changes_ic_values():
    kwargs = dict(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1e-3), n_collocation=10, n_ic=8, n_bc=4)
    cross = sample_points(**kwargs).ic_values
    circles = sample_points(**kwargs, ic="two_circles").ic_values
    assert not torch.allclose(cross, circles)
