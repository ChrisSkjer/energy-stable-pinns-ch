"""Tests for src.pinn.ic_torch: the torch u0 used by --hard-ic is the same
field as the numpy ICs that give the IC targets and the FEM start."""

import numpy as np
import pytest
import torch

from src.common.initial_conditions import (
    INITIAL_CONDITIONS,
    INITIAL_CONDITIONS_1D,
    initial_condition,
    initial_condition_1d,
)
from src.pinn.ic_torch import (
    TORCH_INITIAL_CONDITIONS,
    TORCH_INITIAL_CONDITIONS_1D,
    InitialConditionField,
    hard_ic_kwargs,
)


def test_registries_cover_every_numpy_ic():
    assert set(TORCH_INITIAL_CONDITIONS) == set(INITIAL_CONDITIONS)
    assert set(TORCH_INITIAL_CONDITIONS_1D) == set(INITIAL_CONDITIONS_1D)


@pytest.mark.parametrize("name", sorted(INITIAL_CONDITIONS))
def test_2d_field_matches_numpy(name):
    xy = np.random.default_rng(0).uniform(0.0, 1.0, size=(2000, 2))
    expected = initial_condition(name, xy[:, 0], xy[:, 1], epsilon=0.05, seed=42)
    field = InitialConditionField(2, name, epsilon=0.05, seed=42)
    got = field(torch.as_tensor(xy, dtype=torch.float32)).squeeze(1).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.parametrize("name", sorted(INITIAL_CONDITIONS_1D))
def test_1d_field_matches_numpy(name):
    x = np.random.default_rng(1).uniform(0.0, 1.0, size=(2000, 1))
    expected = initial_condition_1d(name, x[:, 0], epsilon=0.05, seed=42)
    field = InitialConditionField(1, name, epsilon=0.05, seed=42)
    got = field(torch.as_tensor(x, dtype=torch.float32)).squeeze(1).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_noise_seed_changes_the_field():
    x = torch.rand(50, 2)
    a = InitialConditionField(2, "noise", 0.05, seed=1)(x)
    b = InitialConditionField(2, "noise", 0.05, seed=2)(x)
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("dim,name", [(2, "cross"), (2, "two_circles"), (2, "noise"), (1, "two_intervals"), (1, "cosine")])
def test_second_derivatives_are_finite(dim, name):
    # The PDE residual takes the laplacian of u0 by autograd, including at
    # kinks and at the circle centres.
    pts = torch.rand(500, dim)
    pts[0] = 0.5  # cross centre / interval gap
    if name == "two_circles":
        pts[1] = torch.tensor([0.28, 0.5])  # exact circle centre
    pts.requires_grad_(True)
    u = InitialConditionField(dim, name, 0.05, seed=42)(pts)
    grad = torch.autograd.grad(u.sum(), pts, create_graph=True)[0]
    second = torch.autograd.grad(grad[:, 0].sum(), pts)[0]
    assert torch.isfinite(grad).all() and torch.isfinite(second).all()


def test_field_has_no_state_and_moves_with_to():
    field = InitialConditionField(2, "noise", 0.05, seed=42)
    assert field.state_dict() == {}
    assert list(field.parameters()) == []
    field = field.to(torch.float64)
    assert field.coeff_a.dtype == torch.float64


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown"):
        InitialConditionField(2, "cosine", 0.05)


def test_hard_ic_kwargs_is_empty_without_the_flag():
    assert hard_ic_kwargs({}) == {}
    assert hard_ic_kwargs({"hard_ic": False, "ic": "cross", "epsilon": 0.05}) == {}
    kwargs = hard_ic_kwargs({"hard_ic": True, "dim": 1, "ic": "cosine", "epsilon": 0.05, "hard_ic_tau": 1e-3})
    assert isinstance(kwargs["hard_ic"], InitialConditionField)
    assert kwargs["hard_ic_tau"] == 1e-3 and kwargs["hard_ic_form"] == "add"
