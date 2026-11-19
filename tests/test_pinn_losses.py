"""Smoke test for src.pinn.losses."""


def test_import():
    from src.pinn import losses  # noqa: F401

    assert hasattr(losses, "pde_u_residual_loss")
    assert hasattr(losses, "pde_mu_residual_loss")
    assert hasattr(losses, "ic_loss")
    assert hasattr(losses, "bc_loss")
    assert hasattr(losses, "energy_stability_loss")


import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.pinn.losses import bc_loss, pde_residual_losses  # noqa: E402


class _ClosedForm(nn.Module):
    """Stand-in "network" with known derivatives, for checking the losses:
    u = e^t prod_i cos(pi x_i), mu = t * x_0^2, over input (x_0, ..., t)."""

    def forward(self, p):
        x, t = p[:, :-1], p[:, -1:]
        u = torch.exp(t) * torch.prod(torch.cos(math.pi * x), dim=1, keepdim=True)
        mu = t * x[:, 0:1] ** 2
        return torch.cat([u, mu], dim=1)


@pytest.mark.parametrize("n_space", [1, 2])
def test_pde_residuals_match_hand_derivatives(n_space):
    eps = 0.05
    p = torch.rand(64, n_space + 1, dtype=torch.float64, requires_grad=True)
    loss_u, loss_mu = pde_residual_losses(_ClosedForm(), p, epsilon=eps)

    x, t = p.detach()[:, :-1], p.detach()[:, -1:]
    u = torch.exp(t) * torch.prod(torch.cos(math.pi * x), dim=1, keepdim=True)
    mu = t * x[:, 0:1] ** 2
    lap_u = -n_space * math.pi**2 * u
    lap_mu = 2 * t
    residual_u = lap_mu - u  # u_t = u
    residual_mu = mu - (u**3 - u) + eps**2 * lap_u
    torch.testing.assert_close(loss_u, torch.mean(residual_u**2))
    torch.testing.assert_close(loss_mu, torch.mean(residual_mu**2))


def test_bc_loss_1d_uses_x_normal_at_both_ends():
    # du/dx = -pi e^t sin(pi x) vanishes at x = 0 and 1; dmu/dx = 2 x t is 0
    # at x = 0 and 2t at x = 1.
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    ends = torch.cat([torch.stack([torch.zeros_like(t), t], 1), torch.stack([torch.ones_like(t), t], 1)])
    loss = bc_loss(_ClosedForm(), ends.requires_grad_(True))
    expected = torch.mean(torch.cat([torch.zeros_like(t), 2 * t]) ** 2)
    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-9)


def test_time_binned_mean_weights_bins_by_share_not_count():
    from src.pinn.losses import time_binned_mean

    # Bin [0, 0.5): one point with value 1; bin [0.5, 1]: three points with value 3.
    t = torch.tensor([[0.1], [0.6], [0.7], [1.0]])
    squared = torch.tensor([[1.0], [3.0], [3.0], [3.0]])
    edges = torch.tensor([0.0, 0.5, 1.0])
    assert time_binned_mean(squared, t, edges).item() == pytest.approx(0.5 * 1 + 0.5 * 3)


def test_time_binned_mean_matches_window_reweight_and_plain_mean():
    from src.pinn.losses import time_binned_mean, window_weighted_mean

    torch.manual_seed(0)
    t = torch.rand(4000, 1)
    squared = torch.rand(4000, 1)
    # Evenly spread points in equal bins: every bin holds the same count, so the plain mean.
    t_even = (torch.arange(4000, dtype=torch.float32).reshape(-1, 1) + 0.5) / 4000
    assert time_binned_mean(squared, t_even, torch.linspace(0, 1, 11)).item() == pytest.approx(squared.mean().item(), rel=1e-5)
    # Two bins at the window edge give the --dense-window-reweight mean.
    window = (0.0, 0.2, 0.2)
    binned = time_binned_mean(squared, t, torch.tensor([0.0, 0.2, 1.0]))
    assert binned.item() == pytest.approx(window_weighted_mean(squared, t, window).item(), rel=1e-5)


def test_time_binned_mean_drops_empty_bins():
    from src.pinn.losses import time_binned_mean

    t = torch.tensor([[0.1], [0.9]])
    squared = torch.tensor([[2.0], [4.0]])
    # The middle bin is empty: the other two share the weight equally.
    assert time_binned_mean(squared, t, torch.linspace(0, 1, 4)).item() == pytest.approx(3.0)


class _Drifting(nn.Module):
    """u = c + b e^{-t} cos(pi x_0) + a t, mu = 0: mass int u dx =
    (c + a t) |Omega| on [0, 1]^n, so it drifts at rate a."""

    def __init__(self, a=0.0, b=1.0, c=0.0):
        super().__init__()
        self.a, self.b, self.c = a, b, c

    def forward(self, p):
        x, t = p[:, :-1], p[:, -1:]
        u = self.c + self.b * torch.exp(-t) * torch.cos(math.pi * x[:, 0:1]) + self.a * t
        return torch.cat([u, torch.zeros_like(u)], dim=1)


@pytest.mark.parametrize("n_space", [1, 2])
def test_trapezoid_weights_sum_to_domain_size_and_integrate_linear_exactly(n_space):
    from src.pinn.losses import quadrature_grid

    lower, upper = (0.0,) * n_space + (0.0,), (2.0,) * n_space + (1.0,)
    grid = quadrature_grid(lower, upper, 9, 4, dtype=torch.float64)
    assert grid.points.shape == (4 * 9**n_space, n_space + 1)
    assert grid.weights.sum().item() == pytest.approx(2.0**n_space)
    torch.testing.assert_close(grid.times, torch.linspace(0, 1, 4, dtype=torch.float64))
    # int x_0 dx over [0, 2]^n = 2 * 2^(n-1) at every slice (trapezoid is exact for linear).
    x0 = grid.points.detach()[:, 0:1]
    torch.testing.assert_close(grid.integrate(x0), torch.full((4,), 2.0**n_space, dtype=torch.float64))


@pytest.mark.parametrize("n_space", [1, 2])
def test_energy_rate_matches_finite_difference_of_quadrature_energy(n_space):
    from src.pinn.losses import QuadratureGrid, energy_rate, quadrature_grid

    eps = 0.1
    model = _ClosedForm()
    grid = quadrature_grid((0.0,) * (n_space + 1), (1.0,) * (n_space + 1), 17, 3, dtype=torch.float64)
    rate = energy_rate(model, grid, epsilon=eps)

    def energy_at(shift):
        p = grid.points.detach().clone()
        p[:, -1] += shift
        p.requires_grad_(True)
        u = model(p)[:, 0:1]
        grad = torch.autograd.grad(u, p, torch.ones_like(u))[0][:, :-1]
        density = 0.25 * (u**2 - 1) ** 2 + 0.5 * eps**2 * (grad**2).sum(1, keepdim=True)
        return QuadratureGrid(p, grid.weights, grid.n_t).integrate(density)

    h = 1e-5
    torch.testing.assert_close(rate, (energy_at(h) - energy_at(-h)) / (2 * h), rtol=1e-6, atol=1e-8)


def test_energy_stability_loss_penalises_only_increasing_energy():
    from src.pinn.losses import energy_rate, energy_stability_loss, quadrature_grid

    # u = e^t cos(pi x): |u| grows past 1 in places, so E rises; 1 + 0.1 e^{-t} cos
    # relaxes to the well u = 1, so E falls.
    grid = quadrature_grid((0.0, 0.0), (1.0, 1.0), 33, 4, dtype=torch.float64)
    rate = energy_rate(_ClosedForm(), grid, epsilon=0.05)
    expected = torch.mean(torch.clamp(rate, min=0) ** 2)
    torch.testing.assert_close(energy_stability_loss(_ClosedForm(), grid, epsilon=0.05), expected)
    assert (rate > 0).any()
    relaxing = _Drifting(b=0.1, c=1.0)
    assert (energy_rate(relaxing, grid, epsilon=0.05) < 0).all()
    assert energy_stability_loss(relaxing, grid, epsilon=0.05).item() == 0.0


@pytest.mark.parametrize("n_space", [1, 2])
def test_mass_conservation_loss(n_space):
    from src.pinn.losses import ic_mass, mass_conservation_loss, quadrature_grid

    grid = quadrature_grid((0.0,) * (n_space + 1), (1.0,) * (n_space + 1), 33, 5, dtype=torch.float64)
    n_slice = grid.weights.numel()
    u0 = _Drifting()(grid.points[:n_slice].detach())[:, 0:1]
    mass0, u0_sq = ic_mass(grid, u0)
    assert mass0 == pytest.approx(0.0, abs=1e-12)  # int cos(pi x) = 0
    assert u0_sq == pytest.approx(0.5, rel=1e-3)  # mean cos^2

    assert mass_conservation_loss(_Drifting(0.0), grid, mass0, u0_sq).item() == pytest.approx(0.0, abs=1e-20)
    # Drift a t at t = 0, 1/4, ..., 1: mean (a t)^2 / mean(u0^2).
    a = 0.3
    t = torch.linspace(0, 1, 5, dtype=torch.float64)
    expected = torch.mean((a * t) ** 2) / u0_sq
    assert mass_conservation_loss(_Drifting(a), grid, mass0, u0_sq).item() == pytest.approx(expected.item())
