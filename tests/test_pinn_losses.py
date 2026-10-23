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
