"""Loss terms for training the Cahn-Hilliard PINN.

Separate functions for the PDE residual, IC/BC terms, and the
energy-stability penalty used only by the enhanced model.
"""

from __future__ import annotations

import torch
from torch import nn

# Cahn-Hilliard interface half-width. This is the single source of truth for
# epsilon: `sampling.sample_points` defaults to it so the initial-condition
# profile stays consistent with the PDE residual (a mismatch here puts the
# IC out of equilibrium and forces the network to resolve a stiff transient
# at t = 0). Pass `epsilon` explicitly through `train.py` rather than
# overriding this default in only one of the two places it's used.
DEFAULT_EPSILON = 0.05


def pde_residual_losses(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals of both equations of the Cahn-Hilliard PDE (mixed u/mu
    formulation), at collocation points, via automatic differentiation:

        u-equation:  u_t = m * laplacian(mu)
        mu-equation: mu = f'(u) - epsilon^2 * laplacian(u)

    Computed together so the forward pass and the first derivatives of u
    and mu are shared between the two residuals.

    Works in any number of space dimensions: the columns of
    `collocation_points` are the spatial coordinates followed by t, so
    (x, t) gives laplacian = u_xx and (x, y, t) gives u_xx + u_yy.

    Args:
        model: the PINN.
        collocation_points: (N, input_dim) interior points, requires_grad=True.
        epsilon: interface half-width.
        m: mobility.

    Returns:
        (loss_pde_u, loss_pde_mu): scalar MSEs of the u- and mu-equation residuals.
    """
    out = model(collocation_points)
    u, mu = out[:,0:1], out[:,1:2]

    grad_u = torch.autograd.grad(u, collocation_points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_mu = torch.autograd.grad(mu, collocation_points, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    u_t = grad_u[:, -1:]

    lap_u = laplacian(grad_u, collocation_points)
    lap_mu = laplacian(grad_mu, collocation_points)

    residual_u = m * lap_mu - u_t

    #f = 1/4 * (u**2 - 1)**2 dobbel well
    f_u = (u**2 - 1) * u
    residual_mu = mu - f_u + epsilon**2 * lap_u

    return torch.mean(residual_u**2), torch.mean(residual_mu**2)


def laplacian(grad: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Spatial laplacian, sum_i d2/dx_i^2, of a field from its gradient.

    Args:
        grad: (N, input_dim) gradient of the field w.r.t. `points`, built
            with create_graph=True; its last column (d/dt) is ignored.
        points: (N, input_dim) the points it was taken at.

    Returns:
        (N, 1) laplacian over the spatial columns (all but the last).
    """
    n_space = points.shape[1] - 1
    second = []
    for i in range(n_space):
        g_i = grad[:, i:i+1]
        second.append(torch.autograd.grad(g_i, points, grad_outputs=torch.ones_like(g_i), create_graph=True)[0][:, i:i+1])
    return sum(second[1:], second[0])


def pde_u_residual_loss(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0) -> torch.Tensor:
    """u-equation residual loss only; see `pde_residual_losses`. Training
    should call `pde_residual_losses` directly to avoid duplicate work."""
    return pde_residual_losses(model, collocation_points, epsilon=epsilon, m=m)[0]


def pde_mu_residual_loss(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0) -> torch.Tensor:
    """mu-equation residual loss only; see `pde_residual_losses`. Training
    should call `pde_residual_losses` directly to avoid duplicate work."""
    return pde_residual_losses(model, collocation_points, epsilon=epsilon, m=m)[1]


def ic_loss(
    model: nn.Module,
    ic_points: torch.Tensor,
    ic_values: torch.Tensor,
) -> torch.Tensor:
    """Initial-condition loss.

    Args:
        model: the PINN.
        ic_points: (N_ic, input_dim) points at t=0.
        ic_values: (N_ic, 1) target u values at ic_points -- the u channel
            only, from `src.common.initial_conditions`. There is no
            target for mu here: its t=0 value is not free to choose, it is
            fixed by the IC through mu = f'(u0) - eps^2 * laplacian(u0).

    Returns:
        Scalar MSE of the IC term.
    """
    u_ic = model(ic_points)[:, 0:1]
    return torch.mean((u_ic - ic_values) ** 2)/torch.mean(ic_values**2)


def bc_loss(model: nn.Module, bc_points: torch.Tensor) -> torch.Tensor:
    """Boundary-condition loss: no-flux, du/dn = dmu/dn = 0 on every wall.

    Args:
        model: the PINN.
        bc_points: (N_bc, input_dim) points on the walls of the spatial box,
            in 2 * n_space equal blocks ordered (axis 0 low, axis 0 high,
            axis 1 low, axis 1 high, ...) -- in 2D left, right, bottom, top;
            in 1D x = 0, x = L. Block k has normal axis k // 2, as built by
            src/pinn/sampling.py.

    Returns:
        Scalar MSE of the BC term.
    """
    out_bc = model(bc_points)
    u_bc, mu_bc = out_bc[:,0:1], out_bc[:,1:2]
    grad_u_bc = torch.autograd.grad(u_bc, bc_points, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
    grad_mu_bc = torch.autograd.grad(mu_bc, bc_points, grad_outputs=torch.ones_like(mu_bc), create_graph=True)[0]

    n_walls = 2 * (bc_points.shape[1] - 1)
    n_edge = bc_points.shape[0] // n_walls
    du_dn = torch.cat([grad_u_bc[k*n_edge:(k+1)*n_edge, k//2:k//2+1] for k in range(n_walls)], dim=0)
    dmu_dn = torch.cat([grad_mu_bc[k*n_edge:(k+1)*n_edge, k//2:k//2+1] for k in range(n_walls)], dim=0)

    return torch.mean(du_dn**2) + torch.mean(dmu_dn**2)


def energy_stability_loss(model: nn.Module, collocation_points: torch.Tensor) -> torch.Tensor:
    """Placeholder energy-dissipation penalty for the enhanced PINN.

    Encourages the Ginzburg-Landau free energy E(t) predicted by the network
    to be non-increasing in time, matching the analytic dissipation property
    of the Cahn-Hilliard equation.

    Args:
        model: the PINN.
        collocation_points: (N, input_dim) points used to estimate dE/dt.

    Returns:
        Scalar penalty, zero when energy is non-increasing.
    """
    # TODO: compute (an estimate of) dE/dt from model predictions
    # TODO: penalize positive dE/dt, e.g. via relu(dE/dt) ** 2
    raise NotImplementedError
