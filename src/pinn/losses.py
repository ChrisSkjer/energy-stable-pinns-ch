"""Loss terms for training the Cahn-Hilliard PINN.

Separate functions for the PDE residual, IC/BC terms, and the
energy-stability penalty used only by the enhanced model.
"""

from __future__ import annotations

import torch
from torch import nn

# Cahn-Hilliard interface half-width. This is the single source of truth for
# epsilon: `sampling.cross_initial_condition` imports it so the initial
# profile stays consistent with the PDE residual (a mismatch here puts the
# IC out of equilibrium and forces the network to resolve a stiff transient
# at t = 0). Pass `epsilon` explicitly through `train.py` rather than
# overriding this default in only one of the two places it's used.
DEFAULT_EPSILON = 0.05


def pde_residual_loss(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0) -> torch.Tensor:
    """Residual of the Cahn-Hilliard PDE (mixed c/mu formulation) at
    collocation points, via automatic differentiation.

    Args:
        model: the PINN.
        collocation_points: (N, input_dim) interior points, requires_grad=True.

    Returns:
        Scalar MSE of the PDE residual.
    """
    out = model(collocation_points)
    u, mu = out[:,0:1], out[:,1:2]

    u_t = torch.autograd.grad(u, collocation_points, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 2:3]
    u_x = torch.autograd.grad(u, collocation_points, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u, collocation_points, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
    u_xx = torch.autograd.grad(u_x, collocation_points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, collocation_points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    mu_x = torch.autograd.grad(mu, collocation_points, grad_outputs=torch.ones_like(mu), create_graph=True)[0][:, 0:1]
    mu_y = torch.autograd.grad(mu, collocation_points, grad_outputs=torch.ones_like(mu), create_graph=True)[0][:, 1:2]
    mu_xx = torch.autograd.grad(mu_x, collocation_points, grad_outputs=torch.ones_like(mu_x), create_graph=True)[0][:, 0:1]
    mu_yy = torch.autograd.grad(mu_y, collocation_points, grad_outputs=torch.ones_like(mu_y), create_graph=True)[0][:, 1:2]

    #f = 1/4 * (u**2 - 1)**2
    f_u = (u**2 - 1) * u

    residual_u = m * (mu_xx + mu_yy) - u_t
    residual_mu = mu - f_u + epsilon**2 * (u_xx + u_yy)
    loss_pde = torch.mean(residual_u**2) + torch.mean(residual_mu**2)

    return loss_pde


def ic_loss(
    model: nn.Module,
    ic_points: torch.Tensor,
    ic_values: torch.Tensor,
) -> torch.Tensor:
    """Initial-condition loss.

    Args:
        model: the PINN.
        ic_points: (N_ic, input_dim) points at t=0.
        ic_values: (N_ic, output_dim) target values at ic_points.

    Returns:
        Scalar MSE of the IC term.
    """
    out_ic = model(ic_points)
    return torch.mean((out_ic - ic_values) ** 2)


def bc_loss(model: nn.Module, bc_points: torch.Tensor) -> torch.Tensor:
    """Boundary-condition loss.

    Args:
        model: the PINN.
        bc_points: (N_bc, input_dim) points on the domain boundary
            (e.g. for periodic or no-flux/Neumann conditions).

    Returns:
        Scalar MSE of the BC term.
    """
    out_bc = model(bc_points)
    u_bc, mu_bc = out_bc[:,0:1], out_bc[:,1:2]
    grad_u_bc = torch.autograd.grad(u_bc, bc_points, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
    grad_mu_bc = torch.autograd.grad(mu_bc, bc_points, grad_outputs=torch.ones_like(mu_bc), create_graph=True)[0]

    n_edge = bc_points.shape[0] // 4
    du_dn = torch.cat([
        grad_u_bc[:n_edge, 0:1],  # left edge (x=0)
        grad_u_bc[n_edge:2*n_edge, 0:1],  # right edge (x=1)
        grad_u_bc[2*n_edge:3*n_edge, 1:2],  # bottom edge (y=0)
        grad_u_bc[3*n_edge:, 1:2],  # top edge (y=1)
    ], dim=0)

    dmu_dn = torch.cat([
        grad_mu_bc[:n_edge, 0:1],  # left edge (x=0)
        grad_mu_bc[n_edge:2*n_edge, 0:1],   # right edge (x=1)
        grad_mu_bc[2*n_edge:3*n_edge, 1:2],  # bottom edge (y=0)
        grad_mu_bc[3*n_edge:, 1 :2],  # top edge (y=1)
    ], dim=0)

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
