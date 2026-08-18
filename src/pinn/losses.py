"""Loss terms for training the Cahn-Hilliard PINN.

Separate functions for the PDE residual, IC/BC terms, and the
energy-stability penalty used only by the enhanced model.
"""

from __future__ import annotations

import torch
from torch import nn


def pde_residual_loss(model: nn.Module, collocation_points: torch.Tensor) -> torch.Tensor:
    """Residual of the Cahn-Hilliard PDE (mixed c/mu formulation) at
    collocation points, via automatic differentiation.

    Args:
        model: the PINN.
        collocation_points: (N, input_dim) interior points, requires_grad=True.

    Returns:
        Scalar MSE of the PDE residual.
    """
    # TODO: compute c, mu = model(collocation_points) (or split network outputs)
    # TODO: compute required derivatives via torch.autograd.grad
    # TODO: assemble the two coupled residuals (c_t - div(M grad(mu)), mu - dF/dc + eps^2 lap(c))
    raise NotImplementedError


def ic_bc_loss(
    model: nn.Module,
    ic_points: torch.Tensor,
    ic_values: torch.Tensor,
    bc_points: torch.Tensor,
) -> torch.Tensor:
    """Initial-condition and boundary-condition loss.

    Args:
        model: the PINN.
        ic_points: (N_ic, input_dim) points at t=0.
        ic_values: (N_ic, output_dim) target values at ic_points.
        bc_points: (N_bc, input_dim) points on the domain boundary
            (e.g. for periodic or no-flux/Neumann conditions).

    Returns:
        Scalar combined IC + BC loss.
    """
    # TODO: MSE between model(ic_points) and ic_values
    # TODO: boundary condition residual (periodic / zero-flux, matching the FEM setup)
    raise NotImplementedError


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
