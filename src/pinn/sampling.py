"""Domain sampling for the Cahn-Hilliard PINN.

Bundles the collocation/IC/BC point tensors a training step needs into one
dataclass, rather than passing them as loose positional arguments (easy to
mis-order) or repurposing argparse.Namespace (meant for CLI args, not data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.pinn.losses import DEFAULT_EPSILON

# "Swiss flag" cross proportions: arms 6 units wide and 20 units long on a
# 32-unit square, rescaled to the unit square.
_IC_ARM_HALF_W = (6.0 / 32.0) / 2.0
_IC_ARM_HALF_L = (20.0 / 32.0) / 2.0
_IC_CENTER = (0.5, 0.5)


@dataclass
class TrainingPoints:
    """Point tensors sampled from the domain for one training run.

    Attributes:
        collocation: (N, 3) interior (x, y, t) points, requires_grad=True.
        ic_points: (N_ic, 3) points at t = t_lo.
        ic_values: (N_ic, 1) target u values at ic_points.
        bc_points: (N_bc, 3) points on the four spatial edges, requires_grad=True.
    """

    collocation: torch.Tensor
    ic_points: torch.Tensor
    ic_values: torch.Tensor
    bc_points: torch.Tensor


def _rect_sdf(px: torch.Tensor, py: torch.Tensor, half_w: float, half_h: float) -> torch.Tensor:
    """Signed distance to an axis-aligned rectangle centred at the origin, negative inside."""
    qx = px.abs() - half_w
    qy = py.abs() - half_h
    outside = torch.sqrt(qx.clamp(min=0.0) ** 2 + qy.clamp(min=0.0) ** 2)
    inside = torch.maximum(qx, qy).clamp(max=0.0)
    return outside + inside


def _cross_sdf(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
    """Signed distance to the union of a horizontal and a vertical bar (a cross)."""
    px = xx - _IC_CENTER[0]
    py = yy - _IC_CENTER[1]
    horizontal = _rect_sdf(px, py, _IC_ARM_HALF_L, _IC_ARM_HALF_W)
    vertical = _rect_sdf(px, py, _IC_ARM_HALF_W, _IC_ARM_HALF_L)
    return torch.minimum(horizontal, vertical)


def cross_initial_condition(points: torch.Tensor, eps: float = DEFAULT_EPSILON) -> torch.Tensor:
    """Swiss-flag cross initial condition: u = +1 on the cross, u = -1 on the
    background, joined by the equilibrium tanh interface profile.

    Args:
        points: (N, 3) tensor of (x, y, t). Assumes the spatial domain is [0, 1]^2.
        eps: interface half-width; must match `epsilon` in pde_residual_loss
            (see `losses.DEFAULT_EPSILON`).

    Returns:
        (N, 1) tensor of u values.
    """
    xx, yy = points[:, 0:1], points[:, 1:2]
    return -torch.tanh(_cross_sdf(xx, yy) / (np.sqrt(2.0) * eps))


def sample_points(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    n_collocation: int,
    n_ic: int,
    n_bc: int,
    device: torch.device | str = "cpu",
    pde_at_t0: int = 0,
    epsilon: float = DEFAULT_EPSILON,
) -> TrainingPoints:
    """Sample collocation, initial-condition, and boundary-condition points.

    Args:
        lower: (x_lo, y_lo, t_lo) domain lower bound.
        upper: (x_hi, y_hi, t_hi) domain upper bound.
        n_collocation: number of random interior points.
        n_ic: side length of the (n_ic x n_ic) grid of initial-condition points.
        n_bc: side length of the grid of boundary points per edge.
        device: device to place the tensors on.
        pde_at_t0: if > 0, also enforce the PDE residual at this many of the
            initial-condition points (helps the residual stay consistent at t=0).
        epsilon: interface half-width for the IC profile; pass the same value
            used for `epsilon` in `pde_residual_loss` to keep the IC in
            equilibrium with the PDE.

    Returns:
        A TrainingPoints bundle.
    """
    x = torch.rand(n_collocation, 1, device=device) * (upper[0] - lower[0]) + lower[0]
    y = torch.rand(n_collocation, 1, device=device) * (upper[1] - lower[1]) + lower[1]
    t = torch.rand(n_collocation, 1, device=device) * (upper[2] - lower[2]) + lower[2]
    x_pde = torch.cat([x, y, t], dim=1)

    grid_x, grid_y = torch.meshgrid(
        torch.linspace(lower[0], upper[0], n_ic, device=device),
        torch.linspace(lower[1], upper[1], n_ic, device=device),
        indexing="ij",
    )
    x_ic = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    x_ic = torch.cat([x_ic, torch.full((x_ic.shape[0], 1), lower[2], device=device)], dim=1)

    if pde_at_t0:
        idx = torch.randperm(x_ic.shape[0], device=device)[:pde_at_t0]
        x_pde = torch.cat([x_pde, x_ic[idx]], dim=0)

    x_pde = x_pde.requires_grad_(True)
    ic_values = cross_initial_condition(x_ic, eps=epsilon)

    x_edge = torch.linspace(lower[0], upper[0], n_bc, device=device)
    y_edge = torch.linspace(lower[1], upper[1], n_bc, device=device)
    t_bc = torch.linspace(lower[2], upper[2], n_bc, device=device)

    grid_y_edge, grid_t_x = torch.meshgrid(y_edge, t_bc, indexing="ij")
    y_flat, t_flat_x = grid_y_edge.reshape(-1), grid_t_x.reshape(-1)

    grid_x_edge, grid_t_y = torch.meshgrid(x_edge, t_bc, indexing="ij")
    x_flat, t_flat_y = grid_x_edge.reshape(-1), grid_t_y.reshape(-1)

    x_bc = torch.cat(
        [
            torch.stack([torch.full_like(y_flat, lower[0]), y_flat, t_flat_x], dim=1),
            torch.stack([torch.full_like(y_flat, upper[0]), y_flat, t_flat_x], dim=1),
            torch.stack([x_flat, torch.full_like(x_flat, lower[1]), t_flat_y], dim=1),
            torch.stack([x_flat, torch.full_like(x_flat, upper[1]), t_flat_y], dim=1),
        ],
        dim=0,
    ).requires_grad_(True)

    return TrainingPoints(collocation=x_pde, ic_points=x_ic, ic_values=ic_values, bc_points=x_bc)
