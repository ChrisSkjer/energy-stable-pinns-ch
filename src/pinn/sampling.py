"""Domain sampling for the Cahn-Hilliard PINN.

Bundles the collocation/IC/BC point tensors a training step needs into one
dataclass, rather than passing them as loose positional arguments (easy to
mis-order) or repurposing argparse.Namespace (meant for CLI args, not data).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.initial_conditions import initial_condition
from src.pinn.losses import DEFAULT_EPSILON


@dataclass
class TrainingPoints:
    """Point tensors sampled from the domain for one training run.

    Attributes:
        collocation: (N + pde_at_t0, 3) PDE residual points -- the interior
            points followed by t0_points -- requires_grad=True.
        ic_points: (N_ic, 3) points at t = t_lo.
        ic_values: (N_ic, 1) target u values at ic_points.
        bc_points: (N_bc, 3) points on the four spatial edges, requires_grad=True.
        t0_points: (pde_at_t0, 3) the IC points also used as PDE points, kept
            so a resample of the interior points can re-append the same ones.
    """

    collocation: torch.Tensor
    ic_points: torch.Tensor
    ic_values: torch.Tensor
    bc_points: torch.Tensor
    t0_points: torch.Tensor


def with_pde_at_t0(collocation: torch.Tensor, t0_points: torch.Tensor) -> torch.Tensor:
    """Append the fixed t=0 PDE points (see sample_points' pde_at_t0) to a
    set of interior collocation points, as one requires_grad leaf tensor.

    Kept separate so train.py can resample the interior points without
    redrawing the t=0 ones.
    """
    return torch.cat([collocation.detach(), t0_points], dim=0).requires_grad_(True)


def sample_points(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    collocation: torch.Tensor,
    n_ic: int,
    n_bc: int,
    device: torch.device | str = "cpu",
    pde_at_t0: int = 0,
    epsilon: float = DEFAULT_EPSILON,
    ic: str = "cross",
    ic_seed: int = 0,
) -> TrainingPoints:
    """Bundle collocation points with sampled initial- and boundary-condition points.

    Args:
        lower: (x_lo, y_lo, t_lo) domain lower bound.
        upper: (x_hi, y_hi, t_hi) domain upper bound.
        collocation: (N, 3) interior PDE points, from a sampler in
            src/pinn/collocation.py (the choice of strategy lives there).
        n_ic: side length of the (n_ic x n_ic) grid of initial-condition points.
        n_bc: side length of the grid of boundary points per edge.
        device: device to place the tensors on.
        pde_at_t0: if > 0, also enforce the PDE residual at this many of the
            initial-condition points (helps the residual stay consistent at t=0).
        epsilon: interface half-width for the IC profile; pass the same value
            used for `epsilon` in `pde_residual_loss` to keep the IC in
            equilibrium with the PDE.
        ic: name of the initial condition, a key of
            `src.common.initial_conditions.INITIAL_CONDITIONS`.
        ic_seed: random seed for the initial condition (only "noise" uses it).

    Returns:
        A TrainingPoints bundle.
    """
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(lower[0], upper[0], n_ic, device=device),
        torch.linspace(lower[1], upper[1], n_ic, device=device),
        indexing="ij",
    )
    x_ic = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    x_ic = torch.cat([x_ic, torch.full((x_ic.shape[0], 1), lower[2], device=device)], dim=1)

    idx = torch.randperm(x_ic.shape[0], device=device)[:pde_at_t0]
    t0_points = x_ic[idx]
    x_pde = with_pde_at_t0(collocation.to(device), t0_points)
    # IC targets need no grad, so evaluating them in numpy is fine.
    x_ic_np = x_ic.cpu().numpy()
    ic_values = torch.as_tensor(
        initial_condition(ic, x_ic_np[:, 0], x_ic_np[:, 1], epsilon=epsilon, seed=ic_seed),
        dtype=x_ic.dtype,
        device=device,
    ).unsqueeze(1)

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

    return TrainingPoints(
        collocation=x_pde, ic_points=x_ic, ic_values=ic_values, bc_points=x_bc, t0_points=t0_points
    )
