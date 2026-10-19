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


def ic_targets(points: torch.Tensor, ic: str, epsilon: float, ic_seed: int = 0) -> torch.Tensor:
    """(N, 1) initial-condition u values at the (x, y) columns of `points`,
    on the same device and dtype.

    IC targets need no grad, so evaluating them in numpy is fine.
    """
    points_np = points.detach().cpu().numpy()
    return torch.as_tensor(
        initial_condition(ic, points_np[:, 0], points_np[:, 1], epsilon=epsilon, seed=ic_seed),
        dtype=points.dtype,
        device=points.device,
    ).unsqueeze(1)


def heldout_ic_bc_points(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    n_ic: int,
    n_bc_per_edge: int,
    seed: int,
    epsilon: float = DEFAULT_EPSILON,
    ic: str = "cross",
    ic_seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed uniform random IC and BC points for held-out ic_loss/bc_loss.

    The training IC/BC points are regular grids, so these probe the error
    between grid nodes (an unbiased estimate of the continuous error) and,
    drawn from their own generators, never coincide with a training point.

    Args:
        lower, upper: (x, y, t) domain bounds.
        n_ic: number of points on the t = t_lo plane.
        n_bc_per_edge: number of points on each of the four spatial edges.
        seed: seed for the IC draw; the BC draw uses seed + 1.
        epsilon, ic, ic_seed: as in sample_points, for the IC targets.
        device: device to place the tensors on.

    Returns:
        (ic_points (n_ic, 3), ic_values (n_ic, 1), bc_points (4 * n_bc_per_edge, 3)).
        bc_points are ordered left, right, bottom, top edge -- the equal-block
        layout bc_loss expects -- with requires_grad=True.
    """
    lo = torch.tensor(lower)
    hi = torch.tensor(upper)

    unit_ic = torch.rand(n_ic, 2, generator=torch.Generator().manual_seed(seed))
    xy_ic = lo[:2] + unit_ic * (hi[:2] - lo[:2])
    ic_points = torch.cat([xy_ic, torch.full((n_ic, 1), lower[2])], dim=1).to(device)
    ic_values = ic_targets(ic_points, ic=ic, epsilon=epsilon, ic_seed=ic_seed)

    bc_gen = torch.Generator().manual_seed(seed + 1)
    edges = []
    # (fixed axis, fixed value, free spatial axis), in bc_loss's edge order.
    for axis, value, free in ((0, lower[0], 1), (0, upper[0], 1), (1, lower[1], 0), (1, upper[1], 0)):
        unit = torch.rand(n_bc_per_edge, 2, generator=bc_gen)
        edge = torch.empty(n_bc_per_edge, 3)
        edge[:, axis] = value
        edge[:, free] = lo[free] + unit[:, 0] * (hi[free] - lo[free])
        edge[:, 2] = lo[2] + unit[:, 1] * (hi[2] - lo[2])
        edges.append(edge)
    bc_points = torch.cat(edges, dim=0).to(device).requires_grad_(True)

    return ic_points, ic_values, bc_points


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
    ic_values = ic_targets(x_ic, ic=ic, epsilon=epsilon, ic_seed=ic_seed)

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
