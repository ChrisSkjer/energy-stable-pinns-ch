"""Domain sampling for the Cahn-Hilliard PINN.

Bundles the collocation/IC/BC point tensors a training step needs into one
dataclass, rather than passing them as loose positional arguments (easy to
mis-order) or repurposing argparse.Namespace (meant for CLI args, not data).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.initial_conditions import initial_condition, initial_condition_1d
from src.pinn.losses import DEFAULT_EPSILON


@dataclass
class TrainingPoints:
    """Point tensors sampled from the domain for one training run.

    d below is the number of input columns: 3 for (x, y, t), 2 for (x, t).

    Attributes:
        collocation: (N + pde_at_t0, d) PDE residual points -- the interior
            points followed by t0_points -- requires_grad=True.
        ic_points: (N_ic, d) points at t = t_lo.
        ic_values: (N_ic, 1) target u values at ic_points.
        bc_points: (N_bc, d) points on the spatial walls, requires_grad=True.
        t0_points: (pde_at_t0, d) the IC points also used as PDE points, kept
            so a resample of the interior points can re-append the same ones.
    """

    collocation: torch.Tensor
    ic_points: torch.Tensor
    ic_values: torch.Tensor
    bc_points: torch.Tensor
    t0_points: torch.Tensor


def domain_bounds(
    dim: int, x_max: float, y_max: float, t_max: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(lower, upper) corners of the (x, y, t) box for dim=2, or the (x, t)
    box for dim=1 (y_max unused). Lower bounds are all 0."""
    upper = (x_max, y_max, t_max) if dim == 2 else (x_max, t_max)
    return (0.0,) * len(upper), upper


def with_pde_at_t0(collocation: torch.Tensor, t0_points: torch.Tensor) -> torch.Tensor:
    """Append the fixed t=0 PDE points (see sample_points' pde_at_t0) to a
    set of interior collocation points, as one requires_grad leaf tensor.

    Kept separate so train.py can resample the interior points without
    redrawing the t=0 ones.
    """
    return torch.cat([collocation.detach(), t0_points], dim=0).requires_grad_(True)


def ic_targets(points: torch.Tensor, ic: str, epsilon: float, ic_seed: int = 0) -> torch.Tensor:
    """(N, 1) initial-condition u values at the spatial columns of `points`,
    on the same device and dtype. (N, 2) points (x, t) use the 1D ICs
    (INITIAL_CONDITIONS_1D), (N, 3) points (x, y, t) the 2D ones.

    IC targets need no grad, so evaluating them in numpy is fine.
    """
    points_np = points.detach().cpu().numpy()
    if points_np.shape[1] == 2:
        values = initial_condition_1d(ic, points_np[:, 0], epsilon=epsilon, seed=ic_seed)
    else:
        values = initial_condition(ic, points_np[:, 0], points_np[:, 1], epsilon=epsilon, seed=ic_seed)
    return torch.as_tensor(values, dtype=points.dtype, device=points.device).unsqueeze(1)


def _walls(n_space: int) -> list[tuple[int, int, list[int]]]:
    """(normal axis, 0 = low / 1 = high side, free axes) for each wall of the
    spatial box, in bc_loss's block order: axis 0 low, axis 0 high, axis 1
    low, ... The free axes are the other spatial axes followed by t."""
    walls = []
    for axis in range(n_space):
        free = [a for a in range(n_space) if a != axis] + [n_space]
        walls.append((axis, 0, free))
        walls.append((axis, 1, free))
    return walls


def heldout_ic_bc_points(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
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
        lower, upper: (x, y, t) domain bounds, or (x, t) in 1D.
        n_ic: number of points on t = t_lo.
        n_bc_per_edge: number of points on each spatial wall (the 4 edges in
            2D, the 2 end points x = 0 and x = L in 1D).
        seed: seed for the IC draw; the BC draw uses seed + 1.
        epsilon, ic, ic_seed: as in sample_points, for the IC targets.
        device: device to place the tensors on.

    Returns:
        (ic_points (n_ic, d), ic_values (n_ic, 1),
        bc_points (2 * n_space * n_bc_per_edge, d)). bc_points are in the
        equal-block wall order bc_loss expects (in 2D left, right, bottom,
        top edge), with requires_grad=True.
    """
    n_space = len(lower) - 1
    lo = torch.tensor(lower)
    hi = torch.tensor(upper)

    unit_ic = torch.rand(n_ic, n_space, generator=torch.Generator().manual_seed(seed))
    x_ic = lo[:n_space] + unit_ic * (hi[:n_space] - lo[:n_space])
    ic_points = torch.cat([x_ic, torch.full((n_ic, 1), lower[-1])], dim=1).to(device)
    ic_values = ic_targets(ic_points, ic=ic, epsilon=epsilon, ic_seed=ic_seed)

    bc_gen = torch.Generator().manual_seed(seed + 1)
    edges = []
    for axis, side, free in _walls(n_space):
        unit = torch.rand(n_bc_per_edge, n_space, generator=bc_gen)
        edge = torch.empty(n_bc_per_edge, n_space + 1)
        edge[:, axis] = (lower, upper)[side][axis]
        for j, a in enumerate(free):
            edge[:, a] = lo[a] + unit[:, j] * (hi[a] - lo[a])
        edges.append(edge)
    bc_points = torch.cat(edges, dim=0).to(device).requires_grad_(True)

    return ic_points, ic_values, bc_points


def sample_points(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
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
        lower: (x_lo, y_lo, t_lo) domain lower bound, or (x_lo, t_lo) in 1D.
        upper: (x_hi, y_hi, t_hi) domain upper bound, or (x_hi, t_hi) in 1D.
        collocation: (N, d) interior PDE points, from a sampler in
            src/pinn/collocation.py (the choice of strategy lives there).
        n_ic: points per spatial axis of the initial-condition grid
            (n_ic x n_ic in 2D, n_ic in 1D).
        n_bc: points per axis of the grid on each wall (n_bc x n_bc over
            (other space axis, t) in 2D, n_bc times at each end in 1D).
        device: device to place the tensors on.
        pde_at_t0: if > 0, also enforce the PDE residual at this many of the
            initial-condition points (helps the residual stay consistent at t=0).
        epsilon: interface half-width for the IC profile; pass the same value
            used for `epsilon` in `pde_residual_loss` to keep the IC in
            equilibrium with the PDE.
        ic: name of the initial condition, a key of
            `src.common.initial_conditions.INITIAL_CONDITIONS` (2D) or
            `INITIAL_CONDITIONS_1D` (1D).
        ic_seed: random seed for the initial condition (only "noise" uses it).

    Returns:
        A TrainingPoints bundle.
    """
    n_space = len(lower) - 1
    axes = [torch.linspace(lower[a], upper[a], n_ic, device=device) for a in range(n_space)]
    grid = torch.meshgrid(*axes, indexing="ij")
    x_ic = torch.stack([g.reshape(-1) for g in grid], dim=1)
    x_ic = torch.cat([x_ic, torch.full((x_ic.shape[0], 1), lower[-1], device=device)], dim=1)

    idx = torch.randperm(x_ic.shape[0], device=device)[:pde_at_t0]
    t0_points = x_ic[idx]
    x_pde = with_pde_at_t0(collocation.to(device), t0_points)
    ic_values = ic_targets(x_ic, ic=ic, epsilon=epsilon, ic_seed=ic_seed)

    bc_axes = [torch.linspace(lower[a], upper[a], n_bc, device=device) for a in range(n_space + 1)]
    walls = []
    for axis, side, free in _walls(n_space):
        free_grid = torch.meshgrid(*[bc_axes[a] for a in free], indexing="ij")
        wall = torch.empty(free_grid[0].numel(), n_space + 1, device=device)
        wall[:, axis] = (lower, upper)[side][axis]
        for a, g in zip(free, free_grid):
            wall[:, a] = g.reshape(-1)
        walls.append(wall)
    x_bc = torch.cat(walls, dim=0).requires_grad_(True)

    return TrainingPoints(
        collocation=x_pde, ic_points=x_ic, ic_values=ic_values, bc_points=x_bc, t0_points=t0_points
    )
