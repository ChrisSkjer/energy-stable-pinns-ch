"""Loss terms for training the Cahn-Hilliard PINN.

Separate functions for the PDE residual, IC/BC terms, and the
energy-stability and mass-conservation penalties used only by the
enhanced model (integrals over a QuadratureGrid of time slices).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Cahn-Hilliard interface half-width. This is the single source of truth for
# epsilon: `sampling.sample_points` defaults to it so the initial-condition
# profile stays consistent with the PDE residual (a mismatch here puts the
# IC out of equilibrium and forces the network to resolve a stiff transient
# at t = 0). Pass `epsilon` explicitly through `train.py` rather than
# overriding this default in only one of the two places it's used.
DEFAULT_EPSILON = 0.05


def window_weighted_mean(squared: torch.Tensor, t: torch.Tensor, window: tuple[float, float, float] | None) -> torch.Tensor:
    """Mean of `squared` over the points, or, with window = (t_start, t_end,
    fraction), the stratified mean

        fraction * mean(inside) + (1 - fraction) * mean(outside)

    where inside means t_start < t < t_end and `fraction` is the window's
    share of the time interval. This estimates the same space-time average
    as uniform points would, so extra points in the window (--dense-window)
    add resolution there without giving the window more weight in the loss.
    Falls back to the plain mean if either side has no points.
    """
    if window is None:
        return torch.mean(squared)
    t_start, t_end, fraction = window
    inside = ((t > t_start) & (t < t_end)).squeeze(-1)
    n_inside = int(inside.sum())
    if n_inside == 0 or n_inside == inside.numel():
        return torch.mean(squared)
    return fraction * torch.mean(squared[inside]) + (1 - fraction) * torch.mean(squared[~inside])


def time_binned_mean(squared: torch.Tensor, t: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Stratified mean of `squared` over time bins,

        sum_k f_k * mean(bin k),   f_k = (edges[k+1] - edges[k]) / (edges[-1] - edges[0]),

    so every bin counts by its share of the time interval, however many points
    it holds. Extra points that cluster in time (--adaptive-reweight-bins) then
    add resolution there without giving that time more weight in the loss; with
    evenly spread points it is (about) the plain mean. Points outside the edges
    go to the first/last bin. Empty bins are dropped and the remaining shares
    renormalised.
    """
    edges = edges.to(t.device, t.dtype)
    n_bins = edges.numel() - 1
    index = torch.bucketize(t.reshape(-1).contiguous(), edges[1:-1], right=True)
    flat = squared.reshape(-1)
    sums = torch.zeros(n_bins, device=flat.device, dtype=flat.dtype).index_add(0, index, flat)
    counts = torch.bincount(index, minlength=n_bins).to(flat.dtype)
    filled = counts > 0
    shares = torch.diff(edges)[filled]
    return torch.sum(shares / shares.sum() * sums[filled] / counts[filled])


def space_binned_mean(values: torch.Tensor, x: torch.Tensor, bins: tuple[tuple[float, ...], tuple[float, ...], int]) -> torch.Tensor:
    """Stratified mean of `values` over K equal cells per spatial axis
    (K x K in 2D, K in 1D), every cell counting the same as it covers the
    same area, however many points it holds:

        mean over the non-empty cells of mean(cell).

    The spatial analogue of time_binned_mean, for the IC loss: extra points
    that cluster in space (--ic-reweight-bins with adaptive IC points) then
    add resolution there without giving that region more weight; on the
    uniform IC grid alone it is (about) the plain mean. Points outside the
    box go to the edge cells.

    Args:
        values: (N, 1) or (N,) pointwise values.
        x: (N, n_space) spatial coordinates.
        bins: (lower, upper, K): spatial box bounds and cells per axis.
    """
    lower, upper, k = bins
    lo = torch.tensor(lower, device=x.device, dtype=x.dtype)
    hi = torch.tensor(upper, device=x.device, dtype=x.dtype)
    cell = torch.clamp(((x - lo) / (hi - lo) * k).long(), 0, k - 1)
    strides = k ** torch.arange(x.shape[1], device=x.device)
    index = (cell * strides).sum(dim=1)
    n_cells = k ** x.shape[1]
    flat = values.reshape(-1)
    sums = torch.zeros(n_cells, device=flat.device, dtype=flat.dtype).index_add(0, index, flat)
    counts = torch.bincount(index, minlength=n_cells).to(flat.dtype)
    filled = counts > 0
    return torch.mean(sums[filled] / counts[filled])


def pde_residual_losses(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0, window: tuple[float, float, float] | None = None, t_bins: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        window: (t_start, t_end, fraction) to reweight a denser time window
            back to its share of the interval (see window_weighted_mean);
            None = plain mean over the points.
        t_bins: (K + 1,) time-bin edges to weight each bin by its share of
            the interval instead (see time_binned_mean); not together with
            `window`.

    Returns:
        (loss_pde_u, loss_pde_mu): scalar MSEs of the u- and mu-equation residuals.
    """
    residual_u, residual_mu = pde_residuals(model, collocation_points, epsilon=epsilon, m=m)
    t = collocation_points[:, -1:]
    if t_bins is not None:
        if window is not None:
            raise ValueError("pass either window or t_bins, not both")
        return time_binned_mean(residual_u**2, t, t_bins), time_binned_mean(residual_mu**2, t, t_bins)
    return window_weighted_mean(residual_u**2, t, window), window_weighted_mean(residual_mu**2, t, window)


def pde_residuals(model: nn.Module, collocation_points: torch.Tensor, epsilon = DEFAULT_EPSILON, m = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Pointwise (N, 1) residuals (residual_u, residual_mu) behind
    `pde_residual_losses`, e.g. to rank points for adaptive sampling.
    `collocation_points` needs requires_grad=True."""
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
    return residual_u, residual_mu


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
    space_bins: tuple[tuple[float, ...], tuple[float, ...], int] | None = None,
) -> torch.Tensor:
    """Initial-condition loss.

    Args:
        model: the PINN.
        ic_points: (N_ic, input_dim) points at t=0.
        ic_values: (N_ic, 1) target u values at ic_points -- the u channel
            only, from `src.common.initial_conditions`. There is no
            target for mu here: its t=0 value is not free to choose, it is
            fixed by the IC through mu = f'(u0) - eps^2 * laplacian(u0).
        space_bins: (lower, upper, K) to take both means as cell-stratified
            means over K cells per spatial axis (see space_binned_mean), so
            clustered adaptive IC points don't raise the term's weight;
            None = plain means.

    Returns:
        Scalar MSE of the IC term, normalised by the mean square of the IC.
    """
    squared = ic_squared_errors(model, ic_points, ic_values)
    if space_bins is not None:
        x = ic_points[:, :-1]
        return space_binned_mean(squared, x, space_bins) / space_binned_mean(ic_values**2, x, space_bins)
    return torch.mean(squared)/torch.mean(ic_values**2)


def ic_squared_errors(model: nn.Module, ic_points: torch.Tensor, ic_values: torch.Tensor) -> torch.Tensor:
    """Pointwise (N_ic, 1) (u(x, 0) - u0(x))^2 behind `ic_loss` (before its
    normalisation by mean(u0^2)), e.g. for self-adaptive point weights."""
    return (model(ic_points)[:, 0:1] - ic_values) ** 2


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
    return torch.mean(bc_squared_normals(model, bc_points))


def bc_squared_normals(model: nn.Module, bc_points: torch.Tensor) -> torch.Tensor:
    """Pointwise (N_bc, 1) (du/dn)^2 + (dmu/dn)^2 behind `bc_loss` (whose
    mean it is), e.g. for self-adaptive point weights. `bc_points` as in
    `bc_loss`, with requires_grad=True."""
    out_bc = model(bc_points)
    u_bc, mu_bc = out_bc[:,0:1], out_bc[:,1:2]
    grad_u_bc = torch.autograd.grad(u_bc, bc_points, grad_outputs=torch.ones_like(u_bc), create_graph=True)[0]
    grad_mu_bc = torch.autograd.grad(mu_bc, bc_points, grad_outputs=torch.ones_like(mu_bc), create_graph=True)[0]

    n_walls = 2 * (bc_points.shape[1] - 1)
    n_edge = bc_points.shape[0] // n_walls
    du_dn = torch.cat([grad_u_bc[k*n_edge:(k+1)*n_edge, k//2:k//2+1] for k in range(n_walls)], dim=0)
    dmu_dn = torch.cat([grad_mu_bc[k*n_edge:(k+1)*n_edge, k//2:k//2+1] for k in range(n_walls)], dim=0)

    return du_dn**2 + dmu_dn**2


@dataclass
class QuadratureGrid:
    """Space-time points for the integral (energy, mass) loss terms: the same
    endpoint-inclusive trapezoid grid in space at each of n_t fixed time
    slices.

    points: (n_t * n_space, input_dim) rows (x, [y,] t), slice by slice,
        requires_grad=True.
    weights: (n_space,) trapezoid weights of the spatial grid; they sum to
        the domain size |Omega|.
    n_t: number of time slices.
    """

    points: torch.Tensor
    weights: torch.Tensor
    n_t: int

    @property
    def times(self) -> torch.Tensor:
        """(n_t,) the time of each slice."""
        return self.points[:: self.weights.numel(), -1].detach()

    def integrate(self, density: torch.Tensor) -> torch.Tensor:
        """(n_t,) trapezoid integral over space of a pointwise (N, 1) density
        on `points`, one per time slice."""
        return density.reshape(self.n_t, -1) @ self.weights


def trapezoid_weights(n: int, lower: tuple[float, ...], upper: tuple[float, ...], device=None, dtype=torch.float32) -> torch.Tensor:
    """Flattened tensor-product trapezoid weights on the grid of n points per
    axis between `lower` and `upper` (endpoints included), in the order of
    torch.meshgrid(..., indexing="ij"). The torch, any-dimension version of
    src/common/metrics.py::trapezoid_weights_2d, so the loss integrates
    exactly as the diagnostics plots do."""
    if n < 2:
        raise ValueError("trapezoid_weights needs at least 2 points per axis")
    weights = torch.ones((), device=device, dtype=dtype)
    for lo, hi in zip(lower, upper):
        w = torch.full((n,), (hi - lo) / (n - 1), device=device, dtype=dtype)
        w[0] *= 0.5
        w[-1] *= 0.5
        weights = (weights.unsqueeze(-1) * w).reshape(-1)
    return weights


def quadrature_grid(lower: tuple[float, ...], upper: tuple[float, ...], n_space: int, n_t: int, device=None, dtype=torch.float32) -> QuadratureGrid:
    """QuadratureGrid with n_space points per spatial axis and n_t evenly
    spaced time slices over [t_lo, t_hi], both ends included.

    Args:
        lower, upper: corners of the (x, [y,] t) box, time last.
    """
    axes = [torch.linspace(lo, hi, n_space, device=device, dtype=dtype) for lo, hi in zip(lower[:-1], upper[:-1])]
    space = torch.stack([g.reshape(-1) for g in torch.meshgrid(*axes, indexing="ij")], dim=1)
    times = torch.linspace(lower[-1], upper[-1], n_t, device=device, dtype=dtype)
    points = torch.cat(
        [torch.cat([space, torch.full((space.shape[0], 1), float(t), device=device, dtype=dtype)], dim=1) for t in times]
    )
    weights = trapezoid_weights(n_space, lower[:-1], upper[:-1], device=device, dtype=dtype)
    return QuadratureGrid(points.requires_grad_(True), weights, n_t)


def energy_rate(model: nn.Module, grid: QuadratureGrid, epsilon=DEFAULT_EPSILON) -> torch.Tensor:
    """dE/dt of the network's Ginzburg-Landau free energy

        E(t) = int F(u) + (eps^2 / 2) |grad u|^2 dx,   F(u) = (u^2 - 1)^2 / 4,

    (the energy src/common/metrics.py::free_energy plots) at each time slice
    of `grid`, differentiated under the integral by autograd:

        dE/dt = int f'(u) u_t + eps^2 grad u . grad u_t dx,   f'(u) = u^3 - u.

    It uses u only, not the mu output, so it is the energy of the u the
    network actually predicts, whatever mu's residual is.

    Returns:
        (n_t,) dE/dt per slice.
    """
    points = grid.points
    u = model(points)[:, 0:1]
    grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u[:, -1:]
    # d/dx_i of u_t, i.e. u_tx_i: the gradient of the energy's gradient term.
    grad_u_t = torch.autograd.grad(u_t, points, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    gradient_term = torch.sum(grad_u[:, :-1] * grad_u_t[:, :-1], dim=1, keepdim=True)
    return grid.integrate((u**3 - u) * u_t + epsilon**2 * gradient_term)


def energy_stability_loss(model: nn.Module, grid: QuadratureGrid, epsilon=DEFAULT_EPSILON) -> torch.Tensor:
    """Energy-dissipation penalty for the enhanced PINN.

    Cahn-Hilliard is a gradient flow, dE/dt = -m int |grad mu|^2 <= 0, so the
    network's free energy must not increase. Penalises the slices where it
    does:

        mean over slices of relu(dE/dt)^2,

    with dE/dt from `energy_rate`. Zero when E is non-increasing at every
    slice; not normalised (dE/dt is in energy per time).

    Args:
        model: the PINN.
        grid: QuadratureGrid of time slices to check dE/dt at.
        epsilon: interface half-width, as in the PDE residual.

    Returns:
        Scalar penalty.
    """
    return torch.mean(torch.relu(energy_rate(model, grid, epsilon)) ** 2)


def total_mass(model: nn.Module, grid: QuadratureGrid) -> torch.Tensor:
    """(n_t,) total mass M(t) = int u dx of the network's u at each slice of
    `grid` (trapezoid rule, as src/common/metrics.py::total_mass)."""
    return grid.integrate(model(grid.points)[:, 0:1])


def mass_conservation_loss(model: nn.Module, grid: QuadratureGrid, mass0: float, u0_sq_mean: float) -> torch.Tensor:
    """Mass-conservation penalty: CH with no-flux walls keeps int u dx at its
    initial value, so

        mean over slices of ((M(t) - M0) / |Omega|)^2 / mean(u0^2),

    with M(t) from `total_mass` and M0 = int u0 dx of the exact IC (see
    `ic_mass`), so a network whose own t = 0 mass is off is penalised too.
    (M - M0) / |Omega| is the drift of the mean of u; dividing its square by
    mean(u0^2) makes it relative, the same normalisation as `ic_loss`.

    Args:
        model: the PINN.
        grid: QuadratureGrid of time slices to check the mass at.
        mass0, u0_sq_mean: from `ic_mass` on the same grid.
    """
    area = grid.weights.sum()
    return torch.mean(((total_mass(model, grid) - mass0) / area) ** 2) / u0_sq_mean


def ic_mass(grid: QuadratureGrid, u0: torch.Tensor) -> tuple[float, float]:
    """(M0, mean(u0^2)) for `mass_conservation_loss`: the trapezoid integral
    of the exact IC and its domain-mean square, on `grid`'s spatial grid.

    Args:
        u0: (n_space, 1) exact IC values at the spatial grid points (one
            slice of grid.points, e.g. from sampling.ic_targets).
    """
    area = grid.weights.sum()
    u0 = u0.reshape(-1).to(grid.weights)
    return float(u0 @ grid.weights), float((u0**2) @ grid.weights / area)
