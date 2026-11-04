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
    index = torch.bucketize(t.reshape(-1), edges[1:-1], right=True)
    flat = squared.reshape(-1)
    sums = torch.zeros(n_bins, device=flat.device, dtype=flat.dtype).index_add(0, index, flat)
    counts = torch.bincount(index, minlength=n_bins).to(flat.dtype)
    filled = counts > 0
    shares = torch.diff(edges)[filled]
    return torch.sum(shares / shares.sum() * sums[filled] / counts[filled])


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
