"""Comparison metrics for PINN vs. FEM Cahn-Hilliard solutions.

Used locally, once both a trained PINN (weights downloaded from Colab) and a
FEM reference solution (generated via src/fem/cahn_hilliard.py) are
available, to answer RQ3 (accuracy, stability, cost).
"""

from __future__ import annotations

import numpy as np


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 error of `prediction` against `reference`.

    Args:
        prediction: array of predicted values (e.g. PINN output on a grid).
        reference: array of reference values (e.g. FEM solution on the same grid).

    Returns:
        ||prediction - reference||_2 / ||reference||_2.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match "
            f"reference shape {reference.shape}"
        )
    ref_norm = np.linalg.norm(reference.ravel())
    if ref_norm == 0.0:
        raise ValueError("reference has zero L2 norm; relative error is undefined")
    return float(np.linalg.norm((prediction - reference).ravel()) / ref_norm)


def trapezoid_weights_2d(
    nx: int,
    ny: int,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Tensor-product trapezoid quadrature weights for a 2D grid.

    Args:
        nx, ny: grid resolution, matching np.linspace(*x_bounds, nx) and
            np.linspace(*y_bounds, ny) -- the same endpoint-inclusive grid
            src/pinn/evaluate.py::evaluate_on_grid builds.
        x_bounds, y_bounds: (lo, hi) domain bounds along each axis.

    Returns:
        (nx, ny) array of weights. Sums to the domain area
        (x_hi - x_lo) * (y_hi - y_lo).

    Half-weight at each edge, full weight elsewhere -- the standard 1D
    trapezoid rule, applied along each axis and outer-producted. Do not
    replace this with a uniform weight (area / (nx*ny)): on an
    endpoint-inclusive grid that double-counts the boundary and is only
    first-order accurate, vs. this rule's much faster convergence (see
    tests/test_common_metrics.py's notebook-bias regression guard).
    """
    if nx < 2 or ny < 2:
        raise ValueError("trapezoid_weights_2d needs at least 2 points per axis")
    wx = np.full(nx, (x_bounds[1] - x_bounds[0]) / (nx - 1))
    wx[0] *= 0.5
    wx[-1] *= 0.5
    wy = np.full(ny, (y_bounds[1] - y_bounds[0]) / (ny - 1))
    wy[0] *= 0.5
    wy[-1] *= 0.5
    return wx[:, None] * wy[None, :]


def free_energy(
    c: np.ndarray, grad_c: np.ndarray, epsilon: float, weight: np.ndarray | float
) -> float:
    """Ginzburg-Landau free energy E = sum_i w_i * [ F(c_i) + (eps^2/2)|grad c_i|^2 ].

    Matches src/fem/cahn_hilliard.py's energy_form term for term
    (F(c) = 1/4*(1-c**2)**2, coefficient epsilon**2/2 on the gradient term),
    so PINN and FEM energy values are directly comparable.

    Args:
        c: (...) phase-field values on a quadrature grid.
        grad_c: c.shape + (ndim,) gradient of c on the same grid -- the last
            axis holds the spatial components only (e.g. (du/dx, du/dy)),
            not a time derivative.
        epsilon: interface-width parameter. Use the run's own epsilon
            (e.g. checkpoint["args"]["epsilon"]), not a hardcoded default --
            a mismatch silently invalidates any PINN/FEM comparison.
        weight: quadrature weights broadcastable to c.shape, e.g. from
            trapezoid_weights_2d. Required: without an explicit weight this
            would silently return an unnormalised sum instead of an integral.

    Returns:
        Scalar free energy estimate.
    """
    c = np.asarray(c, dtype=np.float64)
    grad_c = np.asarray(grad_c, dtype=np.float64)
    if grad_c.shape[:-1] != c.shape:
        raise ValueError(
            f"grad_c shape {grad_c.shape} does not match c shape {c.shape} + (ndim,)"
        )
    density = 0.25 * (1.0 - c**2) ** 2 + 0.5 * epsilon**2 * np.sum(grad_c**2, axis=-1)
    return float(np.sum(density * weight))


def total_mass(c: np.ndarray, weight: np.ndarray | float) -> float:
    """Total mass m = sum_i w_i * c_i, matching src/fem/cahn_hilliard.py's
    mass_form = c*dx (a plain, signed integral of c -- not an absolute value).

    Args:
        c: phase-field values on a quadrature grid.
        weight: quadrature weights broadcastable to c.shape, e.g. from
            trapezoid_weights_2d.

    Returns:
        Scalar total mass.
    """
    return float(np.sum(np.asarray(c, dtype=np.float64) * weight))


def mass_conservation_error(
    c_t: np.ndarray, c_0: np.ndarray, weight: np.ndarray | float
) -> float:
    """Deviation of total mass (integral of c) at time t from its initial value.

    Args:
        c_t: phase-field values at time t.
        c_0: phase-field values at t=0, on the same grid as c_t.
        weight: quadrature weights broadcastable to c_t.shape, e.g. from
            trapezoid_weights_2d. Required for the same reason as in
            total_mass: without it the "mass" is a grid-size-dependent sum.

    Returns:
        Absolute difference in total mass between c_t and c_0.

    For the full mass-vs-time series that plotting.plot_mass_conservation
    wants, read the total_mass column of a run's diagnostics.csv (see
    src/pinn/diagnostics.py) instead -- that's computed via total_mass()
    above, not this function.
    """
    if np.shape(c_t) != np.shape(c_0):
        raise ValueError(
            f"c_t shape {np.shape(c_t)} does not match c_0 shape {np.shape(c_0)}"
        )
    return abs(total_mass(c_t, weight) - total_mass(c_0, weight))
