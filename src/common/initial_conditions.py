"""Initial conditions u(x, y, t=0) for the Cahn-Hilliard runs.

Single source of truth shared by the PINN (src/pinn/sampling.py) and the FEM
solver (src/fem/cahn_hilliard.py), so a PINN run and its FEM ground truth
start from exactly the same field -- pass the same `--ic` (and seed) to both.

Numpy only: the FEM conda env has no torch.

Every IC has the signature `fn(x, y, *, epsilon, seed) -> ndarray` (same
shape as x), assumes the spatial domain is the unit square [0, 1]^2, and is
consistent with the no-flux boundary conditions (flat at the walls).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

InitialCondition = Callable[..., np.ndarray]


def _tanh_profile(sdf: np.ndarray, epsilon: float) -> np.ndarray:
    """Equilibrium Cahn-Hilliard interface profile u = -tanh(d / (sqrt(2) epsilon)).

    d is the signed distance to the interface, negative inside, so u = +1
    inside the shape and -1 outside. Using the PDE's own epsilon means the
    interface starts at its preferred width: no fast thickening transient at
    t = 0, so the early dynamics are pure curvature-driven motion.
    """
    return -np.tanh(sdf / (np.sqrt(2.0) * epsilon))


# --- "Swiss flag" cross -----------------------------------------------------
# Proportions follow the Swiss flag: arms 6 units wide and 20 units long on a
# 32-unit square, rescaled to the unit square. That gives an area fraction of
# ~0.199, i.e. a limiting circle of radius sqrt(0.199/pi) ~ 0.25.
_CROSS_ARM_HALF_W = (6.0 / 32.0) / 2.0  # half-width of an arm  = 0.09375
_CROSS_ARM_HALF_L = (20.0 / 32.0) / 2.0  # half-length of an arm = 0.3125
_CROSS_CENTER = (0.5, 0.5)


def _rect_sdf(px, py, half_w, half_h):
    """Exact signed distance to an axis-aligned rectangle centred at the
    origin, negative inside. The usual 2D box SDF: the first term measures the
    distance once outside, the second is the (negative) distance to the nearest
    edge while inside."""
    qx = np.abs(px) - half_w
    qy = np.abs(py) - half_h
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def _cross_sdf(x, y):
    """Signed distance to the cross, i.e. the union of a horizontal and a
    vertical bar. A union of SDFs is min(...): exact outside the cross, and a
    slight under-estimate only in the small pocket near the re-entrant corners
    -- harmless once it is squashed through a tanh."""
    px = x - _CROSS_CENTER[0]
    py = y - _CROSS_CENTER[1]
    horizontal = _rect_sdf(px, py, _CROSS_ARM_HALF_L, _CROSS_ARM_HALF_W)
    vertical = _rect_sdf(px, py, _CROSS_ARM_HALF_W, _CROSS_ARM_HALF_L)
    return np.minimum(horizontal, vertical)


def cross(x, y, *, epsilon: float, seed: int = 0) -> np.ndarray:
    """Swiss-flag cross: u = +1 on the cross, -1 on the background.

    The four re-entrant corners fill in, the arm tips retract, and the cross
    relaxes towards a circle of the same area (CH conserves mass). `seed` is
    unused.
    """
    return _tanh_profile(_cross_sdf(x, y), epsilon)


# --- two circles --------------------------------------------------------------
# Unequal radii, so coarsening (Ostwald ripening) is visible: the small circle
# shrinks and feeds the large one. Both sit clear of the walls (>= 0.13) and
# of each other (gap 0.20, ~3 interface widths at eps=0.05) so the interfaces
# don't overlap and merge at t = 0.
_CIRCLES = (((0.28, 0.5), 0.15), ((0.74, 0.5), 0.11))


def two_circles(x, y, *, epsilon: float, seed: int = 0) -> np.ndarray:
    """Two disks of different radius: u = +1 inside, -1 outside. `seed` is unused."""
    sdf = np.minimum.reduce([np.hypot(x - cx, y - cy) - r for (cx, cy), r in _CIRCLES])
    return _tanh_profile(sdf, epsilon)


# --- smooth random noise -------------------------------------------------------
_NOISE_MEAN = 0.0
_NOISE_AMPLITUDE = 0.05
_NOISE_MAX_MODE = 8  # shortest wavelength 2/8 = 0.25


def noise(x, y, *, epsilon: float = 0.0, seed: int = 0) -> np.ndarray:
    """Small random perturbation around a uniform mixture (spinodal decomposition).

    u = mean + amplitude * sum_{k,l} a_kl cos(k pi x) cos(l pi y), with seeded
    random coefficients over the low modes 0 <= k, l <= _NOISE_MAX_MODE.

    Smooth and deterministic in (x, y, seed) rather than i.i.d. per point, so
    the PINN's IC grid and the FEM mesh nodes -- different point sets -- see
    the same field. Each cosine mode has zero normal derivative on the walls
    (no-flux BC holds exactly), and the (0, 0) mode is dropped so the field
    integrates to exactly `mean`. The coefficients are scaled so that
    sum |a_kl| = 1, which bounds |u - mean| <= amplitude everywhere.

    `epsilon` is unused: there is no interface yet.
    """
    rng = np.random.default_rng(seed)
    n = _NOISE_MAX_MODE + 1
    coeffs = rng.uniform(-1.0, 1.0, size=(n, n))
    coeffs[0, 0] = 0.0
    coeffs /= np.abs(coeffs).sum()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    modes = np.arange(n)
    cos_x = np.cos(np.pi * modes * x[..., None])  # (..., n)
    cos_y = np.cos(np.pi * modes * y[..., None])  # (..., n)
    field = np.einsum("...k,kl,...l->...", cos_x, coeffs, cos_y)
    return _NOISE_MEAN + _NOISE_AMPLITUDE * field


INITIAL_CONDITIONS: dict[str, InitialCondition] = {
    "cross": cross,
    "two_circles": two_circles,
    "noise": noise,
}


def initial_condition(name: str, x, y, *, epsilon: float, seed: int = 0) -> np.ndarray:
    """Evaluate the initial condition `name` at points (x, y).

    Args:
        name: a key of INITIAL_CONDITIONS.
        x, y: arrays of spatial coordinates in [0, 1], same shape.
        epsilon: interface half-width; must match the PDE's epsilon.
        seed: random seed, used only by "noise".

    Returns:
        Array of u values, same shape as x.
    """
    try:
        fn = INITIAL_CONDITIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown initial condition {name!r}; choose from {sorted(INITIAL_CONDITIONS)}"
        ) from None
    return fn(x, y, epsilon=epsilon, seed=seed)
