"""Shared plotting helpers for PINN and FEM Cahn-Hilliard fields.

Framework-agnostic: takes plain numpy arrays only, so this module needs
neither torch (src/pinn) nor dolfinx/pyvista (src/fem) as a dependency.
Fields come from src/pinn/evaluate.py (via its .npz output) or from a FEM
run's saved arrays; scalar diagnostics come from a FEM run's
cahn_hilliard_diagnostics.csv (see src/fem/cahn_hilliard.py's
_DiagnosticsLogger: columns t, free_energy, total_mass).
"""

from __future__ import annotations

import numpy as np


def phase_colormap():
    """Diverging blue<->red colormap for c/u in [-1, 1], neutral gray at 0.

    Port of src/fem/cahn_hilliard.py::_phase_colormap, so PINN and FEM plots
    share one visual convention. Consider moving the implementation here and
    having cahn_hilliard.py import it back, rather than duplicating it --
    but that module also falls back to a plain "coolwarm" string when
    matplotlib isn't importable (pyvista ships its own colormap resolution),
    which may or may not be a concern here.
    """
    # TODO
    raise NotImplementedError


def plot_field(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    ax=None,
    title: str | None = None,
):
    """Render one snapshot of a phase field (PINN u or FEM c) on a shared colormap.

    Args:
        x, y: (nx, ny) meshgrid coordinates.
        c: (nx, ny) field values.
        ax: existing matplotlib axes to draw on, or None to create a new figure.
        title: optional axes title, e.g. "PINN, t=0.10".

    Returns:
        The axes drawn on (so callers can compose subplots).
    """
    # TODO: ax.pcolormesh / contourf with phase_colormap(), clim=[-1, 1], colorbar
    raise NotImplementedError


def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    c_pinn: np.ndarray,
    c_fem: np.ndarray,
    epsilon: float | None = None,
):
    """Side-by-side PINN field, FEM field, and their difference, at one time snapshot.

    Visual counterpart to metrics.relative_l2_error -- keep both in view
    when writing this one, they should tell the same story two ways.

    OPEN QUESTION: src/fem/cahn_hilliard.py currently only saves PNG frames
    (via pyvista) and scalar diagnostics, not raw (nx, ny) field arrays -- so
    there's nothing to pass as c_fem yet without adding an array dump to the
    FEM solver (e.g. alongside _FrameWriter.save, or as an .npz per logged
    step). Needs deciding before this function has real input to consume.
    """
    # TODO: 1x3 subplot: PINN field, FEM field, (PINN - FEM) on a separate
    # diverging colormap centered at 0
    raise NotImplementedError


def load_fem_diagnostics(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a FEM run's (t, free_energy, total_mass) log.

    Args:
        csv_path: path to a cahn_hilliard_diagnostics.csv written by
            src/fem/cahn_hilliard.py's _DiagnosticsLogger.

    Returns:
        (t, free_energy, total_mass) arrays.
    """
    # TODO: np.loadtxt(csv_path, delimiter=",", skiprows=1, unpack=True)
    raise NotImplementedError


def plot_energy_dissipation(
    t_pinn: np.ndarray,
    energy_pinn: np.ndarray,
    t_fem: np.ndarray,
    energy_fem: np.ndarray,
    ax=None,
):
    """Overlay PINN vs. FEM free-energy decay over time.

    Cahn-Hilliard dissipates free energy monotonically -- this is the usual
    sanity check for whether the energy-stability penalty (see
    src/pinn/losses.py::energy_stability_loss) is actually doing its job,
    vs. the baseline PINN run without --energy-penalty.

    energy_pinn: needs computing from the PINN's own u, mu output --
    metrics.free_energy (currently also a stub) is presumably the source
    for this once implemented.
    """
    # TODO: line plot of both energy curves against t, shared axes
    raise NotImplementedError


def plot_mass_conservation(
    t_pinn: np.ndarray,
    mass_pinn: np.ndarray,
    t_fem: np.ndarray,
    mass_fem: np.ndarray,
    ax=None,
):
    """Overlay PINN vs. FEM total mass over time (both should stay ~constant).

    Consider plotting deviation from each series' t=0 mass rather than raw
    mass -- easier to read both curves on one axis if their absolute mass
    values differ slightly (e.g. from grid discretisation of the same IC).
    """
    # TODO
    raise NotImplementedError
