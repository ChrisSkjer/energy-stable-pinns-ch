"""Shared plotting helpers for PINN and FEM Cahn-Hilliard fields.

Framework-agnostic: takes plain numpy arrays only, so this module needs
neither torch (src/pinn) nor dolfinx/pyvista (src/fem) as a dependency.
Fields come from src/pinn/evaluate.py (via its .npz output) or from a FEM
run's saved arrays; scalar diagnostics come from a FEM run's
cahn_hilliard_diagnostics.csv (see src/fem/cahn_hilliard.py's
_DiagnosticsLogger) or a PINN run's diagnostics.csv (see
src/pinn/diagnostics.py) -- both share the same three columns:
t, free_energy, total_mass.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def phase_colormap():
    """Diverging blue<->red colormap for c/u in [-1, 1], neutral gray at 0.

    Port of src/fem/cahn_hilliard.py::_phase_colormap, so PINN and FEM plots
    share one visual convention. Kept as a separate copy rather than having
    cahn_hilliard.py import this one -- that module runs fine without
    matplotlib (pyvista resolves its own colormaps), and this copy's fallback
    keeps that true here too.
    """
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ModuleNotFoundError:
        return "coolwarm"
    # blue (c=-1) -> neutral gray (c=0) -> red (c=+1), equal arms
    return LinearSegmentedColormap.from_list(
        "phase", ["#2a78d6", "#f0efec", "#e34948"]
    )


def plot_field(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    ax=None,
    title: str | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap=None,
    colorbar_label: str | None = None,
):
    """Render one snapshot of a scalar field (PINN u/mu or FEM c) on a shared colormap.

    Args:
        x, y: (nx, ny) meshgrid coordinates.
        c: (nx, ny) field values.
        ax: existing matplotlib axes to draw on, or None to create a new figure.
        title: optional axes title, e.g. "PINN, t=0.10".
        vmin, vmax: colormap bounds. Default to [-1, 1], which fits the
            phase field u but not the chemical potential mu (unbounded) --
            pass an explicit symmetric range for mu, e.g. from a quantile of
            its magnitude.
        cmap: matplotlib colormap, or None to use phase_colormap() (the
            default, intended for a field bounded in [-1, 1]).
        colorbar_label: optional label for the colorbar, e.g. "mu".

    Returns:
        The axes drawn on (so callers can compose subplots).
    """
    if ax is None:
        _, ax = plt.subplots()

    mesh = ax.pcolormesh(
        x, y, c, cmap=(phase_colormap() if cmap is None else cmap), vmin=vmin, vmax=vmax, shading="auto"
    )
    cbar = ax.figure.colorbar(mesh, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    return ax


def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    c_pinn: np.ndarray,
    c_fem: np.ndarray,
    epsilon: float | None = None,
    **field_kwargs,
):
    """Side-by-side PINN field, FEM field, and their difference, at one time snapshot.

    Extra field_kwargs (vmin, vmax, cmap, colorbar_label) go to plot_field
    for the PINN and FEM panels, e.g. to plot mu instead of u.

    Visual counterpart to metrics.relative_l2_error -- keep both in view
    when writing this one, they should tell the same story two ways.

    x, y, c_fem come from a FEM run's fem_fields.npz (see
    src/fem/cahn_hilliard.py::_FieldRecorder); c_pinn from a PINN run's
    evaluation.npz (see src/pinn/evaluate.py::save_evaluation) evaluated on
    the same x, y grid. src/pinn/plot_results.py wires the two together.

    Returns:
        (fig, (ax_pinn, ax_fem, ax_diff)).
    """
    fig, (ax_pinn, ax_fem, ax_diff) = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    plot_field(x, y, c_pinn, ax=ax_pinn, title="PINN", **field_kwargs)
    plot_field(x, y, c_fem, ax=ax_fem, title="FEM", **field_kwargs)

    diff = c_pinn - c_fem
    # Symmetric bounds around 0, sized to this diff's own range (unlike the
    # phase fields, which share a fixed [-1, 1] scale) -- and never exactly
    # 0, which would otherwise hand pcolormesh a degenerate [0, 0] color range.
    diff_bound = max(float(np.abs(diff).max()), 1e-12)
    mesh = ax_diff.pcolormesh(
        x, y, diff, cmap="RdBu_r", vmin=-diff_bound, vmax=diff_bound, shading="auto"
    )
    fig.colorbar(mesh, ax=ax_diff)
    ax_diff.set_aspect("equal")
    ax_diff.set_xlabel("x")
    ax_diff.set_ylabel("y")
    ax_diff.set_title("PINN - FEM")

    if epsilon is not None:
        fig.suptitle(f"epsilon = {epsilon:g}")

    return fig, (ax_pinn, ax_fem, ax_diff)


def plot_loss_history(history: dict[str, list[float]], ax=None):
    """Plot per-term PINN training loss curves (log scale) vs. optimizer step.

    Args:
        history: dict of per-component loss histories, e.g. the "history"
            dict bundled into a checkpoint by
            src/pinn/train.py::save_checkpoint (read via
            torch.load(checkpoint_path)["history"]). Keys are "pde", "ic",
            "bc", "total", and "energy" when --energy-penalty was enabled
            for that run; each component's values already include its
            --*-weight multiplier, i.e. these are the same numbers
            train.py prints during training, not the raw unweighted terms.
            Adam and L-BFGS steps are appended to the same flat lists, so
            there's no marker here for where L-BFGS refinement takes over.
        ax: existing matplotlib axes to draw on, or None to create a new figure.

    Returns:
        The axes the loss curves were drawn on.

    For the run's settings (network size, domain, loss weights, etc.), see
    src/pinn/run_summary.py::format_run_summary -- written to a separate
    run_summary.txt file rather than onto this plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name in ("pde", "ic", "bc", "energy"):
        if name in history:
            ax.semilogy(history[name], lw=1.5, alpha=0.8, label=name)
    if "total" in history:
        ax.semilogy(history["total"], lw=2.5, color="black", label="total")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("weighted loss")
    ax.set_title("Training loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def load_diagnostics(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a (t, free_energy, total_mass) diagnostics log.

    Args:
        csv_path: path to a cahn_hilliard_diagnostics.csv written by
            src/fem/cahn_hilliard.py's _DiagnosticsLogger, or a
            diagnostics.csv written by src/pinn/diagnostics.py -- same
            three-column format.

    Returns:
        (t, free_energy, total_mass) arrays, each at least 1-D (a
        single-row file would otherwise come back as 0-d scalars, which
        breaks plot_mass_conservation's mass[0] indexing).
    """
    t, energy, mass = np.loadtxt(csv_path, delimiter=",", skiprows=1, unpack=True)
    return np.atleast_1d(t), np.atleast_1d(energy), np.atleast_1d(mass)


def plot_energy_dissipation(
    t_pinn: np.ndarray | None = None,
    energy_pinn: np.ndarray | None = None,
    t_fem: np.ndarray | None = None,
    energy_fem: np.ndarray | None = None,
    ax=None,
):
    """Plot PINN and/or FEM free-energy decay over time.

    Cahn-Hilliard dissipates free energy monotonically -- this is the usual
    sanity check for whether the energy-stability penalty (see
    src/pinn/losses.py::energy_stability_loss) is actually doing its job,
    vs. the baseline PINN run without --energy-penalty.

    Either series is optional: pass only t_pinn/energy_pinn to check a PINN
    run on its own (e.g. before a FEM reference exists), or only
    t_fem/energy_fem for the reverse. Passing both overlays them for
    comparison, as before. At least one pair is required.

    energy_pinn: computed from the PINN's own u output via
    src/pinn/diagnostics.py (metrics.free_energy under the hood), typically
    read back from a run's diagnostics.csv via load_diagnostics.
    """
    if t_pinn is None and t_fem is None:
        raise ValueError(
            "plot_energy_dissipation needs at least one of (t_pinn, energy_pinn) "
            "or (t_fem, energy_fem)"
        )

    if ax is None:
        _, ax = plt.subplots()

    if t_pinn is not None:
        ax.plot(t_pinn, energy_pinn, lw=2, label="PINN")
    if t_fem is not None:
        ax.plot(t_fem, energy_fem, lw=2, label="FEM")
    ax.set_xlabel("t")
    ax.set_ylabel("free energy E(t)")
    ax.set_title("Energy dissipation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_mass_conservation(
    t_pinn: np.ndarray | None = None,
    mass_pinn: np.ndarray | None = None,
    t_fem: np.ndarray | None = None,
    mass_fem: np.ndarray | None = None,
    ax=None,
):
    """Plot PINN and/or FEM total mass over time (should stay ~constant).

    Either series is optional, same as plot_energy_dissipation -- at least
    one of (t_pinn, mass_pinn) or (t_fem, mass_fem) is required.

    Plots deviation from each series' own t=0 mass rather than raw mass --
    easier to read both curves on one axis if their absolute mass values
    differ slightly (e.g. from grid discretisation of the same IC).
    """
    if t_pinn is None and t_fem is None:
        raise ValueError(
            "plot_mass_conservation needs at least one of (t_pinn, mass_pinn) "
            "or (t_fem, mass_fem)"
        )

    if ax is None:
        _, ax = plt.subplots()

    if t_pinn is not None:
        mass_pinn = np.asarray(mass_pinn)
        ax.plot(t_pinn, mass_pinn - mass_pinn[0], lw=2, label="PINN")
    if t_fem is not None:
        mass_fem = np.asarray(mass_fem)
        ax.plot(t_fem, mass_fem - mass_fem[0], lw=2, label="FEM")
    ax.axhline(0.0, lw=1, color="0.6", zorder=0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"mass drift  $\int u\,dA - \int u_0\,dA$")
    ax.set_title("Mass conservation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
