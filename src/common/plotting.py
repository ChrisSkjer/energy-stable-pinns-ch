"""Shared plotting helpers for PINN and FEM Cahn-Hilliard fields.

Framework-agnostic: takes plain numpy arrays only, so this module needs
neither torch (src/pinn) nor dolfinx/pyvista (src/fem) as a dependency.
Fields come from src/pinn/evaluate.py (via its .npz output) or from a FEM
run's saved arrays; scalar diagnostics come from a FEM run's
cahn_hilliard_diagnostics.csv (see src/fem/output.py's
_DiagnosticsLogger) or a PINN run's diagnostics.csv (see
src/pinn/diagnostics.py) -- both share the same three columns:
t, free_energy, total_mass.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.common.metrics import relative_l2_error

# x axis of every plot indexed like a checkpoint's history: one entry per Adam
# epoch, then one per L-BFGS closure call -- line-search trial points
# included, which is why the L-BFGS part of a loss curve can spike.
STEP_LABEL = "step (Adam epochs, then L-BFGS loss evaluations)"


def phase_colormap():
    """Diverging blue<->red colormap for c/u in [-1, 1], neutral gray at 0.

    Port of src/fem/output.py::_phase_colormap, so PINN and FEM plots
    share one visual convention. Kept as a separate copy rather than having
    src/fem/output.py import this one -- that module runs fine without
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
        x, y, c, cmap=(phase_colormap() if cmap is None else cmap), vmin=vmin, vmax=vmax, shading="auto",
        rasterized=True,  # one embedded image in PDF output, not one vector quad per cell
    )
    # Cells outside [vmin, vmax] are drawn in the end colours, so without an
    # arrow on the colour bar a u of 1.3 would read as 1.0.
    below, above = np.nanmin(c) < vmin, np.nanmax(c) > vmax
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    cbar = ax.figure.colorbar(mesh, ax=ax, extend=extend)
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
    title: str | None = None,
    **field_kwargs,
):
    """Side-by-side PINN field, FEM field, and their difference, at one time snapshot.

    title (e.g. "t = 0.005") and epsilon go into the figure title. Extra
    field_kwargs (vmin, vmax, cmap, colorbar_label) go to plot_field for the
    PINN and FEM panels, e.g. to plot mu instead of u.

    Visual counterpart to metrics.relative_l2_error, whose value is shown
    in the difference panel's title so each figure carries its own number.

    x, y, c_fem come from a FEM run's fem_fields.npz (see
    src/fem/output.py::_FieldRecorder); c_pinn from a PINN run's
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
        x, y, diff, cmap="RdBu_r", vmin=-diff_bound, vmax=diff_bound, shading="auto",
        rasterized=True,
    )
    fig.colorbar(mesh, ax=ax_diff)
    ax_diff.set_aspect("equal")
    ax_diff.set_xlabel("x")
    ax_diff.set_ylabel("y")
    try:
        rel_l2 = relative_l2_error(c_pinn, c_fem)
        ax_diff.set_title(f"PINN - FEM  (rel. L2 = {rel_l2:.2%})")
    except ValueError:
        # FEM field is identically zero -- relative error undefined.
        ax_diff.set_title("PINN - FEM")

    if epsilon is not None:
        title = f"{title}, epsilon = {epsilon:g}" if title else f"epsilon = {epsilon:g}"
    if title is not None:
        fig.suptitle(title)

    return fig, (ax_pinn, ax_fem, ax_diff)


def plot_line_profile(
    x: np.ndarray,
    y: np.ndarray,
    c_pinn: np.ndarray,
    c_fem: np.ndarray,
    x_line: float | None = 0.25,
    title: str | None = None,
    y_line: float | None = None,
):
    """PINN and FEM field along one straight line, at one time snapshot:
    the vertical line x = x_line, or the horizontal line y = y_line if given.

    Zooms in on the diffuse interface: the 2D comparison panels show where
    the phases are, this shows the shape of the transition between them
    (width, steepness, over/undershoot past +-1) for both solvers on one axis.

    Args:
        x, y: (nx, ny) meshgrid coordinates (indexing="ij", as written by
            src/pinn/evaluate.py and src/fem/output.py's _FieldRecorder).
        c_pinn, c_fem: (nx, ny) field values on that same grid.
        x_line: x position of a vertical line; snapped to the closest grid column.
        title: optional figure title, e.g. "t=0.005".
        y_line: y position of a horizontal line; snapped to the closest grid
            row. Takes precedence over x_line, e.g. y_line=0.5 cuts through
            both circles of the "two_circles" initial condition.

    Returns:
        (fig, (ax_field, ax_profile)): the FEM field with the line marked,
        and the PINN/FEM profiles along it.
    """
    if y_line is not None:
        j = int(np.argmin(np.abs(y[0, :] - y_line)))
        snapped = float(y[0, j])
        along, cut = x[:, j], np.s_[:, j]
        fixed, free = "y", "x"
    elif x_line is not None:
        i = int(np.argmin(np.abs(x[:, 0] - x_line)))
        snapped = float(x[i, 0])
        along, cut = y[i, :], np.s_[i, :]
        fixed, free = "x", "y"
    else:
        raise ValueError("plot_line_profile needs x_line or y_line")

    fig, (ax_field, ax_profile) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.4]}
    )

    plot_field(x, y, c_fem, ax=ax_field, title="FEM")
    mark_line = ax_field.axhline if fixed == "y" else ax_field.axvline
    mark_line(snapped, color="black", lw=1.5, ls="--")

    ax_profile.plot(along, c_fem[cut], lw=2, label="FEM")
    ax_profile.plot(along, c_pinn[cut], lw=2, ls="--", label="PINN")
    ax_profile.axhline(1.0, lw=1, color="0.6", zorder=0)
    ax_profile.axhline(-1.0, lw=1, color="0.6", zorder=0)
    ax_profile.set_xlabel(free)
    point = f"x={snapped:.3g}, y" if fixed == "x" else f"x, y={snapped:.3g}"
    ax_profile.set_ylabel(f"u({point})")
    ax_profile.set_title(f"Profile along {fixed} = {snapped:.3g}")
    ax_profile.grid(True, alpha=0.3)
    ax_profile.legend()

    if title is not None:
        fig.suptitle(title)

    return fig, (ax_field, ax_profile)


def plot_heldout_residual(
    heldout: dict[str, list[float]],
    history: dict[str, list[float]],
    pde_u_weight: float = 1.0,
    pde_mu_weight: float = 1.0,
    ax=None,
):
    """Unweighted PDE residuals on held-out points vs. on the training
    collocation points, against optimizer step (log scale).

    A widening gap between the solid (held-out) and faded (training) curve
    of the same colour is the network overfitting its collocation points.

    Args:
        heldout: the "heldout" entry of a train.py checkpoint saved with
            --heldout-every: "step" (index into history), "pde_u", "pde_mu"
            (unweighted).
        history: the same checkpoint's "history"; its pde_u/pde_mu are
            weighted, so they are divided by the weights below.
        pde_u_weight, pde_mu_weight: the run's --pde-u-weight/--pde-mu-weight.
        ax: existing matplotlib axes to draw on, or None to create a new figure.

    Returns:
        The axes the curves were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, weight, color in (("pde_u", pde_u_weight, "C0"), ("pde_mu", pde_mu_weight, "C1")):
        if name in history:
            train = np.asarray(history[name]) / weight
            ax.semilogy(train, lw=1, alpha=0.4, color=color, label=f"{name} (training points)")
        ax.semilogy(heldout["step"], heldout[name], lw=2, marker=".", color=color, label=f"{name} (held-out)")

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("unweighted residual (MSE)")
    ax.set_title("PDE residual: training vs. held-out points")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_heldout_ic_bc(
    heldout: dict[str, list[float]],
    history: dict[str, list[float]],
    ic_weight: float = 1.0,
    bc_weight: float = 1.0,
    ax=None,
):
    """Unweighted IC and BC losses on held-out points vs. on the training
    IC/BC grids, against optimizer step (log scale).

    The companion of plot_heldout_residual: the training IC/BC points are
    regular grids, so a widening gap between the solid (held-out) and faded
    (training) curve means the network fits the condition only at the grid
    nodes.

    Args:
        heldout: the "heldout" entry of a train.py checkpoint saved with
            --heldout-every, with "step", "ic" and "bc" (unweighted).
        history: the same checkpoint's "history"; its ic/bc are weighted, so
            they are divided by the weights below.
        ic_weight, bc_weight: the run's --ic-weight/--bc-weight.
        ax: existing matplotlib axes to draw on, or None to create a new figure.

    Returns:
        The axes the curves were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, weight, color in (("ic", ic_weight, "C2"), ("bc", bc_weight, "C3")):
        if name in history:
            train = np.asarray(history[name]) / weight
            ax.semilogy(train, lw=1, alpha=0.4, color=color, label=f"{name} (training points)")
        ax.semilogy(heldout["step"], heldout[name], lw=2, marker=".", color=color, label=f"{name} (held-out)")

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("unweighted loss (MSE)")
    ax.set_title("IC/BC loss: training vs. held-out points")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_relative_l2_error(
    t: np.ndarray,
    errors: dict[str, np.ndarray],
    ax=None,
):
    """Plot PINN-vs-FEM relative L2 error (metrics.relative_l2_error) over time.

    Args:
        t: (nt,) evaluation times.
        errors: field name -> (nt,) relative L2 errors at those times, e.g.
            {"u": err_u, "mu": err_mu}. NaN entries (error undefined at that
            time) are left as gaps in the curve.
        ax: existing matplotlib axes to draw on, or None to create a new figure.

    Log-scale y axis: PINN error typically grows by orders of magnitude over
    the time window, and this is where that shows up -- the energy and mass
    curves can look fine while the field itself has drifted from FEM.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, err in errors.items():
        ax.semilogy(t, err, lw=2, marker="o", ms=4, label=name)
    ax.set_xlabel("t")
    ax.set_ylabel(r"relative L2 error  $\|f_{PINN} - f_{FEM}\|_2 / \|f_{FEM}\|_2$")
    ax.set_title("PINN vs. FEM relative L2 error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_loss_history(history: dict[str, list[float]], ax=None, adam_steps: int | None = None):
    """Plot per-term PINN training loss curves (log scale) vs. optimizer step.

    Args:
        history: dict of per-component loss histories, e.g. the "history"
            dict bundled into a checkpoint by
            src/pinn/train.py::save_checkpoint (read via
            torch.load(checkpoint_path)["history"]). Keys are "pde_u",
            "pde_mu", "ic", "bc", "total", and "energy" when
            --energy-penalty was enabled for that run (checkpoints from
            before the PDE split have a single "pde" key instead); each component's values already include its
            --*-weight multiplier, i.e. these are the same numbers
            train.py prints during training, not the raw unweighted terms.
            Adam and L-BFGS steps are appended to the same flat lists.
        ax: existing matplotlib axes to draw on, or None to create a new figure.
        adam_steps: how many leading entries are Adam epochs (see
            run_summary.adam_step_count); draws a dotted line where L-BFGS
            takes over. None draws no line.

    Returns:
        The axes the loss curves were drawn on.

    For the run's settings (network size, domain, loss weights, etc.), see
    src/pinn/run_summary.py::format_run_summary -- written to a separate
    run_summary.txt file rather than onto this plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name in ("pde", "pde_u", "pde_mu", "ic", "bc", "energy"):
        if name in history:
            ax.semilogy(history[name], lw=1.5, alpha=0.8, label=name)
    if "total" in history:
        ax.semilogy(history["total"], lw=2.5, color="black", label="total")
    if adam_steps is not None and 0 < adam_steps < len(history.get("total", [])):
        ax.axvline(adam_steps, lw=1, ls=":", color="0.4", label="Adam -> L-BFGS")

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("weighted loss")
    ax.set_title("Training loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def load_diagnostics(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a (t, free_energy, total_mass) diagnostics log.

    Args:
        csv_path: path to a cahn_hilliard_diagnostics.csv written by
            src/fem/output.py's _DiagnosticsLogger, or a
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


# Loss-term colours, matching plot_heldout_residual/plot_heldout_ic_bc.
_TERM_COLORS = {"pde": "C0", "pde_u": "C0", "pde_mu": "C1", "ic": "C2", "bc": "C3", "energy": "C4", "total": "black"}
_TERM_ORDER = ("pde", "pde_u", "pde_mu", "ic", "bc", "energy", "total")


def _ordered_terms(names) -> list[str]:
    """Known loss terms in _TERM_ORDER, then any others in their given order."""
    names = list(names)
    return [t for t in _TERM_ORDER if t in names] + [t for t in names if t not in _TERM_ORDER]


def _term_color(term: str, terms: list[str]) -> str:
    return _TERM_COLORS.get(term, f"C{5 + terms.index(term) % 5}")


def _layer_colors(n: int):
    """One hue, light (input layer) -> dark (output layer): layers are ordered."""
    return plt.cm.Blues(np.linspace(0.35, 1.0, n))


def _mark_adam_lbfgs(ax, adam_steps: int | None, steps) -> None:
    if adam_steps is not None and len(steps) and steps[0] < adam_steps <= steps[-1]:
        ax.axvline(adam_steps, lw=1, ls=":", color="0.4", label="Adam -> L-BFGS")


def plot_layer_grad_norms(layer_stats: dict, adam_steps: int | None = None):
    """Per-layer gradient norm ||dL_term/dW_l|| vs. step, one panel per loss term.

    Args:
        layer_stats: the "layer_stats" entry of a checkpoint trained with
            --layer-stats-every (src/pinn/layer_stats.py::LayerStatsLog.as_dict).
        adam_steps: where L-BFGS takes over (dotted line), or None.

    Vanishing gradients: the light (input-side) curves sitting orders of
    magnitude below the dark (output-side) ones. All panels share one y axis,
    so they also show which term dominates the update.

    Returns:
        The figure.
    """
    terms = _ordered_terms(layer_stats["grad_norm"])
    steps = np.asarray(layer_stats["step"])
    names = layer_stats["layers"]
    colors = _layer_colors(len(names))
    ncols = min(3, len(terms))
    nrows = int(np.ceil(len(terms) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True, squeeze=False
    )
    for ax, term in zip(axes.flat, terms):
        norms = np.asarray(layer_stats["grad_norm"][term])
        for j, name in enumerate(names):
            ax.semilogy(steps, norms[:, j], lw=1.5, color=colors[j], label=name)
        _mark_adam_lbfgs(ax, adam_steps, steps)
        ax.set_title(term)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes.flat[len(terms):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|\partial L / \partial W_l\|_2$")
    for ax in axes[-1, :]:
        ax.set_xlabel("step")
    fig.suptitle("Gradient norm per layer, by loss term (weighted)")
    fig.tight_layout()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="layer", fontsize="small", loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def plot_grad_depth_profile(layer_stats: dict, term: str = "total", n_curves: int = 6, ax=None):
    """Gradient norm against layer depth, one curve per logged step.

    The direct vanishing/exploding-gradient picture: a flat profile means
    every layer receives a comparable gradient; a profile rising steeply
    towards "out" means the input-side layers are starved. Earlier records
    are lighter, later ones darker.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    norms = np.asarray(layer_stats["grad_norm"][term])
    steps = layer_stats["step"]
    phases = layer_stats["phase"]
    picks = np.unique(np.linspace(0, len(steps) - 1, min(n_curves, len(steps))).round().astype(int))
    colors = plt.cm.Greys(np.linspace(0.35, 1.0, len(picks)))
    depth = np.arange(len(layer_stats["layers"]))
    for color, k in zip(colors, picks):
        ax.semilogy(depth, norms[k], lw=2, marker="o", ms=5, color=color, label=f"step {steps[k]} ({phases[k]})")
    ax.set_xticks(depth, layer_stats["layers"])
    ax.set_xlabel("layer (input -> output)")
    ax.set_ylabel(r"$\|\partial L / \partial W_l\|_2$")
    ax.set_title(f"Gradient norm vs. depth ({term} loss)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    return ax


def plot_update_ratio(layer_stats: dict, adam_steps: int | None = None, ax=None):
    """Relative weight update ||dW_l|| / ||W_l|| of one optimizer step, per layer.

    What a layer actually moves by (Adam normalises gradients, so a small
    gradient alone doesn't prove a layer is frozen). The shaded band
    1e-4..1e-2 is the usual healthy Adam range; curves far below it are
    layers that have stopped learning. One L-BFGS step runs up to 20 inner
    iterations, so its ratios are naturally larger.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    ratios = np.asarray(layer_stats["update_ratio"])
    steps = np.asarray(layer_stats["step"][: len(ratios)])
    names = layer_stats["layers"]
    colors = _layer_colors(len(names))
    ax.axhspan(1e-4, 1e-2, color="0.92", zorder=0)
    for j, name in enumerate(names):
        ax.semilogy(steps, ratios[:, j], lw=1.5, color=colors[j], label=name)
    _mark_adam_lbfgs(ax, adam_steps, steps)
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(r"$\|\Delta W_l\|_2 / \|W_l\|_2$ per step")
    ax.set_title("Relative weight update per layer")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="layer", fontsize="small")
    return ax


def plot_activation_stats(layer_stats: dict, adam_steps: int | None = None):
    """Hidden-layer tanh activations: saturated fraction (left) and std (right).

    Saturated units (|tanh| > 0.99) pass back < 2% of their gradient, so a
    rising saturation curve is the mechanism behind vanishing gradients; a
    collapsing std means the layer's output has become nearly constant.

    Returns:
        The figure.
    """
    steps = np.asarray(layer_stats["step"])
    sat = np.asarray(layer_stats["act_saturation"])
    std = np.asarray(layer_stats["act_std"])
    colors = _layer_colors(sat.shape[1])
    fig, (ax_sat, ax_std) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for j in range(sat.shape[1]):
        ax_sat.plot(steps, 100 * sat[:, j], lw=1.5, color=colors[j], label=f"hidden {j + 1}")
        ax_std.plot(steps, std[:, j], lw=1.5, color=colors[j], label=f"hidden {j + 1}")
    for ax in (ax_sat, ax_std):
        _mark_adam_lbfgs(ax, adam_steps, steps)
        ax.set_xlabel("step")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    ax_sat.set_ylabel("saturated units (%)")
    ax_sat.set_title(r"Saturation ($|\tanh| > 0.99$)")
    ax_std.set_ylabel("std of activations")
    ax_std.set_title("Activation spread")
    ax_sat.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_gradient_balance(layer_stats: dict, adam_steps: int | None = None, ax=None):
    """Whole-network gradient norm of each (weighted) loss term vs. step.

    If one term's curve sits far above the rest, the optimizer is effectively
    minimising that term alone -- the loss weights are out of balance even
    if every term's *value* looks reasonable.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()
    steps = np.asarray(layer_stats["step"])
    terms = [t for t in _ordered_terms(layer_stats["grad_norm"]) if t != "total"]
    for term in terms:
        norms = np.asarray(layer_stats["grad_norm"][term])
        ax.semilogy(steps, np.sqrt((norms**2).sum(axis=1)), lw=2, color=_term_color(term, terms), label=term)
    _mark_adam_lbfgs(ax, adam_steps, steps)
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(r"$\|\nabla_W L_{term}\|_2$ (all weight matrices)")
    ax.set_title("Gradient balance between loss terms")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_gradient_histograms(grads: dict[str, list[np.ndarray]], layer_names: list[str], bins: int = 80):
    """Distribution of dL_term/dW entries per layer, one panel per layer,
    one curve per loss term (Wang, Teng & Perdikaris 2021, Fig. 2 style).

    Args:
        grads: term -> per-layer 1-D arrays of gradient entries
            (src/pinn/layer_stats.py::per_term_gradients, as numpy).
        layer_names: panel labels, input layer first.

    Binned by magnitude, log10|dL/dW|, since the terms' gradients typically
    differ by orders of magnitude: a term whose curve sits far to the left of
    the others contributes almost nothing to the update in that layer. All
    panels share one x axis, so the panels can also be compared across depth
    -- curves drifting left towards the input layer is vanishing gradients.

    Returns:
        The figure.
    """
    n = len(layer_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), sharex=True, sharey=True, squeeze=False
    )
    terms = _ordered_terms(grads)
    log_mag = {t: [np.log10(np.abs(g[g != 0])) for g in grads[t]] for t in terms}
    finite = np.concatenate([a for t in terms for a in log_mag[t]])
    lo, hi = np.quantile(finite, [0.001, 1.0]) if finite.size else (-1.0, 1.0)
    edges = np.linspace(np.floor(lo), np.ceil(hi), bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for j, ax in enumerate(axes.flat):
        if j >= n:
            ax.set_visible(False)
            continue
        for t in terms:
            hist, _ = np.histogram(log_mag[t][j], bins=edges, density=True)
            ax.plot(centers, hist, lw=1.5, color=_term_color(t, terms), label=t)
        ax.set_title(f"layer {layer_names[j]}")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize="small")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10} |\partial L / \partial W|$")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle("Gradient distribution per layer, by loss term (weighted)")
    fig.tight_layout()
    return fig
