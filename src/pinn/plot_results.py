"""Generate and save diagnostic plots from a PINN evaluation, checkpoint,
and/or diagnostics log.

Thin CLI wrapper, same shape as train.py/evaluate.py: does no computation of
its own, only orchestrates src/pinn/evaluate.py's saved .npz field arrays,
src/pinn/train.py's checkpointed loss history, and src/pinn/diagnostics.py's
diagnostics.csv through src/common/plotting.py, then writes PDFs (or PNGs
with --format png; plus one text file) to disk. Run as a module from the repo root:

    python -m src.pinn.plot_results --evaluation evaluation.npz --checkpoint checkpoint.pt

Each of --evaluation, --checkpoint, --diagnostics, and --fem-diagnostics is
independently optional (at least one is required); pass several to get
everything from one run in one place. --output-dir defaults to a plots/
subfolder next to whichever of --evaluation/--checkpoint/--diagnostics it's
given (see run_paths.infer_run_dir), so plots land alongside the checkpoint
and evaluation they came from without saying so explicitly. If --diagnostics
is omitted, a diagnostics.csv inside that same run folder is picked up
automatically when present, and likewise the FEM run's
cahn_hilliard_diagnostics.csv next to --fem-fields when --fem-diagnostics
is omitted.

PINN u/mu field snapshots (field_u_*, field_mu_*) are opt-in: pass
--field-plots together with --evaluation.

PINN vs. FEM field comparison: pass --evaluation together with --fem-fields
(a fem_fields.npz written by src/fem/cahn_hilliard.py, see
src/fem/output.py::_FieldRecorder) to get plotting.plot_comparison plots, one per PINN
evaluation time, a profile_*.pdf of u along the vertical line
x = --profile-x, or the horizontal line y = --profile-y if given (PINN
and FEM on one axis, to see the shape of the transition), plus a relative_l2_error.pdf of the error over time. Requires both to share the same (x, y) grid:
evaluate with --nx/--ny equal to fem_fields.npz's x.shape, which is the FEM
run's --nx/--ny + 1 for files saved since the grid sits on the mesh vertices
(scripts/analyze_run.py reads it from the file).
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.initial_conditions import initial_condition
from src.common.metrics import relative_l2_error
from src.common.plotting import (
    load_diagnostics,
    plot_comparison,
    plot_energy_dissipation,
    plot_field,
    plot_heldout_residual,
    plot_line_profile,
    plot_loss_history,
    plot_mass_conservation,
    plot_relative_l2_error,
)
from src.pinn.run_paths import infer_run_dir
from src.pinn.run_summary import adam_step_count, format_run_summary

# Relative L2 distance between a FEM run's u(t=0) and a PINN run's analytic
# IC above which the two can't be the same setup. The same IC lands at 0.1-0.4%
# (0 on the nodal grid of a current fem_fields.npz); epsilon off by 0.01 already
# gives ~5%, another IC or noise seed far more.
FEM_IC_MISMATCH = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a PINN evaluation (.npz) and/or training history (.pt checkpoint)"
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default=None,
        help="Path to a .npz written by src/pinn/evaluate.py (x, y, t, u, mu). "
        "Used by --fem-fields comparisons, and by --field-plots.",
    )
    parser.add_argument(
        "--field-plots",
        action="store_true",
        help="With --evaluation: also save one PINN u and one mu field snapshot "
        "per saved time (field_u_*, field_mu_*). Off by default to keep output small.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint written by src/pinn/train.py. "
        "Produces a training-loss-vs-step plot from its bundled history, and "
        "heldout_residual.* when it was trained with --heldout-every.",
    )
    parser.add_argument(
        "--diagnostics",
        type=str,
        default=None,
        help="Path to a diagnostics.csv written by src/pinn/diagnostics.py. "
        "Produces energy-dissipation and mass-conservation plots. If omitted, "
        "a diagnostics.csv inside the inferred run folder is used when present.",
    )
    parser.add_argument(
        "--fem-diagnostics",
        type=str,
        default=None,
        help="Path to a cahn_hilliard_diagnostics.csv from a FEM run "
        "(src/fem/cahn_hilliard.py), to overlay on the energy/mass plots. "
        "If omitted, the cahn_hilliard_diagnostics.csv next to --fem-fields "
        "is used when present.",
    )
    parser.add_argument(
        "--fem-fields",
        type=str,
        default=None,
        help="Path to a fem_fields.npz from a FEM run (src/fem/output.py's "
        "_FieldRecorder). Requires --evaluation: produces one PINN-vs-FEM "
        "comparison plot per PINN evaluation time, matched to the closest FEM "
        "snapshot time.",
    )
    parser.add_argument(
        "--profile-x",
        type=float,
        default=0.25,
        help="x position of the vertical line for the PINN-vs-FEM u profile "
        "plots (profile_*.pdf, with --fem-fields). Snapped to the nearest grid column.",
    )
    parser.add_argument(
        "--profile-y",
        type=float,
        default=None,
        help="y position of a horizontal line for the profile plots instead of "
        "the vertical --profile-x one, e.g. 0.5 to cut through both circles of "
        "the two_circles initial condition. Snapped to the nearest grid row.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots into. Defaults to a plots/ subfolder inside "
        "the evaluation's (or else the checkpoint's, or else the diagnostics') "
        "own run folder (results/pinn_models/<run-name>/).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="pdf",
        help="Output file format. pdf (default) keeps axes/text as vector and "
        "embeds the (rasterized) field plots as images at --dpi, and is smaller "
        "than png on disk.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Resolution of PNGs, and of the rasterized fields inside PDFs. "
        "Defaults to 150 for png and 300 for pdf.",
    )
    args = parser.parse_args()
    if (
        args.evaluation is None
        and args.checkpoint is None
        and args.diagnostics is None
        and args.fem_diagnostics is None
    ):
        parser.error(
            "pass at least one of --evaluation, --checkpoint, --diagnostics, or --fem-diagnostics"
        )
    if args.fem_fields is not None and args.evaluation is None:
        parser.error("--fem-fields requires --evaluation")
    if args.field_plots and args.evaluation is None:
        parser.error("--field-plots requires --evaluation")
    if args.dpi is None:
        args.dpi = 300 if args.format == "pdf" else 150

    if args.evaluation is not None:
        run_dir = os.path.dirname(os.path.abspath(args.evaluation))
    elif args.checkpoint is not None:
        run_dir = infer_run_dir(args.checkpoint)
    elif args.diagnostics is not None:
        run_dir = os.path.dirname(os.path.abspath(args.diagnostics))
    else:
        run_dir = os.path.dirname(os.path.abspath(args.fem_diagnostics))

    if args.output_dir is None:
        args.output_dir = os.path.join(run_dir, "plots")

    if args.diagnostics is None:
        candidate = os.path.join(run_dir, "diagnostics.csv")
        if os.path.exists(candidate):
            args.diagnostics = candidate

    # Pick up the FEM run's diagnostics alongside its fields, so a --fem-fields
    # call doesn't redraw energy/mass plots without the FEM curve (overwriting
    # an earlier call's two-curve versions).
    if args.fem_diagnostics is None and args.fem_fields is not None:
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(args.fem_fields)), "cahn_hilliard_diagnostics.csv"
        )
        if os.path.exists(candidate):
            args.fem_diagnostics = candidate

    return args


def plot_evaluation(
    evaluation_path: str, output_dir: str, dpi: int = 150, fmt: str = "pdf"
) -> list[str]:
    """Save one u and one mu field snapshot per saved time in an evaluate.py .npz.

    mu snapshots share one symmetric color scale across all saved times
    (from a robust quantile of |mu|, not max() -- a single boundary spike
    would otherwise flatten every frame's scale), since unlike u, mu is
    unbounded and has no fixed [-1, 1] range to use.

    Returns the list of saved paths, in time order (all u snapshots,
    then all mu snapshots).
    """
    with np.load(evaluation_path) as data:
        x, y, t, u, mu = data["x"], data["y"], data["t"], data["u"], data["mu"]

    saved = []
    for idx, t_val in enumerate(t):
        ax = plot_field(x, y, u[idx], title=f"PINN u, t={t_val:g}")
        path = os.path.join(output_dir, f"field_u_{idx:03d}_t{t_val:g}.{fmt}")
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(ax.figure)
        saved.append(path)

    mu_bound = max(float(np.quantile(np.abs(mu), 0.99)), 1e-12)
    for idx, t_val in enumerate(t):
        ax = plot_field(
            x,
            y,
            mu[idx],
            title=f"PINN mu, t={t_val:g}",
            vmin=-mu_bound,
            vmax=mu_bound,
            cmap="RdBu_r",
            colorbar_label="mu",
        )
        path = os.path.join(output_dir, f"field_mu_{idx:03d}_t{t_val:g}.{fmt}")
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(ax.figure)
        saved.append(path)

    return saved


def plot_field_comparison(
    evaluation_path: str,
    fem_fields_path: str,
    output_dir: str,
    dpi: int = 150,
    profile_x: float = 0.25,
    fmt: str = "pdf",
    profile_y: float | None = None,
    train_args: dict | None = None,
) -> list[str]:
    """Save one PINN-vs-FEM comparison plot (plotting.plot_comparison) per
    PINN evaluation time, matched to the closest FEM snapshot time in
    fem_fields.npz.

    Raises if the two .npz files aren't on the same (x, y) grid -- re-run
    `python -m src.pinn.evaluate` with --nx/--ny equal to fem_fields.npz's
    x.shape otherwise (scripts/analyze_run.py does this for you).

    train_args (the PINN checkpoint's "args"), when given, are used to check
    that the FEM run started from this run's initial condition: fem_fields.npz
    records no ic/epsilon/seed, so a reference picked by folder name could be
    the wrong one. A mismatch warns rather than raises.

    The mu comparison is skipped (with a warning) at FEM snapshots where mu is
    identically zero: the FEM solver only sets u at t = 0, so its mu(0) = 0
    is not a state to compare a PINN against. (The IC's actual mu(0) is
    singular at the IC's kinks -- the cross's corners, the circle centres --
    so no smooth network could match it anyway.)

    Also saves, per matched time, a profile_*.pdf of u along the vertical
    line x = profile_x, or the horizontal line y = profile_y if given
    (plotting.plot_line_profile), and
    relative_l2_error.pdf: metrics.relative_l2_error of u and mu
    against FEM at each of those same matched times.

    Returns the list of saved paths: comparisons and profiles in PINN
    time order, then the relative L2 error plot.
    """
    # Read each array once: indexing an open NpzFile re-reads the whole array
    # from the zip on every access, and the loop below indexes per time.
    with np.load(evaluation_path) as f:
        pinn = dict(f)
    with np.load(fem_fields_path) as f:
        fem = dict(f)

    if pinn["x"].shape != fem["x"].shape or not np.allclose(pinn["x"], fem["x"]) or not np.allclose(
        pinn["y"], fem["y"]
    ):
        raise ValueError(
            f"{evaluation_path} and {fem_fields_path} are on different (x, y) grids "
            "-- re-run src/pinn/evaluate.py with --nx/--ny matching the FEM run's."
        )

    fem_t = fem["t"]
    if train_args is not None and fem_t[0] == 0.0:
        # Runs from before --ic existed all used the cross.
        ic = train_args.get("ic") or "cross"
        expected = initial_condition(
            ic, fem["x"], fem["y"], epsilon=train_args["epsilon"], seed=train_args.get("ic_seed", 0)
        )
        mismatch = relative_l2_error(fem["u"][0], expected)
        if mismatch > FEM_IC_MISMATCH:
            warnings.warn(
                f"{fem_fields_path} does not start from this run's initial condition "
                f"(ic={ic}, epsilon={train_args['epsilon']:g}, seed={train_args.get('ic_seed', 0)}): "
                f"relative L2 distance {mismatch:.1%} at t=0. The comparison is against a "
                "different setup -- pick the FEM run made with the same --ic/--epsilon/--seed."
            )

    # One shared symmetric mu scale for PINN and FEM, as in plot_evaluation.
    mu_bound = max(float(np.quantile(np.abs(fem["mu"]), 0.99)), 1e-12)
    mu_kwargs = dict(vmin=-mu_bound, vmax=mu_bound, cmap="RdBu_r", colorbar_label="mu")

    saved = []
    errors = {"u": np.full(len(pinn["t"]), np.nan), "mu": np.full(len(pinn["t"]), np.nan)}
    for idx, t_val in enumerate(pinn["t"]):
        fem_idx = int(np.argmin(np.abs(fem_t - t_val)))
        matched_t = fem_t[fem_idx]
        title = f"t = {t_val:g}"
        if abs(matched_t - t_val) > 1e-9:
            warnings.warn(
                f"PINN t={t_val:g} has no exact FEM snapshot; using closest one "
                f"at t={matched_t:g} instead"
            )
            title += f" (FEM at t = {matched_t:g})"
        fig, _ = plot_comparison(pinn["x"], pinn["y"], pinn["u"][idx], fem["u"][fem_idx], title=title)
        path = os.path.join(output_dir, f"comparison_{idx:03d}_t{t_val:g}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        if np.any(fem["mu"][fem_idx]):
            fig, _ = plot_comparison(
                pinn["x"], pinn["y"], pinn["mu"][idx], fem["mu"][fem_idx], title=title, **mu_kwargs
            )
            path = os.path.join(output_dir, f"comparison_mu_{idx:03d}_t{t_val:g}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
        else:
            warnings.warn(
                f"FEM mu is identically zero at t={matched_t:g} (the solver only sets u at "
                f"t=0); skipping comparison_mu_{idx:03d} and the mu error at t={t_val:g}"
            )

        fig, _ = plot_line_profile(
            pinn["x"], pinn["y"], pinn["u"][idx], fem["u"][fem_idx],
            x_line=profile_x, y_line=profile_y, title=f"t={t_val:g}",
        )
        path = os.path.join(output_dir, f"profile_{idx:03d}_t{t_val:g}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        for name in errors:
            try:
                errors[name][idx] = relative_l2_error(pinn[name][idx], fem[name][fem_idx])
            except ValueError:
                pass  # FEM field identically zero here; leave NaN (a gap in the plot)

    ax = plot_relative_l2_error(pinn["t"], errors)
    path = os.path.join(output_dir, f"relative_l2_error.{fmt}")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    saved.append(path)

    return saved


def plot_training_history(
    checkpoint_path: str, output_dir: str, dpi: int = 150, fmt: str = "pdf"
) -> str:
    """Save a loss-vs-optimizer-step plot from a train.py checkpoint's bundled history.

    Returns the saved path.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint["history"]
    adam_steps = adam_step_count(checkpoint["args"], history, checkpoint.get("adam_steps"))
    ax = plot_loss_history(history, adam_steps=adam_steps)
    path = os.path.join(output_dir, f"loss_history.{fmt}")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    return path


def plot_heldout(checkpoint_path: str, output_dir: str, dpi: int = 150, fmt: str = "pdf") -> str | None:
    """Save heldout_residual.<fmt> from a checkpoint trained with
    --heldout-every (plotting.plot_heldout_residual).

    Returns the saved path, or None when the checkpoint has no held-out log.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "heldout" not in checkpoint:
        return None
    args = checkpoint["args"]
    ax = plot_heldout_residual(
        checkpoint["heldout"],
        checkpoint["history"],
        pde_u_weight=args.get("pde_u_weight", 1.0),
        pde_mu_weight=args.get("pde_mu_weight", 1.0),
    )
    path = os.path.join(output_dir, f"heldout_residual.{fmt}")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    return path


def write_run_summary(checkpoint_path: str, output_dir: str) -> str:
    """Write <output_dir>/run_summary.txt from a train.py checkpoint --
    network size, domain, sampling, optimizer, loss weights, parameter
    count, and each loss term's final value (see run_summary.format_run_summary).

    Returns the saved path.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    text = format_run_summary(
        checkpoint["args"],
        model_state=checkpoint["model_state"],
        history=checkpoint["history"],
        source=os.path.abspath(checkpoint_path),
        adam_steps=checkpoint.get("adam_steps"),
    )
    path = os.path.join(output_dir, "run_summary.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return path


def plot_diagnostics(
    diagnostics_path: str | None,
    fem_diagnostics_path: str | None,
    output_dir: str,
    dpi: int = 150,
    fmt: str = "pdf",
) -> list[str]:
    """Save energy-dissipation and mass-conservation plots from one or both
    diagnostics CSVs (src/pinn/diagnostics.py's diagnostics.csv and/or a FEM
    run's cahn_hilliard_diagnostics.csv). At least one path is required.

    Returns the list of saved paths.
    """
    t_p = e_p = m_p = None
    if diagnostics_path is not None:
        t_p, e_p, m_p = load_diagnostics(diagnostics_path)

    t_f = e_f = m_f = None
    if fem_diagnostics_path is not None:
        t_f, e_f, m_f = load_diagnostics(fem_diagnostics_path)
        if t_p is not None:
            # A FEM run's own time window is typically much longer than a
            # PINN run's t_max -- clip so the PINN curve isn't compressed
            # to a single pixel against it. With a tolerance: the FEM's
            # accumulated t (0.009999999999999995) and a PINN t_max that went
            # through float32 (0.009999999776) both mean t = 0.01.
            keep = (t_f <= t_p[-1]) | np.isclose(t_f, t_p[-1], rtol=1e-6, atol=0.0)
            if keep.sum() < 2:
                warnings.warn(
                    f"FEM diagnostics has {int(keep.sum())} sample(s) at t <= {t_p[-1]:g}; "
                    "the overlay will be empty. Re-run the FEM solver with a matching "
                    "--t-final and --epsilon."
                )
            t_f, e_f, m_f = t_f[keep], e_f[keep], m_f[keep]

    saved = []

    ax = plot_energy_dissipation(t_p, e_p, t_f, e_f)
    path = os.path.join(output_dir, f"energy_dissipation.{fmt}")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    saved.append(path)

    ax = plot_mass_conservation(t_p, m_p, t_f, m_f)
    path = os.path.join(output_dir, f"mass_conservation.{fmt}")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    saved.append(path)

    return saved


if __name__ == "__main__":
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    saved_paths = []
    if cli_args.field_plots:
        saved_paths += plot_evaluation(
            cli_args.evaluation, cli_args.output_dir, dpi=cli_args.dpi, fmt=cli_args.format
        )
    if cli_args.checkpoint is not None:
        saved_paths.append(
            plot_training_history(
                cli_args.checkpoint, cli_args.output_dir, dpi=cli_args.dpi, fmt=cli_args.format
            )
        )
        saved_paths.append(write_run_summary(cli_args.checkpoint, cli_args.output_dir))
        heldout_path = plot_heldout(
            cli_args.checkpoint, cli_args.output_dir, dpi=cli_args.dpi, fmt=cli_args.format
        )
        if heldout_path is not None:
            saved_paths.append(heldout_path)
    if cli_args.diagnostics is not None or cli_args.fem_diagnostics is not None:
        saved_paths += plot_diagnostics(
            cli_args.diagnostics,
            cli_args.fem_diagnostics,
            cli_args.output_dir,
            dpi=cli_args.dpi,
            fmt=cli_args.format,
        )
    if cli_args.fem_fields is not None:
        train_args = None
        if cli_args.checkpoint is not None:
            train_args = torch.load(cli_args.checkpoint, map_location="cpu")["args"]
        saved_paths += plot_field_comparison(
            cli_args.evaluation,
            cli_args.fem_fields,
            cli_args.output_dir,
            dpi=cli_args.dpi,
            profile_x=cli_args.profile_x,
            fmt=cli_args.format,
            profile_y=cli_args.profile_y,
            train_args=train_args,
        )

    for path in saved_paths:
        print(f"saved {path}")
