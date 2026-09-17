"""Generate and save diagnostic plots from a PINN evaluation, checkpoint,
and/or diagnostics log.

Thin CLI wrapper, same shape as train.py/evaluate.py: does no computation of
its own, only orchestrates src/pinn/evaluate.py's saved .npz field arrays,
src/pinn/train.py's checkpointed loss history, and src/pinn/diagnostics.py's
diagnostics.csv through src/common/plotting.py, then writes PNGs (and one
text file) to disk. Run as a module from the repo root:

    python -m src.pinn.plot_results --evaluation evaluation.npz --checkpoint checkpoint.pt

Each of --evaluation, --checkpoint, --diagnostics, and --fem-diagnostics is
independently optional (at least one is required); pass several to get
everything from one run in one place. --output-dir defaults to a plots/
subfolder next to whichever of --evaluation/--checkpoint/--diagnostics it's
given (see run_paths.infer_run_dir), so plots land alongside the checkpoint
and evaluation they came from without saying so explicitly. If --diagnostics
is omitted, a diagnostics.csv inside that same run folder is picked up
automatically when present.

PINN vs. FEM field comparison isn't wired up yet: plotting.plot_comparison
needs raw FEM field arrays, and src/fem/cahn_hilliard.py currently saves
only PNG frames and the diagnostics CSV, not an array dump.
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.plotting import (
    load_diagnostics,
    plot_energy_dissipation,
    plot_field,
    plot_loss_history,
    plot_mass_conservation,
)
from src.pinn.run_paths import infer_run_dir
from src.pinn.run_summary import format_run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a PINN evaluation (.npz) and/or training history (.pt checkpoint)"
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default=None,
        help="Path to a .npz written by src/pinn/evaluate.py (x, y, t, u, mu). "
        "Produces one u field snapshot per saved time.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint written by src/pinn/train.py. "
        "Produces a training-loss-vs-step plot from its bundled history.",
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
        "Never auto-detected -- pass it explicitly.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save PNGs into. Defaults to a plots/ subfolder inside "
        "the evaluation's (or else the checkpoint's, or else the diagnostics') "
        "own run folder (results/pinn_models/<run-name>/).",
    )
    parser.add_argument("--dpi", type=int, default=150)
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

    return args


def plot_evaluation(evaluation_path: str, output_dir: str, dpi: int = 150) -> list[str]:
    """Save one u and one mu field snapshot per saved time in an evaluate.py .npz.

    mu snapshots share one symmetric color scale across all saved times
    (from a robust quantile of |mu|, not max() -- a single boundary spike
    would otherwise flatten every frame's scale), since unlike u, mu is
    unbounded and has no fixed [-1, 1] range to use.

    Returns the list of saved PNG paths, in time order (all u snapshots,
    then all mu snapshots).
    """
    data = np.load(evaluation_path)
    x, y, t, u, mu = data["x"], data["y"], data["t"], data["u"], data["mu"]

    saved = []
    for idx, t_val in enumerate(t):
        ax = plot_field(x, y, u[idx], title=f"PINN u, t={t_val:g}")
        path = os.path.join(output_dir, f"field_u_{idx:03d}_t{t_val:g}.png")
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
        path = os.path.join(output_dir, f"field_mu_{idx:03d}_t{t_val:g}.png")
        ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(ax.figure)
        saved.append(path)

    return saved


def plot_training_history(checkpoint_path: str, output_dir: str, dpi: int = 150) -> str:
    """Save a loss-vs-optimizer-step plot from a train.py checkpoint's bundled history.

    Returns the saved PNG path.
    """
    history = torch.load(checkpoint_path, map_location="cpu")["history"]
    ax = plot_loss_history(history)
    path = os.path.join(output_dir, "loss_history.png")
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
) -> list[str]:
    """Save energy-dissipation and mass-conservation PNGs from one or both
    diagnostics CSVs (src/pinn/diagnostics.py's diagnostics.csv and/or a FEM
    run's cahn_hilliard_diagnostics.csv). At least one path is required.

    Returns the list of saved PNG paths.
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
            # to a single pixel against it.
            keep = t_f <= t_p[-1]
            if keep.sum() < 2:
                warnings.warn(
                    f"FEM diagnostics has {int(keep.sum())} sample(s) at t <= {t_p[-1]:g}; "
                    "the overlay will be empty. Re-run the FEM solver with a matching "
                    "--t-final and --epsilon."
                )
            t_f, e_f, m_f = t_f[keep], e_f[keep], m_f[keep]

    saved = []

    ax = plot_energy_dissipation(t_p, e_p, t_f, e_f)
    path = os.path.join(output_dir, "energy_dissipation.png")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    saved.append(path)

    ax = plot_mass_conservation(t_p, m_p, t_f, m_f)
    path = os.path.join(output_dir, "mass_conservation.png")
    ax.figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)
    saved.append(path)

    return saved


if __name__ == "__main__":
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)

    saved_paths = []
    if cli_args.evaluation is not None:
        saved_paths += plot_evaluation(cli_args.evaluation, cli_args.output_dir, dpi=cli_args.dpi)
    if cli_args.checkpoint is not None:
        saved_paths.append(
            plot_training_history(cli_args.checkpoint, cli_args.output_dir, dpi=cli_args.dpi)
        )
        saved_paths.append(write_run_summary(cli_args.checkpoint, cli_args.output_dir))
    if cli_args.diagnostics is not None or cli_args.fem_diagnostics is not None:
        saved_paths += plot_diagnostics(
            cli_args.diagnostics, cli_args.fem_diagnostics, cli_args.output_dir, dpi=cli_args.dpi
        )

    for path in saved_paths:
        print(f"saved {path}")
