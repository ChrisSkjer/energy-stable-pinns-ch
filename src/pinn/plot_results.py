"""Generate and save diagnostic plots from a PINN evaluation and/or checkpoint.

Thin CLI wrapper, same shape as train.py/evaluate.py: does no computation of
its own, only orchestrates src/pinn/evaluate.py's saved .npz field arrays and
src/pinn/train.py's checkpointed loss history through src/common/plotting.py,
then writes PNGs to disk. Run as a module from the repo root:

    python -m src.pinn.plot_results --evaluation evaluation.npz --checkpoint checkpoint.pt

Either --evaluation or --checkpoint alone is fine (each produces its own
plots independently); pass both to get everything from one run in one place.
--output-dir defaults to a plots/ subfolder next to whichever of those two
paths it's given (see run_paths.infer_run_dir), so plots land alongside the
checkpoint and evaluation they came from without saying so explicitly.

Only the u (phase) field is plotted, via plotting.plot_field -- that
function's colormap is fixed to [-1, 1], which fits u but not mu (chemical
potential, unbounded), so mu snapshots aren't produced here. PINN vs. FEM
comparison plots and energy/mass diagnostics aren't wired up yet either:
plotting.plot_comparison needs FEM field arrays that src/fem/cahn_hilliard.py
doesn't save, and plotting.plot_energy_dissipation/plot_mass_conservation
need src/common/metrics.py's free_energy/mass_conservation_error, which are
still NotImplementedError stubs.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.plotting import plot_field, plot_loss_history
from src.pinn.run_paths import infer_run_dir


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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save PNGs into. Defaults to a plots/ subfolder inside "
        "the evaluation's (or else the checkpoint's) own run folder "
        "(results/pinn_models/<run-name>/).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    if args.evaluation is None and args.checkpoint is None:
        parser.error("pass at least one of --evaluation or --checkpoint")
    if args.output_dir is None:
        if args.evaluation is not None:
            run_dir = os.path.dirname(os.path.abspath(args.evaluation))
        else:
            run_dir = infer_run_dir(args.checkpoint)
        args.output_dir = os.path.join(run_dir, "plots")
    return args


def plot_evaluation(evaluation_path: str, output_dir: str, dpi: int = 150) -> list[str]:
    """Save one u field snapshot per saved time in an evaluate.py .npz.

    Returns the list of saved PNG paths, in time order.
    """
    data = np.load(evaluation_path)
    x, y, t, u = data["x"], data["y"], data["t"], data["u"]

    saved = []
    for idx, t_val in enumerate(t):
        ax = plot_field(x, y, u[idx], title=f"PINN u, t={t_val:g}")
        path = os.path.join(output_dir, f"field_u_{idx:03d}_t{t_val:g}.png")
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

    for path in saved_paths:
        print(f"saved {path}")
