"""Run the full evaluate -> diagnostics -> plot -> metrics pipeline for one trained run.

Wraps the three CLIs documented in the README's "Evaluating and plotting a
trained checkpoint" section so the run name is typed once instead of being
threaded through three separate --checkpoint-path/--evaluation flags. Each
step writes into results/pinn_models/<run-name>/ exactly as it does when
invoked by hand -- this only saves the retyping, it changes no defaults.
The one exception is --fem-fields: the comparison needs the PINN evaluated on
the FEM run's own (x, y) grid, so evaluate's --nx/--ny are read from that
fem_fields.npz instead of left at evaluate's defaults.

A --dim 1 run (read from the checkpoint) goes through src.pinn.plot_1d
instead of evaluate + diagnostics + the field plots: it evaluates on the
FEM run's own vertices and times when --fem-fields (a src.fem.cahn_hilliard
--dim 1 run) is given, and writes the PINN-vs-FEM space-time comparison and
relative L2 error plots. plot_results then adds the loss curves and
run_metrics the scalar metrics, as in 2D. --t, --profile-x/-y and
--field-plots are 2D-only and ignored.

Usage:
    python scripts/analyze_run.py demo_run
    python scripts/analyze_run.py eps0.01_h64x4 --device cuda --t 0.0 0.01 0.05
    python scripts/analyze_run.py demo_run --fem-diagnostics results/eps0.05_nx96_dt2e-4/cahn_hilliard_diagnostics.csv
    python scripts/analyze_run.py demo_run --fem-fields results/eps0.05_nx96_dt2e-4/fem_fields.npz
    python scripts/analyze_run.py demo_run --format png
    python scripts/analyze_run.py demo_run --field-plots
    python scripts/analyze_run.py demo_run --fem-fields results/eps0.05_nx100_dte-4_two_cricles/fem_fields.npz --profile-y 0.5
    python scripts/analyze_run.py <1d run> --fem-fields results/eps0.05_nx1000_dt1e-5_t0.1_dim1_cosine/fem_fields.npz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pinn.run_paths import final_path_for  # noqa: E402


N_DEFAULT_SNAPSHOTS = 5


def default_snapshot_times(t_max: float) -> list[float]:
    """Evenly spaced snapshot times covering the whole trained window,
    0, t_max/4, ..., t_max (0.0025 steps for t_max = 0.01). Rounded so
    linspace's last-bit noise doesn't break the exact match against the
    FEM snapshot times."""
    return [round(float(t), 12) for t in np.linspace(0.0, t_max, N_DEFAULT_SNAPSHOTS)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate, compute diagnostics for, and plot one trained PINN run"
    )
    parser.add_argument("run_name", help="Run folder under results/pinn_models/ (e.g. demo_run)")
    parser.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=None,
        help="Snapshot times to evaluate at -- must lie within the run's trained [0, t_max]. "
        f"Defaults to {N_DEFAULT_SNAPSHOTS} evenly spaced times over [0, t_max], read from the checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' locally, 'cuda' on Colab.")
    parser.add_argument(
        "--fem-diagnostics",
        type=str,
        default=None,
        help="Path to a FEM cahn_hilliard_diagnostics.csv to overlay on the "
        "energy/mass plots, for comparing the PINN against the FEM reference.",
    )
    parser.add_argument(
        "--fem-fields",
        type=str,
        default=None,
        help="Path to a FEM fem_fields.npz, for PINN-vs-FEM comparison plots of u "
        "and mu. The PINN is evaluated on that file's grid, so --nx/--ny match.",
    )
    parser.add_argument(
        "--profile-x",
        type=float,
        default=0.25,
        help="With --fem-fields: x position of the vertical line for the "
        "PINN-vs-FEM u profile plots.",
    )
    parser.add_argument(
        "--profile-y",
        type=float,
        default=None,
        help="With --fem-fields: y position of a horizontal line to use instead "
        "of the vertical --profile-x one (e.g. 0.5 for two_circles).",
    )
    parser.add_argument(
        "--field-plots",
        action="store_true",
        help="Also save the PINN u and mu field snapshots (off by default).",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="pdf",
        help="Plot file format, passed on to plot_results.",
    )
    return parser.parse_args()


def run_step(label: str, module: str, module_args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([sys.executable, "-m", module, *module_args], cwd=REPO_ROOT, check=True)


def analyze_1d(args: argparse.Namespace, checkpoint: str) -> None:
    """plot_1d (fields, FEM comparison, diagnostics.csv) -> plot_results
    (loss curves) -> run_metrics, for a --dim 1 run."""
    plot_1d_args = ["--checkpoint", checkpoint, "--device", args.device, "--format", args.format]
    if args.fem_fields:
        if not Path(args.fem_fields).exists():
            sys.exit(f"No FEM fields file at {args.fem_fields}.")
        plot_1d_args += ["--fem-fields", args.fem_fields]
        if args.fem_diagnostics is None:
            candidate = Path(args.fem_fields).with_name("cahn_hilliard_diagnostics.csv")
            if candidate.exists():
                args.fem_diagnostics = str(candidate)
    if args.fem_diagnostics:
        plot_1d_args += ["--fem-diagnostics", args.fem_diagnostics]
    run_step("plot_1d", "src.pinn.plot_1d", plot_1d_args)
    # plot_results redraws energy_dissipation/mass_conservation from the
    # diagnostics.csv plot_1d just wrote, so it needs the FEM curves too
    results_args = ["--checkpoint", checkpoint, "--format", args.format]
    if args.fem_diagnostics:
        results_args += ["--fem-diagnostics", args.fem_diagnostics]
    run_step("plot_results", "src.pinn.plot_results", results_args)

    metrics_args = ["--run-dir", str(Path(checkpoint).parent)]
    if args.fem_fields:
        metrics_args += ["--fem-fields", args.fem_fields]
    if args.fem_diagnostics:
        metrics_args += ["--fem-diagnostics", args.fem_diagnostics]
    run_step("metrics", "src.pinn.run_metrics", metrics_args)
    print(f"\nDone -- plots in {Path(checkpoint).parent / 'plots'}")


def main() -> None:
    args = parse_args()

    checkpoint = final_path_for(args.run_name)
    if not (REPO_ROOT / checkpoint).exists():
        sys.exit(
            f"No checkpoint at {checkpoint}. Train the run first, or -- if it was "
            f"trained on Colab -- pull it down with scripts/pull_colab_model.py."
        )
    train_args = torch.load(REPO_ROOT / checkpoint, map_location="cpu")["args"]
    if train_args.get("dim", 2) == 1:
        analyze_1d(args, checkpoint)
        return

    if args.t is None:
        args.t = default_snapshot_times(train_args["t_max"])
    evaluate_args = ["--checkpoint-path", checkpoint, "--device", args.device, "--t", *[str(t) for t in args.t]]
    if args.fem_fields:
        if not Path(args.fem_fields).exists():
            sys.exit(f"No FEM fields file at {args.fem_fields}.")
        with np.load(args.fem_fields) as fem:
            nx, ny = fem["x"].shape
        evaluate_args += ["--nx", str(nx), "--ny", str(ny)]
    run_step("evaluate", "src.pinn.evaluate", evaluate_args)
    run_step(
        "diagnostics",
        "src.pinn.diagnostics",
        ["--checkpoint-path", checkpoint, "--device", args.device],
    )

    plot_args = [
        "--evaluation",
        str(Path(checkpoint).with_name("evaluation.npz")),
        "--checkpoint",
        checkpoint,
        "--format",
        args.format,
    ]
    if args.field_plots:
        plot_args.append("--field-plots")
    if args.fem_diagnostics:
        plot_args += ["--fem-diagnostics", args.fem_diagnostics]
    if args.fem_fields:
        plot_args += ["--fem-fields", args.fem_fields, "--profile-x", str(args.profile_x)]
        if args.profile_y is not None:
            plot_args += ["--profile-y", str(args.profile_y)]
    run_step("plot_results", "src.pinn.plot_results", plot_args)

    metrics_args = ["--run-dir", str(Path(checkpoint).parent)]
    if args.fem_fields:
        metrics_args += ["--fem-fields", args.fem_fields]
    if args.fem_diagnostics:
        metrics_args += ["--fem-diagnostics", args.fem_diagnostics]
    run_step("metrics", "src.pinn.run_metrics", metrics_args)

    print(f"\nDone -- plots in {Path(checkpoint).parent / 'plots'}")


if __name__ == "__main__":
    main()
