"""Run the full evaluate -> diagnostics -> plot pipeline for one trained run.

Wraps the three CLIs documented in the README's "Evaluating and plotting a
trained checkpoint" section so the run name is typed once instead of being
threaded through three separate --checkpoint-path/--evaluation flags. Each
step writes into results/pinn_models/<run-name>/ exactly as it does when
invoked by hand -- this only saves the retyping, it changes no defaults.
The one exception is --fem-fields: the comparison needs the PINN evaluated on
the FEM run's own (x, y) grid, so evaluate's --nx/--ny are read from that
fem_fields.npz instead of left at evaluate's defaults.

Usage:
    python scripts/analyze_run.py demo_run
    python scripts/analyze_run.py eps0.01_h64x4 --device cuda --t 0.0 0.01 0.05
    python scripts/analyze_run.py demo_run --fem-diagnostics results/eps0.05_nx96_dt2e-4/cahn_hilliard_diagnostics.csv
    python scripts/analyze_run.py demo_run --fem-fields results/eps0.05_nx96_dt2e-4/fem_fields.npz
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pinn.run_paths import final_path_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate, compute diagnostics for, and plot one trained PINN run"
    )
    parser.add_argument("run_name", help="Run folder under results/pinn_models/ (e.g. demo_run)")
    parser.add_argument(
        "--t",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005],
        help="Snapshot times to evaluate at -- must lie within the run's trained [0, t_max]",
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
    return parser.parse_args()


def run_step(label: str, module: str, module_args: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([sys.executable, "-m", module, *module_args], cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()

    checkpoint = final_path_for(args.run_name)
    if not (REPO_ROOT / checkpoint).exists():
        sys.exit(
            f"No checkpoint at {checkpoint}. Train the run first, or -- if it was "
            f"trained on Colab -- pull it down with scripts/pull_colab_model.py."
        )

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
    ]
    if args.fem_diagnostics:
        plot_args += ["--fem-diagnostics", args.fem_diagnostics]
    if args.fem_fields:
        plot_args += ["--fem-fields", args.fem_fields]
    run_step("plot_results", "src.pinn.plot_results", plot_args)

    print(f"\nDone -- plots in {Path(checkpoint).parent / 'plots'}")


if __name__ == "__main__":
    main()
