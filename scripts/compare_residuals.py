"""Compare trained runs on PDE residuals at fresh points they never trained on.

The training loss only measures the residual at a run's own collocation
points, so it can't tell a well-trained network from one that overfits
those points. This evaluates every run on the *same* fresh uniform random
points (fixed --seed), so runs trained with different sampling strategies
(--collocation, --resample-every) are compared on equal terms:

  - pde_u / pde_mu: unweighted residual MSE, training points vs. fresh
    points, and the fresh/train ratio (>> 1 means overfitting)
  - ic: IC error on a dense 300x300 grid at t=0 vs. the training value
  - pde_u per t-window, log-spaced towards t=0, where the residual tends to
    hide (few uniform points land there)

Run from the repo root, with run folder names under results/pinn_models/
or glob patterns:

    .venv\\Scripts\\python.exe scripts/compare_residuals.py run_a run_b
    .venv\\Scripts\\python.exe scripts/compare_residuals.py "*ic-cross" --csv results/residuals_cross.csv
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.initial_conditions import initial_condition  # noqa: E402
from src.pinn.collocation import heldout_points  # noqa: E402
from src.pinn.evaluate import load_model  # noqa: E402
from src.pinn.losses import pde_residual_losses  # noqa: E402
from src.pinn.run_paths import RUNS_ROOT  # noqa: E402

# t-window edges as fractions of t_max: dense near t=0, where a uniform
# draw puts almost no points.
T_EDGE_FRACTIONS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare runs on fresh-point PDE residuals")
    parser.add_argument("runs", nargs="+", help="Run folder names under results/pinn_models/, or glob patterns")
    parser.add_argument("--n", type=int, default=40000, help="Fresh points for the overall residual")
    parser.add_argument("--n-window", type=int, default=8000, help="Fresh points per t-window")
    parser.add_argument("--seed", type=int, default=123, help="Seed for the fresh points (same for every run)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--csv", type=str, default=None, help="Also write the results to this CSV")
    return parser.parse_args()


def resolve_runs(patterns: list[str]) -> list[Path]:
    runs = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(REPO_ROOT, RUNS_ROOT, pattern)))
        runs += [Path(m) for m in matches if (Path(m) / "final.pt").exists()]
    if not runs:
        sys.exit(f"No runs with a final.pt match {patterns} under {RUNS_ROOT}/")
    return runs


def compare_run(run: Path, args: argparse.Namespace) -> dict:
    ckpt_path = run / "final.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    train_args, history = checkpoint["args"], checkpoint["history"]
    model = load_model(str(ckpt_path), device=args.device)
    eps, t_max = train_args["epsilon"], train_args["t_max"]
    upper = (train_args["x_max"], train_args["y_max"], t_max)

    row = {"run": run.name, "collocation": train_args.get("collocation", "random"),
           "resample_every": train_args.get("resample_every", 0)}

    # Training-point values: history is weighted, so undo the weights.
    row["train_pde_u"] = history["pde_u"][-1] / train_args["pde_u_weight"]
    row["train_pde_mu"] = history["pde_mu"][-1] / train_args["pde_mu_weight"]
    row["train_ic"] = history["ic"][-1] / train_args["ic_weight"]

    fresh = heldout_points((0.0, 0.0, 0.0), upper, args.n, seed=args.seed, device=args.device)
    loss_u, loss_mu = pde_residual_losses(model, fresh, epsilon=eps)
    row["fresh_pde_u"], row["fresh_pde_mu"] = loss_u.item(), loss_mu.item()

    # IC error on a 300x300 grid (finer than any training IC grid), with the
    # same normalisation as losses.ic_loss.
    g = np.linspace(0.0, 1.0, 301)
    X, Y = np.meshgrid(g * upper[0], g * upper[1], indexing="ij")
    u0 = initial_condition(train_args["ic"], X.ravel(), Y.ravel(), epsilon=eps, seed=train_args["ic_seed"])
    with torch.no_grad():
        xyt = torch.tensor(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1), dtype=torch.float32)
        u_pred = model(xyt.to(args.device))[:, 0].cpu().numpy()
    row["fresh_ic"] = float(np.mean((u_pred - u0) ** 2) / np.mean(u0**2))

    for k, (lo, hi) in enumerate(zip(T_EDGE_FRACTIONS[:-1], T_EDGE_FRACTIONS[1:])):
        pts = heldout_points(
            (0.0, 0.0, lo * t_max), (upper[0], upper[1], hi * t_max),
            args.n_window, seed=args.seed + 1 + k, device=args.device,
        )
        row[f"pde_u_t[{lo:g},{hi:g}]"] = pde_residual_losses(model, pts, epsilon=eps)[0].item()
    return row


def print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(f"\n## {row['run']}  (collocation={row['collocation']}, resample_every={row['resample_every']})")
        print(f"  {'':8s} {'train':>10s} {'fresh':>10s} {'fresh/train':>12s}")
        for term in ("pde_u", "pde_mu", "ic"):
            train, fresh = row[f"train_{term}"], row[f"fresh_{term}"]
            print(f"  {term:8s} {train:10.3e} {fresh:10.3e} {fresh / train:12.1f}")
        windows = [k for k in row if k.startswith("pde_u_t[")]
        print("  fresh pde_u by t/t_max window:")
        for key in windows:
            print(f"    {key[len('pde_u_t'):]:>14s}  {row[key]:.3e}")


def main() -> None:
    args = parse_args()
    rows = [compare_run(run, args) for run in resolve_runs(args.runs)]
    print_rows(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
