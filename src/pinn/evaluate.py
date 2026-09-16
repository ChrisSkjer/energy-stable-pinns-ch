"""Load a trained PINN checkpoint and evaluate it on a grid.

Produces plain numpy arrays only (no plotting here) so that the output can
feed src/common/plotting.py or src/common/metrics.py without either of
those needing a torch import. See docs/colab_workflow.md step 5.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.pinn.model import PINN
from src.pinn.run_paths import infer_run_dir


def parse_args() -> argparse.Namespace:
    """CLI flags for evaluation.

    Architecture and domain bounds are no longer passed here -- load_model
    reads them straight out of the checkpoint (see train.py::save_checkpoint),
    so they can't drift out of sync with what a given checkpoint actually
    was trained with. Only evaluation-specific settings are needed:
      --checkpoint-path (required)
      --nx, --ny (grid resolution)
      --t (one or more snapshot times to evaluate at)
      --device
      --output (.npz path to write, see save_evaluation; defaults to
          evaluation.npz inside the checkpoint's own run folder -- see
          src/pinn/run_paths.py)
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained Cahn-Hilliard PINN checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--nx", type=int, default=100, help="Grid resolution in x")
    parser.add_argument("--ny", type=int, default=100, help="Grid resolution in y")
    parser.add_argument(
        "--t",
        type=float,
        nargs="+",
        required=True,
        help="One or more snapshot times to evaluate at",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' locally, 'cuda' on Colab.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=".npz path to write, see save_evaluation. Defaults to evaluation.npz "
        "inside the checkpoint's own run folder (results/pinn_models/<run-name>/).",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(infer_run_dir(args.checkpoint_path), "evaluation.npz")
    return args


def load_model(checkpoint_path: str, device: torch.device | str = "cpu") -> PINN:
    """Reconstruct a PINN from a checkpoint saved by train.py's save_checkpoint.

    Architecture (hidden_layers, hidden_width) and domain bounds
    (x_max, y_max, t_max) are read from the checkpoint's bundled "args"
    instead of being passed in by the caller -- the checkpoint is the only
    source of truth for what it was actually trained with, so there's no
    way to accidentally reconstruct the wrong shape.

    The checkpoint also carries a "history" entry (loss per optimizer
    step); read it directly with `torch.load(checkpoint_path)["history"]`
    if you want the training curve.

    Should end with model.eval() -- disables any train-only behaviour
    (none yet, but cheap insurance) and signals intent at the call site.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint["args"]
    model = PINN(
        lower_bound=(0.0, 0.0, 0.0),
        upper_bound=(train_args["x_max"], train_args["y_max"], train_args["t_max"]),
        input_dim=3,
        output_dim=2,  # u (phase field) and mu (chemical potential), matching train.py
        hidden_layers=train_args["hidden_layers"],
        hidden_width=train_args["hidden_width"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_on_grid(
    model: PINN,
    t: float,
    nx: int,
    ny: int,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on a regular (nx, ny) grid at a fixed time t.

    Args:
        model: a loaded, .eval()'d PINN.
        t: the fixed time to evaluate at.
        nx, ny: grid resolution.
        device: device to run the forward pass on.

    Returns:
        (X, Y, U, MU): (nx, ny) meshgrid coordinate arrays and the model's u,
        mu predictions reshaped to match -- plain numpy, detached, on CPU.

    Domain bounds come from the model's own lower_bound/upper_bound buffers
    (restored from the checkpoint by load_model), not a separately-passed
    x_max/y_max, so the grid can't be built over the wrong physical domain.

    Inference only: unlike src/pinn/sampling.py's collocation points, these
    grid points should NOT have requires_grad_(True) -- no gradients needed,
    and torch.no_grad() should wrap the forward pass.
    """
    x_lo, y_lo, _ = model.lower_bound.tolist()
    x_hi, y_hi, _ = model.upper_bound.tolist()
    x = np.linspace(x_lo, x_hi, nx)
    y = np.linspace(y_lo, y_hi, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    points = np.stack([X.reshape(-1), Y.reshape(-1), np.full(X.size, t)], axis=1)
    points_t = torch.as_tensor(points, dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(points_t)

    output_np = output.cpu().numpy()
    U = output_np[:, 0].reshape(nx, ny)
    MU = output_np[:, 1].reshape(nx, ny)
    return X, Y, U, MU


def evaluate_over_time(
    model: PINN,
    t_values: np.ndarray,
    nx: int,
    ny: int,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like evaluate_on_grid, but stacked over multiple times.

    For an animation, or for a t-series comparison against a FEM run's
    cahn_hilliard_diagnostics.csv (t, free_energy, total_mass columns).

    Returns:
        (X, Y, U, MU) where X, Y are (nx, ny) and U, MU are
        (len(t_values), nx, ny).
    """
    X = Y = None
    u_frames = []
    mu_frames = []
    for t in t_values:
        X, Y, u, mu = evaluate_on_grid(model, float(t), nx, ny, device=device)
        u_frames.append(u)
        mu_frames.append(mu)

    U = np.stack(u_frames, axis=0)
    MU = np.stack(mu_frames, axis=0)
    return X, Y, U, MU


def save_evaluation(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    mu: np.ndarray,
) -> None:
    """Save evaluated fields to an .npz.

    Lets plotting/comparison scripts work from disk without needing the
    model, the checkpoint, or torch at all -- mirrors data/ and results/
    being regenerable, gitignored outputs (see .gitignore).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, x=x, y=y, t=t, u=u, mu=mu)


if __name__ == "__main__":
    cli_args = parse_args()

    model = load_model(cli_args.checkpoint_path, device=cli_args.device)

    t_values = np.array(cli_args.t)
    grid_x, grid_y, grid_u, grid_mu = evaluate_over_time(
        model,
        t_values=t_values,
        nx=cli_args.nx,
        ny=cli_args.ny,
        device=cli_args.device,
    )

    save_evaluation(cli_args.output, grid_x, grid_y, t_values, grid_u, grid_mu)
    print(f"saved {cli_args.output}")
