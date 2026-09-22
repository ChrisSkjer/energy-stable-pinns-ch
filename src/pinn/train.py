"""Training loop for the Cahn-Hilliard PINN (baseline and enhanced).

Adam followed by L-BFGS. Runs end-to-end on CPU for a few iterations as a
local smoke test; real training happens on Colab (GPU) — see
docs/colab_workflow.md.
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch

from src.pinn.losses import DEFAULT_EPSILON, bc_loss, energy_stability_loss, ic_loss, pde_residual_loss
from src.pinn.model import PINN
from src.pinn.run_paths import checkpoints_dir_for, final_path_for, run_dir_for
from src.pinn.sampling import TrainingPoints, sample_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cahn-Hilliard PINN")
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--hidden-width", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100, help="Adam epochs before L-BFGS")
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=70,
        help="L-BFGS refinement steps. Each one is a .step(closure) call "
        "running up to 20 inner iterations, so this is an upper bound of "
        "~20x that many closure evaluations -- see --lbfgs-patience.",
    )
    parser.add_argument(
        "--lbfgs-tol",
        type=float,
        default=1e-6,
        help="Relative improvement in the total loss below which an L-BFGS "
        "step counts as stalled (early-stopping threshold).",
    )
    parser.add_argument(
        "--lbfgs-patience",
        type=int,
        default=3,
        help="Stop L-BFGS after this many consecutive stalled steps. "
        "0 disables early stopping and always runs --lbfgs-steps.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--energy-penalty",
        action="store_true",
        help="Enable the energy-stability penalty term (enhanced model). "
        "Omit for the baseline PINN.",
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=1.0, help="Weight on the PDE residual loss term")
    parser.add_argument("--ic-weight", type=float, default=100, help="Weight on the initial-condition loss term")
    parser.add_argument("--bc-weight", type=float, default=10, help="Weight on the boundary-condition loss term")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Cahn-Hilliard interface half-width, used for both the PDE "
        "residual and the initial-condition profile.",
    )
    parser.add_argument("--x-max", type=float, default=1.0, help="Domain upper bound in x (lower is 0)")
    parser.add_argument("--y-max", type=float, default=1.0, help="Domain upper bound in y (lower is 0)")
    parser.add_argument("--t-max", type=float, default=5e-3, help="Domain upper bound in t (lower is 0)")
    parser.add_argument("--n-collocation", type=int, default=10000, help="Random interior PDE points")
    parser.add_argument("--n-ic", type=int, default=100, help="Side length of the IC point grid")
    parser.add_argument("--n-bc", type=int, default=100, help="Side length of the BC point grid, per edge")
    parser.add_argument(
        "--pde-at-t0",
        type=int,
        default=2000,
        help="Also enforce the PDE residual at this many t=0 points",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' locally, 'cuda' on Colab.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Epochs between checkpoints")
    parser.add_argument("--log-every", type=int, default=100, help="Epochs between progress prints")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Unique name for this run's output folder, "
        "results/pinn_models/<run-name>/. Defaults to a timestamp.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a handful of iterations only")
    return parser.parse_args()


def is_improvement(total: float, best: float | None, tol: float) -> bool:
    """Whether `total` beats the best total loss so far by enough to count.

    Args:
        total: the total loss after the L-BFGS step just taken.
        best: the lowest total seen so far, or None before any step.
        tol: minimum *relative* improvement, as a fraction of |best|.
            Relative rather than absolute so the same threshold works
            whether a run plateaus at 1e-1 or 1e-6.

    Returns:
        True for the first step (best is None) and whenever `total` is at
        least `tol * abs(best)` below `best`. False for NaN, since every
        comparison against NaN is False -- a diverged run reads as stalled
        rather than as endless improvement.
    """
    if best is None:
        return True
    return total < best - tol * abs(best)


def train(args: argparse.Namespace) -> None:
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = checkpoints_dir_for(run_name)
    final_path = final_path_for(run_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"run directory: {run_dir_for(run_name)}")

    device = torch.device(args.device)
    lower = (0.0, 0.0, 0.0)
    upper = (args.x_max, args.y_max, args.t_max)

    model = PINN(
        lower_bound=lower,
        upper_bound=upper,
        input_dim=3,
        output_dim=2,  # u (phase field) and mu (chemical potential)
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)

    n_collocation = 100 if args.smoke_test else args.n_collocation
    n_ic = 10 if args.smoke_test else args.n_ic
    n_bc = 10 if args.smoke_test else args.n_bc
    points = sample_points(
        lower=lower,
        upper=upper,
        n_collocation=n_collocation,
        n_ic=n_ic,
        n_bc=n_bc,
        device=device,
        pde_at_t0=args.pde_at_t0,
        epsilon=args.epsilon,
    )

    def compute_loss(points: TrainingPoints) -> dict[str, torch.Tensor]:
        losses = {
            "pde": args.pde_weight * pde_residual_loss(model, points.collocation, epsilon=args.epsilon),
            "ic": args.ic_weight * ic_loss(model, points.ic_points, points.ic_values),
            "bc": args.bc_weight * bc_loss(model, points.bc_points),
        }
        if args.energy_penalty:
            losses["energy"] = args.energy_weight * energy_stability_loss(model, points.collocation)
        losses["total"] = sum(losses.values())
        return losses

    def format_losses(losses: dict[str, torch.Tensor]) -> str:
        return " ".join(f"{name} {value.item():.4e}" for name, value in losses.items())

    epochs = 5 if args.smoke_test else args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: dict[str, list[float]] = {}

    def record(losses: dict[str, torch.Tensor]) -> None:
        for name, value in losses.items():
            history.setdefault(name, []).append(value.item())

    for epoch in range(epochs):
        optimizer.zero_grad()
        losses = compute_loss(points)
        losses["total"].backward()
        optimizer.step()
        record(losses)
        if epoch % args.log_every == 0 or epoch == epochs - 1:
            print(f"[adam] epoch {epoch:5d}/{epochs} {format_losses(losses)}")
        if epoch % args.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_step{len(history['total']):06d}.pt")
            save_checkpoint(model, checkpoint_path, args, history)

    lbfgs_steps = 2 if args.smoke_test else args.lbfgs_steps
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
    last_losses: dict[str, torch.Tensor] = {}

    def closure() -> torch.Tensor:
        nonlocal last_losses
        lbfgs.zero_grad()
        losses = compute_loss(points)
        losses["total"].backward()
        record(losses)
        last_losses = losses
        return losses["total"]

    best_total: float | None = None
    stalled = 0
    for lbfgs_step in range(lbfgs_steps):
        lbfgs.step(closure)
        print(f"[lbfgs] step {lbfgs_step + 1:3d}/{lbfgs_steps} {format_losses(last_losses)}")
        if lbfgs_step % args.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_step{len(history['total']):06d}.pt")
            save_checkpoint(model, checkpoint_path, args, history)

        total = last_losses["total"].item()
        if is_improvement(total, best_total, args.lbfgs_tol):
            best_total = total
            stalled = 0
        else:
            stalled += 1
            if args.lbfgs_patience and stalled >= args.lbfgs_patience:
                print(
                    f"[lbfgs] early stop after step {lbfgs_step + 1}/{lbfgs_steps}: "
                    f"total loss improved by less than {args.lbfgs_tol:g} (relative) "
                    f"for {stalled} consecutive steps"
                )
                break

    save_checkpoint(model, final_path, args, history)
    print(f"final model: {final_path}")


def save_checkpoint(
    model: torch.nn.Module, path: str, args: argparse.Namespace, history: dict[str, list[float]]
) -> None:
    """Save model weights plus the metadata needed to reload them.

    Bundling `args` (the exact CLI flags this run used) and `history` (each
    weighted loss component -- "pde", "ic", "bc", "total", and "energy" when
    --energy-penalty is enabled -- per optimizer step) alongside the weights
    means evaluate.py can reconstruct the right architecture and domain
    bounds on its own instead of having them re-supplied by hand -- and the
    loss curve survives past the training process instead of being discarded
    when it exits.
    """
    torch.save({"model_state": model.state_dict(), "args": vars(args), "history": history}, path)


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
