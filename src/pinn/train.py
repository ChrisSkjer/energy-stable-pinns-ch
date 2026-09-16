"""Training loop for the Cahn-Hilliard PINN (baseline and enhanced).

Adam followed by L-BFGS. Runs end-to-end on CPU for a few iterations as a
local smoke test; real training happens on Colab (GPU) — see
docs/colab_workflow.md.
"""

from __future__ import annotations

import argparse

import torch

from src.pinn.losses import DEFAULT_EPSILON, energy_stability_loss, ic_bc_loss, pde_residual_loss
from src.pinn.model import PINN
from src.pinn.sampling import TrainingPoints, sample_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cahn-Hilliard PINN")
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100, help="Adam epochs before L-BFGS")
    parser.add_argument("--lbfgs-steps", type=int, default=10, help="L-BFGS refinement steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--energy-penalty",
        action="store_true",
        help="Enable the energy-stability penalty term (enhanced model). "
        "Omit for the baseline PINN.",
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
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
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--smoke-test", action="store_true", help="Run a handful of iterations only")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
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

    def compute_loss(points: TrainingPoints) -> torch.Tensor:
        loss = pde_residual_loss(model, points.collocation, epsilon=args.epsilon) + ic_bc_loss(
            model, points.ic_points, points.ic_values, points.bc_points
        )
        if args.energy_penalty:
            loss = loss + args.energy_weight * energy_stability_loss(model, points.collocation)
        return loss

    epochs = 5 if args.smoke_test else args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(points)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if epoch % args.checkpoint_every == 0:
            save_checkpoint(model, args.checkpoint_path, args, history)

    lbfgs_steps = 2 if args.smoke_test else args.lbfgs_steps
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = compute_loss(points)
        loss.backward()
        history.append(loss.item())
        return loss

    for lbfgs_step in range(lbfgs_steps):
        lbfgs.step(closure)
        if lbfgs_step % args.checkpoint_every == 0:
            save_checkpoint(model, args.checkpoint_path, args, history)

    save_checkpoint(model, args.checkpoint_path, args, history)


def save_checkpoint(
    model: torch.nn.Module, path: str, args: argparse.Namespace, history: list[float]
) -> None:
    """Save model weights plus the metadata needed to reload them.

    Bundling `args` (the exact CLI flags this run used) and `history` (loss
    per optimizer step) alongside the weights means evaluate.py can
    reconstruct the right architecture and domain bounds on its own instead
    of having them re-supplied by hand -- and the loss curve survives past
    the training process instead of being discarded when it exits.
    """
    torch.save({"model_state": model.state_dict(), "args": vars(args), "history": history}, path)


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
