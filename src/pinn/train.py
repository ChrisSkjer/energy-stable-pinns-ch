"""Training loop for the Cahn-Hilliard PINN (baseline and enhanced).

Adam followed by L-BFGS. Runs end-to-end on CPU for a few iterations as a
local smoke test; real training happens on Colab (GPU) — see
docs/colab_workflow.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.pinn.losses import energy_stability_loss, ic_bc_loss, pde_residual_loss
from src.pinn.model import PINN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cahn-Hilliard PINN")
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100, help="Adam epochs before L-BFGS")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--energy-penalty",
        action="store_true",
        help="Enable the energy-stability penalty term (enhanced model). "
        "Omit for the baseline PINN.",
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
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
    model = PINN(hidden_layers=args.hidden_layers, hidden_width=args.hidden_width).to(device)

    # TODO: sample/load collocation, IC, and BC points for the domain and time window
    # TODO: instantiate Adam optimizer
    # TODO: Adam training loop for args.epochs, combining pde_residual_loss + ic_bc_loss
    #       (+ args.energy_weight * energy_stability_loss if args.energy_penalty)
    # TODO: periodic checkpointing every args.checkpoint_every epochs (Colab sessions
    #       can disconnect — do not only checkpoint at the end)
    # TODO: switch to L-BFGS for a final refinement phase
    # TODO: final checkpoint save to args.checkpoint_path

    raise NotImplementedError


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model weights to `path`."""
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
