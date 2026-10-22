"""Per-layer gradient snapshot of a trained PINN, for checkpoints trained
without --layer-stats-every (or to look at full distributions, not just norms).

Rebuilds the run's training points from its saved args, evaluates every loss
term at the checkpoint's weights, and saves to <run>/plots/:

- gradient_histograms.<fmt>: distribution of dL_term/dW per layer, one curve
  per loss term (Wang, Teng & Perdikaris 2021, "Understanding and mitigating
  gradient pathologies in PINNs", Fig. 2 style)
- gradient_snapshot.txt: per-layer gradient norms per term, the input/output
  layer gradient ratio, and tanh saturation per hidden layer

This is a single snapshot at one set of weights. For how the same quantities
evolve during training, train with --layer-stats-every and use plot_results.

Usage:
    python scripts/inspect_gradients.py <run-name>
    python scripts/inspect_gradients.py <run-name> --checkpoint checkpoints/checkpoint_step001000.pt
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.plotting import plot_gradient_histograms  # noqa: E402
from src.pinn.collocation import make_sampler  # noqa: E402
from src.pinn.evaluate import load_model  # noqa: E402
from src.pinn.layer_stats import activation_stats, layer_names, per_term_gradients  # noqa: E402
from src.pinn.losses import DEFAULT_EPSILON, bc_loss, ic_loss, pde_residual_losses  # noqa: E402
from src.pinn.run_paths import run_dir_for  # noqa: E402
from src.pinn.sampling import sample_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-layer gradient snapshot of a trained PINN")
    parser.add_argument("run_name", help="Run folder under results/pinn_models/")
    parser.add_argument(
        "--checkpoint", default="final.pt", help="Checkpoint file inside the run folder (default final.pt)"
    )
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=None,
        help="Interior points to evaluate the PDE terms on. Defaults to the run's own "
        "--n-collocation; lower it if the CPU pass is too slow.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    return parser.parse_args()


def training_losses(
    model, train_args: dict, n_collocation: int, device: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The run's weighted loss terms at the model's current weights, on a
    freshly drawn set of training points with the run's own settings, plus
    the collocation points they were evaluated on."""
    lower = (0.0, 0.0, 0.0)
    upper = (train_args["x_max"], train_args["y_max"], train_args["t_max"])
    epsilon = train_args.get("epsilon", DEFAULT_EPSILON)
    torch.manual_seed(train_args.get("sampling_seed", 0))
    sampler = make_sampler(
        train_args.get("collocation", "random"),
        lower,
        upper,
        n_collocation,
        seed=train_args.get("sampling_seed", 0),
        grid_shape=train_args.get("grid_shape"),
        device=device,
    )
    points = sample_points(
        lower=lower,
        upper=upper,
        collocation=sampler.sample(),
        n_ic=train_args["n_ic"],
        n_bc=train_args["n_bc"],
        device=device,
        pde_at_t0=train_args.get("pde_at_t0", 0),
        epsilon=epsilon,
        ic=train_args.get("ic") or "cross",
        ic_seed=train_args.get("ic_seed", 0),
    )
    loss_u, loss_mu = pde_residual_losses(model, points.collocation, epsilon=epsilon)
    losses = {
        "pde_u": train_args.get("pde_u_weight", 1.0) * loss_u,
        "pde_mu": train_args.get("pde_mu_weight", 1.0) * loss_mu,
        "ic": train_args.get("ic_weight", 1.0) * ic_loss(model, points.ic_points, points.ic_values),
        "bc": train_args.get("bc_weight", 1.0) * bc_loss(model, points.bc_points),
    }
    return losses, points.collocation


def format_snapshot(names: list[str], grads: dict[str, list[np.ndarray]], acts: dict, losses: dict) -> str:
    lines = ["per-layer gradient norms ||dL_term/dW_l||_2 (weighted terms)", ""]
    header = f"{'term':>8} {'loss':>11} " + " ".join(f"{n:>10}" for n in names) + f" {'in/out':>10}"
    lines.append(header)
    total = [sum(grads[t][j] for t in grads) for j in range(len(names))]
    for term, per_layer in list(grads.items()) + [("total", total)]:
        norms = [float(np.linalg.norm(g)) for g in per_layer]
        loss = sum(v.item() for v in losses.values()) if term == "total" else losses[term].item()
        ratio = norms[0] / norms[-1] if norms[-1] > 0 else float("nan")
        lines.append(
            f"{term:>8} {loss:11.4e} " + " ".join(f"{v:10.3e}" for v in norms) + f" {ratio:10.3e}"
        )
    lines += [
        "",
        "in/out: input-layer over output-layer gradient norm. Values far below 1",
        "(e.g. < 1e-2) mean the input-side layers get a vanishing share of the signal.",
        "",
        "hidden-layer tanh activations",
        f"{'layer':>8} {'std':>10} {'saturated':>10}",
    ]
    for j, (std, sat) in enumerate(zip(acts["std"], acts["saturation"])):
        lines.append(f"{'hidden ' + str(j + 1):>8} {std:10.4f} {100 * sat:9.2f}%")
    lines.append("")
    lines.append("saturated: |tanh| > 0.99, i.e. the unit passes back < 2% of its gradient.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = REPO_ROOT / run_dir_for(args.run_name)
    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.exists():
        sys.exit(f"No checkpoint at {checkpoint_path}")
    train_args = torch.load(checkpoint_path, map_location="cpu")["args"]
    model = load_model(str(checkpoint_path), device=args.device)
    n_collocation = args.n_collocation or train_args["n_collocation"]

    losses, collocation = training_losses(model, train_args, n_collocation, args.device)
    grads = {k: [g.numpy() for g in v] for k, v in per_term_gradients(model, losses).items()}
    acts = activation_stats(model, collocation[:5000])
    names = layer_names(model)

    out_dir = run_dir / "plots"
    os.makedirs(out_dir, exist_ok=True)
    text = format_snapshot(names, grads, acts, losses)
    stem = Path(args.checkpoint).stem
    suffix = "" if stem == "final" else f"_{stem}"
    text_path = out_dir / f"gradient_snapshot{suffix}.txt"
    text_path.write_text(f"checkpoint: {checkpoint_path}\n\n{text}\n", encoding="utf-8")
    print(text)

    fig = plot_gradient_histograms(grads, names)
    fig_path = out_dir / f"gradient_histograms{suffix}.{args.format}"
    fig.savefig(fig_path, dpi=300 if args.format == "pdf" else 150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {text_path}\nsaved {fig_path}")


if __name__ == "__main__":
    main()
