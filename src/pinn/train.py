"""Training loop for the Cahn-Hilliard PINN (baseline and enhanced).

Adam followed by L-BFGS. Runs end-to-end on CPU for a few iterations as a
local smoke test; real training happens on Colab (GPU) — see
docs/colab_workflow.md.
"""

from __future__ import annotations

import argparse
import datetime
import os
import time

import torch

from src.common.initial_conditions import INITIAL_CONDITIONS, INITIAL_CONDITIONS_1D
from src.pinn.losses import (
    DEFAULT_EPSILON,
    bc_loss,
    energy_stability_loss,
    ic_loss,
    pde_residual_losses,
)
from src.pinn.layer_stats import LayerStatsLog, update_ratios, weight_snapshot
from src.pinn.model import PINN
from src.pinn.run_paths import checkpoints_dir_for, final_path_for, run_dir_for
from src.pinn.collocation import COLLOCATION_STRATEGIES, RESAMPLABLE, heldout_points, make_sampler
from src.pinn.sampling import TrainingPoints, domain_bounds, heldout_ic_bc_points, sample_points, with_pde_at_t0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cahn-Hilliard PINN")
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        choices=(1, 2),
        help="Space dimension: 2 trains u(x, y, t) on [0, x-max] x [0, y-max], "
        "1 trains u(x, t) on [0, x-max] (--y-max is then ignored). The network "
        "input is (x, [y,] t); the output is (u, mu) either way.",
    )
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
        "--adam-tol",
        type=float,
        default=1e-4,
        help="Relative improvement in the total loss below which an Adam "
        "epoch counts as stalled. Shared by --plateau-patience and "
        "--adam-patience.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=0,
        help="ReduceLROnPlateau: multiply the Adam lr by --plateau-factor "
        "after this many stalled epochs. 0 disables (constant lr).",
    )
    parser.add_argument("--plateau-factor", type=float, default=0.5, help="LR multiplier on each plateau")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for the plateau-reduced lr")
    parser.add_argument(
        "--adam-patience",
        type=int,
        default=0,
        help="Stop Adam after this many consecutive stalled epochs and restore "
        "the best weights seen before moving on to L-BFGS. 0 disables. "
        "Set it well above --plateau-patience so the lr gets a chance to drop "
        "a few times first.",
    )
    parser.add_argument(
        "--energy-penalty",
        action="store_true",
        help="Enable the energy-stability penalty term (enhanced model). "
        "Omit for the baseline PINN.",
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument(
        "--pde-u-weight",
        type=float,
        default=1.0,
        help="Weight on the u-equation PDE residual, m*laplacian(mu) - u_t",
    )
    parser.add_argument(
        "--pde-mu-weight",
        type=float,
        default=100.0,
        help="Weight on the mu-equation PDE residual, mu - f'(u) + eps^2*laplacian(u)",
    )
    parser.add_argument("--ic-weight", type=float, default=100, help="Weight on the initial-condition loss term")
    parser.add_argument("--bc-weight", type=float, default=10, help="Weight on the boundary-condition loss term")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="Cahn-Hilliard interface half-width, used for both the PDE "
        "residual and the initial-condition profile.",
    )
    parser.add_argument(
        "--ic",
        type=str,
        default=None,
        help="Initial condition (see src/common/initial_conditions.py). "
        f"--dim 2: {sorted(INITIAL_CONDITIONS)}, default 'cross' -- 'cross' "
        "(Swiss-flag cross relaxing to a circle), 'two_circles' (coarsening, "
        "the small circle feeds the big one), 'noise' (smooth random "
        "perturbation, spinodal decomposition). "
        f"--dim 1: {sorted(INITIAL_CONDITIONS_1D)}, default 'cosine' -- "
        "'step' (stationary equilibrium interface), 'two_intervals', 'noise', "
        "'cosine' (single growing mode with an analytic early-time solution), "
        "'cosine_large' (-cos(4 pi x), the Wight & Zhao 1D IC). "
        "Use the same --ic for the FEM ground truth.",
    )
    parser.add_argument(
        "--ic-seed",
        type=int,
        default=42,
        help="Random seed for --ic noise; must match the FEM --seed (default 42 "
        "on both) for a valid comparison.",
    )
    parser.add_argument("--x-max", type=float, default=1.0, help="Domain upper bound in x (lower is 0)")
    parser.add_argument("--y-max", type=float, default=1.0, help="Domain upper bound in y (lower is 0)")
    parser.add_argument("--t-max", type=float, default=5e-3, help="Domain upper bound in t (lower is 0)")
    parser.add_argument(
        "--n-collocation",
        type=int,
        default=10000,
        help="Interior PDE points (for --collocation grid: only picks the "
        "default --grid-shape, round(n^(1/(dim+1))) per axis)",
    )
    parser.add_argument(
        "--collocation",
        type=str,
        default="random",
        choices=sorted(COLLOCATION_STRATEGIES),
        help="Collocation point strategy (see src/pinn/collocation.py): "
        "'random' (independent uniform draws), 'sobol' (scrambled Sobol "
        "quasi-random, more evenly spread), 'grid' (cell-centred regular "
        "(x, [y,] t) grid).",
    )
    parser.add_argument(
        "--resample-every",
        type=int,
        default=0,
        help="Redraw the interior collocation points every this many Adam "
        "epochs (random/sobol only; 0 = never). L-BFGS always keeps the "
        "last set fixed: its line search needs a deterministic loss.",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Points per axis for --collocation grid: NX NY NT, or NX NT for "
        "--dim 1. Defaults to round(n_collocation^(1/(dim+1))) on each axis.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Seed for the collocation draw (and the t=0 PDE point choice), "
        "so a run's points can be reproduced.",
    )
    parser.add_argument(
        "--heldout-every",
        type=int,
        default=0,
        help="Log the unweighted PDE residuals, IC loss and BC loss on fixed "
        "sets of fresh uniform random points (never trained on) every this "
        "many Adam epochs and L-BFGS steps. "
        "0 = off. Stored under the checkpoint's 'heldout' key.",
    )
    parser.add_argument("--heldout-n", type=int, default=20000, help="Number of held-out points")
    parser.add_argument("--heldout-n-ic", type=int, default=10000, help="Number of held-out IC points on t=0")
    parser.add_argument(
        "--heldout-n-bc", type=int, default=2500, help="Number of held-out BC points per spatial edge"
    )
    parser.add_argument(
        "--layer-stats-every",
        type=int,
        default=0,
        help="Log per-layer statistics (gradient norm of each loss term, "
        "weight norm, update ratio, tanh saturation -- see "
        "src/pinn/layer_stats.py) every this many Adam epochs and L-BFGS "
        "steps, to check for vanishing gradients. 0 = off. Costs one extra "
        "backward pass per loss term at each logged step. Stored under the "
        "checkpoint's 'layer_stats' key.",
    )
    parser.add_argument(
        "--n-ic", type=int, default=100, help="IC points per spatial axis (n x n in 2D, n in 1D)"
    )
    parser.add_argument(
        "--n-bc",
        type=int,
        default=100,
        help="BC points per axis on each wall (n x n over (space, t) per edge "
        "in 2D, n times at each end in 1D)",
    )
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
    args = parser.parse_args()
    if args.resample_every and args.collocation not in RESAMPLABLE:
        parser.error(f"--resample-every needs a random strategy, not --collocation {args.collocation}")
    if args.grid_shape is not None and args.collocation != "grid":
        parser.error("--grid-shape only applies to --collocation grid")
    if args.grid_shape is not None and len(args.grid_shape) != args.dim + 1:
        parser.error(f"--grid-shape needs {args.dim + 1} values for --dim {args.dim}")
    ics = INITIAL_CONDITIONS if args.dim == 2 else INITIAL_CONDITIONS_1D
    if args.ic is None:
        args.ic = "cross" if args.dim == 2 else "cosine"
    if args.ic not in ics:
        parser.error(f"--ic {args.ic!r} is not a {args.dim}D initial condition; choose from {sorted(ics)}")
    return args


def is_improvement(total: float, best: float | None, tol: float) -> bool:
    """Whether `total` beats the best total loss so far by enough to count.

    Args:
        total: the total loss after the Adam epoch or L-BFGS step just taken.
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
    lower, upper = domain_bounds(args.dim, args.x_max, args.y_max, args.t_max)

    model = PINN(
        lower_bound=lower,
        upper_bound=upper,
        input_dim=len(lower),
        output_dim=2,  # u (phase field) and mu (chemical potential)
        hidden_layers=args.hidden_layers,
        hidden_width=args.hidden_width,
    ).to(device)

    n_collocation = 100 if args.smoke_test else args.n_collocation
    n_ic = 10 if args.smoke_test else args.n_ic
    n_bc = 10 if args.smoke_test else args.n_bc
    # Seeds the t=0 PDE point choice in sample_points; the collocation
    # samplers carry their own generators.
    torch.manual_seed(args.sampling_seed)
    sampler = make_sampler(
        args.collocation,
        lower,
        upper,
        n_collocation,
        seed=args.sampling_seed,
        resample_every=args.resample_every,
        grid_shape=args.grid_shape,
        device=device,
    )
    collocation = sampler.sample()
    print(f"collocation: {args.collocation}, {collocation.shape[0]} interior points")
    points = sample_points(
        lower=lower,
        upper=upper,
        collocation=collocation,
        n_ic=n_ic,
        n_bc=n_bc,
        device=device,
        pde_at_t0=args.pde_at_t0,
        epsilon=args.epsilon,
        ic=args.ic,
        ic_seed=args.ic_seed,
    )

    def compute_loss(points: TrainingPoints) -> dict[str, torch.Tensor]:
        loss_pde_u, loss_pde_mu = pde_residual_losses(model, points.collocation, epsilon=args.epsilon)
        losses = {
            "pde_u": args.pde_u_weight * loss_pde_u,
            "pde_mu": args.pde_mu_weight * loss_pde_mu,
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
    scheduler = None
    if args.plateau_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.adam_tol,
            threshold_mode="rel",
            min_lr=args.min_lr,
        )
    history: dict[str, list[float]] = {}

    def record(losses: dict[str, torch.Tensor]) -> None:
        for name, value in losses.items():
            history.setdefault(name, []).append(value.item())

    # Unweighted PDE residuals and IC/BC losses on fixed fresh points,
    # against the history step they were taken at -- the training loss alone
    # can't show the network overfitting its own collocation/IC/BC points.
    heldout: dict[str, list[float]] | None = None
    if args.heldout_every:
        heldout = {"step": [], "pde_u": [], "pde_mu": [], "ic": [], "bc": []}
        x_heldout = heldout_points(lower, upper, args.heldout_n, seed=args.sampling_seed + 1, device=device)
        ic_heldout, ic_values_heldout, bc_heldout = heldout_ic_bc_points(
            lower,
            upper,
            args.heldout_n_ic,
            args.heldout_n_bc,
            seed=args.sampling_seed + 2,
            epsilon=args.epsilon,
            ic=args.ic,
            ic_seed=args.ic_seed,
            device=device,
        )

    def record_heldout() -> None:
        loss_u, loss_mu = pde_residual_losses(model, x_heldout, epsilon=args.epsilon)
        heldout["step"].append(len(history["total"]))
        heldout["pde_u"].append(loss_u.item())
        heldout["pde_mu"].append(loss_mu.item())
        heldout["ic"].append(ic_loss(model, ic_heldout, ic_values_heldout).item())
        heldout["bc"].append(bc_loss(model, bc_heldout).item())

    layer_stats = LayerStatsLog(model) if args.layer_stats_every else None
    # Wall-clock seconds per optimizer phase (including logging/checkpoint
    # overhead), for comparing training cost against FEM.
    timing = {"adam_seconds": 0.0, "lbfgs_seconds": 0.0}

    def checkpoint_extras() -> dict:
        extras = {"timing": dict(timing)}
        if layer_stats is not None:
            extras["layer_stats"] = layer_stats.as_dict()
        return extras

    def activation_points() -> torch.Tensor:
        return points.collocation[:5000]

    best_adam_total: float | None = None
    best_adam_state: dict[str, torch.Tensor] | None = None
    adam_stalled = 0
    adam_start = time.perf_counter()
    for epoch in range(epochs):
        stats_now = layer_stats is not None and (
            epoch % args.layer_stats_every == 0 or epoch == epochs - 1
        )
        optimizer.zero_grad()
        losses = compute_loss(points)
        if stats_now:
            # Index of this epoch's entry in history: the stats belong to the
            # weights the loss was evaluated at, i.e. before the update.
            layer_stats.record(len(history.get("total", [])), "adam", model, losses, activation_points())
            weights_before = weight_snapshot(model)
        losses["total"].backward()
        # The loss belongs to the weights *before* this update, so snapshot
        # them now in case they turn out to be the best seen.
        if args.adam_patience:
            pre_step_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        optimizer.step()
        if stats_now:
            layer_stats.add_update_ratio(update_ratios(model, weights_before))
        record(losses)
        total = losses["total"].item()
        lr = optimizer.param_groups[0]["lr"]
        if epoch % args.log_every == 0 or epoch == epochs - 1:
            print(f"[adam] epoch {epoch:5d}/{epochs} lr {lr:.2e} {format_losses(losses)}")
        if heldout is not None and (epoch % args.heldout_every == 0 or epoch == epochs - 1):
            record_heldout()
        if epoch % args.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_step{len(history['total']):06d}.pt")
            timing["adam_seconds"] = time.perf_counter() - adam_start
            save_checkpoint(
                model, checkpoint_path, args, history, heldout,
                adam_steps=len(history["total"]), extras=checkpoint_extras(),
            )

        new_collocation = sampler.update(model, epoch + 1)
        if new_collocation is not None:
            points.collocation = with_pde_at_t0(new_collocation, points.t0_points)

        if scheduler is not None:
            scheduler.step(total)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < lr:
                print(f"[adam] epoch {epoch:5d}/{epochs} plateau: lr {lr:.2e} -> {new_lr:.2e}")

        if args.adam_patience:
            if is_improvement(total, best_adam_total, args.adam_tol):
                best_adam_total = total
                best_adam_state = pre_step_state
                adam_stalled = 0
            else:
                adam_stalled += 1
                if adam_stalled >= args.adam_patience:
                    print(
                        f"[adam] early stop after epoch {epoch}/{epochs}: total loss improved by "
                        f"less than {args.adam_tol:g} (relative) for {adam_stalled} consecutive epochs"
                    )
                    break
    timing["adam_seconds"] = time.perf_counter() - adam_start

    if best_adam_state is not None:
        model.load_state_dict(best_adam_state)
        print(f"[adam] restored best weights (total {best_adam_total:.4e})")
    # --adam-patience can end Adam before --epochs, so history alone can't
    # tell where L-BFGS took over; checkpoints carry it from here on.
    adam_steps = len(history.get("total", []))

    lbfgs_steps = 2 if args.smoke_test else args.lbfgs_steps
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
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
    lbfgs_start = time.perf_counter()
    for lbfgs_step in range(lbfgs_steps):
        stats_now = layer_stats is not None and lbfgs_step % args.layer_stats_every == 0
        if stats_now:
            # The step's first closure call evaluates these same weights, so
            # this lines up with that history entry.
            layer_stats.record(len(history["total"]), "lbfgs", model, compute_loss(points), activation_points())
            weights_before = weight_snapshot(model)
        lbfgs.step(closure)
        if stats_now:
            layer_stats.add_update_ratio(update_ratios(model, weights_before))
        timing["lbfgs_seconds"] = time.perf_counter() - lbfgs_start
        print(f"[lbfgs] step {lbfgs_step + 1:3d}/{lbfgs_steps} {format_losses(last_losses)}")
        if heldout is not None and (lbfgs_step + 1) % args.heldout_every == 0:
            record_heldout()
        if lbfgs_step % args.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_step{len(history['total']):06d}.pt")
            save_checkpoint(
                model, checkpoint_path, args, history, heldout, adam_steps=adam_steps, extras=checkpoint_extras()
            )

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

    if heldout is not None and (not heldout["step"] or heldout["step"][-1] != len(history["total"])):
        record_heldout()
    save_checkpoint(model, final_path, args, history, heldout, adam_steps=adam_steps, extras=checkpoint_extras())
    print(f"final model: {final_path}")


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    args: argparse.Namespace,
    history: dict[str, list[float]],
    heldout: dict[str, list[float]] | None = None,
    adam_steps: int | None = None,
    extras: dict | None = None,
) -> None:
    """Save model weights plus the metadata needed to reload them.

    Bundling `args` (the exact CLI flags this run used) and `history` (each
    weighted loss component -- "pde_u", "pde_mu", "ic", "bc", "total", and "energy" when
    --energy-penalty is enabled -- per optimizer step) alongside the weights
    means evaluate.py can reconstruct the right architecture and domain
    bounds on its own instead of having them re-supplied by hand -- and the
    loss curve survives past the training process instead of being discarded
    when it exits.

    `heldout` (with --heldout-every) is saved under its own key rather than
    in `history`: it is logged every K steps, not every step, so it can't
    share history's one-value-per-step lists. Keys are "step" (index into
    history), "pde_u" and "pde_mu" (unweighted).

    `adam_steps` is saved as "adam_steps": how many leading history entries
    are Adam epochs, the rest being L-BFGS closure evaluations (see
    src/pinn/run_summary.py::adam_step_count).

    `extras` are merged in as further top-level keys: train.py passes
    "timing" (adam_seconds, lbfgs_seconds) and, with --layer-stats-every,
    "layer_stats" (src/pinn/layer_stats.py::LayerStatsLog.as_dict).
    """
    checkpoint = {"model_state": model.state_dict(), "args": vars(args), "history": history}
    if heldout is not None:
        checkpoint["heldout"] = heldout
    if adam_steps is not None:
        checkpoint["adam_steps"] = adam_steps
    if extras:
        checkpoint.update(extras)
    torch.save(checkpoint, path)


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
