"""Evaluate and plot a 1D (--dim 1) PINN checkpoint.

The 1D counterpart of evaluate.py + diagnostics.py + plot_results.py's field
plots, in one step: in 1D the whole (x, t) field fits in one forward pass, so
there is no separate evaluation stage. Run as a module from the repo root:

    python -m src.pinn.plot_1d --checkpoint results/pinn_models/<run>/final.pt

Writes into <run>/plots/ (or --output-dir):
    space_time_u, space_time_mu   u and mu over the whole (x, t) window
    profiles_u                    u(x) at a few times, with the IC (and, for
                                  --ic cosine, the analytic linear solution)
    energy_dissipation            free energy E(t) (metrics.free_energy)
    mass_conservation             drift of the integral of u from t = 0
and evaluation_1d.npz (x, t, u, mu, free_energy, total_mass) next to the
checkpoint's run folder. Loss curves, held-out and layer-stats plots of a 1D
run come from plot_results.py --checkpoint as usual.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.initial_conditions import cosine_growth_rate, initial_condition_1d
from src.common.metrics import free_energy, total_mass
from src.common.plotting import phase_colormap, plot_energy_dissipation, plot_mass_conservation
from src.pinn.evaluate import evaluate_space_time_1d, load_model
from src.pinn.run_paths import infer_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and plot a 1D Cahn-Hilliard PINN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="A .pt from train.py --dim 1")
    parser.add_argument("--nx", type=int, default=401, help="Grid points in x")
    parser.add_argument("--nt", type=int, default=201, help="Grid points in t")
    parser.add_argument("--n-profiles", type=int, default=5, help="Number of u(x) snapshot times")
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <run>/plots/")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    parser.add_argument("--dpi", type=int, default=None, help="Defaults to 150 for png, 300 for pdf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.dpi is None:
        args.dpi = 300 if args.format == "pdf" else 150
    if args.output_dir is None:
        args.output_dir = os.path.join(infer_run_dir(args.checkpoint), "plots")
    return args


def trapezoid_weights_1d(nx: int, x_bounds: tuple[float, float]) -> np.ndarray:
    """1D trapezoid weights for np.linspace(*x_bounds, nx); see
    metrics.trapezoid_weights_2d for why not a uniform weight."""
    weight = np.full(nx, (x_bounds[1] - x_bounds[0]) / (nx - 1))
    weight[[0, -1]] /= 2
    return weight


def plot_space_time(x, t, field, title, **imshow_kwargs):
    """Field over (x, t), x horizontal and t upwards."""
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(
        field, origin="lower", aspect="auto", extent=(x[0], x[-1], t[0], t[-1]), interpolation="nearest",
        **imshow_kwargs,
    )
    image.set_rasterized(True)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    return fig


def plot_profiles(x, t, u, k_snap, train_args):
    """u(x) at the times t[k_snap], with the IC and, for the cosine IC, the
    analytic linear solution A exp(sigma t) cos(k pi x) (dashed)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epsilon = train_args["epsilon"]
    u0 = initial_condition_1d(train_args["ic"], x, epsilon=epsilon, seed=train_args["ic_seed"])
    ax.plot(x, u0, color="0.6", lw=3, label="IC $u_0$", zorder=0)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_snap)))
    for color, k in zip(colors, k_snap):
        ax.plot(x, u[k], color=color, lw=1.8, label=f"PINN t={t[k]:.3g}")
        if train_args["ic"] == "cosine":
            ax.plot(x, u0 * np.exp(cosine_growth_rate(epsilon) * t[k]), color=color, lw=1, ls="--")
    if train_args["ic"] == "cosine":
        ax.plot([], [], color="0.3", lw=1, ls="--", label="linear theory")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"u(x) profiles, ic={train_args['ic']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def main(args: argparse.Namespace) -> None:
    train_args = torch.load(args.checkpoint, map_location="cpu")["args"]
    if train_args.get("dim", 2) != 1:
        raise SystemExit(f"{args.checkpoint} is not a --dim 1 checkpoint; use src.pinn.plot_results")
    model = load_model(args.checkpoint, device=args.device)
    x, t, u, mu, u_x = evaluate_space_time_1d(model, args.nx, args.nt, device=args.device)

    weight = trapezoid_weights_1d(args.nx, (x[0], x[-1]))
    energy = np.array([free_energy(u[k], u_x[k][:, None], train_args["epsilon"], weight) for k in range(len(t))])
    mass = np.array([total_mass(u[k], weight) for k in range(len(t))])

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(os.path.dirname(args.output_dir), "evaluation_1d.npz")
    np.savez(npz_path, x=x, t=t, u=u, mu=mu, free_energy=energy, total_mass=mass)
    print(f"saved {npz_path}")

    mu_bound = max(float(np.quantile(np.abs(mu), 0.99)), 1e-12)
    k_snap = np.linspace(0, len(t) - 1, args.n_profiles).round().astype(int)
    figures = {
        "space_time_u": plot_space_time(x, t, u, "PINN u(x, t)", cmap=phase_colormap(), vmin=-1, vmax=1),
        "space_time_mu": plot_space_time(
            x, t, mu, r"PINN $\mu$(x, t)", cmap="RdBu_r", vmin=-mu_bound, vmax=mu_bound
        ),
        "profiles_u": plot_profiles(x, t, u, k_snap, train_args),
        "energy_dissipation": plot_energy_dissipation(t, energy).figure,
        "mass_conservation": plot_mass_conservation(t, mass).figure,
    }
    for name, fig in figures.items():
        path = os.path.join(args.output_dir, f"{name}.{args.format}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main(parse_args())
