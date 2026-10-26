"""Evaluate and plot a 1D (--dim 1) PINN checkpoint, optionally against FEM.

The 1D counterpart of evaluate.py + diagnostics.py + plot_results.py's field
plots, in one step: in 1D the whole (x, t) field fits in one forward pass, so
there is no separate evaluation stage. Run as a module from the repo root:

    python -m src.pinn.plot_1d --checkpoint results/pinn_models/<run>/final.pt
    python -m src.pinn.plot_1d --checkpoint results/pinn_models/<run>/final.pt \\
        --fem-fields results/eps0.05_nx1000_dt1e-5_t0.1_dim1_cosine/fem_fields.npz

Writes into <run>/plots/ (or --output-dir):
    space_time_u, space_time_mu   u and mu over the whole (x, t) window
    profiles_u                    u(x) at a few times, with the IC (and, for
                                  --ic cosine, the analytic linear solution)
    energy_dissipation            free energy E(t) (metrics.free_energy)
    mass_conservation             drift of the integral of u from t = 0
and, with --fem-fields (a 1D FEM run, src.fem.cahn_hilliard --dim 1):
    compare_space_time_u/_mu      PINN | FEM | PINN - FEM over (x, t)
    rel_l2_vs_fem                 relative L2 error of u and mu against t
plus FEM curves on profiles_u, energy_dissipation and mass_conservation.

With --fem-fields the PINN is evaluated on the FEM run's own vertices and
(every few) snapshot times, so the errors involve no interpolation.

Also writes, next to plots/: evaluation_1d.npz (x, t, u, mu, free_energy,
total_mass, and with FEM rel_l2_u, rel_l2_mu per time) and diagnostics.csv
(t, free_energy, total_mass, same format as diagnostics.py), so
run_metrics.py reads a 1D run like a 2D one. Loss curves, held-out and
layer-stats plots come from plot_results.py --checkpoint as usual; all of it
in one go via scripts/analyze_run.py.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.initial_conditions import cosine_growth_rate, initial_condition_1d
from src.common.metrics import free_energy, relative_l2_error, total_mass
from src.common.plotting import load_diagnostics, phase_colormap, plot_energy_dissipation, plot_mass_conservation
from src.pinn.evaluate import evaluate_grid_1d, evaluate_space_time_1d, load_model
from src.pinn.run_paths import infer_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and plot a 1D Cahn-Hilliard PINN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="A .pt from train.py --dim 1")
    parser.add_argument("--nx", type=int, default=401, help="Grid points in x (ignored with --fem-fields)")
    parser.add_argument(
        "--nt",
        type=int,
        default=201,
        help="Grid points in t. With --fem-fields: about this many FEM snapshots, evenly strided",
    )
    parser.add_argument("--n-profiles", type=int, default=5, help="Number of u(x) snapshot times")
    parser.add_argument(
        "--fem-fields",
        type=str,
        default=None,
        help="fem_fields.npz of a 1D FEM run (src.fem.cahn_hilliard --dim 1) with the same IC and epsilon",
    )
    parser.add_argument(
        "--fem-diagnostics",
        type=str,
        default=None,
        help="FEM cahn_hilliard_diagnostics.csv. Defaults to the one next to --fem-fields, when present.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Defaults to <run>/plots/")
    parser.add_argument("--format", choices=["png", "pdf"], default="pdf")
    parser.add_argument("--dpi", type=int, default=None, help="Defaults to 150 for png, 300 for pdf")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    if args.dpi is None:
        args.dpi = 300 if args.format == "pdf" else 150
    if args.output_dir is None:
        args.output_dir = os.path.join(infer_run_dir(args.checkpoint), "plots")
    if args.fem_diagnostics is None and args.fem_fields is not None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.fem_fields)), "cahn_hilliard_diagnostics.csv")
        if os.path.exists(candidate):
            args.fem_diagnostics = candidate
    return args


def trapezoid_weights_1d(nx: int, x_bounds: tuple[float, float]) -> np.ndarray:
    """1D trapezoid weights for np.linspace(*x_bounds, nx); see
    metrics.trapezoid_weights_2d for why not a uniform weight."""
    weight = np.full(nx, (x_bounds[1] - x_bounds[0]) / (nx - 1))
    weight[[0, -1]] /= 2
    return weight


def load_fem_1d(path: str, t_max: float, nt: int) -> dict:
    """The snapshots of a 1D fem_fields.npz inside the PINN's [0, t_max],
    strided down to about nt of them (always keeping the first)."""
    with np.load(path) as f:
        fem = dict(f)
    if fem["x"].ndim != 1:
        raise SystemExit(f"{path} is a 2D FEM run; pass a src.fem.cahn_hilliard --dim 1 one")
    inside = np.flatnonzero(fem["t"] <= t_max * (1 + 1e-9))
    stride = max(1, int(round((len(inside) - 1) / max(nt - 1, 1))))
    keep = inside[::stride]
    return {"x": fem["x"], "t": fem["t"][keep], "u": fem["u"][keep], "mu": fem["mu"][keep]}


def relative_l2_per_time(pinn: np.ndarray, fem: np.ndarray) -> np.ndarray:
    """Relative L2 error of each row (time) of pinn against fem; NaN where the
    FEM row is identically zero (the FEM solver's mu(t=0) = 0)."""
    return np.array([relative_l2_error(p, f) if np.any(f) else np.nan for p, f in zip(pinn, fem)])


def plot_space_time(x, t, field, title, **imshow_kwargs):
    """Field over (x, t), x horizontal and t upwards."""
    fig, ax = plt.subplots(figsize=(6, 4))
    _space_time_panel(fig, ax, x, t, field, title, **imshow_kwargs)
    return fig


def _space_time_panel(fig, ax, x, t, field, title, **imshow_kwargs):
    image = ax.imshow(
        field, origin="lower", aspect="auto", extent=(x[0], x[-1], t[0], t[-1]), interpolation="nearest",
        **imshow_kwargs,
    )
    image.set_rasterized(True)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)


def plot_compare_space_time(x, t, pinn, fem, name, rel_l2, **imshow_kwargs):
    """PINN | FEM | PINN - FEM over (x, t) for one field, the error panel on
    a symmetric scale of its own."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    _space_time_panel(fig, axes[0], x, t, pinn, f"PINN {name}", **imshow_kwargs)
    _space_time_panel(fig, axes[1], x, t, fem, f"FEM {name}", **imshow_kwargs)
    error = pinn - fem
    bound = max(float(np.max(np.abs(error))), 1e-12)
    _space_time_panel(
        fig, axes[2], x, t, error, f"PINN - FEM (rel. L2 {rel_l2:.2e})", cmap="RdBu_r", vmin=-bound, vmax=bound
    )
    return fig


def plot_rel_l2(t, rel_l2_u, rel_l2_mu, overall_u, overall_mu):
    """Relative L2 error against FEM at each time, log scale."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(t, rel_l2_u, lw=2, label=f"u (space-time {overall_u:.2e})")
    ax.semilogy(t, rel_l2_mu, lw=2, label=rf"$\mu$ (space-time {overall_mu:.2e})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|\cdot_{PINN} - \cdot_{FEM}\|_2 / \|\cdot_{FEM}\|_2$")
    ax.set_title("Relative L2 error vs FEM")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig


def plot_profiles(x, t, u, k_snap, train_args, u_fem=None):
    """u(x) at the times t[k_snap], with the IC, the FEM solution (dotted,
    when given) and, for the cosine IC, the analytic linear solution
    A exp(sigma t) cos(k pi x) (dashed)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epsilon = train_args["epsilon"]
    u0 = initial_condition_1d(train_args["ic"], x, epsilon=epsilon, seed=train_args["ic_seed"])
    ax.plot(x, u0, color="0.6", lw=3, label="IC $u_0$", zorder=0)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(k_snap)))
    for color, k in zip(colors, k_snap):
        ax.plot(x, u[k], color=color, lw=1.8, label=f"PINN t={t[k]:.3g}")
        if u_fem is not None:
            ax.plot(x, u_fem[k], color="black", lw=1, ls=":")
        if train_args["ic"] == "cosine":
            ax.plot(x, u0 * np.exp(cosine_growth_rate(epsilon) * t[k]), color=color, lw=1, ls="--")
    if u_fem is not None:
        ax.plot([], [], color="black", lw=1, ls=":", label="FEM")
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

    fem = None
    if args.fem_fields:
        fem = load_fem_1d(args.fem_fields, model.upper_bound[-1].item(), args.nt)
        x_lo, x_hi = model.lower_bound[0].item(), model.upper_bound[0].item()
        if not (np.isclose(fem["x"][0], x_lo) and np.isclose(fem["x"][-1], x_hi)):
            raise SystemExit(f"FEM x range [{fem['x'][0]}, {fem['x'][-1]}] != PINN's [{x_lo}, {x_hi}]")
        x, t = fem["x"], fem["t"]
        u, mu, u_x = evaluate_grid_1d(model, x, t, device=args.device)
    else:
        x, t, u, mu, u_x = evaluate_space_time_1d(model, args.nx, args.nt, device=args.device)

    weight = trapezoid_weights_1d(len(x), (x[0], x[-1]))
    energy = np.array([free_energy(u[k], u_x[k][:, None], train_args["epsilon"], weight) for k in range(len(t))])
    mass = np.array([total_mass(u[k], weight) for k in range(len(t))])

    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.dirname(os.path.abspath(args.output_dir))
    arrays = dict(x=x, t=t, u=u, mu=mu, free_energy=energy, total_mass=mass)
    if fem is not None:
        arrays["rel_l2_u"] = relative_l2_per_time(u, fem["u"])
        arrays["rel_l2_mu"] = relative_l2_per_time(mu, fem["mu"])
        overall_u = relative_l2_error(u, fem["u"])
        # mu(t=0) of the FEM is 0 by construction, so leave that row out
        has_mu = np.any(fem["mu"], axis=1)
        overall_mu = relative_l2_error(mu[has_mu], fem["mu"][has_mu])
        print(f"relative L2 vs FEM over (x, t): u {overall_u:.4e}, mu {overall_mu:.4e}")
    npz_path = os.path.join(run_dir, "evaluation_1d.npz")
    np.savez(npz_path, **arrays)
    print(f"saved {npz_path}")
    diag_path = os.path.join(run_dir, "diagnostics.csv")
    with open(diag_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("t,free_energy,total_mass\n")
        for t_i, e_i, m_i in zip(t, energy, mass):
            fh.write(f"{t_i:.16e},{e_i:.16e},{m_i:.16e}\n")
    print(f"saved {diag_path}")

    t_fem_diag = e_fem = m_fem = None
    if args.fem_diagnostics:
        t_fem_diag, e_fem, m_fem = load_diagnostics(args.fem_diagnostics)
        inside = t_fem_diag <= t[-1] * (1 + 1e-9)
        t_fem_diag, e_fem, m_fem = t_fem_diag[inside], e_fem[inside], m_fem[inside]

    mu_bound = max(float(np.quantile(np.abs(mu), 0.99)), 1e-12)
    phase_kwargs = dict(cmap=phase_colormap(), vmin=-1, vmax=1)
    mu_kwargs = dict(cmap="RdBu_r", vmin=-mu_bound, vmax=mu_bound)
    k_snap = np.linspace(0, len(t) - 1, args.n_profiles).round().astype(int)
    figures = {
        "space_time_u": plot_space_time(x, t, u, "PINN u(x, t)", **phase_kwargs),
        "space_time_mu": plot_space_time(x, t, mu, r"PINN $\mu$(x, t)", **mu_kwargs),
        "profiles_u": plot_profiles(x, t, u, k_snap, train_args, None if fem is None else fem["u"]),
        "energy_dissipation": plot_energy_dissipation(t, energy, t_fem_diag, e_fem).figure,
        "mass_conservation": plot_mass_conservation(t, mass, t_fem_diag, m_fem).figure,
    }
    if fem is not None:
        figures["compare_space_time_u"] = plot_compare_space_time(x, t, u, fem["u"], "u", overall_u, **phase_kwargs)
        figures["compare_space_time_mu"] = plot_compare_space_time(
            x, t, mu, fem["mu"], r"$\mu$", overall_mu, **mu_kwargs
        )
        figures["rel_l2_vs_fem"] = plot_rel_l2(t, arrays["rel_l2_u"], arrays["rel_l2_mu"], overall_u, overall_mu)
    for name, fig in figures.items():
        path = os.path.join(args.output_dir, f"{name}.{args.format}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {path}")


if __name__ == "__main__":
    main(parse_args())
