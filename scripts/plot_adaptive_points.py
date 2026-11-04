"""Where the adaptive sampler puts its points: the points added by the runs in
plot_adaptive.RUNS, stored in final.pt under "adaptive" (points, history step).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_adaptive_points.py
Writes to OUT:
  adaptive_points_t.pdf: share of added points per log-spaced t-bin, per run,
    against what uniform sampling in t would put there,
  adaptive_points_xy.pdf: added points in (x, y), coloured by t, one panel per
    run, with the FEM interface u = 0 at t = 0 and at T_LATE,
  adaptive_points_u.pdf: distribution of |u_FEM| at the added points (nearest
    FEM node and snapshot; the dt = 1e-6 EARLY_FEM run for t <= its end), against
    uniform points: |u| near 0 is the interface.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common.plotting import LINE_COLORS, LINE_MARKERS, LINE_STYLES  # noqa: E402  (also sets the colour-blind-safe cycle)
from plot_adaptive import OUT, RUNS  # noqa: E402
from plot_lbfgs_points import EARLY_FEM, FEM  # noqa: E402
from src.pinn.run_paths import run_dir_for  # noqa: E402

T_FLOOR = 1e-6  # points at t = 0 are drawn in the first bin
T_LATE = 1e-2


def load_added(run):
    checkpoint = torch.load(Path(run_dir_for(run)) / "final.pt", map_location="cpu", weights_only=False)
    added = checkpoint["adaptive"]
    return added["points"].numpy(), added["step"].numpy(), checkpoint["args"]


def plot_t_hist(runs, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    t_max = None
    for k, (label, run) in enumerate(runs.items()):
        points, _, args = load_added(run)
        t_max = args["t_max"]
        edges = np.logspace(np.log10(T_FLOOR), np.log10(t_max), 13)
        counts, _ = np.histogram(np.clip(points[:, 2], T_FLOOR, None), bins=edges)
        centers = np.sqrt(edges[:-1] * edges[1:])
        ax.semilogx(centers, counts / counts.sum(), linestyle=LINE_STYLES[k], color=LINE_COLORS[k],
                    marker=LINE_MARKERS[k], markersize=4, label=label)
    uniform = np.diff(edges) / t_max
    uniform[0] += T_FLOOR / t_max
    ax.semilogx(centers, uniform, lw=2, color="0.6", label="uniform in $t$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("share of added points per bin")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def plot_xy(runs, out):
    fem = dict(np.load(FEM))
    late = int(np.argmin(np.abs(fem["t"] - T_LATE)))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for ax, (label, run) in zip(axes.flat, runs.items()):
        points, _, args = load_added(run)
        order = np.argsort(-points[:, 2])  # earliest points on top
        sc = ax.scatter(points[order, 0], points[order, 1], c=np.clip(points[order, 2], T_FLOOR, None),
                        s=4, cmap="viridis", norm=LogNorm(T_FLOOR, args["t_max"]))
        ax.contour(fem["x"], fem["y"], fem["u"][0], levels=[0], colors="black", linewidths=1)
        ax.contour(fem["x"], fem["y"], fem["u"][late], levels=[0], colors=LINE_COLORS[1],
                   linewidths=1.5, linestyles="--")
        ax.set_title(label.replace(", ", ",\n", 1), fontsize=8)
        ax.set_xlim(0, args["x_max"])
        ax.set_ylim(0, args["y_max"])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.plot([], [], color="black", lw=1, label="FEM $u = 0$ at $t = 0$")
    legend_ax.plot([], [], color=LINE_COLORS[1], lw=1.5, ls="--", label=f"FEM $u = 0$ at $t = {fem['t'][late]:g}$")
    legend_ax.legend(loc="center", fontsize=8, frameon=False)
    fig.colorbar(sc, ax=axes, shrink=0.7, label="$t$ of added point")
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def fem_abs_u(fem, early, points):
    """|u| of the FEM solution at the nearest snapshot and grid node to each point,
    from `early` where it covers the point's t (FEM's snapshots are 5e-4 apart,
    so most added points would otherwise get the t = 0 IC)."""
    abs_u = _nearest_abs_u(fem, points)
    in_early = points[:, 2] <= early["t"][-1]
    abs_u[in_early] = _nearest_abs_u(early, points[in_early])
    return abs_u


def _nearest_abs_u(fem, points):
    k = np.abs(fem["t"][None, :] - points[:, 2:3]).argmin(axis=1)
    n = fem["x"].shape[0] - 1
    i = np.rint(points[:, 0] / fem["x"].max() * n).astype(int)
    j = np.rint(points[:, 1] / fem["y"].max() * n).astype(int)
    if fem["x"][0, 1] != fem["x"][0, 0]:  # x along axis 1 (xy indexing)
        i, j = j, i
    return np.abs(fem["u"][k, i, j])


def plot_u_hist(runs, out, n_uniform=200000, seed=0):
    fem, early = dict(np.load(FEM)), dict(np.load(EARLY_FEM))
    edges = np.linspace(0, 1.05, 22)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, (label, run) in enumerate(runs.items()):
        points, _, args = load_added(run)
        abs_u = fem_abs_u(fem, early, points)
        counts, _ = np.histogram(np.clip(abs_u, 0, edges[-1] - 1e-9), bins=edges)
        ax.plot(centers, counts / counts.sum(), linestyle=LINE_STYLES[k], color=LINE_COLORS[k],
                marker=LINE_MARKERS[k], markersize=4, label=label)
        print(f"{label}: share with |u| < 0.5: {np.mean(abs_u < 0.5):.2f}")
    upper = np.array([args["x_max"], args["y_max"], args["t_max"]])
    uniform = np.random.default_rng(seed).random((n_uniform, 3)) * upper
    abs_u = fem_abs_u(fem, early, uniform)
    counts, _ = np.histogram(np.clip(abs_u, 0, edges[-1] - 1e-9), bins=edges)
    ax.plot(centers, counts / counts.sum(), lw=2, color="0.6", label="uniform in $(x, y, t)$")
    print(f"uniform: share with |u| < 0.5: {np.mean(abs_u < 0.5):.2f}")
    ax.set_xlabel("$|u_{\\mathrm{FEM}}|$ at the point (0 = interface, 1 = pure phase)")
    ax.set_ylabel("share of points per bin")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    OUT.mkdir(exist_ok=True)
    plot_t_hist(RUNS, OUT / "adaptive_points_t.pdf")
    plot_xy(RUNS, OUT / "adaptive_points_xy.pdf")
    plot_u_hist(RUNS, OUT / "adaptive_points_u.pdf")


if __name__ == "__main__":
    main()
