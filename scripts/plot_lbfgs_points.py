"""Compare runs with a denser, fresh L-BFGS point set (--lbfgs-n-collocation)
against the run that keeps Adam's last 20000 points.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_lbfgs_points.py
Writes lbfgs_points_l2.pdf (relative L2 error vs FEM over time, at every FEM
snapshot in the training window) and lbfgs_points_heldout.pdf (held-out vs
training PDE residuals per step) to OUT, and the same pair as lbfgs_long_*.pdf
for LONG_RUNS: the 100000-point run trained for 400 L-BFGS steps instead of 120,
plus lbfgs_long_early_l2.pdf: LONG_RUNS against the dt = 1e-6 FEM run on t <= 2e-4,
and lbfgs_long_heldout_windows.pdf: their held-out PDE residuals per t-window,
and lbfgs_resample_l2.pdf: RESAMPLE_RUNS, the fixed sets against resampling
during L-BFGS (--lbfgs-resample-every), and lbfgs_resample_long_l2.pdf:
LONG_RUNS plus the 400-step run resampled every 20 steps, and
lbfgs_budget400_l2.pdf: BUDGET_RUNS. `--resample` makes only those three.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.plotting import LINE_COLORS, LINE_MARKERS, LINE_STYLES  # noqa: E402  (also sets the colour-blind-safe cycle)
from plot_ic_pretrain import rel_l2_series  # noqa: E402
from compare_residuals import T_EDGE_FRACTIONS  # noqa: E402
from src.pinn.collocation import heldout_points  # noqa: E402
from src.pinn.evaluate import load_model  # noqa: E402
from src.pinn.losses import pde_residual_losses  # noqa: E402

MODELS = Path("results/pinn_models")
BASE = "eps0.05_h100x5_t0_0.01_p20000_wmu10000_icw10000_nopdeat-t0_icnorm_e2000-lb120_sobol_heldout"
RUNS = {
    "L-BFGS on Adam's 20000 points": f"{BASE}_ic-cross",
    "L-BFGS on 60000 new points": f"{BASE}_lbp60000_ic-cross",
    "L-BFGS on 100000 new points": f"{BASE}_lbp100000_ic-cross",
}
LONG_RUNS = {
    "Adam's 20000 points, 120 L-BFGS steps": f"{BASE}_ic-cross",
    "100000 new points, 120 L-BFGS steps": f"{BASE}_lbp100000_ic-cross",
    "100000 new points, 400 L-BFGS steps": f"{BASE.replace('-lb120', '-lb400')}_lbp100000_ic-cross",
}
# Fixed sets (solid) against redrawing Adam's 20000-point Sobol sequence every K
# L-BFGS steps (--lbfgs-resample-every, dashed); more runs than LINE_COLORS has,
# so the styles are set per run: (color, line style, marker).
RESAMPLE_RUNS = {
    "fixed: Adam's 20000 points": f"{BASE}_ic-cross",
    "fixed: 60000 new points": f"{BASE}_lbp60000_ic-cross",
    "fixed: 100000 new points": f"{BASE}_lbp100000_ic-cross",
    "20000 points, resampled every step": f"{BASE}_lbrs1_ic-cross",
    "20000 points, resampled every 5 steps": f"{BASE}_lbrs5_ic-cross",
    "20000 points, resampled every 10 steps": f"{BASE}_lbrs10_ic-cross",
    "20000 points, resampled every 20 steps": f"{BASE}_lbrs20_ic-cross",
}
RESAMPLE_STYLES = [
    (LINE_COLORS[4], "-", "o"),
    (LINE_COLORS[2], "-", "^"),
    (LINE_COLORS[0], "-", "s"),
    (LINE_COLORS[1], "--", "D"),
    (LINE_COLORS[3], "--", "v"),
    (LINE_COLORS[1], ":", "x"),
    (LINE_COLORS[4], "-.", "P"),
]
# LONG_RUNS plus resampling every 20 steps on the same 2000 Adam + 400 L-BFGS budget
LONG_RESAMPLE_RUNS = {
    **LONG_RUNS,
    "20000 points resampled every 20 steps, 400 L-BFGS steps": f"{BASE.replace('-lb120', '-lb400')}_lbrs20_ic-cross",
}
# All on 2000 Adam + 400 L-BFGS steps. The only 400-step run on Adam's fixed
# 20000 points was trained with wmu = icw = 3000, not 10000 as the other two.
BUDGET_RUNS = {
    "fixed: Adam's 20000 points (weights 3000)":
        "eps0.05_h100x5_t0_0.01_p20000_wmu3000_icw3000_nopdeat-t0_icnorm_e2000-lb400_sobol_heldout_ic-cross",
    "fixed: 100000 new points (weights $10^4$)": LONG_RUNS["100000 new points, 400 L-BFGS steps"],
    "20000 points resampled every 20 steps (weights $10^4$)":
        LONG_RESAMPLE_RUNS["20000 points resampled every 20 steps, 400 L-BFGS steps"],
}
FEM = Path("results/eps0.05_nx100_dte-4/fem_fields.npz")
# dt = 1e-6, every step saved, t <= 2e-4: resolves the initial layer before FEM's first snapshot
EARLY_FEM = Path("results/eps0.05_nx100_dt1e-6_t2e-4_early/fem_fields.npz")
OUT = MODELS / "_lbfgs_points"
FIRST_STEP = 1500


def plot_l2(runs, out, fem_path=FEM, early=False, styles=None):
    """With early=True: log t axis (t = 0 dropped), sparse markers, and a line at t = 1e-4,
    where the held-out residual rise ends. styles: optional (color, line style, marker)
    per run, instead of the k-th LINE_COLORS/LINE_STYLES/LINE_MARKERS."""
    fem = dict(np.load(fem_path))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for k, (label, run) in enumerate(runs.items()):
        t, errors = rel_l2_series(run, fem)
        keep = t > 0 if early else slice(None)
        color, style, marker = styles[k] if styles else (LINE_COLORS[k], LINE_STYLES[k], LINE_MARKERS[k])
        for ax, name in zip(axes, ["u", "mu"]):
            ax.plot(t[keep], errors[name][keep], style, color=color,
                    marker=marker, markersize=4, markevery=0.1 if early else 1, label=label)
        print(f"{label}: u mean {np.nanmean(errors['u']):.3f}%, mu mean {np.nanmean(errors['mu']):.3f}%")
    for ax, name in zip(axes, ["u", "\\mu"]):
        if early:
            ax.set_xscale("log")
            ax.axvline(1e-4, lw=1, ls=":", color="0.4")
        ax.set_title(f"${name}$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("relative $L^2$ error vs FEM [%]")
        ax.grid(alpha=0.3)
    # one shared legend below both panels, so it never covers the curves
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def plot_heldout(runs, out):
    """Unweighted PDE residuals: held-out points (markers) and the training
    points (thin line, divided by the loss weight), per history step, from
    FIRST_STEP on: late Adam and all of L-BFGS, without the early Adam spikes."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for k, (label, run) in enumerate(runs.items()):
        checkpoint = torch.load(MODELS / run / "final.pt", map_location="cpu", weights_only=False)
        args, history, heldout = checkpoint["args"], checkpoint["history"], checkpoint["heldout"]
        for ax, name in zip(axes, ["pde_u", "pde_mu"]):
            train = np.array(history[name]) / args[f"{name}_weight"]
            steps = np.arange(len(train))
            ax.semilogy(steps[FIRST_STEP:], train[FIRST_STEP:], "-", lw=0.6, alpha=0.6, color=LINE_COLORS[k])
            held_steps, held = np.array(heldout["step"]), np.array(heldout[name])
            keep = held_steps >= FIRST_STEP
            ax.semilogy(held_steps[keep], held[keep], LINE_STYLES[k], lw=1.5, color=LINE_COLORS[k],
                        marker=LINE_MARKERS[k], markersize=4, markevery=20, label=label)
        adam_steps = checkpoint["adam_steps"]  # the same 2000 Adam epochs in every run
    for ax, title in zip(axes, ["$u$-equation residual", "$\\mu$-equation residual"]):
        ax.axvline(adam_steps, lw=1, ls=":", color="0.4")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("mean squared residual (unweighted)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def plot_heldout_windows(runs, out, n_window=8000, seed=123):
    """Unweighted PDE residuals of the final models on fresh uniform points in each
    t-window of compare_residuals.py (log-spaced towards t = 0). The same points
    for every run, seeded as in compare_residuals.py."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    labels = None
    for k, (label, run) in enumerate(runs.items()):
        checkpoint = torch.load(MODELS / run / "final.pt", map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        model = load_model(str(MODELS / run / "final.pt"))
        t_max = args["t_max"]
        upper = (args["x_max"], args["y_max"], t_max)
        res = {"pde_u": [], "pde_mu": []}
        labels = []
        for j, (lo, hi) in enumerate(zip(T_EDGE_FRACTIONS[:-1], T_EDGE_FRACTIONS[1:])):
            pts = heldout_points((0.0, 0.0, lo * t_max), (upper[0], upper[1], hi * t_max),
                                 n_window, seed=seed + 1 + j)
            loss_u, loss_mu = pde_residual_losses(model, pts, epsilon=args["epsilon"])
            res["pde_u"].append(loss_u.item())
            res["pde_mu"].append(loss_mu.item())
            labels.append(f"[{lo * t_max:.0e}, {hi * t_max:.0e}]".replace("0e+00", "0"))
        print(f"{label}: pde_u {np.array(res['pde_u'])}, pde_mu {np.array(res['pde_mu'])}")
        x = np.arange(len(labels))
        for ax, name in zip(axes, ["pde_u", "pde_mu"]):
            ax.semilogy(x, res[name], LINE_STYLES[k], color=LINE_COLORS[k],
                        marker=LINE_MARKERS[k], markersize=5, label=label)
    for ax, title in zip(axes, ["$u$-equation residual", "$\\mu$-equation residual"]):
        ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("time window $[t_0, t_1]$")
        ax.set_ylabel("held-out mean squared residual")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def main():
    OUT.mkdir(exist_ok=True)
    plot_l2(RUNS, OUT / "lbfgs_points_l2.pdf")
    plot_heldout(RUNS, OUT / "lbfgs_points_heldout.pdf")
    plot_l2(LONG_RUNS, OUT / "lbfgs_long_l2.pdf")
    plot_heldout(LONG_RUNS, OUT / "lbfgs_long_heldout.pdf")
    plot_l2(LONG_RUNS, OUT / "lbfgs_long_early_l2.pdf", EARLY_FEM, early=True)
    plot_heldout_windows(LONG_RUNS, OUT / "lbfgs_long_heldout_windows.pdf")
    plot_l2(RESAMPLE_RUNS, OUT / "lbfgs_resample_l2.pdf", styles=RESAMPLE_STYLES)
    plot_l2(LONG_RESAMPLE_RUNS, OUT / "lbfgs_resample_long_l2.pdf")
    plot_l2(BUDGET_RUNS, OUT / "lbfgs_budget400_l2.pdf")


def resample_only():
    OUT.mkdir(exist_ok=True)
    plot_l2(RESAMPLE_RUNS, OUT / "lbfgs_resample_l2.pdf", styles=RESAMPLE_STYLES)
    plot_l2(LONG_RESAMPLE_RUNS, OUT / "lbfgs_resample_long_l2.pdf")
    plot_l2(BUDGET_RUNS, OUT / "lbfgs_budget400_l2.pdf")


if __name__ == "__main__":
    resample_only() if "--resample" in sys.argv else main()
