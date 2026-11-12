"""Why self-adaptive (SA) point weights did not help: where the learned weights
end up, and the u error with and without SA on the cross and noise ICs.

SA runs: sigmoid mask m = 100 sigma(lambda), Adam ascent lr 1e-2, frozen in L-BFGS.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_sa_failure.py
Writes sa_failure.pdf to OUT:
  (a) learned weight m(lambda) per PDE point against t (median and 10-90% band
      in log-spaced t bins), with the cap 100 marked,
  (b) cross IC: relative L2 error of u vs FEM, plain / adaptive points, with and without SA,
  (c) noise IC with hard IC: the same, with and without SA.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common.plotting import LINE_COLORS, LINE_MARKERS  # noqa: E402  (also sets the colour-blind-safe cycle)
from plot_ic_pretrain import rel_l2_series  # noqa: E402
from src.pinn.run_paths import run_dir_for  # noqa: E402

MODELS = Path("results/pinn_models")
OUT = MODELS / "_self_adaptive"
SA = "sapdeu-sig100-lr1e-2"
SA_CAP = 100.0

CROSS = "eps0.05_h100x5_t0_0.01_p20000_wmu10000_icw10000_nopdeat-t0_icnorm_e2000-lb400_sobol_heldout"
ADAPT = "adlb24n500m2000rw100"
CROSS_RUNS = {  # label: (run, colour, line style, marker)
    "plain": (f"{CROSS}_ic-cross", LINE_COLORS[4], "-", LINE_MARKERS[4]),
    "plain + SA on $u$-PDE": (f"{CROSS}_{SA}_ic-cross", LINE_COLORS[4], "--", "x"),
    "plain, 120 L-BFGS steps": (
        f"{CROSS.replace('-lb400', '-lb120')}_wseed0_ic-cross", LINE_COLORS[3], ":", LINE_MARKERS[3]),
    "adaptive points": (f"{CROSS}_{ADAPT}_ic-cross", LINE_COLORS[0], "-", LINE_MARKERS[0]),
    "adaptive points + SA on $u$-PDE": (f"{CROSS}_{ADAPT}_{SA}_ic-cross", LINE_COLORS[0], "--", "D"),
}
NOISE = "eps0.05_h100x5_t0_0.01_p20000_wmu3000_icw3000_nopdeat-t0_icnorm_e2000-lb400_random_heldout_hardic"
NOISE_RUNS = {
    "hard IC": (f"{NOISE}_amp3.5_ic-noise", LINE_COLORS[1], "-", LINE_MARKERS[1]),
    "hard IC + SA on $u$- and $\\mu$-PDE": (f"{NOISE}_sapdeupdemu-sig100-lr1e-2_amp3.5_ic-noise",
                                           LINE_COLORS[1], "--", "x"),
}
# (a): which SA weights to show. Not the adaptive + SA run: its 2000 added points
# arrive during L-BFGS, when the weights are frozen, so they all keep one value.
WEIGHT_CURVES = [  # label, run, term, colour, line style
    ("cross, plain + SA: $u$-PDE", CROSS_RUNS["plain + SA on $u$-PDE"][0], "pde_u", LINE_COLORS[4], "-"),
    ("noise, hard IC + SA: $u$-PDE", NOISE_RUNS["hard IC + SA on $u$- and $\\mu$-PDE"][0], "pde_u", LINE_COLORS[1], "-."),
    ("noise, hard IC + SA: $\\mu$-PDE", NOISE_RUNS["hard IC + SA on $u$- and $\\mu$-PDE"][0], "pde_mu", LINE_COLORS[3], ":"),
]
CROSS_FEM = Path("results/eps0.05_nx100_dte-4/fem_fields.npz")
NOISE_FEM = Path("results/eps0.05_nx100_dt1e-5_t0.01_noise_amp3.5/fem_fields.npz")


def plot_weights(ax):
    edges = np.logspace(-5, -2, 10)
    centres = np.sqrt(edges[:-1] * edges[1:])
    edges[0] = 0.0  # the first bin also takes the few points with t < 1e-5
    for label, run, term, color, style in WEIGHT_CURVES:
        sa = torch.load(Path(run_dir_for(run)) / "final.pt", map_location="cpu", weights_only=False)["sa"][term]
        w = sa["weights"].numpy().ravel()
        t = sa["points"].numpy()[:, -1]
        stats = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = w[(t >= lo) & (t < hi)]
            stats.append(np.percentile(sel, [10, 50, 90]) if sel.size else [np.nan] * 3)
        stats = np.array(stats)
        ax.plot(centres, stats[:, 1], linestyle=style, color=color, lw=1.6, label=label)
        ax.fill_between(centres, stats[:, 0], stats[:, 2], color=color, alpha=0.15, lw=0)
        print(f"{label}: median {np.median(w):.1f}, 10-90% [{np.percentile(w, 10):.1f}, {np.percentile(w, 90):.1f}]")
    ax.axhline(SA_CAP, color="0.3", lw=1, ls=":")
    ax.text(1.2e-5, SA_CAP * 0.8, "cap $m = 100$", fontsize=8, va="top", color="0.3")
    ax.axhline(1.0, color="0.3", lw=1, ls=":")
    ax.text(1.2e-5, 1.15, "initial $m = 1$", fontsize=8, va="bottom", color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$ of the PDE point")
    ax.set_ylabel("learned weight $m(\\lambda_i)$")
    ax.set_title("(a) learned SA weights")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)


def plot_errors(ax, runs, fem_path, title):
    fem = dict(np.load(fem_path))
    for label, (run, color, style, marker) in runs.items():
        t, errors = rel_l2_series(run, fem)
        ax.plot(t, errors["u"], linestyle=style, color=color, marker=marker, markersize=4, label=label)
        print(f"{title} | {label}: u mean {np.nanmean(errors['u']):.2f}%, mu mean {np.nanmean(errors['mu']):.2f}%")
    ax.set_xlabel("$t$")
    ax.set_ylabel("relative $L^2$ error of $u$ vs FEM [%]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    plot_weights(axes[0])
    plot_errors(axes[1], CROSS_RUNS, CROSS_FEM, "(b) cross IC")
    plot_errors(axes[2], NOISE_RUNS, NOISE_FEM, "(c) noise IC, hard IC")
    fig.tight_layout()
    out = OUT / "sa_failure.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
