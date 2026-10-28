"""Compare the loss-weight sweep runs against FEM (relative L2 error vs time).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_weight_sweep.py
Writes ic_weight_sweep.pdf and mu_weight_sweep.pdf to the sweep folder, and
mu_weight_sweep_equal_ic.pdf (IC weight = mu weight, up to 1e5) to MU_CHECK.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.plotting import LINE_COLORS, LINE_MARKERS  # noqa: E402  (also sets the colour-blind-safe cycle)

SWEEP = Path("results/pinn_models/_sweep_weights_icnorm")
MU_CHECK = Path("results/pinn_models/_sweep_weights_mu_check")
FEM = {
    "cross": Path("results/eps0.05_nx100_dte-4/fem_fields.npz"),
    "two_circles": Path("results/eps0.05_nx100_dte-4_two_cricles/fem_fields.npz"),
}
TITLES = {"cross": "Cross", "two_circles": "Two circles"}


def run_dir(ic, icw, wmu):
    """The run folder: the first sweep's naming, or the newer one in MU_CHECK."""
    old = SWEEP / f"eps0.05_h100x5_t0_0.01_p20000_wmu{wmu}_nopdeat-t0_icnorm_ic-{ic}_icw{icw}"
    if old.exists():
        return old
    return MU_CHECK / f"eps0.05_h100x5_t0_0.01_p20000_wmu{wmu}_icw{icw}_nopdeat-t0_icnorm_e1000-lb100_random_ic-{ic}"


def rel_l2(ic, icw, wmu, field):
    """Relative L2 error of `field` against FEM at the run's evaluation times."""
    pinn = np.load(run_dir(ic, icw, wmu) / "evaluation.npz")
    fem = np.load(FEM[ic])
    # FEM mu(0) = 0, so the mu error is only meaningful for t > 0
    first = 1 if field == "mu" else 0
    times = pinn["t"][first:]
    errors = []
    for k, t in enumerate(times, start=first):
        j = np.argmin(np.abs(fem["t"] - t))
        ref = fem[field][j]
        errors.append(np.linalg.norm(pinn[field][k] - ref) / np.linalg.norm(ref))
    return times, np.array(errors) * 100


def plot_sweep(configs, fields, out):
    fig, axes = plt.subplots(len(fields), 2, figsize=(9, 3.2 * len(fields)),
                             squeeze=False, sharex=True)
    for row, field in enumerate(fields):
        for col, ic in enumerate(FEM):
            ax = axes[row, col]
            for k, (icw, wmu, style) in enumerate(configs):
                t, err = rel_l2(ic, icw, wmu, field)
                ax.plot(t, err, style, color=LINE_COLORS[k], marker=LINE_MARKERS[k], label=f"IC {icw}, $\\mu$ {wmu}")
            ax.set_title(f"{TITLES[ic]}: ${'u' if field == 'u' else chr(92) + 'mu'}$")
            ax.set_ylabel("relative $L^2$ error vs FEM [%]")
            ax.grid(alpha=0.3)
            if row == len(fields) - 1:
                ax.set_xlabel("$t$")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def plot_total_loss(configs, out):
    """Weighted total training loss per run, one panel per IC."""
    import torch

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, ic in zip(axes, FEM):
        for k, (icw, wmu, style) in enumerate(configs):
            history = torch.load(run_dir(ic, icw, wmu) / "final.pt", map_location="cpu", weights_only=False)["history"]
            ax.semilogy(history["total"], style, lw=1.2, color=LINE_COLORS[k], label=f"IC {icw}, $\\mu$ {wmu}")
        # every run here trained 1000 Adam epochs before switching to L-BFGS
        ax.axvline(1000, lw=1, ls=":", color="0.4")
        ax.set_title(TITLES[ic])
        ax.set_xlabel("step (Adam epochs, then L-BFGS loss evaluations)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("total weighted loss")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_sweep([(100, 300, "-"), (1000, 300, "-"), (10000, 300, "-")],
               ["u"], SWEEP / "ic_weight_sweep.pdf")
    plot_sweep([(1000, 300, "-"), (1000, 1000, "-"), (1000, 3000, "-"), (3000, 3000, "--")],
               ["u", "mu"], SWEEP / "mu_weight_sweep.pdf")
    plot_sweep([(1000, 1000, "-"), (3000, 3000, "-"), (10000, 10000, "--"),
                (30000, 30000, "--"), (100000, 100000, "--")],
               ["u", "mu"], MU_CHECK / "mu_weight_sweep_equal_ic.pdf")
    plot_total_loss([(1000, 1000, "-"), (3000, 3000, "-"), (10000, 10000, "--"),
                     (30000, 30000, "--"), (100000, 100000, "--")],
                    MU_CHECK / "mu_weight_sweep_equal_ic_total_loss.pdf")
