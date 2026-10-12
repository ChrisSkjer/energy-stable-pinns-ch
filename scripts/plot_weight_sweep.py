"""Compare the loss-weight sweep runs against FEM (relative L2 error vs time).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_weight_sweep.py
Writes ic_weight_sweep.pdf and mu_weight_sweep.pdf to the sweep folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SWEEP = Path("results/pinn_models/_sweep_weights_icnorm")
FEM = {
    "cross": Path("results/eps0.05_nx100_dte-4/fem_fields.npz"),
    "two_circles": Path("results/eps0.05_nx100_dte-4_two_cricles/fem_fields.npz"),
}
TITLES = {"cross": "Cross", "two_circles": "Two circles"}


def run_dir(ic, icw, wmu):
    return SWEEP / f"eps0.05_h100x5_t0_0.01_p20000_wmu{wmu}_nopdeat-t0_icnorm_ic-{ic}_icw{icw}"


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
            for icw, wmu, style in configs:
                t, err = rel_l2(ic, icw, wmu, field)
                ax.plot(t, err, style, marker="o", label=f"IC {icw}, $\\mu$ {wmu}")
            ax.set_title(f"{TITLES[ic]}: ${'u' if field == 'u' else chr(92) + 'mu'}$")
            ax.set_ylabel("relative $L^2$ error vs FEM [%]")
            ax.grid(alpha=0.3)
            if row == len(fields) - 1:
                ax.set_xlabel("$t$")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_sweep([(100, 300, "-"), (1000, 300, "-"), (10000, 300, "-")],
               ["u"], SWEEP / "ic_weight_sweep.pdf")
    plot_sweep([(1000, 300, "-"), (1000, 1000, "-"), (1000, 3000, "-"), (3000, 3000, "--")],
               ["u", "mu"], SWEEP / "mu_weight_sweep.pdf")
