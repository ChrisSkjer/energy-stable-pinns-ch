"""Compare the collocation sampling strategies against FEM (relative L2 error vs time).

Uses the cross runs trained with held-out residual tracking (`*_heldout_ic-cross`).
Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_sampling_comparison.py
Writes sampling_l2_error.pdf to results/pinn_models/_sampling_heldout/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = Path("results/pinn_models")
OUT = MODELS / "_sampling_heldout"
FEM = Path("results/eps0.05_nx100_dte-4/fem_fields.npz")
PREFIX = "eps0.05_h100x5_t0_0.01_p20000_wmu3000_icw3000_nopdeat-t0_icnorm_e2000-"

# (run suffix, label, color, linestyle)
RUNS = [
    ("lb400_grid", "grid", "C0", "-"),
    ("lb400_random", "random", "C1", "-"),
    ("lb400_random-rs100", "random, resample 100", "C1", "--"),
    ("lb400_sobol", "Sobol", "C2", "-"),
    ("lb400_sobol-rs100", "Sobol, resample 100", "C2", "--"),
    ("lb100_grid", "grid, 100 L-BFGS", "C0", ":"),
]


def rel_l2(suffix, field):
    """Relative L2 error of `field` against FEM at the run's evaluation times."""
    pinn = np.load(MODELS / f"{PREFIX}{suffix}_heldout_ic-cross" / "evaluation.npz")
    fem = np.load(FEM)
    # FEM mu(0) = 0, so the mu error is only meaningful for t > 0
    first = 1 if field == "mu" else 0
    times = pinn["t"][first:]
    errors = []
    for k, t in enumerate(times, start=first):
        j = np.argmin(np.abs(fem["t"] - t))
        ref = fem[field][j]
        errors.append(np.linalg.norm(pinn[field][k] - ref) / np.linalg.norm(ref))
    return times, np.array(errors) * 100


def main():
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, field, title in zip(axes, ["u", "mu"], ["$u$", "$\\mu$"]):
        for suffix, label, color, style in RUNS:
            t, err = rel_l2(suffix, field)
            ax.plot(t, err, style, color=color, marker="o", label=label)
        ax.set_title(f"Cross: {title}")
        ax.set_xlabel("$t$")
        ax.set_ylabel("relative $L^2$ error vs FEM [%]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = OUT / "sampling_l2_error.pdf"
    fig.savefig(out)
    print(f"saved {out}")
    for suffix, label, *_ in RUNS:
        print(f"{label:>22}: u {np.round(rel_l2(suffix, 'u')[1], 2)}  mu {np.round(rel_l2(suffix, 'mu')[1], 2)}")


if __name__ == "__main__":
    main()
