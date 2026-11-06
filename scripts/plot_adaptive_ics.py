"""Relative L2 error of the reweighted adaptive-sampling runs, one figure per IC
(each against its own FEM reference), all 2000 Adam + 400 L-BFGS steps.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_adaptive_ics.py
Writes adaptive_l2_cross.pdf and adaptive_l2_two_circles.pdf to OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lbfgs_points import MODELS, plot_l2  # noqa: E402

PREFIX = "eps0.05_h100x5_t0_0.01_p{p}_wmu10000_icw10000_nopdeat-t0_icnorm_e2000-lb400_sobol_heldout"
# label -> (n_colloc, adaptive token)
SCHEDULES = {
    "L-BFGS: 300 points every 100 steps": (20000, "adlb100n300rw100"),
    "L-BFGS: 500 points every 24 steps, max 2000": (20000, "adlb24n500m2000rw100"),
    "Adam: 250 every 500 epochs + L-BFGS: 250 every 80 steps, max 2000": (20000, "ad500lb80n250m2000rw100"),
    "15000 start; Adam: 500 every 1000 epochs + L-BFGS: 500 every 40 steps, max 5000": (15000, "ad1000lb40n500m5000rw100"),
}
FEM = {
    "cross": Path("results/eps0.05_nx100_dte-4/fem_fields.npz"),
    "two_circles": Path("results/eps0.05_nx100_dte-4_two_cricles/fem_fields.npz"),
}
OUT = MODELS / "_adaptive"


def main():
    OUT.mkdir(exist_ok=True)
    for ic, fem in FEM.items():
        runs = {}
        for label, (p, token) in SCHEDULES.items():
            run = f"{PREFIX.format(p=p)}_{token}_ic-{ic}"
            if (MODELS / run / "evaluation.npz").exists():  # no adlb100n300rw100 run for two_circles
                runs[label] = run
        plot_l2(runs, OUT / f"adaptive_l2_{ic}.pdf", fem_path=fem, title=f"IC {ic.replace('_', ' ')}")


if __name__ == "__main__":
    main()
