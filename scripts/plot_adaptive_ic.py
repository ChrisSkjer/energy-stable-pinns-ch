"""Adaptive IC points (--ic-adaptive-every / --lbfgs-ic-adaptive-every, with and
without --ic-reweight-bins) on top of the best adaptive PDE run
(adlb24n500m2000rw100), all 2000 Adam + 400 L-BFGS steps.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_adaptive_ic.py
Writes adaptive_ic_l2_cross.pdf and adaptive_ic_l2_noise.pdf (relative L2
error vs FEM over time) to OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lbfgs_points import MODELS, plot_l2  # noqa: E402

PREFIX = "_adaptive/eps0.05_h100x5_t0_0.01_p20000_wmu{w}_icw{w}_nopdeat-t0_icnorm_e2000-lb400_{s}_heldout_adlb24n500m2000rw100"
CROSS = PREFIX.format(w=10000, s="sobol")
NOISE = PREFIX.format(w=3000, s="random")
CROSS_RUNS = {
    "no adaptive IC points": f"{CROSS}_ic-cross",
    "500 IC points at the end of Adam, not reweighted": f"{CROSS}_icad2000n500_ic-cross",
    "500 IC points every 24 L-BFGS steps, max 2000, reweighted (50 x 50 cells)": f"{CROSS}_icadlb24n500m2000rw50_ic-cross",
}
NOISE_RUNS = {
    "no adaptive IC points": f"{NOISE}_amp3.5_ic-noise",
    "500 IC points every 24 L-BFGS steps, max 2000, reweighted (50 x 50 cells)": f"{NOISE}_icadlb24n500m2000rw50_amp3.5_ic-noise",
}
OUT = MODELS / "_adaptive"


def main():
    plot_l2(CROSS_RUNS, OUT / "adaptive_ic_l2_cross.pdf", fem_path=Path("results/eps0.05_nx100_dte-4/fem_fields.npz"), title="IC cross")
    plot_l2(NOISE_RUNS, OUT / "adaptive_ic_l2_noise.pdf", fem_path=Path("results/eps0.05_nx100_dt1e-5_t0.01_noise_amp3.5/fem_fields.npz"), title="IC noise (amplitude 3.5)")


if __name__ == "__main__":
    main()
