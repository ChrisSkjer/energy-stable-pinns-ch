"""Compare adaptive sampling without and with time-binned reweighting
(--adaptive-reweight-bins) on the cross IC, both 2000 Adam + 400 L-BFGS steps.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_adaptive_reweight.py
Writes adaptive_reweight_l2.pdf (relative L2 error vs FEM over time) to OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lbfgs_points import BASE, MODELS, plot_l2  # noqa: E402

LB400 = BASE.replace("-lb120", "-lb400")
RUNS = {
    "not reweighted (every 25 steps)": f"{LB400}_adlb25n500m2000_ic-cross",
    "reweighted over 100 time bins (every 24 steps)": f"{LB400}_adlb24n500m2000rw100_ic-cross",
}
OUT = MODELS / "_adaptive"


def main():
    OUT.mkdir(exist_ok=True)
    plot_l2(RUNS, OUT / "adaptive_reweight_l2.pdf", title="500 points per round, max 2000, in L-BFGS")


if __name__ == "__main__":
    main()
