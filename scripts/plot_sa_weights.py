"""Compare self-adaptive point weights (--sa-terms) with the plain run on the
cross IC, all Sobol 20000 points, w_mu = w_ic = 1e4, 2000 Adam + 400 L-BFGS.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_sa_weights.py
Writes sa_weights_l2.pdf (relative L2 error vs FEM over time) to OUT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lbfgs_points import BASE, MODELS, plot_l2  # noqa: E402
from src.common.plotting import LINE_COLORS, LINE_MARKERS, LINE_STYLES  # noqa: E402

LB400 = BASE.replace("-lb120", "-lb400")
RUNS = {
    "plain (no SA)": f"{LB400}_ic-cross",
    "SA on IC, $m = \\lambda^2$": f"{LB400}_saic-poly2_ic-cross",
    "SA on IC, $m = 10\\,\\sigma(\\lambda)$": f"{LB400}_saic-sig10_ic-cross",
    "SA on IC and $u$-PDE, $m = 10\\,\\sigma(\\lambda)$": f"{LB400}_saic-pdeu-sig10_ic-cross",
}
# The plain run as the black solid reference, the SA runs in the palette.
STYLES = [(LINE_COLORS[4], LINE_STYLES[0], LINE_MARKERS[4])] + [
    (LINE_COLORS[k], LINE_STYLES[k + 1], LINE_MARKERS[k]) for k in range(3)
]
OUT = MODELS / "_self_adaptive"


def main():
    OUT.mkdir(exist_ok=True)
    plot_l2(RUNS, OUT / "sa_weights_l2.pdf", styles=STYLES, title="Self-adaptive weights, 2000 Adam + 400 L-BFGS")


if __name__ == "__main__":
    main()
