"""Effect of the dense PDE-point window on t in [0, 1e-4] (--dense-window 0 1e-4)
on the cross IC, against the runs without it.

Run from the repo root, optionally naming the groups to plot (default: all):
    .venv\\Scripts\\python.exe scripts/plot_dense_window.py [short] [long] [reweight] [reweight_long]
Writes to OUT, for SHORT_RUNS (2000 Adam + 120 L-BFGS steps), LONG_RUNS
(the 800-step window run against the 400-step 100000-point run) and
REWEIGHT_RUNS / REWEIGHT_LONG_RUNS (--dense-window-reweight with 120 / 400 L-BFGS
steps; L2 plots only, titled with the budget):
  dense_window_{short,long}_l2.pdf: relative L2 error vs FEM over the whole window,
  dense_window_{short,long}_early_l2.pdf: the same against the dt = 1e-6 FEM run
    on t <= 2e-4 (log t), where the window acts,
  dense_window_{short,long}_heldout.pdf: held-out vs training PDE residual per step,
  dense_window_{short,long}_heldout_windows.pdf: held-out PDE residual of the
    final models per t-window.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_lbfgs_points import (  # noqa: E402
    BASE, EARLY_FEM, MODELS, plot_heldout, plot_heldout_windows, plot_l2,
)

LB400 = BASE.replace("-lb120", "-lb400")
DW = "dw0-1e-4"
SHORT_RUNS = {
    "no window, Adam's 20000 points": f"{BASE}_ic-cross",
    "no window, 100000 new L-BFGS points": f"{BASE}_lbp100000_ic-cross",
    "window, 2000 points": f"{BASE}_{DW}n2000_ic-cross",
    "window, 20000 points": f"{BASE}_{DW}n20000_ic-cross",
    "window, 2000 points, IC weight $3\\cdot10^4$": f"{BASE.replace('icw10000', 'icw30000')}_{DW}n2000_ic-cross",
}
# The 800-step run also has 5000 Adam epochs and resamples every 50 L-BFGS steps,
# so the 400-step window run is there as the same-budget reference.
LONG_RUNS = {
    "no window, 100000 new points, 2000 Adam + 400 L-BFGS": f"{LB400}_lbp100000_ic-cross",
    "window, 2000 points, 2000 Adam + 400 L-BFGS": f"{LB400}_{DW}n2000_ic-cross",
    "window, 2000 points, 5000 Adam + 800 L-BFGS, resampled every 50":
        f"{BASE.replace('e2000-lb120', 'e5000-lb800')}_lbrs50_{DW}n2000_ic-cross",
}
# --dense-window-reweight against the unweighted window and no window, all on
# Adam's 20000 points with 2000 Adam + 120 L-BFGS steps.
REWEIGHT_RUNS = {
    "no window": SHORT_RUNS["no window, Adam's 20000 points"],
    "window $0<t<10^{-4}$": SHORT_RUNS["window, 2000 points"],
    "window $0<t<10^{-4}$, IC weight $3\\cdot10^4$": SHORT_RUNS["window, 2000 points, IC weight $3\\cdot10^4$"],
    "window $0<t<10^{-4}$, reweighted": f"{BASE}_{DW}n2000rw_ic-cross",
    "window $10^{-5}<t<10^{-4}$, reweighted": f"{BASE}_dw1e-5-1e-4n2000rw_ic-cross",
}
# The same on 2000 Adam + 400 L-BFGS steps, against 100000 fresh L-BFGS points.
REWEIGHT_LONG_RUNS = {
    "no window, 100000 new L-BFGS points": LONG_RUNS["no window, 100000 new points, 2000 Adam + 400 L-BFGS"],
    "window $0<t<10^{-4}$": LONG_RUNS["window, 2000 points, 2000 Adam + 400 L-BFGS"],
    "window $0<t<10^{-4}$, reweighted": f"{LB400}_{DW}n2000rw_ic-cross",
}
# Figure title per L2-only group, so the budget is stated on the plot.
TITLES = {
    "reweight": "2000 Adam epochs + 120 L-BFGS steps",
    "reweight_long": "2000 Adam epochs + 400 L-BFGS steps",
}
OUT = MODELS / "_dense_window"


def main():
    OUT.mkdir(exist_ok=True)
    groups = {"short": SHORT_RUNS, "long": LONG_RUNS, "reweight": REWEIGHT_RUNS,
              "reweight_long": REWEIGHT_LONG_RUNS}
    for name in sys.argv[1:] or groups:
        runs = groups[name]
        title = TITLES.get(name)
        plot_l2(runs, OUT / f"dense_window_{name}_l2.pdf", title=title)
        plot_l2(runs, OUT / f"dense_window_{name}_early_l2.pdf", EARLY_FEM, early=True, title=title)
        if name in TITLES:
            continue
        plot_heldout(runs, OUT / f"dense_window_{name}_heldout.pdf")
        plot_heldout_windows(runs, OUT / f"dense_window_{name}_heldout_windows.pdf")


if __name__ == "__main__":
    main()
