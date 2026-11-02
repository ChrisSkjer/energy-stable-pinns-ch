"""Effect of the dense PDE-point window on t in [0, 1e-4] (--dense-window 0 1e-4)
on the cross IC, against the runs without it.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_dense_window.py
Writes to OUT, for SHORT_RUNS (2000 Adam + 120 L-BFGS steps) and LONG_RUNS
(the 800-step window run against the 400-step 100000-point run):
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
OUT = MODELS / "_dense_window"


def main():
    OUT.mkdir(exist_ok=True)
    for name, runs in [("short", SHORT_RUNS), ("long", LONG_RUNS)]:
        plot_l2(runs, OUT / f"dense_window_{name}_l2.pdf")
        plot_l2(runs, OUT / f"dense_window_{name}_early_l2.pdf", EARLY_FEM, early=True)
        plot_heldout(runs, OUT / f"dense_window_{name}_heldout.pdf")
        plot_heldout_windows(runs, OUT / f"dense_window_{name}_heldout_windows.pdf")


if __name__ == "__main__":
    main()
