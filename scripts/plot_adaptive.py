"""Compare the runs with adaptive PDE-point sampling (--adaptive-every /
--lbfgs-adaptive-every) on the cross IC, all with 2000 Adam + 400 L-BFGS steps.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_adaptive.py
Writes to OUT:
  adaptive_l2.pdf: relative L2 error vs FEM over time,
  adaptive_loss.pdf: total training loss per history step.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.common.plotting import LINE_COLORS, LINE_MARKERS, LINE_STYLES  # noqa: E402  (also sets the colour-blind-safe cycle)
from plot_lbfgs_points import BASE, MODELS, plot_l2  # noqa: E402
from src.pinn.run_paths import run_dir_for  # noqa: E402

LB400 = BASE.replace("-lb120", "-lb400")
RUNS = {
    "Adam: 500 points every 500 epochs": f"{LB400}_ad500n500_ic-cross",
    "L-BFGS: 300 points every 100 steps": f"{LB400}_adlb100n300_ic-cross",
    "L-BFGS: 300 points every 100 steps, $u$-residual score": f"{LB400}_adlb100n300-u_ic-cross",
    "L-BFGS: 500 points every 50 steps, max 2000": f"{LB400}_adlb50n500m2000_ic-cross",
    "L-BFGS: 500 points every 25 steps, max 2000": f"{LB400}_adlb25n500m2000_ic-cross",
}
OUT = MODELS / "_adaptive"


def plot_loss(runs, out):
    """Total weighted training loss per history step (Adam epochs, then L-BFGS
    closure evaluations), with a line at the Adam -> L-BFGS switch."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    adam_steps = set()
    for k, (label, run) in enumerate(runs.items()):
        checkpoint = torch.load(Path(run_dir_for(run)) / "final.pt", map_location="cpu", weights_only=False)
        adam_steps.add(checkpoint["adam_steps"])
        total = np.array(checkpoint["history"]["total"])
        ax.semilogy(np.arange(len(total)), total, linestyle=LINE_STYLES[k], lw=1, color=LINE_COLORS[k],
                    marker=LINE_MARKERS[k], markersize=4, markevery=len(total) // 10, label=label)
    for step in adam_steps:
        ax.axvline(step, lw=1, ls=":", color="0.4")
    ax.set_xlabel("step (Adam epoch, then L-BFGS closure evaluation)")
    ax.set_ylabel("total loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


def main():
    OUT.mkdir(exist_ok=True)
    plot_l2(RUNS, OUT / "adaptive_l2.pdf", title="2000 Adam epochs + 400 L-BFGS steps")
    plot_loss(RUNS, OUT / "adaptive_loss.pdf")


if __name__ == "__main__":
    main()
