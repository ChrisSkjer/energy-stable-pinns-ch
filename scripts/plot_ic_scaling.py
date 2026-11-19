"""1D random cosine IC at a high and a low amplitude: same curve, different scale.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_ic_scaling.py
Writes Report/Research_Notes/figures/_ic_scaling/ic_scaling.pdf with three panels:
(a) high amplitude (max |u0| = 1), (b) low amplitude, each with its own y-axis,
(c) both on the same y-axis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.plotting import LINE_COLORS, LINE_STYLES  # noqa: E402

OUT = ROOT / "Report/Research_Notes/figures/_ic_scaling"
LOW_AMP = 0.2

modes = np.arange(1, 8 + 1)
rng = np.random.default_rng(42)
coeffs = rng.uniform(-1, 1, size=modes.size)
coeffs /= np.abs(coeffs).sum()
x = np.linspace(0, 1, 400)
base = np.cos(np.pi * modes * x[..., None]) @ coeffs

high_amp = 1.0 / np.abs(base).max()  # puts the field in [-1, 1]
high = high_amp * base
low = LOW_AMP * base

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
axes[0].plot(x, high, color=LINE_COLORS[0], ls=LINE_STYLES[0])
axes[0].set_title(f"(a) $A = {high_amp:.2f}$")
axes[1].plot(x, low, color=LINE_COLORS[1], ls=LINE_STYLES[1])
axes[1].set_title(f"(b) $A = {LOW_AMP}$")
axes[2].plot(x, high, color=LINE_COLORS[0], ls=LINE_STYLES[0], label=f"$A = {high_amp:.2f}$")
axes[2].plot(x, low, color=LINE_COLORS[1], ls=LINE_STYLES[1], label=f"$A = {LOW_AMP}$")
axes[2].set_ylim(-1.1, 1.1)
axes[2].set_title("(c) same axis")
axes[2].legend(loc="upper right", fontsize=9)
for ax in axes:
    ax.set_xlabel("$x$")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("$u_0$")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "ic_scaling.pdf")
print(f"high amplitude {high_amp:.3f}, wrote {OUT / 'ic_scaling.pdf'}")
