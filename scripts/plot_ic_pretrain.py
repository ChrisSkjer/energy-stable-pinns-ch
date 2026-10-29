"""Compare the IC-pretraining runs against FEM (relative L2 error vs time).

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_ic_pretrain.py
Writes ic_pretrain_l2.pdf to OUT. Each model is evaluated at every FEM snapshot
in its training window, not just the 4 times in its evaluation.npz.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.metrics import relative_l2_error  # noqa: E402
from src.common.plotting import LINE_COLORS, LINE_MARKERS, LINE_STYLES  # noqa: E402  (also sets the colour-blind-safe cycle)
from src.pinn.evaluate import evaluate_over_time, load_model  # noqa: E402

MODELS = Path("results/pinn_models")
BASE = "eps0.05_h100x5_t0_0.01_p20000_wmu10000_icw10000_nopdeat-t0_icnorm_e2000-lb120_sobol_heldout"
RUNS = {
    "no pretraining": f"{BASE}_ic-cross",
    "500 epochs IC": f"{BASE}_pre500_ic-cross",
    "500 epochs IC + BC": f"{BASE}_pre500bc_ic-cross",
}
FEM = Path("results/eps0.05_nx100_dte-4/fem_fields.npz")
OUT = MODELS / "_ic_pretrain"


def rel_l2_series(run, fem):
    """Relative L2 error [%] of u and mu at every FEM time in the run's window."""
    model = load_model(str(MODELS / run / "final.pt"))
    t_max = model.upper_bound[-1].item()
    idx = np.nonzero(fem["t"] <= t_max * (1 + 1e-6))[0]
    nx, ny = fem["x"].shape
    _, _, u, mu = evaluate_over_time(model, fem["t"][idx], nx, ny)
    errors = {
        "u": np.array([relative_l2_error(u[k], fem["u"][j]) for k, j in enumerate(idx)]),
        # FEM mu(0) = 0, so the mu error is only meaningful for t > 0
        "mu": np.array([relative_l2_error(mu[k], fem["mu"][j]) if fem["t"][j] > 0 else np.nan
                        for k, j in enumerate(idx)]),
    }
    return fem["t"][idx], {name: e * 100 for name, e in errors.items()}


def plot_loss_history(out):
    """Weighted IC and total loss per run, with the pretraining epochs at negative steps."""
    import torch

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    pretrain_len = 0
    for k, (label, run) in enumerate(RUNS.items()):
        checkpoint = torch.load(MODELS / run / "final.pt", map_location="cpu", weights_only=False)
        history, pretrain = checkpoint["history"], checkpoint.get("ic_pretrain", {})
        n_pre = len(pretrain.get("total", []))
        pretrain_len = max(pretrain_len, n_pre)
        for ax, name in zip(axes, ["ic", "total"]):
            # the pretraining "total" is only IC (+ BC), so it joins the main curve with a jump
            values = pretrain.get(name, []) + history[name]
            ax.semilogy(np.arange(len(values)) - n_pre, values, LINE_STYLES[k], lw=1,
                        color=LINE_COLORS[k], label=label)
        adam_steps = checkpoint["adam_steps"]
    for ax, title in zip(axes, ["weighted IC loss", "total weighted loss"]):
        ax.axvspan(-pretrain_len, 0, color="0.9", lw=0)
        ax.axvline(adam_steps, lw=1, ls=":", color="0.4")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"saved {out}")


def main():
    plot_loss_history(OUT / "ic_pretrain_loss.pdf")
    fem = dict(np.load(FEM))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for k, (label, run) in enumerate(RUNS.items()):
        t, errors = rel_l2_series(run, fem)
        for ax, name in zip(axes, ["u", "mu"]):
            ax.plot(t, errors[name], LINE_STYLES[k], color=LINE_COLORS[k],
                    marker=LINE_MARKERS[k], markersize=4, label=label)
        print(f"{label}: u mean {np.nanmean(errors['u']):.3f}%, mu mean {np.nanmean(errors['mu']):.3f}%")
    for ax, name in zip(axes, ["u", "\\mu"]):
        ax.set_title(f"${name}$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("relative $L^2$ error vs FEM [%]")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    out = OUT / "ic_pretrain_l2.pdf"
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
