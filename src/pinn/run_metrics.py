"""Scalar performance metrics for one trained PINN run, in one file.

Collects, from whatever a run folder already holds, the numbers worth
comparing across runs -- the plots show the curves, this condenses each into
a few scalars:

- physics (diagnostics.csv): does the PINN dissipate energy monotonically
  and conserve mass? (RQ1)
- field bounds (evaluation.npz): does u leave the physical range [-1, 1]?
- accuracy vs. FEM (with --fem-fields / --fem-diagnostics): relative L2 and
  max-norm errors of u and mu, error in the phase fraction, error in the
  free-energy curve (RQ3)
- generalization (checkpoint "heldout"): held-out over training loss per term
- training (checkpoint): final losses, parameter count, wall-clock time
- network health (checkpoint "layer_stats"): input/output gradient ratio,
  tanh saturation, slowest-moving layer

Each group is left out when its input file is missing, so this runs on any
run folder. Writes metrics.json (for cross-run tables) and metrics.txt.

    python -m src.pinn.run_metrics --run-dir results/pinn_models/<run>
    python -m src.pinn.run_metrics --run-dir results/pinn_models/<run> --fem-fields results/<fem>/fem_fields.npz
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch

from src.common.metrics import relative_l2_error
from src.common.plotting import load_diagnostics
from src.pinn.run_summary import adam_step_count, count_parameters

# An energy increase smaller than this fraction of |E(0)| is quadrature noise,
# not a violation of dissipation.
ENERGY_INCREASE_TOL = 1e-6


def physics_metrics(t: np.ndarray, energy: np.ndarray, mass: np.ndarray) -> dict:
    """Energy-dissipation and mass-conservation metrics from a diagnostics log.

    Returns:
        energy_increase_fraction: share of consecutive time intervals over
            which E increases (by more than ENERGY_INCREASE_TOL * |E(0)|).
            Exact Cahn-Hilliard dynamics give 0.
        energy_max_rise_rel: largest climb of E above its running minimum,
            relative to |E(0)| -- how far the worst violation goes back up.
        energy_decay_rel: (E(0) - E(t_end)) / |E(0)|.
        mass_drift_max_abs: max_t |m(t) - m(0)|.
        mass_drift_max_rel: the same over |m(0)|; omitted when m(0) ~ 0
            (symmetric ICs), where the ratio is meaningless.
    """
    e0 = abs(energy[0]) or 1.0
    d_e = np.diff(energy)
    running_min = np.minimum.accumulate(energy)
    out = {
        "energy_increase_fraction": float(np.mean(d_e > ENERGY_INCREASE_TOL * e0)) if d_e.size else 0.0,
        "energy_max_rise_rel": float(np.max(energy - running_min) / e0),
        "energy_decay_rel": float((energy[0] - energy[-1]) / e0),
        "mass_drift_max_abs": float(np.max(np.abs(mass - mass[0]))),
    }
    if abs(mass[0]) > 1e-8:
        out["mass_drift_max_rel"] = out["mass_drift_max_abs"] / abs(mass[0])
    return out


def bounds_metrics(u: np.ndarray) -> dict:
    """u_overshoot: max |u| - 1 over every snapshot (> 0 means u left the
    physical range [-1, 1]; small overshoots near interfaces are normal)."""
    return {"u_overshoot": float(np.max(np.abs(u)) - 1.0)}


def fem_field_metrics(pinn: dict, fem: dict) -> dict:
    """Pointwise PINN-vs-FEM errors at each PINN evaluation time, matched to
    the closest FEM snapshot (same grid required, as in plot_results).

    Returns per-time lists plus summaries:
        t, rel_l2_u, rel_l2_mu, linf_u, phase_fraction_error -- per time;
        rel_l2_u_mean/max, rel_l2_mu_mean/max, linf_u_max,
        phase_fraction_error_max -- summaries. phase_fraction_error is
        |area(u > 0)_PINN - area(u > 0)_FEM| as a fraction of the domain: it
        tracks whether the two agree on how much of each phase there is,
        independent of interface sharpness.
    mu errors are NaN at FEM snapshots where mu is identically zero (the
    FEM solver's mu(0) = 0), and left out of the summaries.
    """
    if pinn["x"].shape != fem["x"].shape or not np.allclose(pinn["x"], fem["x"]):
        raise ValueError("PINN evaluation and FEM fields are on different grids -- see analyze_run.py --fem-fields")
    rows = {"t": [], "rel_l2_u": [], "rel_l2_mu": [], "linf_u": [], "phase_fraction_error": []}
    for idx, t_val in enumerate(pinn["t"]):
        k = int(np.argmin(np.abs(fem["t"] - t_val)))
        if abs(fem["t"][k] - t_val) > 1e-9:
            warnings.warn(f"PINN t={t_val:g}: using closest FEM snapshot t={fem['t'][k]:g}")
        u_p, u_f = pinn["u"][idx], fem["u"][k]
        rows["t"].append(float(t_val))
        rows["rel_l2_u"].append(relative_l2_error(u_p, u_f))
        rows["rel_l2_mu"].append(
            relative_l2_error(pinn["mu"][idx], fem["mu"][k]) if np.any(fem["mu"][k]) else float("nan")
        )
        rows["linf_u"].append(float(np.max(np.abs(u_p - u_f))))
        rows["phase_fraction_error"].append(float(abs(np.mean(u_p > 0) - np.mean(u_f > 0))))
    out = dict(rows)
    for name in ("rel_l2_u", "rel_l2_mu"):
        values = np.asarray(rows[name])
        values = values[np.isfinite(values)]
        if values.size:
            out[f"{name}_mean"] = float(values.mean())
            out[f"{name}_max"] = float(values.max())
    out["linf_u_max"] = max(rows["linf_u"])
    out["phase_fraction_error_max"] = max(rows["phase_fraction_error"])
    return out


def fem_energy_metrics(t_p: np.ndarray, e_p: np.ndarray, t_f: np.ndarray, e_f: np.ndarray) -> dict:
    """Free-energy curve error against FEM over their common time window,
    with FEM linearly interpolated to the PINN's times.

    Returns:
        energy_error_max_rel: max_t |E_PINN - E_FEM| / |E_FEM(0)|.
        energy_error_end_rel: the same at the last common time.
    """
    keep = t_p <= t_f[-1] * (1 + 1e-6)
    e_ref = np.interp(t_p[keep], t_f, e_f)
    err = np.abs(e_p[keep] - e_ref) / (abs(e_f[0]) or 1.0)
    return {"energy_error_max_rel": float(err.max()), "energy_error_end_rel": float(err[-1])}


def checkpoint_metrics(checkpoint: dict) -> dict:
    """Training, generalization and network-health metrics stored in a
    train.py checkpoint."""
    args, history = checkpoint["args"], checkpoint["history"]
    out: dict = {"parameters": count_parameters(checkpoint["model_state"])}
    out["final_loss"] = {k: v[-1] for k, v in history.items() if v}
    out["steps"] = len(history.get("total", []))
    out["adam_steps"] = adam_step_count(args, history, checkpoint.get("adam_steps"))
    if "timing" in checkpoint:
        out["train_seconds"] = checkpoint["timing"]["adam_seconds"] + checkpoint["timing"]["lbfgs_seconds"]
        out.update(checkpoint["timing"])

    heldout = checkpoint.get("heldout")
    if heldout:
        weights = {
            "pde_u": args.get("pde_u_weight", 1.0),
            "pde_mu": args.get("pde_mu_weight", 1.0),
            "ic": args.get("ic_weight", 1.0),
            "bc": args.get("bc_weight", 1.0),
        }
        # history[k][step] is the loss at the weights *before* step, and
        # heldout["step"] the number of steps taken, so the matching training
        # value is one entry earlier -- clamped for the final record.
        step = min(heldout["step"][-1], len(history["total"])) - 1
        gap = {}
        for name, w in weights.items():
            if name in heldout and name in history and history[name][step] > 0:
                gap[name] = heldout[name][-1] / (history[name][step] / w)
        out["heldout_over_train"] = gap
        out["heldout_final"] = {k: v[-1] for k, v in heldout.items() if k != "step"}

    stats = checkpoint.get("layer_stats")
    if stats and stats["step"]:
        total = stats["grad_norm"]["total"][-1]
        out["grad_in_over_out"] = total[0] / total[-1] if total[-1] > 0 else float("nan")
        out["grad_min_over_max_layer"] = min(total) / max(total) if max(total) > 0 else float("nan")
        out["act_saturation_mean"] = float(np.mean(stats["act_saturation"][-1]))
        out["act_saturation_max"] = float(np.max(stats["act_saturation"][-1]))
        if stats["update_ratio"]:
            adam_rows = [r for r, p in zip(stats["update_ratio"], stats["phase"]) if p == "adam"]
            if adam_rows:
                out["adam_update_ratio_min_layer_final"] = float(min(adam_rows[-1]))
    return out


def compute_run_metrics(
    run_dir: str,
    checkpoint_path: str | None = None,
    fem_fields_path: str | None = None,
    fem_diagnostics_path: str | None = None,
) -> dict:
    """Every metric group whose inputs exist in run_dir (see module docstring)."""
    metrics: dict = {"run": os.path.basename(os.path.normpath(run_dir))}
    checkpoint_path = checkpoint_path or os.path.join(run_dir, "final.pt")
    if os.path.exists(checkpoint_path):
        metrics["training"] = checkpoint_metrics(torch.load(checkpoint_path, map_location="cpu"))

    diag_path = os.path.join(run_dir, "diagnostics.csv")
    t_p = e_p = None
    if os.path.exists(diag_path):
        t_p, e_p, m_p = load_diagnostics(diag_path)
        metrics["physics"] = physics_metrics(t_p, e_p, m_p)

    eval_path = os.path.join(run_dir, "evaluation.npz")
    pinn = None
    if os.path.exists(eval_path):
        with np.load(eval_path) as f:
            pinn = dict(f)
        metrics["bounds"] = bounds_metrics(pinn["u"])

    if fem_fields_path is not None and pinn is not None:
        with np.load(fem_fields_path) as f:
            fem = dict(f)
        metrics["vs_fem_fields"] = fem_field_metrics(pinn, fem)
        metrics["vs_fem_fields"]["reference"] = os.path.abspath(fem_fields_path)
    if fem_diagnostics_path is not None and t_p is not None:
        t_f, e_f, _ = load_diagnostics(fem_diagnostics_path)
        metrics["vs_fem_energy"] = fem_energy_metrics(t_p, e_p, t_f, e_f)
        metrics["vs_fem_energy"]["reference"] = os.path.abspath(fem_diagnostics_path)
    return metrics


def format_metrics(metrics: dict) -> str:
    """Readable text version of compute_run_metrics' output (per-time lists
    are left to metrics.json)."""
    lines = [f"metrics: {metrics['run']}", ""]
    for group, values in metrics.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"{group}:")
        for key, value in values.items():
            if isinstance(value, list) or key == "reference":
                continue
            if isinstance(value, dict):
                inner = ", ".join(f"{k} {v:.3e}" for k, v in value.items())
                lines.append(f"  {key}: {inner}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4e}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalar performance metrics for one PINN run")
    parser.add_argument("--run-dir", required=True, help="results/pinn_models/<run-name>")
    parser.add_argument("--checkpoint", default=None, help="Defaults to <run-dir>/final.pt")
    parser.add_argument("--fem-fields", default=None, help="FEM fem_fields.npz on the evaluation's grid")
    parser.add_argument(
        "--fem-diagnostics",
        default=None,
        help="FEM cahn_hilliard_diagnostics.csv. Defaults to the one next to --fem-fields, when present.",
    )
    args = parser.parse_args()
    if args.fem_diagnostics is None and args.fem_fields is not None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.fem_fields)), "cahn_hilliard_diagnostics.csv")
        if os.path.exists(candidate):
            args.fem_diagnostics = candidate
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    result = compute_run_metrics(cli_args.run_dir, cli_args.checkpoint, cli_args.fem_fields, cli_args.fem_diagnostics)
    json_path = os.path.join(cli_args.run_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, allow_nan=True)
    text = format_metrics(result)
    with open(os.path.join(cli_args.run_dir, "metrics.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nsaved {json_path}")
