"""Shared per-run output folder convention for src/pinn/*.

train.py creates, per run:

    results/pinn_models/<run_name>/
        checkpoints/checkpoint_step<N>.pt   (periodic, one per --checkpoint-every)
        final.pt                            (weights once the full run finishes)

evaluate.py, diagnostics.py, and plot_results.py default their own outputs
(evaluation.npz, diagnostics.csv, plots/run_summary.txt, plots/*.png) into
that same folder by walking back up from whichever checkpoint file they were
pointed at, via infer_run_dir -- so a checkpoint, its evaluation, its
diagnostics, and its plots end up findable in one place without retyping the
run name in four separate flags.
"""

from __future__ import annotations

import os

RUNS_ROOT = os.path.join("results", "pinn_models")


def run_dir_for(run_name: str) -> str:
    return os.path.join(RUNS_ROOT, run_name)


def checkpoints_dir_for(run_name: str) -> str:
    return os.path.join(run_dir_for(run_name), "checkpoints")


def final_path_for(run_name: str) -> str:
    return os.path.join(run_dir_for(run_name), "final.pt")


def infer_run_dir(checkpoint_path: str) -> str:
    """Recover a run's folder from one of its checkpoint files.

    Checkpoints saved by train.py live at either <run_dir>/final.pt or
    <run_dir>/checkpoints/checkpoint_step<N>.pt, so the run folder is the
    checkpoint's parent directory, or its grandparent when the parent is
    itself named "checkpoints". Falls back to the checkpoint's own parent
    directory for any checkpoint that doesn't follow this layout (e.g. one
    saved before this convention existed, or passed in from elsewhere) --
    still a reasonable place to write outputs alongside it.
    """
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(parent) == "checkpoints":
        return os.path.dirname(parent)
    return parent
