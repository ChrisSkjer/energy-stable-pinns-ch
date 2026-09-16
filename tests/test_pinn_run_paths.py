"""Tests for src.pinn.run_paths."""

import os

from src.pinn.run_paths import checkpoints_dir_for, final_path_for, infer_run_dir, run_dir_for


def test_run_dir_layout():
    run_dir = run_dir_for("my_run")
    assert run_dir == os.path.join("results", "pinn_models", "my_run")
    assert checkpoints_dir_for("my_run") == os.path.join(run_dir, "checkpoints")
    assert final_path_for("my_run") == os.path.join(run_dir, "final.pt")


def test_infer_run_dir_from_final_checkpoint():
    run_dir = run_dir_for("my_run")
    inferred = infer_run_dir(final_path_for("my_run"))
    assert inferred == os.path.abspath(run_dir)


def test_infer_run_dir_from_periodic_checkpoint():
    run_dir = run_dir_for("my_run")
    checkpoint = os.path.join(checkpoints_dir_for("my_run"), "checkpoint_step000500.pt")
    assert infer_run_dir(checkpoint) == os.path.abspath(run_dir)
