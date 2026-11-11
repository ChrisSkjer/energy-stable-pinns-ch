"""Test --save-at-lbfgs-steps in src/pinn/train.py."""

import os

import pytest
import torch


def test_save_at_lbfgs_steps_writes_the_checkpoint_after_exactly_n_steps(monkeypatch, tmp_path):
    from src.pinn import train as train_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--smoke-test", "--device", "cpu", "--run-name", "at", "--heldout-every", "5",
         "--heldout-n", "50", "--heldout-n-ic", "20", "--heldout-n-bc", "5", "--save-at-lbfgs-steps", "1"],
    )
    train_module.train(train_module.parse_args())
    ckpt = torch.load(os.path.join("results", "pinn_models", "at", "checkpoints", "lbfgs_step0001.pt"), weights_only=False)
    final = torch.load(os.path.join("results", "pinn_models", "at", "final.pt"), weights_only=False)
    assert ckpt["adam_steps"] == final["adam_steps"]
    assert ckpt["adam_steps"] < len(ckpt["history"]["total"]) < len(final["history"]["total"])
    assert ckpt["heldout"]["step"][-1] == len(ckpt["history"]["total"])


def test_save_at_lbfgs_steps_must_fit_the_budget(monkeypatch):
    from src.pinn.train import parse_args

    monkeypatch.setattr("sys.argv", ["train", "--lbfgs-steps", "100", "--save-at-lbfgs-steps", "400"])
    with pytest.raises(SystemExit):
        parse_args()
