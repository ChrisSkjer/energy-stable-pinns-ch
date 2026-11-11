"""Tests for self-adaptive point weights (src/pinn/self_adaptive.py and its
wiring in src/pinn/train.py)."""

import glob
import os

import pytest
import torch
from torch import nn

from src.pinn.losses import bc_loss, bc_squared_normals, ic_loss, ic_squared_errors
from src.pinn.self_adaptive import SelfAdaptiveWeights, mask, unit_init


@pytest.mark.parametrize("kind", ["polynomial", "sigmoid"])
def test_masks_are_positive_increasing_and_start_at_one(kind):
    lam = torch.linspace(0.1, 5.0, 50)
    m = mask(lam, kind, q=2.0, scale=10.0)
    assert torch.all(m > 0)
    assert torch.all(torch.diff(m) > 0)
    torch.testing.assert_close(mask(torch.tensor(unit_init(kind, 2.0, 10.0)), kind, 2.0, 10.0), torch.tensor(1.0))


def test_sigmoid_mask_is_bounded_by_its_scale():
    assert mask(torch.tensor(50.0), "sigmoid", scale=7.0).item() == pytest.approx(7.0)


def test_term_at_unit_weights_equals_the_plain_mean_and_logs_it():
    sa = SelfAdaptiveWeights({"ic": 8}, kind="sigmoid", scale=4.0)
    squared = torch.rand(8, 1)
    out = sa.term("ic", squared, 3.0)
    torch.testing.assert_close(out, 3.0 * squared.mean())
    torch.testing.assert_close(sa.plain["ic"], 3.0 * squared.mean())
    # Terms that are not self-adaptive are plain means.
    torch.testing.assert_close(sa.term("bc", squared, 2.0), 2.0 * squared.mean())
    assert "bc" not in sa.plain


def test_term_rejects_a_changed_point_set():
    sa = SelfAdaptiveWeights({"pde_u": 8})
    with pytest.raises(RuntimeError):
        sa.term("pde_u", torch.rand(9, 1))


def test_ascent_raises_the_weight_of_large_residuals_more():
    sa = SelfAdaptiveWeights({"ic": 2})
    opt = torch.optim.SGD(sa.parameters(), lr=1.0, maximize=True)
    squared = torch.tensor([[1.0], [0.01]])
    for _ in range(5):
        opt.zero_grad()
        sa.term("ic", squared).backward()
        opt.step()
    w = sa.weights("ic").detach()
    assert w[0] > w[1] > 1.0


def test_pointwise_helpers_match_the_losses():
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2))
    ic_points = torch.rand(16, 3)
    ic_points[:, -1] = 0
    ic_values = torch.rand(16, 1)
    torch.testing.assert_close(
        ic_squared_errors(model, ic_points, ic_values).mean() / ic_values.pow(2).mean(),
        ic_loss(model, ic_points, ic_values),
    )
    bc_points = torch.rand(16, 3, requires_grad=True)  # 4 walls x 4 points
    torch.testing.assert_close(bc_squared_normals(model, bc_points).mean(), bc_loss(model, bc_points))


def test_parse_args_rejects_moving_points_of_a_self_adaptive_term(monkeypatch):
    from src.pinn.train import parse_args

    for bad in (
        ["--sa-terms", "pde_u", "--resample-every", "10"],
        ["--sa-terms", "pde_mu", "--lbfgs-n-collocation", "100"],
        ["--sa-terms", "pde_u", "--collocation", "sobol", "--adaptive-every", "10"],
        ["--sa-terms", "ic", "--hard-ic"],
        ["--sa-terms", "ic", "--ic-adaptive-every", "10"],
        ["--sa-terms", "ic", "--sa-mask", "sigmoid", "--sa-mask-scale", "1"],
    ):
        monkeypatch.setattr("sys.argv", ["train", *bad])
        with pytest.raises(SystemExit):
            parse_args()
    # IC-only SA leaves the PDE points free to move.
    monkeypatch.setattr("sys.argv", ["train", "--sa-terms", "ic", "--lbfgs-n-collocation", "100", "--lr", "2e-3"])
    args = parse_args()
    assert args.sa_terms == ["ic"] and args.sa_lr == 2e-3


@pytest.mark.parametrize("sa_mask", ["polynomial", "sigmoid"])
def test_smoke_run_trains_lambdas_in_adam_and_freezes_them_in_lbfgs(monkeypatch, tmp_path, sa_mask):
    from src.pinn import train as train_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train", "--smoke-test", "--device", "cpu", "--run-name", "sa", "--no-heldout",
            "--checkpoint-every", "1", "--sa-terms", "ic", "pde_u", "--sa-mask", sa_mask, "--sa-lr", "0.1",
        ],
    )
    train_module.train(train_module.parse_args())

    checkpoints = sorted(glob.glob(os.path.join("results", "pinn_models", "sa", "checkpoints", "*.pt")))
    first = torch.load(checkpoints[0], weights_only=False)
    final = torch.load(os.path.join("results", "pinn_models", "sa", "final.pt"), weights_only=False)
    adam_end = torch.load(checkpoints[-1], weights_only=False)  # saved during L-BFGS (frozen)
    for term in ("ic", "pde_u"):
        w0, w_end = first["sa"][term]["weights"], final["sa"][term]["weights"]
        assert w0.shape[0] == first["sa"][term]["points"].shape[0]
        assert torch.all(w_end > 1.0)  # grown from m = 1 by ascent
        torch.testing.assert_close(adam_end["sa"][term]["weights"], w_end)
    history = final["history"]
    assert {"sa_ic", "sa_pde_u", "ic", "pde_u", "total"} <= set(history)
    assert "sa_pde_mu" not in history
