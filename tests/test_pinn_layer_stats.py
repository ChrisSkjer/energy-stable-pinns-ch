"""Tests for src.pinn.layer_stats, its plots, and src.pinn.run_metrics."""

import numpy as np
import pytest
import torch

from src.common import plotting
from src.pinn import layer_stats as ls
from src.pinn.model import PINN
from src.pinn.run_metrics import fem_energy_metrics, fem_field_metrics, physics_metrics


def _model(hidden_layers=3, width=8):
    torch.manual_seed(0)
    return PINN((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), input_dim=3, output_dim=2,
                hidden_layers=hidden_layers, hidden_width=width)


def _losses(model):
    x = torch.rand(32, 3)
    out = model(x)
    losses = {"a": (out[:, 0] ** 2).mean(), "b": 3.0 * (out[:, 1] - 1).pow(2).mean()}
    losses["total"] = losses["a"] + losses["b"]
    return losses, x


def test_layer_names_cover_every_linear_layer():
    model = _model(hidden_layers=3)
    assert ls.layer_names(model) == ["in", "h1", "h2", "out"]
    assert len(ls.linear_layers(model)) == 4


def test_gradient_norms_match_a_plain_backward_pass():
    model = _model()
    losses, _ = _losses(model)
    norms = ls.per_term_gradient_norms(model, losses)
    # The graph is retained, so the training backward still works afterwards.
    losses["b"].backward()
    expected = [layer.weight.grad.norm().item() for layer in ls.linear_layers(model)]
    assert norms["b"] == pytest.approx(expected, rel=1e-6)
    assert set(norms) == {"a", "b", "total"}


def test_per_term_gradients_skip_total_and_flatten():
    model = _model()
    losses, _ = _losses(model)
    grads = ls.per_term_gradients(model, losses)
    assert set(grads) == {"a", "b"}
    assert [g.numel() for g in grads["a"]] == [l.weight.numel() for l in ls.linear_layers(model)]


def test_update_ratios():
    model = _model()
    before = ls.weight_snapshot(model)
    assert ls.update_ratios(model, before) == pytest.approx([0.0] * 4)
    with torch.no_grad():
        for layer in ls.linear_layers(model):
            layer.weight.mul_(1.1)
    assert ls.update_ratios(model, before) == pytest.approx([0.1] * 4, rel=1e-5)


def test_activation_stats_detect_saturation():
    model = _model(hidden_layers=2)
    x = torch.rand(256, 3)
    assert max(ls.activation_stats(model, x)["saturation"]) < 0.5
    with torch.no_grad():
        for layer in ls.linear_layers(model):
            layer.weight.mul_(100.0)
    stats = ls.activation_stats(model, x)
    assert len(stats["saturation"]) == 2
    assert min(stats["saturation"]) > 0.8


def test_log_records_aligned_lists():
    model = _model()
    log = ls.LayerStatsLog(model)
    for step in (0, 5):
        losses, x = _losses(model)
        log.record(step, "adam", model, losses, x)
        log.add_update_ratio([1e-3] * 4)
    d = log.as_dict()
    assert d["step"] == [0, 5]
    assert np.asarray(d["grad_norm"]["total"]).shape == (2, 4)
    assert np.asarray(d["act_saturation"]).shape == (2, 3)
    assert np.asarray(d["update_ratio"]).shape == (2, 4)


def test_layer_stat_plots_run():
    model = _model()
    log = ls.LayerStatsLog(model)
    for step in range(4):
        losses, x = _losses(model)
        log.record(step * 10, "adam" if step < 3 else "lbfgs", model, losses, x)
        log.add_update_ratio([1e-3] * 4)
    d = log.as_dict()
    plotting.plot_layer_grad_norms(d, adam_steps=25)
    plotting.plot_grad_depth_profile(d)
    plotting.plot_update_ratio(d, adam_steps=25)
    plotting.plot_activation_stats(d)
    plotting.plot_gradient_balance(d)
    losses, _ = _losses(model)
    grads = {k: [g.numpy() for g in v] for k, v in ls.per_term_gradients(model, losses).items()}
    plotting.plot_gradient_histograms(grads, ls.layer_names(model))


def test_physics_metrics_monotone_energy():
    t = np.linspace(0, 1, 11)
    energy = 2.0 - t
    mass = np.full_like(t, 0.5)
    m = physics_metrics(t, energy, mass)
    assert m["energy_increase_fraction"] == 0.0
    assert m["energy_max_rise_rel"] == 0.0
    assert m["energy_decay_rel"] == pytest.approx(0.5)
    assert m["mass_drift_max_abs"] == 0.0


def test_physics_metrics_measure_a_bump():
    t = np.arange(5.0)
    energy = np.array([1.0, 0.8, 0.9, 0.7, 0.6])  # one rise of 0.1
    mass = np.array([0.0, 0.0, 0.02, 0.0, -0.01])
    m = physics_metrics(t, energy, mass)
    assert m["energy_increase_fraction"] == pytest.approx(0.25)
    assert m["energy_max_rise_rel"] == pytest.approx(0.1)
    assert m["mass_drift_max_abs"] == pytest.approx(0.02)
    assert "mass_drift_max_rel" not in m  # m(0) = 0


def test_fem_field_metrics():
    X, Y = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9), indexing="ij")
    u = np.stack([np.tanh(5 * (X - 0.5)), np.tanh(5 * (X - 0.4))])
    mu = np.stack([np.zeros_like(X), np.cos(X)])
    fem = {"x": X, "y": Y, "t": np.array([0.0, 1.0]), "u": u, "mu": mu}
    pinn = {"x": X, "y": Y, "t": np.array([0.0, 1.0]), "u": u + 0.01, "mu": mu}
    m = fem_field_metrics(pinn, fem)
    assert m["linf_u_max"] == pytest.approx(0.01)
    assert np.isnan(m["rel_l2_mu"][0])  # FEM mu(0) = 0
    assert m["rel_l2_mu_max"] == 0.0


def test_fem_field_metrics_1d_and_spacetime():
    # 1D layout (x vector, u (nt, nx)); PINN at every other FEM snapshot.
    x = np.linspace(0, 1, 11)
    t_f = np.array([0.0, 0.5, 1.0])
    u_f = np.stack([np.cos(np.pi * x) * (1 + k) for k in range(3)])
    mu_f = np.stack([np.zeros_like(x), np.sin(np.pi * x), 2 * np.sin(np.pi * x)])
    fem = {"x": x, "t": t_f, "u": u_f, "mu": mu_f}
    pinn = {"x": x, "t": t_f[[0, 2]], "u": 1.1 * u_f[[0, 2]], "mu": mu_f[[0, 2]]}
    m = fem_field_metrics(pinn, fem)
    assert m["rel_l2_u_spacetime"] == pytest.approx(0.1)
    assert m["rel_l2_mu_spacetime"] == 0.0  # the all-zero mu(0) row is left out


def test_fem_energy_metrics_interpolate():
    t_f = np.linspace(0, 2, 21)
    e_f = 1.0 - 0.1 * t_f
    t_p = np.array([0.0, 0.55, 1.0])
    m = fem_energy_metrics(t_p, 1.0 - 0.1 * t_p + np.array([0, 0.02, 0.01]), t_f, e_f)
    assert m["energy_error_max_rel"] == pytest.approx(0.02)
    assert m["energy_error_end_rel"] == pytest.approx(0.01)
