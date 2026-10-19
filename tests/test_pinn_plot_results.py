"""Tests for src.pinn.plot_results: the right data reaches each plot.

Synthetic evaluation.npz / fem_fields.npz / diagnostics.csv files, built so
that every time snapshot is different from every other: a PINN time matched
to the wrong FEM snapshot, a u passed where mu belongs, or a wrong error
value all show up as an exact-equality failure on what the plotting
functions were handed.
"""

import os
import warnings

import numpy as np
import pytest
import torch

from src.common.initial_conditions import initial_condition
from src.common.metrics import relative_l2_error
from src.pinn import plot_results
from src.pinn.diagnostics import save_diagnostics
from src.pinn.evaluate import save_evaluation


def _grid(nx=9, ny=7):
    return np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny), indexing="ij")


def _fem_file(path, X, Y, t, mu0_zero=True):
    """FEM-like fields whose every snapshot differs; mu(0) = 0 like FEM runs
    from before the solver initialized mu."""
    u = np.stack([np.tanh(4 * (X - 0.3 - k)) * 0.9 for k in t])
    mu = np.stack([np.sin(3 * X + 5 * k) * np.cos(2 * Y) + 0.2 for k in t])
    if mu0_zero:
        mu[0] = 0.0
    np.savez(path, x=X, y=Y, t=np.asarray(t), u=u, mu=mu)
    return u, mu


class _Recorder:
    """Wraps a plotting function: records each call's arguments, then calls it."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


@pytest.fixture
def recorders(monkeypatch):
    names = ("plot_comparison", "plot_line_profile", "plot_relative_l2_error", "plot_field",
             "plot_energy_dissipation", "plot_mass_conservation", "plot_loss_history")
    recs = {name: _Recorder(getattr(plot_results, name)) for name in names}
    for name, rec in recs.items():
        monkeypatch.setattr(plot_results, name, rec)
    return recs


def test_field_comparison_pairs_each_pinn_time_with_its_fem_snapshot(tmp_path, recorders):
    X, Y = _grid()
    fem_t = [0.0, 0.001, 0.002, 0.003]
    fem_u, fem_mu = _fem_file(tmp_path / "fem_fields.npz", X, Y, fem_t)
    pinn_t = np.array([0.0, 0.002, 0.003])
    match = [0, 2, 3]
    pinn_u = fem_u[match] + 0.01 * np.sin(7 * X)[None]
    pinn_mu = fem_mu[match] + 0.05 * np.cos(5 * Y)[None]
    save_evaluation(str(tmp_path / "evaluation.npz"), X, Y, pinn_t, pinn_u, pinn_mu)

    with pytest.warns(UserWarning, match="FEM mu is identically zero at t=0"):
        saved = plot_results.plot_field_comparison(
            str(tmp_path / "evaluation.npz"), str(tmp_path / "fem_fields.npz"), str(tmp_path), dpi=20, fmt="png"
        )

    # u comparison for every time; mu comparison only where FEM mu was set.
    calls = recorders["plot_comparison"].calls
    u_calls = [c for c in calls if "vmin" not in c[1]]
    mu_calls = [c for c in calls if "vmin" in c[1]]
    assert len(u_calls) == 3 and len(mu_calls) == 2
    for k, (args, kwargs) in enumerate(u_calls):
        np.testing.assert_array_equal(args[2], pinn_u[k])
        np.testing.assert_array_equal(args[3], fem_u[match[k]])
        assert kwargs["title"] == f"t = {pinn_t[k]:g}"
    for k, (args, _) in zip((1, 2), mu_calls):
        np.testing.assert_array_equal(args[2], pinn_mu[k])
        np.testing.assert_array_equal(args[3], fem_mu[match[k]])

    # Profiles are of u, PINN first and FEM second.
    for k, (args, _) in enumerate(recorders["plot_line_profile"].calls):
        np.testing.assert_array_equal(args[2], pinn_u[k])
        np.testing.assert_array_equal(args[3], fem_u[match[k]])

    # The error curve: one value per PINN time, mu undefined (a gap) at t=0.
    ((t_arg, errors), _), = recorders["plot_relative_l2_error"].calls
    np.testing.assert_array_equal(t_arg, pinn_t)
    expected_u = [relative_l2_error(pinn_u[k], fem_u[match[k]]) for k in range(3)]
    np.testing.assert_allclose(errors["u"], expected_u)
    assert np.isnan(errors["mu"][0])
    np.testing.assert_allclose(errors["mu"][1:], [relative_l2_error(pinn_mu[k], fem_mu[match[k]]) for k in (1, 2)])

    names = sorted(p.split("\\")[-1].split("/")[-1] for p in saved)
    assert "comparison_mu_000_t0.png" not in names
    assert {"comparison_000_t0.png", "comparison_mu_001_t0.002.png", "profile_002_t0.003.png",
            "relative_l2_error.png"} <= set(names)


def test_field_comparison_flags_an_inexact_time_match_in_the_title(tmp_path, recorders):
    X, Y = _grid()
    fem_u, _ = _fem_file(tmp_path / "fem_fields.npz", X, Y, [0.0, 0.001, 0.002], mu0_zero=False)
    save_evaluation(str(tmp_path / "evaluation.npz"), X, Y, np.array([0.0018]), fem_u[[2]], fem_u[[2]])

    with pytest.warns(UserWarning, match="no exact FEM snapshot"):
        plot_results.plot_field_comparison(
            str(tmp_path / "evaluation.npz"), str(tmp_path / "fem_fields.npz"), str(tmp_path), dpi=20, fmt="png"
        )
    args, kwargs = recorders["plot_comparison"].calls[0]
    np.testing.assert_array_equal(args[3], fem_u[2])
    assert kwargs["title"] == "t = 0.0018 (FEM at t = 0.002)"


def test_field_comparison_rejects_a_different_grid(tmp_path):
    X, Y = _grid(9, 7)
    _fem_file(tmp_path / "fem_fields.npz", X, Y, [0.0])
    X8, Y8 = _grid(8, 7)
    save_evaluation(str(tmp_path / "evaluation.npz"), X8, Y8, np.array([0.0]), X8[None], X8[None])
    with pytest.raises(ValueError, match="different"):
        plot_results.plot_field_comparison(
            str(tmp_path / "evaluation.npz"), str(tmp_path / "fem_fields.npz"), str(tmp_path)
        )


def _ic_files(tmp_path, fem_ic, fem_eps=0.05):
    X, Y = _grid(41, 41)
    u0 = initial_condition(fem_ic, X, Y, epsilon=fem_eps, seed=42)
    np.savez(tmp_path / "fem_fields.npz", x=X, y=Y, t=np.array([0.0]), u=u0[None], mu=np.ones((1, 41, 41)))
    # A PINN close to, not equal to, the FEM field: a zero error can't go on the log axis.
    save_evaluation(str(tmp_path / "evaluation.npz"), X, Y, np.array([0.0]), u0[None] + 0.01, np.full((1, 41, 41), 1.1))


@pytest.mark.parametrize(("fem_ic", "fem_eps"), [("two_circles", 0.05), ("cross", 0.04), ("noise", 0.05)])
def test_field_comparison_warns_when_fem_starts_from_another_setup(tmp_path, fem_ic, fem_eps):
    _ic_files(tmp_path, fem_ic, fem_eps)
    with pytest.warns(UserWarning, match="does not start from this run's initial condition"):
        plot_results.plot_field_comparison(
            str(tmp_path / "evaluation.npz"), str(tmp_path / "fem_fields.npz"), str(tmp_path), dpi=20,
            fmt="png", train_args={"ic": "cross", "epsilon": 0.05, "ic_seed": 42},
        )


@pytest.mark.parametrize("ic", ["cross", "two_circles", "noise"])
def test_field_comparison_accepts_fem_from_the_same_setup(tmp_path, ic):
    _ic_files(tmp_path, ic)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_results.plot_field_comparison(
            str(tmp_path / "evaluation.npz"), str(tmp_path / "fem_fields.npz"), str(tmp_path), dpi=20,
            fmt="png", train_args={"ic": ic, "epsilon": 0.05, "ic_seed": 42},
        )
    assert not [w for w in caught if "initial condition" in str(w.message)]


def test_plot_diagnostics_overlays_fem_up_to_the_pinn_t_max(tmp_path, recorders):
    # The PINN CSV of a t_max=0.01 run written before diagnostics.py read t_max
    # in float64 ends at float32(0.01); FEM's accumulated t ends at 0.0099...95.
    t_p = np.linspace(0.0, float(np.float32(0.01)), 11)
    save_diagnostics(str(tmp_path / "diagnostics.csv"), t_p, 0.1 - t_p, -0.57 + 0 * t_p)
    t_f = np.cumsum([0.0] + [1e-4] * 200)[::5]  # t += dt per step, logged every 5th: 0 .. 0.02
    assert t_f[20] > t_p[-1]  # a strict t_f <= t_p[-1] clip drops FEM's t = 0.01 sample
    save_diagnostics(str(tmp_path / "fem.csv"), t_f, 0.12 - t_f, -0.5693 + 0 * t_f)

    plot_results.plot_diagnostics(str(tmp_path / "diagnostics.csv"), str(tmp_path / "fem.csv"), str(tmp_path), fmt="png")

    (args, _), = recorders["plot_energy_dissipation"].calls
    np.testing.assert_array_equal(args[0], t_p)
    np.testing.assert_array_equal(args[2], t_f[:21])  # through t = 0.01, not 0.0095
    np.testing.assert_allclose(args[3], 0.12 - t_f[:21])
    (args, _), = recorders["plot_mass_conservation"].calls
    np.testing.assert_array_equal(args[2], t_f[:21])


def test_plot_evaluation_titles_files_and_mu_scale_follow_each_time(tmp_path, recorders):
    X, Y = _grid()
    t = np.array([0.0, 0.002])
    u = np.stack([0.5 * X, -0.5 * Y])
    mu = np.stack([3.0 * X, -2.0 * Y])
    save_evaluation(str(tmp_path / "evaluation.npz"), X, Y, t, u, mu)

    saved = plot_results.plot_evaluation(str(tmp_path / "evaluation.npz"), str(tmp_path), dpi=20, fmt="png")

    calls = recorders["plot_field"].calls
    for k in range(2):
        np.testing.assert_array_equal(calls[k][0][2], u[k])
        assert calls[k][1]["title"] == f"PINN u, t={t[k]:g}"
        np.testing.assert_array_equal(calls[2 + k][0][2], mu[k])
        assert calls[2 + k][1]["title"] == f"PINN mu, t={t[k]:g}"
        assert calls[2 + k][1]["vmin"] == -calls[2 + k][1]["vmax"]  # one symmetric scale
    assert [p.split("\\")[-1].split("/")[-1] for p in saved] == [
        "field_u_000_t0.png", "field_u_001_t0.002.png", "field_mu_000_t0.png", "field_mu_001_t0.002.png",
    ]


def _checkpoint(path, history, extra=None, **args):
    payload = {"model_state": {}, "args": dict(epochs=5, smoke_test=False, **args), "history": history}
    payload.update(extra or {})
    torch.save(payload, path)
    return str(path)


@pytest.mark.parametrize(
    ("extra", "args", "boundary"),
    [
        ({"adam_steps": 3}, {"adam_patience": 2}, 3),  # recorded by train.py: exact
        ({}, {}, 5),  # older checkpoint, Adam ran all --epochs
        ({}, {"adam_patience": 2}, None),  # older, Adam may have stopped early: unknown
    ],
)
def test_training_history_marks_where_lbfgs_takes_over(tmp_path, recorders, extra, args, boundary):
    history = {"total": [5.0, 4.0, 3.0, 2.5, 2.4, 2.0, 1.9, 1.8]}
    path = _checkpoint(tmp_path / "final.pt", history, extra, **args)
    plot_results.plot_training_history(path, str(tmp_path), dpi=20, fmt="png")
    (call_args, kwargs), = recorders["plot_loss_history"].calls
    assert call_args[0] == history
    assert kwargs["adam_steps"] == boundary


@pytest.mark.parametrize(
    ("heldout", "expected"),
    [
        (
            {"step": [1, 2], "pde_u": [1.0, 0.5], "pde_mu": [1.0, 0.5], "ic": [0.3, 0.2], "bc": [0.1, 0.1]},
            ["heldout_residual.png", "heldout_ic_bc.png"],
        ),
        ({"step": [1, 2], "pde_u": [1.0, 0.5], "pde_mu": [1.0, 0.5]}, ["heldout_residual.png"]),  # older run
        (None, []),  # trained without --heldout-every
    ],
)
def test_plot_heldout_saves_ic_bc_plot_only_when_logged(tmp_path, heldout, expected):
    history = {"pde_u": [1.0, 0.5], "pde_mu": [1.0, 0.5], "ic": [1.0, 0.5], "bc": [1.0, 0.5], "total": [4.0, 2.0]}
    extra = None if heldout is None else {"heldout": heldout}
    path = _checkpoint(tmp_path / "final.pt", history, extra)
    saved = plot_results.plot_heldout(path, str(tmp_path), dpi=20, fmt="png")
    assert [os.path.basename(p) for p in saved] == expected
    assert all(os.path.exists(p) for p in saved)
