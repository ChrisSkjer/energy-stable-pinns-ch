"""Tests for src.common.plotting: each plot shows the data it says it shows.

Field plots are checked on the rendered pixels -- the colour drawn at data
position (x, y) must be the colour map's colour for the field's value there --
so a transposed grid, a flipped axis, a swapped panel or a wrong colour scale
fails even when every array has the right shape. Line plots are checked
through their Line2D data, which is exactly what gets drawn.
"""

import numpy as np
import pytest

from src.common import plotting
from src.common.metrics import relative_l2_error


def _grid(nx=6, ny=4, x_max=1.0, y_max=2.0):
    # Non-square in point count and extent, so a transposed grid can't pass.
    return np.meshgrid(np.linspace(0.0, x_max, nx), np.linspace(0.0, y_max, ny), indexing="ij")


def _mesh(ax):
    (mesh,) = ax.collections
    return mesh


def _assert_shows(ax, X, Y, C, cells=None):
    """Every value C[i, j] (or those in `cells`) is drawn at (X[i, j], Y[i, j]),
    in its own colour."""
    ax.figure.canvas.draw()
    buf = np.asarray(ax.figure.canvas.buffer_rgba())
    mesh = _mesh(ax)
    for i, j in np.ndindex(C.shape) if cells is None else cells:
        px, py = ax.transData.transform((X[i, j], Y[i, j]))
        drawn = buf[buf.shape[0] - 1 - int(py), int(px), :3] / 255.0
        expected = mesh.cmap(mesh.norm(C[i, j]))[:3]
        np.testing.assert_allclose(drawn, expected, atol=2.5 / 255, err_msg=f"cell ({i}, {j})")


def _lines(ax):
    return {line.get_label(): line for line in ax.get_lines()}


def test_phase_colormap_is_blue_at_minus_one_and_red_at_plus_one():
    # Same convention (and colours) as the FEM frames' colormap in
    # src/fem/cahn_hilliard.py, so PINN and FEM pictures read the same way.
    from matplotlib.colors import to_rgb

    cmap = plotting.phase_colormap()
    np.testing.assert_allclose(cmap(0.0)[:3], to_rgb("#2a78d6"), atol=1e-2)  # u = -1
    np.testing.assert_allclose(cmap(0.5)[:3], to_rgb("#f0efec"), atol=1e-2)  # u = 0
    np.testing.assert_allclose(cmap(1.0)[:3], to_rgb("#e34948"), atol=1e-2)  # u = +1


def test_plot_field_draws_each_value_at_its_own_position():
    X, Y = _grid()
    C = np.linspace(-0.9, 0.9, X.size).reshape(X.shape)  # every cell different
    ax = plotting.plot_field(X, Y, C)
    _assert_shows(ax, X, Y, C)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x", "y")
    assert _mesh(ax).norm.vmin == -1.0 and _mesh(ax).norm.vmax == 1.0


@pytest.mark.parametrize(
    ("low", "high", "extend"),
    [(-0.5, 0.5, "neither"), (-0.5, 1.3, "max"), (-1.2, 0.5, "min"), (-1.2, 1.3, "both")],
)
def test_plot_field_colour_bar_flags_clipped_values(low, high, extend):
    X, Y = _grid()
    C = np.linspace(low, high, X.size).reshape(X.shape)
    assert _mesh(plotting.plot_field(X, Y, C)).colorbar.extend == extend


def test_plot_comparison_panels_show_pinn_fem_and_their_difference():
    X, Y = _grid()
    pinn = 0.8 * np.tanh(3.0 * (X - 0.5))
    fem = 0.6 * np.cos(2.0 * Y) + 0.1 * X
    fig, (ax_pinn, ax_fem, ax_diff) = plotting.plot_comparison(X, Y, pinn, fem, title="t = 0.005")

    _assert_shows(ax_pinn, X, Y, pinn)
    _assert_shows(ax_fem, X, Y, fem)
    _assert_shows(ax_diff, X, Y, pinn - fem)  # PINN minus FEM, as the title says
    bound = np.abs(pinn - fem).max()
    assert _mesh(ax_diff).norm.vmin == pytest.approx(-bound)
    assert _mesh(ax_diff).norm.vmax == pytest.approx(bound)
    assert (ax_pinn.get_title(), ax_fem.get_title()) == ("PINN", "FEM")
    assert ax_diff.get_title() == f"PINN - FEM  (rel. L2 = {relative_l2_error(pinn, fem):.2%})"
    assert fig.get_suptitle() == "t = 0.005"


def test_plot_comparison_forwards_mu_colour_scale_to_both_field_panels():
    X, Y = _grid()
    fig, (ax_pinn, ax_fem, _) = plotting.plot_comparison(
        X, Y, 3 * X, 2 * Y, vmin=-4.0, vmax=4.0, cmap="RdBu_r", epsilon=0.05
    )
    for ax in (ax_pinn, ax_fem):
        assert (_mesh(ax).norm.vmin, _mesh(ax).norm.vmax) == (-4.0, 4.0)
        assert _mesh(ax).cmap.name == "RdBu_r"
    _assert_shows(ax_pinn, X, Y, 3 * X)
    assert fig.get_suptitle() == "epsilon = 0.05"


def test_plot_comparison_title_survives_zero_fem_field():
    X, Y = _grid()
    _, (_, _, ax_diff) = plotting.plot_comparison(X, Y, 0.1 * X, np.zeros_like(X))
    assert ax_diff.get_title() == "PINN - FEM"


def test_plot_line_profile_vertical_line():
    X, Y = _grid(nx=11, ny=21, x_max=1.0, y_max=1.0)
    fem = np.cos(np.pi * X) * np.sin(np.pi * Y)
    pinn = fem + 0.1 * X + 0.05 * Y  # differs from FEM, so a swap shows
    fig, (ax_field, ax_profile) = plotting.plot_line_profile(X, Y, pinn, fem, x_line=0.33)

    i = 3  # x = 0.3 is the grid column closest to 0.33
    lines = _lines(ax_profile)
    np.testing.assert_array_equal(lines["FEM"].get_xdata(), Y[i, :])
    np.testing.assert_array_equal(lines["FEM"].get_ydata(), fem[i, :])
    np.testing.assert_array_equal(lines["PINN"].get_xdata(), Y[i, :])
    np.testing.assert_array_equal(lines["PINN"].get_ydata(), pinn[i, :])
    assert ax_profile.get_xlabel() == "y"
    assert ax_profile.get_title() == "Profile along x = 0.3"
    # The field panel is the FEM field (probed off the cut line, which covers
    # column i), with the cut drawn at the snapped x.
    marker = ax_field.get_lines()[-1]
    np.testing.assert_allclose(marker.get_xdata(), [0.3, 0.3])
    _assert_shows(ax_field, X, Y, fem, cells=[c for c in np.ndindex(fem.shape) if c[0] != i])


def test_plot_line_profile_horizontal_line_wins_over_vertical():
    X, Y = _grid(nx=11, ny=21, x_max=1.0, y_max=1.0)
    fem = np.cos(np.pi * X) * np.sin(np.pi * Y)
    pinn = fem - 0.2 * X
    _, (ax_field, ax_profile) = plotting.plot_line_profile(X, Y, pinn, fem, x_line=0.25, y_line=0.52)

    j = 10  # y = 0.5
    lines = _lines(ax_profile)
    np.testing.assert_array_equal(lines["FEM"].get_xdata(), X[:, j])
    np.testing.assert_array_equal(lines["FEM"].get_ydata(), fem[:, j])
    np.testing.assert_array_equal(lines["PINN"].get_ydata(), pinn[:, j])
    assert ax_profile.get_xlabel() == "x"
    np.testing.assert_allclose(ax_field.get_lines()[-1].get_ydata(), [0.5, 0.5])


def test_plot_relative_l2_error_plots_given_errors_with_gaps():
    t = np.array([0.0, 0.002, 0.005])
    errors = {"u": np.array([0.01, 0.02, 0.03]), "mu": np.array([np.nan, 0.1, 0.2])}
    ax = plotting.plot_relative_l2_error(t, errors)

    lines = _lines(ax)
    for name, err in errors.items():
        np.testing.assert_array_equal(lines[name].get_xdata(), t)
        np.testing.assert_array_equal(lines[name].get_ydata(), err)
    assert ax.get_yscale() == "log"


_HISTORY = {
    "pde_u": [4.0, 2.0, 1.0, 0.9, 0.8, 1.5, 0.7],
    "pde_mu": [1.0, 0.5, 0.4, 0.3, 0.2, 0.3, 0.1],
    "ic": [3.0, 2.0, 1.0, 0.5, 0.4, 0.6, 0.3],
    "bc": [0.1, 0.1, 0.05, 0.05, 0.04, 0.04, 0.03],
}
_HISTORY["total"] = list(np.sum([_HISTORY[k] for k in _HISTORY], axis=0))


def test_plot_loss_history_plots_each_term_against_its_step():
    ax = plotting.plot_loss_history(_HISTORY, adam_steps=5)

    lines = _lines(ax)
    for name, values in _HISTORY.items():
        np.testing.assert_array_equal(lines[name].get_xdata(), np.arange(len(values)))
        np.testing.assert_array_equal(lines[name].get_ydata(), values)
    np.testing.assert_array_equal(lines["Adam -> L-BFGS"].get_xdata(), [5, 5])
    assert ax.get_yscale() == "log"


def test_plot_loss_history_without_adam_steps_draws_no_boundary():
    assert "Adam -> L-BFGS" not in _lines(plotting.plot_loss_history(_HISTORY))


def test_plot_heldout_residual_unweights_training_curves():
    history = {"pde_u": [2.0, 1.0], "pde_mu": [300.0, 30.0], "total": [302.0, 31.0]}
    heldout = {"step": [1, 2], "pde_u": [5.0, 4.0], "pde_mu": [0.5, 0.4]}
    ax = plotting.plot_heldout_residual(heldout, history, pde_u_weight=2.0, pde_mu_weight=300.0)

    lines = _lines(ax)
    np.testing.assert_allclose(lines["pde_u (training points)"].get_ydata(), [1.0, 0.5])
    np.testing.assert_allclose(lines["pde_mu (training points)"].get_ydata(), [1.0, 0.1])
    np.testing.assert_array_equal(lines["pde_u (held-out)"].get_xdata(), [1, 2])
    np.testing.assert_array_equal(lines["pde_u (held-out)"].get_ydata(), [5.0, 4.0])
    np.testing.assert_array_equal(lines["pde_mu (held-out)"].get_ydata(), [0.5, 0.4])


def test_energy_plot_shows_each_series_under_its_own_label():
    t_p, e_p = np.linspace(0.0, 0.01, 5), np.array([0.11, 0.10, 0.09, 0.085, 0.08])
    t_f, e_f = np.linspace(0.0, 0.01, 3), np.array([0.12, 0.10, 0.07])
    lines = _lines(plotting.plot_energy_dissipation(t_p, e_p, t_f, e_f))
    np.testing.assert_array_equal(lines["PINN"].get_xdata(), t_p)
    np.testing.assert_array_equal(lines["PINN"].get_ydata(), e_p)
    np.testing.assert_array_equal(lines["FEM"].get_xdata(), t_f)
    np.testing.assert_array_equal(lines["FEM"].get_ydata(), e_f)


def test_mass_plot_shows_drift_from_each_series_own_start():
    t_p, m_p = np.linspace(0.0, 0.01, 4), np.array([-0.57, -0.569, -0.571, -0.572])
    t_f, m_f = np.linspace(0.0, 0.01, 3), np.array([-0.5693, -0.5693, -0.5692])
    lines = _lines(plotting.plot_mass_conservation(t_p, m_p, t_f, m_f))
    np.testing.assert_allclose(lines["PINN"].get_ydata(), m_p - m_p[0])
    np.testing.assert_allclose(lines["FEM"].get_ydata(), m_f - m_f[0])
    np.testing.assert_array_equal(lines["FEM"].get_xdata(), t_f)
