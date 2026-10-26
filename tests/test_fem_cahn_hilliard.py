"""Tests for src.fem.cahn_hilliard.

NOTE: importing this module requires dolfinx (the fenicsx-env conda
environment). Skipped automatically if dolfinx is not importable, e.g. when
running the PINN-only test suite on Colab or in the Windows venv.
"""

import warnings

import numpy as np
import pytest

from src.common.initial_conditions import cosine_growth_rate, initial_condition, initial_condition_1d


def test_import():
    pytest.importorskip("dolfinx")
    from src.fem import cahn_hilliard  # noqa: F401

    assert hasattr(cahn_hilliard, "CahnHilliardConfig")
    assert hasattr(cahn_hilliard, "solve")


def _solve(tmp_path, **overrides):
    pytest.importorskip("dolfinx")
    pytest.importorskip("scipy")
    from src.fem.cahn_hilliard import CahnHilliardConfig, solve

    settings = dict(nx=20, ny=20, dt=1e-4, t_final=1e-4, viz_every=1, log_every=1, visualize=False)
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a coarse mesh and a short t_final are flagged, on purpose here
        solve(CahnHilliardConfig(**settings), output_dir=str(tmp_path))
    with np.load(tmp_path / "fem_fields.npz") as f:
        return dict(f)


def test_saved_fields_are_the_nodal_values(tmp_path):
    fields = _solve(tmp_path, ic="two_circles")
    # create_unit_square(20, 20) has 21 vertices per side, and the grid is on them.
    assert fields["x"].shape == fields["y"].shape == (21, 21)
    np.testing.assert_allclose(fields["x"][:, 0], np.linspace(0, 1, 21))
    np.testing.assert_allclose(fields["y"][0, :], np.linspace(0, 1, 21))
    ic = initial_condition("two_circles", fields["x"], fields["y"], epsilon=0.05)
    np.testing.assert_allclose(fields["u"][0], ic, atol=1e-12)


def test_first_steps_are_backward_euler(tmp_path):
    # One step with the backward-Euler start is one step of theta = 1.
    started = _solve(tmp_path / "a", startup_steps=1)
    implicit = _solve(tmp_path / "b", startup_steps=0, theta=1.0)
    np.testing.assert_allclose(started["u"][-1], implicit["u"][-1], atol=1e-10)
    crank_nicolson = _solve(tmp_path / "c", startup_steps=0)
    assert not np.allclose(started["u"][-1], crank_nicolson["u"][-1], atol=1e-6)


def test_1d_saved_fields_are_the_nodal_values(tmp_path):
    fields = _solve(tmp_path, dim=1, nx=200, ic="two_intervals")
    np.testing.assert_allclose(fields["x"], np.linspace(0, 1, 201))
    assert fields["u"].shape == fields["mu"].shape == (2, 201)
    ic = initial_condition_1d("two_intervals", fields["x"], epsilon=0.05)
    np.testing.assert_allclose(fields["u"][0], ic, atol=1e-12)


def test_1d_cosine_grows_at_the_linear_rate(tmp_path):
    # A = 0.01 stays in the linear regime: u(x, t) = u0(x) exp(sigma t).
    t_final = 5e-3
    fields = _solve(tmp_path, dim=1, nx=400, ic="cosine", dt=1e-5, t_final=t_final, viz_every=500)
    exact = fields["u"][0] * np.exp(cosine_growth_rate(0.05) * t_final)
    np.testing.assert_allclose(fields["t"][-1], t_final)
    assert np.linalg.norm(fields["u"][-1] - exact) / np.linalg.norm(exact) < 1e-2


def test_1d_writes_space_time_frames(tmp_path):
    _solve(tmp_path, dim=1, nx=50, ic="step", visualize=True)
    assert (tmp_path / "frames" / "space_time_u.png").exists()
    assert (tmp_path / "frames" / "space_time_mu.png").exists()


def _render_frame(tmp_path, field):
    pytest.importorskip("dolfinx")
    pytest.importorskip("pyvista")
    import matplotlib.image as mpimg
    from dolfinx.fem import Function

    from src.fem.cahn_hilliard import CahnHilliardConfig, build_function_space, build_mesh
    from src.fem.output import _FrameWriter

    ME = build_function_space(build_mesh(CahnHilliardConfig(nx=20, ny=20)))
    V0, dofs0 = ME.sub(0).collapse()
    u = Function(ME)
    u.sub(0).interpolate(field)
    _FrameWriter(str(tmp_path), 1, V0, dofs0).save(0, 0.0, u.x.array)
    img = mpimg.imread(tmp_path / "frames" / "frame_000000.png")[..., :3]
    # The domain square: the non-white run through the image centre, left of
    # the colour bar (which sits in the right ~20% of the window).
    cut = img.shape[1] * 4 // 5
    rows = np.where((img[:, img.shape[1] // 3, :] < 0.95).any(axis=1))[0]
    rows = rows[rows > img.shape[0] // 10]  # skip the "time:" label
    cols = np.where((img[img.shape[0] // 2, :cut, :] < 0.95).any(axis=1))[0]
    return img, (rows[0] + 5, rows[-1] - 5), (cols[0] + 5, cols[-1] - 5)


def test_frames_put_x_right_and_y_up_in_colour_bar_colours(tmp_path):
    from matplotlib.colors import to_rgb

    blue, red = np.array(to_rgb("#2a78d6")), np.array(to_rgb("#e34948"))

    img, _, (left, right) = _render_frame(tmp_path / "x", lambda x: 2.0 * x[0] - 1.0)
    mid = img.shape[0] // 2
    np.testing.assert_allclose(img[mid, left], blue, atol=0.05)  # x = 0: u = -1
    np.testing.assert_allclose(img[mid, right], red, atol=0.05)  # x = 1: u = +1

    img, (top, bottom), (left, right) = _render_frame(tmp_path / "y", lambda x: 2.0 * x[1] - 1.0)
    centre = (left + right) // 2
    np.testing.assert_allclose(img[bottom, centre], blue, atol=0.05)  # y = 0
    np.testing.assert_allclose(img[top, centre], red, atol=0.05)  # y = 1
