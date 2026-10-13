"""Tests for src.fem.cahn_hilliard.

NOTE: importing this module requires dolfinx (the fenicsx-env conda
environment). Skipped automatically if dolfinx is not importable, e.g. when
running the PINN-only test suite on Colab or in the Windows venv.
"""

import warnings

import numpy as np
import pytest

from src.common.initial_conditions import initial_condition


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


def _render_frame(tmp_path, field):
    pytest.importorskip("dolfinx")
    pytest.importorskip("pyvista")
    import matplotlib.image as mpimg
    from dolfinx.fem import Function

    from src.fem.cahn_hilliard import CahnHilliardConfig, _FrameWriter, build_function_space, build_mesh

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
