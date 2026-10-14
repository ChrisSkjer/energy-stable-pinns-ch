"""Everything the Cahn-Hilliard FEM solver writes to disk.

- cahn_hilliard_diagnostics.csv: t, free_energy, total_mass (_DiagnosticsLogger)
- frames/frame_*.png: rendered snapshots of c (_FrameWriter, needs pyvista)
- fem_fields.npz: (x, y, t, u, mu) on the grid of mesh vertices (_FieldRecorder)

RunOutputs bundles the three so the solver makes one call per time step.
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import numpy as np

from dolfinx import plot

try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    _HAVE_PYVISTA = True
except ModuleNotFoundError:
    _HAVE_PYVISTA = False

DIAGNOSTICS_FILE = "cahn_hilliard_diagnostics.csv"


class _DiagnosticsLogger:
    """Buffered CSV writer for scalar time-step diagnostics."""

    def __init__(self, output_dir: str, every: int) -> None:
        if every < 1:
            raise ValueError("log_every must be at least 1")
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.file: TextIO = (path / DIAGNOSTICS_FILE).open(
            "w", encoding="utf-8", buffering=64 * 1024
        )
        self.file.write("t,free_energy,total_mass\n")

    def log(self, step: int, t: float, energy: float, mass: float) -> None:
        if step % self.every == 0:
            self.file.write(f"{t:.16e},{energy:.16e},{mass:.16e}\n")

    def close(self) -> None:
        self.file.close()


def _phase_colormap():
    """Diverging blue<->red colormap for c in [-1, 1], neutral gray at c=0.

    c is a *polarity* field, not a magnitude one: the two phases sit at c=-1
    and c=+1 either side of a meaningful midpoint at c=0. A sequential map
    (matplotlib's default viridis) implies a low-to-high ramp and buries that
    symmetry -- the c=0 mixed state lands mid-ramp instead of reading as
    "neither phase". A diverging map with a neutral midpoint puts the two
    phases at opposite poles and lets interfaces show up as the gray band.
    """
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ModuleNotFoundError:  # pyvista ships equivalent built-ins
        return "coolwarm"
    # blue (c=-1) -> neutral gray (c=0) -> red (c=+1), equal arms
    return LinearSegmentedColormap.from_list(
        "phase", ["#2a78d6", "#f0efec", "#e34948"]
    )


class _FrameWriter:
    """Off-screen PNG snapshots of the concentration field, saved to disk."""

    def __init__(
        self, output_dir: str, every: int, V0, dofs, show_gridpoints: bool = False
    ) -> None:
        if every < 1:
            raise ValueError("viz_every must be at least 1")
        if not _HAVE_PYVISTA:
            raise ModuleNotFoundError(
                "visualize=True requires pyvista (conda install -n fenicsx-env "
                "-c conda-forge pyvista)"
            )
        self.every = every
        # some DOLFINx versions wrap the dof array in a single-element list
        self.dofs = dofs[0] if isinstance(dofs, list) else dofs
        self.show_gridpoints = show_gridpoints
        self.frames_dir = Path(output_dir) / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        topology, cell_types, x = plot.vtk_mesh(V0)
        self.grid = pv.UnstructuredGrid(topology, cell_types, x)

        self.cmap = _phase_colormap()
        # Vertical bar down the right-hand side. The pyvista default is a wide
        # horizontal bar with its title centred *above* the ticks, which puts
        # the title "c" straight on top of the "0.00" label; going vertical
        # removes the collision and stops the bar crowding the square domain.
        self.scalar_bar_args = dict(
            title="c",
            vertical=True,
            position_x=0.86,
            position_y=0.22,
            width=0.05,
            height=0.56,
            n_labels=5,
            fmt="%.1f",  # -1.0/-0.5/0.0/0.5/1.0, not the default -1.00/-0.500
            title_font_size=18,
            label_font_size=15,
            color="black",
        )

    def save(self, step: int, t: float, u_array) -> None:
        if step % self.every != 0:
            return
        self.grid.point_data["c"] = u_array[self.dofs].real
        self.grid.set_active_scalars("c")
        plotter = pv.Plotter(off_screen=True, window_size=(900, 800))
        plotter.set_background("white")
        plotter.add_mesh(
            self.grid,
            cmap=self.cmap,
            clim=[-1, 1],
            scalar_bar_args=self.scalar_bar_args,
            # Lighting shades a flat 2D mesh ~25% darker than the colour bar,
            # so c could no longer be read off the bar.
            lighting=False,
        )
        if self.show_gridpoints:
            plotter.add_points(
                self.grid.points,
                color="black",
                point_size=3,
                render_points_as_spheres=True,
            )
        # Camera on +z: x to the right, y up. view_xy(negative=True) looks from
        # -z and mirrors x, which put the large two_circles disk on the right.
        plotter.view_xy()
        plotter.add_text(
            f"time: {t:.2e}", font_size=12, color="black", name="timelabel"
        )
        plotter.screenshot(str(self.frames_dir / f"frame_{step:06d}.png"))
        plotter.close()


def _interpolate_to_grid(coords: np.ndarray, values: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Scatter DOF values onto a regular grid via linear interpolation (exact
    at grid points that coincide with a DOF, as _FieldRecorder's all do).

    Falls back to nearest-neighbor for any grid point griddata leaves as NaN
    -- happens occasionally right at the domain boundary, where a target
    point sits exactly on (rather than safely inside) the DOF convex hull.
    """
    from scipy.interpolate import griddata

    grid = griddata(coords, values, (X, Y), method="linear")
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid[nan_mask] = griddata(coords, values, (X[nan_mask], Y[nan_mask]), method="nearest")
    return grid


class _FieldRecorder:
    """Collects (t, u, mu) snapshots on a regular grid, for saving to an .npz
    that PINN vs. FEM comparisons can load directly.

    Uses the same (x, y, t, u, mu) layout as src/pinn/evaluate.py's
    save_evaluation, so src/common/plotting.py::plot_comparison can consume
    either one -- x, y are (nx+1, ny+1) meshgrids (np.meshgrid(...,
    indexing="ij")) and u, mu are (len(t), nx+1, ny+1).

    nx+1 points per side, because create_unit_square(nx, ny) has nx cells and
    so nx+1 vertices per side: the grid then sits exactly on the P1 dofs and
    the values saved are the nodal values themselves. (An nx-point grid falls
    between the vertices and adds an interpolation error -- 0.37% relative L2
    for the cross IC at eps=0.05 -- to every PINN-vs-FEM comparison.) The
    dofs are not in raster order, so each snapshot is still scattered onto
    the grid with scipy.interpolate.griddata rather than reshaped directly.
    """

    def __init__(self, every: int, V0, dofs0, V1, dofs1, nx: int, ny: int) -> None:
        if every < 1:
            raise ValueError("viz_every must be at least 1")
        self.every = every
        self.dofs0 = dofs0[0] if isinstance(dofs0, list) else dofs0
        self.dofs1 = dofs1[0] if isinstance(dofs1, list) else dofs1
        self.coords0 = V0.tabulate_dof_coordinates()[:, :2]
        self.coords1 = V1.tabulate_dof_coordinates()[:, :2]
        x = np.linspace(0.0, 1.0, nx + 1)
        y = np.linspace(0.0, 1.0, ny + 1)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.times: list[float] = []
        self.u_frames: list[np.ndarray] = []
        self.mu_frames: list[np.ndarray] = []

    def save(self, step: int, t: float, u_array: np.ndarray) -> None:
        if step % self.every != 0:
            return
        u_vals = u_array[self.dofs0].real
        mu_vals = u_array[self.dofs1].real
        self.times.append(t)
        self.u_frames.append(_interpolate_to_grid(self.coords0, u_vals, self.X, self.Y))
        self.mu_frames.append(_interpolate_to_grid(self.coords1, mu_vals, self.X, self.Y))

    def write(self, output_dir: str) -> Path:
        path = Path(output_dir) / "fem_fields.npz"
        np.savez(
            path,
            x=self.X,
            y=self.Y,
            t=np.array(self.times),
            u=np.stack(self.u_frames, axis=0),
            mu=np.stack(self.mu_frames, axis=0),
        )
        return path


def check_output_dir(output_dir: str, overwrite: bool) -> None:
    """Refuse to reuse an output_dir that already has a diagnostics log,
    unless overwrite -- protects a run that took hours from being silently
    clobbered by a later one that forgot to pass a fresh --output-dir."""
    existing_log = Path(output_dir) / DIAGNOSTICS_FILE
    if existing_log.exists() and not overwrite:
        raise FileExistsError(
            f"{existing_log} already has results. Pass a different --output-dir, "
            f"or --overwrite if you mean to replace it."
        )


class RunOutputs:
    """The diagnostics log, plus frames (if config.visualize) and the field
    recorder (if config.save_fields), behind one record()/close() pair.

    `ME` is the mixed (c, mu) function space; sub(0) is c, sub(1) is mu.
    """

    def __init__(self, config, output_dir: str, ME) -> None:
        self.output_dir = output_dir
        self.frames = None
        self.fields = None
        if config.visualize or config.save_fields:
            V0, dofs0 = ME.sub(0).collapse()
        if config.visualize:
            self.frames = _FrameWriter(
                output_dir, config.viz_every, V0, dofs0, config.show_gridpoints
            )
        if config.save_fields:
            V1, dofs1 = ME.sub(1).collapse()
            self.fields = _FieldRecorder(
                config.viz_every, V0, dofs0, V1, dofs1, config.nx, config.ny
            )
        self.diagnostics = _DiagnosticsLogger(output_dir, config.log_every)

    @property
    def diagnostics_path(self) -> Path:
        return Path(self.output_dir) / DIAGNOSTICS_FILE

    def record(self, step: int, t: float, u_array, energy: float, mass: float) -> None:
        """Log and snapshot the accepted state at `step`; each output keeps its own cadence."""
        self.diagnostics.log(step, t, energy, mass)
        if self.frames is not None:
            self.frames.save(step, t, u_array)
        if self.fields is not None:
            self.fields.save(step, t, u_array)

    def close(self) -> None:
        self.diagnostics.close()
        if self.fields is not None and self.fields.times:
            self.fields.write(self.output_dir)
