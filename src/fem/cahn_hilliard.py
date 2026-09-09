"""DOLFINx solver for the Cahn-Hilliard equation (mixed formulation).

Ground-truth baseline for the PINN comparisons. Structure follows the
official DOLFINx Cahn-Hilliard demo: the phase field c and chemical
potential mu are solved as a coupled mixed (c, mu) system with a
Crank-Nicolson-type time-stepping scheme and Newton nonlinear solves.

NOTE: legacy FEniCS (`dolfin`) demos are NOT API-compatible with DOLFINx
(`dolfinx`) — only DOLFINx-specific references apply here.

Run in the `fenicsx-env` conda environment (see environment.yml).
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, get_type_hints

import numpy as np
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_real_type, plot
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_unit_square

try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    _HAVE_PYVISTA = True
except ModuleNotFoundError:
    _HAVE_PYVISTA = False


@dataclass
class CahnHilliardConfig:
    """Parameters for a Cahn-Hilliard FEM run.

    NOTE: nx/ny, epsilon, dt and t_final are physically coupled and cannot be
    chosen independently. Three constraints tie them together:

    1. The mesh must resolve the interface: h <~ epsilon/4, i.e. nx >~ 4/epsilon.
    2. Spinodal decomposition runs on the timescale set by the bulk well. For
       f = 1/4 (1 - c^2)^2 linear stability gives a fastest growth rate
       sigma_max = mobility/(4*epsilon^2), so the timescale is
       tau = 4*epsilon^2/mobility. Take dt ~ tau/50 and t_final ~ 50*tau --
       phase separation completes in roughly 30-60 tau.
    3. epsilon must be well below the domain size or the phases never form.
       The fastest-growing wavelength is ~8.9*epsilon, so on the unit square
       epsilon=0.1 gives a single blob that stalls at c ~ +-0.87 and never
       reaches the pure phases; epsilon <= 0.05 separates properly.

    Beware of copying dt/t_final from the DOLFINx Cahn-Hilliard demo: it uses
    f = 100*c^2*(1-c)^2 on c in [0, 1], a well ~100x stiffer than the one here,
    so its dt=5e-6 / t_final=2.5e-4 are ~4 orders of magnitude too short for
    this f and produce a field that visibly never changes.

    The defaults below are a fast *smoke* configuration (epsilon=0.05, ~2
    blobs per side, tau=0.01). A resolved thesis run at epsilon=0.02 needs
    roughly `--nx 200 --ny 200 --dt 3e-5 --t-final 0.4`, which is far slower.

    When `adaptive_dt` is set the solver retries a failed step with a halved
    dt rather than aborting, so a slightly-too-large dt self-corrects.
    """

    nx: int = 96
    ny: int = 96
    epsilon: float = 0.01  # interface-width parameter; needs nx >~ 4/epsilon
    mobility: float = 1.0
    dt: float = 2.0e-4  # ~tau/50; initial/maximum step, reduced if adaptive_dt
    t_final: float = 0.5  # ~50*tau, enough for full separation at epsilon=0.05
    theta: float = 0.5  # Crank-Nicolson parameter
    log_every: int = 1  # write diagnostics every N time steps
    seed: int = 42  # random seed for the initial condition
    visualize: bool = True  # save concentration-field PNG frames
    viz_every: int = 25  # save a frame every N time steps (if visualize)
    show_gridpoints: bool = False  # overlay mesh vertices on the solution frames
    adaptive_dt: bool = True  # halve dt and retry when a Newton solve fails
    dt_min: float = 1.0e-12  # give up if an adaptive step falls below this


class _DiagnosticsLogger:
    """Buffered CSV writer for scalar time-step diagnostics."""

    def __init__(self, output_dir: str, every: int) -> None:
        if every < 1:
            raise ValueError("log_every must be at least 1")
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.file: TextIO = (path / "cahn_hilliard_diagnostics.csv").open(
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
        )
        if self.show_gridpoints:
            plotter.add_points(
                self.grid.points,
                color="black",
                point_size=3,
                render_points_as_spheres=True,
            )
        plotter.view_xy(negative=True)
        plotter.add_text(
            f"time: {t:.2e}", font_size=12, color="black", name="timelabel"
        )
        plotter.screenshot(str(self.frames_dir / f"frame_{step:06d}.png"))
        plotter.close()


def parse_args() -> argparse.Namespace:
    """CLI flags mirrored 1:1 from CahnHilliardConfig fields, so they can't drift."""
    parser = argparse.ArgumentParser(description="Run the Cahn-Hilliard FEM solver")
    hints = get_type_hints(CahnHilliardConfig)
    for f in dataclasses.fields(CahnHilliardConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if hints[f.name] is bool:
            # BooleanOptionalAction so True-by-default flags stay switchable
            # off (--visualize / --no-visualize)
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=f.default
            )
        else:
            parser.add_argument(flag, type=hints[f.name], default=f.default)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an output-dir that already has results in it.",
    )
    return parser.parse_args()


def build_mesh(config: CahnHilliardConfig):
    """Create the DOLFINx mesh for the given configuration."""
    return create_unit_square(MPI.COMM_WORLD, config.nx, config.ny, CellType.triangle)


def build_function_space(mesh):
    """Build the mixed (c, mu) function space."""
    P1 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    return functionspace(mesh, mixed_element([P1, P1]))


def _spinodal_initial_condition(seed: int, k_max: int = 4):
    """Smooth low-wavenumber perturbation around c=0, for the CH IC.

    The bulk free energy is the symmetric double well f = 1/4 (1 - c^2)^2
    with minima at c = -1 and c = +1 (the two phases), so a spinodal
    quench is seeded from the symmetric mixture c = 0 with a small
    perturbation.

    Independent per-vertex noise (one random value per mesh vertex, as in
    the official DOLFINx demo) makes the initial field rougher as the mesh
    is refined -- its gradient scales like noise_amplitude/h, so an
    undamped Newton step diverges on fine meshes. A sum of a few low
    wavenumber cosine modes is smooth and mesh-resolution-independent
    instead, and is also the textbook way to seed CH spinodal decomposition
    (small-wavenumber perturbations are what linear stability analysis says
    triggers it).
    """
    rng = np.random.default_rng(seed)
    modes = [
        (kx, ky, rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2 * np.pi))
        for kx in range(1, k_max + 1)
        for ky in range(1, k_max + 1)
    ]

    def ic(x):
        val = np.zeros_like(x[0])
        for kx, ky, amp, phase in modes:
            val += amp * np.cos(2 * np.pi * kx * x[0] + 2 * np.pi * ky * x[1] + phase)
        return 0.05 * val / np.max(np.abs(val))

    return ic


def solve(
    config: CahnHilliardConfig, output_dir: str = "data", overwrite: bool = False
) -> Path:
    """Run the Cahn-Hilliard time-stepping loop.

    Logs free energy E(t) and total mass over time to `output_dir` (stays
    local — this data never needs to leave this machine).

    Args:
        config: solver configuration (mesh resolution, epsilon, mobility, dt, t_final).
        output_dir: directory to write time-series logs / solution snapshots to.
        overwrite: if False (default), refuse to run when output_dir already
            has a diagnostics log in it -- protects a run that took hours
            from being silently clobbered by a later one that forgot to pass
            a fresh --output-dir.

    Returns:
        Path to the diagnostics CSV log (t, free_energy, total_mass).
    """
    existing_log = Path(output_dir) / "cahn_hilliard_diagnostics.csv"
    if existing_log.exists() and not overwrite:
        raise FileExistsError(
            f"{existing_log} already has results. Pass a different --output-dir, "
            f"or --overwrite if you mean to replace it."
        )

    # interface needs several elements across it (~epsilon/4) or the Newton
    # solver tends to diverge on the rough random initial condition
    h = 1.0 / max(config.nx, config.ny)
    if h > config.epsilon / 4:
        warnings.warn(
            f"mesh element size (~{h:.4g}) is coarse relative to epsilon "
            f"({config.epsilon:.4g}); the interface may be under-resolved. "
            f"Increase nx/ny to ~{4 / config.epsilon:.0f}+ or use a larger epsilon.",
            stacklevel=2,
        )

    # Spinodal timescale for f = 1/4 (1 - c^2)^2: linear stability about c=0
    # gives growth rate sigma(k) = M k^2 (1 - lam k^2) with lam = epsilon^2,
    # peaking at sigma_max = M/(4 epsilon^2). Both guards below catch silent
    # "the field never changes" runs, which are easy to get by borrowing
    # dt/t_final from the DOLFINx demo (its well is ~100x stiffer than this one).
    tau = 4 * config.epsilon**2 / config.mobility
    if config.t_final < 20 * tau:
        warnings.warn(
            f"t_final ({config.t_final:.4g}) is short next to the spinodal "
            f"timescale tau={tau:.4g}; the field will barely move. Phase "
            f"separation needs roughly 30-60 tau, i.e. t_final >~ {30 * tau:.3g}.",
            stacklevel=2,
        )

    # the fastest-growing wavelength is 2*pi*sqrt(2)*epsilon; if that is not
    # comfortably inside the unit domain the solution stalls as a single smooth
    # blob at |c| ~ 0.87 instead of separating into the c = -1 / +1 phases
    fastest_wavelength = 2 * np.pi * np.sqrt(2) * config.epsilon
    if fastest_wavelength > 0.5:
        warnings.warn(
            f"epsilon ({config.epsilon:.4g}) is large for the unit square: the "
            f"fastest-growing wavelength is ~{fastest_wavelength:.3g}, so the "
            f"interface spans the domain and c will stall short of +-1 rather "
            f"than separating. Use epsilon <~ 0.05.",
            stacklevel=2,
        )

    mesh = build_mesh(config)
    ME = build_function_space(mesh)

    q, v = ufl.TestFunctions(ME)

    u = Function(ME)  # current solution (c_{n+1}, mu_{n+1})
    u0 = Function(ME)  # previous step (c_n, mu_n)

    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)

    u.sub(0).interpolate(_spinodal_initial_condition(config.seed))
    u.x.scatter_forward()

    c = ufl.variable(c)
    # symmetric double-well bulk free-energy density; phases at c = -1, +1
    f = 1 / 4 * (1 - c**2) ** 2  # kan endre, se notat
    dfdc = ufl.diff(f, c)

    lam = config.epsilon**2  # interfacial-energy coefficient
    mu_mid = (1.0 - config.theta) * mu0 + config.theta * mu

    # dt lives in the form as a Constant so adaptive stepping can change it
    # without rebuilding (and recompiling) the variational form each time
    dt_const = Constant(mesh, default_real_type(config.dt))

    F0 = (
        ufl.inner(c, q) * ufl.dx
        - ufl.inner(c0, q) * ufl.dx
        + dt_const * config.mobility * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
    )
    F1 = (
        ufl.inner(mu, v) * ufl.dx
        - ufl.inner(dfdc, v) * ufl.dx
        - lam * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
    )
    F = F0 + F1

    use_superlu = PETSc.IntType == np.int64
    sys = PETSc.Sys()
    if sys.hasExternalPackage("mumps") and not use_superlu:
        linear_solver = "mumps"
    elif sys.hasExternalPackage("superlu_dist"):
        linear_solver = "superlu_dist"
    else:
        linear_solver = "petsc"

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
        "snes_atol": 0,
        "snes_rtol": 0,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": linear_solver,
    }
    problem = NonlinearProblem(
        F, u, petsc_options_prefix="ch_solve_", petsc_options=petsc_options
    )

    # free energy E(t) = interfacial term + bulk double-well term
    energy_form = form(
        (lam / 2) * ufl.inner(ufl.grad(c), ufl.grad(c)) * ufl.dx + f * ufl.dx
    )
    mass_form = form(c * ufl.dx)

    def energy_and_mass() -> tuple[float, float]:
        e = mesh.comm.allreduce(assemble_scalar(energy_form), op=MPI.SUM)
        m = mesh.comm.allreduce(assemble_scalar(mass_form), op=MPI.SUM)
        return e, m

    u0.x.array[:] = u.x.array

    frames = None
    if config.visualize:
        V0, dofs = ME.sub(0).collapse()
        frames = _FrameWriter(
            output_dir, config.viz_every, V0, dofs, config.show_gridpoints
        )

    diagnostics = _DiagnosticsLogger(output_dir, config.log_every)
    try:
        t = 0.0
        step = 0
        e, m = energy_and_mass()
        diagnostics.log(step, t, e, m)
        if frames is not None:
            frames.save(step, t, u.x.array)

        dt_current = config.dt
        successes_at_dt = 0

        while t < config.t_final:
            # never step past t_final
            dt_current = min(dt_current, config.t_final - t)
            dt_const.value = dt_current

            # a failed Newton solve leaves u polluted, so always (re)start the
            # attempt from the last accepted state
            u.x.array[:] = u0.x.array
            problem.solve()
            converged_reason = problem.solver.getConvergedReason()

            if converged_reason <= 0:
                if not config.adaptive_dt or dt_current / 2 < config.dt_min:
                    raise RuntimeError(
                        f"Newton solver failed to converge at step {step} "
                        f"(t={t}, dt={dt_current:.3g}): reason={converged_reason}. "
                        f"Try a smaller --dt, a finer mesh, or a larger --epsilon."
                    )
                dt_current /= 2
                successes_at_dt = 0
                continue

            t += dt_current
            u0.x.array[:] = u.x.array
            step += 1
            e, m = energy_and_mass()
            diagnostics.log(step, t, e, m)
            if frames is not None:
                frames.save(step, t, u.x.array)

            # creep back toward the requested dt once the step is settled again
            successes_at_dt += 1
            if config.adaptive_dt and successes_at_dt >= 5 and dt_current < config.dt:
                dt_current = min(2 * dt_current, config.dt)
                successes_at_dt = 0
    finally:
        diagnostics.close()

    return Path(output_dir) / "cahn_hilliard_diagnostics.csv"

if __name__ == "__main__":
    cli_args = parse_args()
    field_names = {f.name for f in dataclasses.fields(CahnHilliardConfig)}
    cfg = CahnHilliardConfig(
        **{k: v for k, v in vars(cli_args).items() if k in field_names}
    )
    solve(cfg, output_dir=cli_args.output_dir, overwrite=cli_args.overwrite)
