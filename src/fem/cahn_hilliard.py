"""DOLFINx solver for the Cahn-Hilliard equation (mixed formulation).

Ground-truth baseline for the PINN comparisons. Structure follows the
official DOLFINx Cahn-Hilliard demo: the phase field c and chemical
potential mu are solved as a coupled mixed (c, mu) system with a
Crank-Nicolson-type time-stepping scheme and Newton nonlinear solves.

This file holds only the math. Run parameters, sanity checks, the CLI and
the PETSc options are in src/fem/config.py. Logging, frames and the
fem_fields.npz dump are in src/fem/output.py.

NOTE: legacy FEniCS (`dolfin`) demos are NOT API-compatible with DOLFINx
(`dolfinx`) — only DOLFINx-specific references apply here.

Run in the `fenicsx-env` conda environment (see environment.yml).
"""

from __future__ import annotations

from pathlib import Path

import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI

from dolfinx import default_real_type
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_unit_square

from src.common.initial_conditions import initial_condition
from src.fem.config import CahnHilliardConfig, check_config, config_from_args, newton_options, parse_args
from src.fem.output import RunOutputs, check_output_dir


def build_mesh(config: CahnHilliardConfig):
    """Create the DOLFINx mesh for the given configuration."""
    return create_unit_square(MPI.COMM_WORLD, config.nx, config.ny, CellType.triangle)


def build_function_space(mesh):
    """Build the mixed (c, mu) function space."""
    P1 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    return functionspace(mesh, mixed_element([P1, P1]))


def solve(
    config: CahnHilliardConfig, output_dir: str = "data", overwrite: bool = False
) -> Path:
    """Run the Cahn-Hilliard time-stepping loop.

    Writes the diagnostics CSV (t, free_energy, total_mass), PNG frames and
    fem_fields.npz to `output_dir`; see src/fem/output.py.

    Args:
        config: solver configuration (mesh resolution, epsilon, mobility, dt, t_final).
        output_dir: directory to write time-series logs / solution snapshots to.
        overwrite: if False (default), refuse to run when output_dir already
            has a diagnostics log in it.

    Returns:
        Path to the diagnostics CSV log (t, free_energy, total_mass).
    """
    check_config(config)
    check_output_dir(output_dir, overwrite)

    mesh = build_mesh(config)
    ME = build_function_space(mesh)

    q, v = ufl.TestFunctions(ME)

    u = Function(ME)  # current solution (c_{n+1}, mu_{n+1})
    u0 = Function(ME)  # previous step (c_n, mu_n)

    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)

    # Shared with the PINN (src/pinn/sampling.py), so both start from the same
    # field. The tanh-profile ICs use the solver's own epsilon, so the
    # interface starts at its equilibrium width.
    u.sub(0).interpolate(
        lambda x: initial_condition(
            config.ic, x[0], x[1], epsilon=config.epsilon, seed=config.seed
        )
    )
    u.x.scatter_forward()

    c = ufl.variable(c)
    # symmetric double-well bulk free-energy density; phases at c = -1, +1
    f = 1 / 4 * (1 - c**2) ** 2  
    dfdc = ufl.diff(f, c)

    lam = config.epsilon**2  # interfacial-energy coefficient

    # dt and theta live in the form as Constants so adaptive stepping and the
    # backward-Euler start can change them without rebuilding (and
    # recompiling) the variational form each time
    dt_const = Constant(mesh, default_real_type(config.dt))
    theta_const = Constant(mesh, default_real_type(config.theta))
    mu_mid = (1.0 - theta_const) * mu0 + theta_const * mu

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

    problem = NonlinearProblem(
        F, u, petsc_options_prefix="ch_solve_", petsc_options=newton_options()
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

    outputs = RunOutputs(config, output_dir, ME)
    try:
        t = 0.0
        step = 0
        outputs.record(step, t, u.x.array, *energy_and_mass())

        dt_current = config.dt
        successes_at_dt = 0

        while t < config.t_final:
            # never step past t_final
            dt_current = min(dt_current, config.t_final - t)
            dt_const.value = dt_current
            # Rannacher start: the first startup_steps steps are backward
            # Euler, the rest use theta. The IC only sets c, so mu starts at 0,
            # and a Crank-Nicolson first step from there applies half the
            # flux: u then lags the true solution by ~dt/2 and mu stays off
            # for many steps (cross IC, dt=1e-4, t <= 0.01: 0.08-0.4% relative
            # L2 error in u, 0.7-6% in mu). Starting Crank-Nicolson from a
            # consistent mu instead rings on the IC's kinks (the cross's
            # corners). Two implicit steps avoid both, cutting those errors to
            # 0.001-0.08% in u and 0.006-1% in mu; startup_steps=0 is the old
            # scheme.
            theta_const.value = 1.0 if step < config.startup_steps else config.theta

            # a failed Newton solve leaves u polluted, so always (re)start the
            # attempt from the last accepted state
            u.x.array[:] = u0.x.array
            u.x.scatter_forward()
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
            outputs.record(step, t, u.x.array, *energy_and_mass())

            # creep back toward the requested dt once the step is settled again
            successes_at_dt += 1
            if config.adaptive_dt and successes_at_dt >= 5 and dt_current < config.dt:
                dt_current = min(2 * dt_current, config.dt)
                successes_at_dt = 0
    finally:
        outputs.close()

    return outputs.diagnostics_path


if __name__ == "__main__":
    cli_args = parse_args()
    solve(config_from_args(cli_args), output_dir=cli_args.output_dir, overwrite=cli_args.overwrite)
