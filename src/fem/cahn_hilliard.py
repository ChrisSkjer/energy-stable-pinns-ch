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

from dataclasses import dataclass


@dataclass
class CahnHilliardConfig:
    """Parameters for a Cahn-Hilliard FEM run."""

    nx: int = 96
    ny: int = 96
    epsilon: float = 0.02  # interface-width parameter
    mobility: float = 1.0
    dt: float = 5.0e-6
    t_final: float = 5.0e-2
    theta: float = 0.5  # Crank-Nicolson parameter


def build_mesh(config: CahnHilliardConfig):
    """Create the DOLFINx mesh for the given configuration."""
    # TODO: dolfinx.mesh.create_unit_square(comm, config.nx, config.ny, ...)
    raise NotImplementedError


def build_function_space(mesh):
    """Build the mixed (c, mu) function space."""
    # TODO: mixed P1/P1 (or similar) function space, matching the DOLFINx CH demo
    raise NotImplementedError


def solve(config: CahnHilliardConfig, output_dir: str = "data"):
    """Run the Cahn-Hilliard time-stepping loop.

    Logs free energy E(t) and total mass over time to `output_dir` (stays
    local — this data never needs to leave this machine).

    Args:
        config: solver configuration (mesh resolution, epsilon, mobility, dt, t_final).
        output_dir: directory to write time-series logs / solution snapshots to.
    """
    # TODO: assemble weak form (mixed formulation), set up Newton solver
    # TODO: time-stepping loop over [0, t_final] with step config.dt
    # TODO: at each step (or every N steps), log t, free energy E(t), total mass to output_dir
    raise NotImplementedError


if __name__ == "__main__":
    solve(CahnHilliardConfig())
