"""Run parameters for the Cahn-Hilliard FEM solver (src/fem/cahn_hilliard.py).

Holds everything about *choosing* a run rather than solving it: the config
dataclass, the CLI that mirrors it, sanity checks on the parameter choice,
and the PETSc Newton/linear-solver options.
"""

from __future__ import annotations

import argparse
import dataclasses
import warnings
from dataclasses import dataclass
from typing import get_type_hints

import numpy as np
from petsc4py import PETSc

from dolfinx import default_real_type

from src.common.initial_conditions import INITIAL_CONDITIONS


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

    The first `startup_steps` steps are backward Euler (theta=1), the rest use
    `theta`; see the comment at the time loop in cahn_hilliard.solve for why.
    """

    nx: int = 100
    ny: int = 100
    epsilon: float = 0.05  # interface-width parameter; needs nx >~ 4/epsilon
    mobility: float = 1.0
    dt: float = 1.0e-4  # ~tau/50; initial/maximum step, reduced if adaptive_dt
    t_final: float = 0.1  # ~50*tau, enough for full separation at epsilon=0.05
    theta: float = 0.5  # Crank-Nicolson parameter
    startup_steps: int = 2  # backward-Euler steps before switching to theta
    log_every: int = 5  # write diagnostics every N time steps
    ic: str = "cross"  # initial condition, see src/common/initial_conditions.py
    seed: int = 42  # random seed for the initial condition (used by ic="noise")
    visualize: bool = True  # save concentration-field PNG frames
    viz_every: int = 5  # save a frame every N time steps (if visualize)
    show_gridpoints: bool = False  # overlay mesh vertices on the solution frames
    adaptive_dt: bool = True  # halve dt and retry when a Newton solve fails
    dt_min: float = 1.0e-12  # give up if an adaptive step falls below this
    save_fields: bool = True  # dump (u, mu) on a regular grid, for PINN comparison


def check_config(config: CahnHilliardConfig) -> None:
    """Reject an unknown IC and warn about parameter choices that silently
    give a useless run (see CahnHilliardConfig's docstring).

    Warnings use stacklevel=3 so they point at the caller of solve(), which
    is where this is called from.
    """
    if config.ic not in INITIAL_CONDITIONS:
        raise ValueError(
            f"unknown initial condition {config.ic!r}; choose from {sorted(INITIAL_CONDITIONS)}"
        )

    # interface needs several elements across it (~epsilon/4) or the Newton
    # solver tends to diverge on the rough random initial condition
    h = 1.0 / max(config.nx, config.ny)
    if h > config.epsilon / 4:
        warnings.warn(
            f"mesh element size (~{h:.4g}) is coarse relative to epsilon "
            f"({config.epsilon:.4g}); the interface may be under-resolved. "
            f"Increase nx/ny to ~{4 / config.epsilon:.0f}+ or use a larger epsilon.",
            stacklevel=3,
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
            stacklevel=3,
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
            stacklevel=3,
        )


def newton_options() -> dict:
    """PETSc options for the Newton solve: line-search Newton with a direct
    LU solve per iteration, using MUMPS or SuperLU_dist when available."""
    use_superlu = PETSc.IntType == np.int64
    sys = PETSc.Sys()
    if sys.hasExternalPackage("mumps") and not use_superlu:
        linear_solver = "mumps"
    elif sys.hasExternalPackage("superlu_dist"):
        linear_solver = "superlu_dist"
    else:
        linear_solver = "petsc"

    return {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
        "snes_atol": 0,
        "snes_rtol": 0,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": linear_solver,
    }


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


def config_from_args(args: argparse.Namespace) -> CahnHilliardConfig:
    """Build a config from parse_args() output, ignoring the non-config flags."""
    field_names = {f.name for f in dataclasses.fields(CahnHilliardConfig)}
    return CahnHilliardConfig(**{k: v for k, v in vars(args).items() if k in field_names})
