"""Comparison metrics for PINN vs. FEM Cahn-Hilliard solutions.

Used locally, once both a trained PINN (weights downloaded from Colab) and a
FEM reference solution (generated via src/fem/cahn_hilliard.py) are
available, to answer RQ3 (accuracy, stability, cost).
"""

from __future__ import annotations

import numpy as np


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Relative L2 error of `prediction` against `reference`.

    Args:
        prediction: array of predicted values (e.g. PINN output on a grid).
        reference: array of reference values (e.g. FEM solution on the same grid).

    Returns:
        ||prediction - reference||_2 / ||reference||_2.
    """
    # TODO: flatten inputs as needed and compute the relative L2 norm
    raise NotImplementedError


def free_energy(c: np.ndarray, grad_c: np.ndarray, epsilon: float) -> float:
    """Ginzburg-Landau free energy E = integral( F(c) + (eps^2/2)|grad c|^2 ) dx.

    Args:
        c: phase-field values on a grid/mesh.
        grad_c: gradient of c on the same grid/mesh.
        epsilon: interface-width parameter.

    Returns:
        Scalar free energy estimate.
    """
    # TODO: implement double-well potential F(c) and quadrature/integration
    raise NotImplementedError


def mass_conservation_error(c_t: np.ndarray, c_0: np.ndarray) -> float:
    """Deviation of total mass (integral of c) at time t from its initial value.

    Args:
        c_t: phase-field values at time t.
        c_0: phase-field values at t=0.

    Returns:
        Absolute difference in total mass between c_t and c_0.
    """
    # TODO: implement mass integral and compare
    raise NotImplementedError
