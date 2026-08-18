"""Smoke test for src.fem.cahn_hilliard.

NOTE: importing this module requires dolfinx (the fenicsx-env conda
environment). Skipped automatically if dolfinx is not importable, e.g. when
running the PINN-only test suite on Colab.
"""

import pytest


def test_import():
    pytest.importorskip("dolfinx")
    from src.fem import cahn_hilliard  # noqa: F401

    assert hasattr(cahn_hilliard, "CahnHilliardConfig")
    assert hasattr(cahn_hilliard, "solve")
