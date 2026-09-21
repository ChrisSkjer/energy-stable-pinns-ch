"""Tests for src.pinn.train's L-BFGS early-stopping predicate."""

import math

from src.pinn.train import is_improvement


def test_first_step_always_improves():
    assert is_improvement(1.0, None, tol=1e-6)


def test_improvement_must_clear_the_relative_threshold():
    # 1% below best, against a 0.1% threshold.
    assert is_improvement(0.99, 1.0, tol=1e-3)
    # 0.01% below best -- real but under the threshold, so it reads as stalled.
    assert not is_improvement(0.9999, 1.0, tol=1e-3)


def test_threshold_scales_with_the_current_loss():
    # The same absolute step (1e-5) clears the threshold at a loss of 1e-3
    # but not at a loss of 1.0 -- the point of a relative tolerance.
    assert is_improvement(1e-3 - 1e-5, 1e-3, tol=1e-3)
    assert not is_improvement(1.0 - 1e-5, 1.0, tol=1e-3)


def test_worse_or_equal_is_not_an_improvement():
    assert not is_improvement(1.0, 1.0, tol=1e-6)
    assert not is_improvement(2.0, 1.0, tol=1e-6)


def test_nan_is_not_an_improvement():
    # A diverged run must count as stalled, not as endless improvement.
    assert not is_improvement(math.nan, 1.0, tol=1e-6)
