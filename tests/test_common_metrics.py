"""Smoke test for src.common.metrics."""


def test_import():
    from src.common import metrics  # noqa: F401

    assert hasattr(metrics, "relative_l2_error")
    assert hasattr(metrics, "free_energy")
    assert hasattr(metrics, "mass_conservation_error")
