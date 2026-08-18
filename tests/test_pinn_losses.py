"""Smoke test for src.pinn.losses."""


def test_import():
    from src.pinn import losses  # noqa: F401

    assert hasattr(losses, "pde_residual_loss")
    assert hasattr(losses, "ic_bc_loss")
    assert hasattr(losses, "energy_stability_loss")
