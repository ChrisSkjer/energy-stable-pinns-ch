"""Smoke test for src.pinn.model."""

import pytest


def test_import():
    from src.pinn import model  # noqa: F401


@pytest.mark.skip(reason="PINN.__init__ is not implemented yet")
def test_forward_shape():
    import torch

    from src.pinn.model import PINN

    net = PINN(input_dim=3, output_dim=1, hidden_layers=2, hidden_width=8)
    x = torch.zeros(5, 3)
    y = net(x)
    assert y.shape == (5, 1)
