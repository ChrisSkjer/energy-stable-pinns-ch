"""Tests for src.pinn.model."""


def test_import():
    from src.pinn import model  # noqa: F401


def test_forward_shape_and_input_normalisation():
    import torch

    from src.pinn.model import PINN

    net = PINN(
        lower_bound=(0.0, 0.0, 0.0), upper_bound=(1.0, 2.0, 0.01), input_dim=3, output_dim=2,
        hidden_layers=2, hidden_width=8,
    )
    assert net(torch.zeros(5, 3)).shape == (5, 2)

    # The first layer sees (x, y, t) scaled to [-1, 1]: the domain's corners
    # and centre map to the cube's.
    seen = []
    net.model[0].register_forward_hook(lambda module, inputs, output: seen.append(inputs[0]))
    net(torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.01], [0.5, 1.0, 0.005]]))
    torch.testing.assert_close(
        seen[0], torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    )
