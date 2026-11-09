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


def _hard_ic_pinn(form="add", tau=0.0, ic="noise"):
    import torch

    from src.pinn.ic_torch import InitialConditionField
    from src.pinn.model import PINN

    torch.manual_seed(0)
    return PINN(
        (0.0, 0.0, 0.0), (1.0, 1.0, 0.01), input_dim=3, output_dim=2, hidden_layers=2, hidden_width=8,
        hard_ic=InitialConditionField(2, ic, 0.05, seed=42), hard_ic_tau=tau, hard_ic_form=form,
    )


def test_hard_ic_holds_exactly_at_t0():
    import pytest
    import torch

    for form, tau in (("add", 0.0), ("add", 1e-3), ("blend", 1e-3)):
        net = _hard_ic_pinn(form, tau)
        xy = torch.rand(50, 2)
        pts = torch.cat([xy, torch.zeros(50, 1)], dim=1)
        torch.testing.assert_close(net(pts)[:, 0:1], net.hard_ic(xy), rtol=0, atol=0)
        later = torch.cat([xy, torch.full((50, 1), 0.005)], dim=1)
        assert not torch.allclose(net(later)[:, 0:1], net.hard_ic(xy))
    with pytest.raises(ValueError, match="blend"):
        _hard_ic_pinn("blend", 0.0)


def test_hard_ic_time_factor_and_forms():
    import math

    import torch

    t = torch.tensor([[0.0], [0.0025], [0.01]])
    torch.testing.assert_close(_hard_ic_pinn("add", 0.0).time_factor(t), t / 0.01)
    expected = 1 - torch.exp(-t / 1e-3)
    torch.testing.assert_close(_hard_ic_pinn("add", 1e-3).time_factor(t), expected)

    # u = u0 + g N (add) and (1 - g) u0 + g N (blend), N the raw network output.
    for form in ("add", "blend"):
        net = _hard_ic_pinn(form, 1e-3)
        pts = torch.tensor([[0.3, 0.7, 0.002]])
        raw = net.model(2 * (pts - net.lower_bound) / (net.upper_bound - net.lower_bound) - 1.0)
        u0 = net.hard_ic(pts[:, :2])
        g = 1 - math.exp(-2.0)
        want = u0 + g * raw[:, 0:1] if form == "add" else (1 - g) * u0 + g * raw[:, 0:1]
        out = net(pts)
        torch.testing.assert_close(out[:, 0:1], want)
        torch.testing.assert_close(out[:, 1:2], raw[:, 1:2])  # mu left free


def test_hard_ic_keeps_the_state_dict_and_the_pde_residual_runs():
    import torch

    from src.pinn.losses import bc_loss, pde_residuals
    from src.pinn.model import PINN

    torch.manual_seed(0)
    plain = PINN((0.0, 0.0, 0.0), (1.0, 1.0, 0.01), input_dim=3, output_dim=2, hidden_layers=2, hidden_width=8)
    for ic in ("noise", "cross", "two_circles"):
        net = _hard_ic_pinn(ic=ic)
        assert net.state_dict().keys() == plain.state_dict().keys()
        net.load_state_dict(plain.state_dict())  # weights interchange
        pts = (torch.rand(64, 3) * torch.tensor([1.0, 1.0, 0.01])).requires_grad_(True)
        r_u, r_mu = pde_residuals(net, pts)
        assert torch.isfinite(r_u).all() and torch.isfinite(r_mu).all()
        bc = torch.rand(40, 3).requires_grad_(True)
        assert torch.isfinite(bc_loss(net, bc))
