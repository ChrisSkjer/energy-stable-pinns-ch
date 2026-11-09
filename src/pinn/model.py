"""Fully-connected PINN model for the Cahn-Hilliard equation.

Maps spatiotemporal coordinates (x, t) or (x, y, t) to the phase-field
variable c (and, depending on formulation, the chemical potential mu).
"""

from __future__ import annotations

import torch
from torch import nn


HARD_IC_FORMS = ("add", "blend")


class PINN(nn.Module):
    """Configurable fully-connected network for PDE solution approximation.

    With `hard_ic` (an InitialConditionField, src/pinn/ic_torch.py) the u
    output (column 0) satisfies the initial condition exactly by construction
    (--hard-ic), with N the network's raw u output:

        form "add":   u = u0(x) + g(t) * N(x, t)
        form "blend": u = u0(x) + g(t) * (N(x, t) - u0(x)) = (1 - g) u0 + g N

        g(t) = t / t_span             for hard_ic_tau = 0
        g(t) = 1 - exp(-t / tau)      for hard_ic_tau = tau > 0

    with t measured from the lower time bound and t_span the time interval,
    so g(0) = 0. "blend" lets kinks of u0 (min/abs in the SDF-based ICs)
    fade out like exp(-t/tau) instead of staying in u for all t; it needs
    tau > 0. mu (column 1) is left free. The field holds no persistent
    tensors, so the state_dict is the same with or without it.
    """

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_layers: int = 5,
        hidden_width: int = 100,
        activation: type[nn.Module] = nn.Tanh,
        hard_ic: nn.Module | None = None,
        hard_ic_tau: float = 0.0,
        hard_ic_form: str = "add",
    ) -> None:
        super().__init__()
        if hard_ic_form not in HARD_IC_FORMS:
            raise ValueError(f"hard_ic_form must be one of {HARD_IC_FORMS}, not {hard_ic_form!r}")
        if hard_ic_tau < 0:
            raise ValueError("hard_ic_tau must be >= 0")
        if hard_ic is not None and hard_ic_form == "blend" and hard_ic_tau == 0:
            raise ValueError("hard_ic_form 'blend' needs hard_ic_tau > 0")
        self.hard_ic = hard_ic
        self.hard_ic_tau = float(hard_ic_tau)
        self.hard_ic_form = hard_ic_form
        self.register_buffer("lower_bound", torch.as_tensor(lower_bound, dtype=torch.float32))
        self.register_buffer("upper_bound", torch.as_tensor(upper_bound, dtype=torch.float32))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.hidden_width = hidden_width

        layers = []
        layers.append(nn.Linear(input_dim, hidden_width))
        layers.append(activation())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation())
        layers.append(nn.Linear(hidden_width, output_dim))
        self.model = nn.Sequential(*layers)

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            x: tensor of shape (N, input_dim) — e.g. columns [x, y, t] or [x, t].

        Returns:
            Tensor of shape (N, output_dim).
        """
        x_normalized = 2 * (x - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        out = self.model(x_normalized)
        if self.hard_ic is None:
            return out
        t = x[:, -1:] - self.lower_bound[-1]
        u0 = self.hard_ic(x[:, :-1])
        n_u = out[:, 0:1]
        g = self.time_factor(t)
        if self.hard_ic_form == "add":
            u = u0 + g * n_u
        else:
            u = u0 + g * (n_u - u0)
        return torch.cat([u, out[:, 1:]], dim=1)

    def time_factor(self, t: torch.Tensor) -> torch.Tensor:
        """g(t) of the hard IC, t measured from the lower time bound."""
        if self.hard_ic_tau == 0:
            return t / (self.upper_bound[-1] - self.lower_bound[-1])
        return -torch.expm1(-t / self.hard_ic_tau)
