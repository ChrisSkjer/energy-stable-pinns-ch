"""Fully-connected PINN model for the Cahn-Hilliard equation.

Maps spatiotemporal coordinates (x, t) or (x, y, t) to the phase-field
variable c (and, depending on formulation, the chemical potential mu).
"""

from __future__ import annotations

import torch
from torch import nn


class PINN(nn.Module):
    """Configurable fully-connected network for PDE solution approximation.

    TODO: decide on output dimension (c only, vs. [c, mu] for the mixed
    formulation) and wire that into `output_dim`.
    """

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        # TODO: build the MLP (input_dim -> [hidden_width] * hidden_layers -> output_dim)
        # TODO: consider Xavier/Glorot init, common for tanh-activated PINNs
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
        return self.model(x_normalized)
