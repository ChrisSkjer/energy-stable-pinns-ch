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
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_layers: int = 4,
        hidden_width: int = 64,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        # TODO: build the MLP (input_dim -> [hidden_width] * hidden_layers -> output_dim)
        # TODO: consider Xavier/Glorot init, common for tanh-activated PINNs
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Args:
            x: tensor of shape (N, input_dim) — e.g. columns [x, y, t] or [x, t].

        Returns:
            Tensor of shape (N, output_dim).
        """
        # TODO: forward pass through the MLP
        raise NotImplementedError
