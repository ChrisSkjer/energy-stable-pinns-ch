"""Self-adaptive point weights (SA-PINN, McClenny & Braga-Neto).

Every training point i of a self-adaptive term gets a trainable lambda_i, and
the term becomes

    L_term = scale * mean_i( m(lambda_i) * squared_i )

The network minimises the total loss; the lambdas *maximise* it (gradient
ascent, a separate optimiser with maximize=True), so points with a large
residual gain weight. dL/dlambda_i = scale * m'(lambda_i) * squared_i / N >= 0,
so the lambdas only grow; the mask m sets how fast and whether they are
bounded:

    polynomial  m(lambda) = lambda^q            (unbounded; the original form)
    sigmoid     m(lambda) = s * sigmoid(lambda)  (bounded by s)

lambda_0 defaults to the value where m(lambda_0) = 1, so a run starts from
the same loss as the plain run and the --*-weight factors keep their meaning.

The lambdas belong to fixed point sets: train.py rejects resampling and
adaptive refinement of a term's points while that term is self-adaptive.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SA_TERMS = ("pde_u", "pde_mu", "ic", "bc")
SA_MASKS = ("polynomial", "sigmoid")


def mask(lam: torch.Tensor, kind: str, q: float = 2.0, scale: float = 10.0) -> torch.Tensor:
    """m(lambda): non-negative, increasing (for lambda >= 0 with polynomial)."""
    if kind == "polynomial":
        return lam**q
    if kind == "sigmoid":
        return scale * torch.sigmoid(lam)
    raise ValueError(f"unknown mask {kind!r}; choose from {SA_MASKS}")


def unit_init(kind: str, q: float = 2.0, scale: float = 10.0) -> float:
    """The lambda with m(lambda) = 1."""
    if kind == "polynomial":
        return 1.0
    if kind == "sigmoid":
        if scale <= 1:
            raise ValueError("sigmoid mask needs scale > 1 to reach m = 1")
        return math.log(1.0 / (scale - 1.0))  # logit(1 / scale)
    raise ValueError(f"unknown mask {kind!r}; choose from {SA_MASKS}")


class SelfAdaptiveWeights(nn.Module):
    """One lambda vector per self-adaptive term.

    Args:
        sizes: term name -> number of training points of that term.
        kind, q, scale: the mask (see `mask`).
        init: lambda_0 for every point; None = `unit_init` (m = 1).
    """

    def __init__(
        self,
        sizes: dict[str, int],
        kind: str = "polynomial",
        q: float = 2.0,
        scale: float = 10.0,
        init: float | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        unknown = set(sizes) - set(SA_TERMS)
        if unknown:
            raise ValueError(f"unknown self-adaptive terms {sorted(unknown)}; choose from {SA_TERMS}")
        self.kind, self.q, self.scale = kind, q, scale
        lam0 = unit_init(kind, q, scale) if init is None else init
        self.lambdas = nn.ParameterDict(
            {name: nn.Parameter(torch.full((n, 1), float(lam0), device=device)) for name, n in sizes.items()}
        )
        # Plain (lambda-free) value of each self-adaptive term from the last
        # `term` call, for logging next to the weighted one.
        self.plain: dict[str, torch.Tensor] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.lambdas

    def weights(self, name: str) -> torch.Tensor | None:
        """(N, 1) m(lambda) for a self-adaptive term, None otherwise."""
        if name not in self.lambdas:
            return None
        return mask(self.lambdas[name], self.kind, self.q, self.scale)

    def term(self, name: str, squared: torch.Tensor, scale: float | torch.Tensor = 1.0) -> torch.Tensor:
        """scale * mean(m(lambda) * squared) for a self-adaptive term (and
        scale * mean(squared) stored in `self.plain[name]`); the plain
        scale * mean(squared) for any other term."""
        plain = scale * torch.mean(squared)
        w = self.weights(name)
        if w is None:
            return plain
        if w.shape[0] != squared.shape[0]:
            raise RuntimeError(
                f"self-adaptive term {name!r} has {w.shape[0]} weights but {squared.shape[0]} points; "
                "its points must stay fixed while it is self-adaptive"
            )
        self.plain[name] = plain.detach()
        return scale * torch.mean(w * squared.reshape(w.shape))

    def freeze(self) -> None:
        """Stop training the lambdas (the L-BFGS phase)."""
        for lam in self.lambdas.values():
            lam.requires_grad_(False)

    def as_dict(self) -> dict[str, torch.Tensor]:
        """term -> m(lambda) on the CPU, for checkpoints."""
        with torch.no_grad():
            return {name: self.weights(name).detach().cpu() for name in self.lambdas}
