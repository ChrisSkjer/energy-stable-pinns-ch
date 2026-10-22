"""Per-layer training statistics: is every layer of the PINN actually learning?

Three complementary views, all per nn.Linear layer (input layer first, output
layer last):

- Gradient norms, split by loss term. ||dL_term/dW_l||_2 for each term
  (pde_u, pde_mu, ic, bc, ...) and for the total. Vanishing gradients show up
  as the first layers' norms sitting orders of magnitude below the last
  layers'; an unbalanced loss shows up as one term's gradient dominating the
  others in every layer (Wang, Teng & Perdikaris 2021, "Understanding and
  mitigating gradient pathologies in PINNs").
- Update ratio ||W_after - W_before|| / ||W_before|| of one optimizer step.
  This is what the layer actually moves by -- Adam rescales gradients, so a
  tiny gradient doesn't by itself mean a frozen layer. Around 1e-3 per Adam
  step is the usual healthy range; 1e-6 and below means the layer has
  effectively stopped learning.
- Activation statistics of each hidden layer's tanh output: standard
  deviation, and the fraction of units saturated (|a| > SATURATION_LEVEL,
  where tanh' < 0.02). Saturated units pass back almost no gradient, which
  is the usual mechanical cause of vanishing gradients in a tanh net.

Only the weight matrices are measured, not the biases: they hold nearly all
parameters, and their norms are what the update-ratio rule of thumb refers to.
"""

from __future__ import annotations

import torch
from torch import nn

# |tanh(z)| above this means tanh'(z) = 1 - a^2 < 0.02: the unit passes back
# under 2% of the gradient it receives.
SATURATION_LEVEL = 0.99


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    """The model's nn.Linear layers in forward order (input layer first)."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def layer_names(model: nn.Module) -> list[str]:
    """Short labels for linear_layers: "in", "h1".."h{n-1}", "out"."""
    n = len(linear_layers(model))
    return ["in"] + [f"h{i}" for i in range(1, n - 1)] + ["out"]


def per_term_gradient_norms(
    model: nn.Module, losses: dict[str, torch.Tensor]
) -> dict[str, list[float]]:
    """||d term / d W_l||_2 for each loss term and each linear layer.

    Args:
        model: the PINN.
        losses: train.py's compute_loss output -- weighted loss terms plus
            "total". The graph must still be alive (call this before
            losses["total"].backward()); it is retained for that backward.

    Returns:
        term name -> per-layer L2 norms, input layer first. Includes "total".
    """
    weights = [layer.weight for layer in linear_layers(model)]
    norms = {}
    for name, value in losses.items():
        grads = torch.autograd.grad(value, weights, retain_graph=True, allow_unused=True)
        norms[name] = [0.0 if g is None else g.norm().item() for g in grads]
    return norms


def per_term_gradients(model: nn.Module, losses: dict[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    """Like per_term_gradient_norms but returns the flattened gradient
    entries themselves, for histograms (see scripts/inspect_gradients.py).
    Skips "total"."""
    weights = [layer.weight for layer in linear_layers(model)]
    out = {}
    for name, value in losses.items():
        if name == "total":
            continue
        grads = torch.autograd.grad(value, weights, retain_graph=True, allow_unused=True)
        out[name] = [
            torch.zeros(w.numel()) if g is None else g.detach().reshape(-1).cpu()
            for g, w in zip(grads, weights)
        ]
    return out


def weight_snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Detached copies of each linear layer's weight, for update_ratios."""
    return [layer.weight.detach().clone() for layer in linear_layers(model)]


def update_ratios(model: nn.Module, before: list[torch.Tensor]) -> list[float]:
    """||W_now - W_before|| / ||W_before|| per linear layer."""
    return [
        ((layer.weight.detach() - w0).norm() / w0.norm().clamp_min(1e-30)).item()
        for layer, w0 in zip(linear_layers(model), before)
    ]


def activation_stats(model: nn.Module, x: torch.Tensor) -> dict[str, list[float]]:
    """Std and saturated fraction of each hidden activation layer's output.

    Args:
        model: the PINN (activations are the non-Linear modules of
            model.model, one after each hidden Linear).
        x: (N, input_dim) points to run the forward pass on.

    Returns:
        {"std": [...], "saturation": [...]}, one entry per hidden layer.
    """
    outputs: list[torch.Tensor] = []
    hooks = [
        m.register_forward_hook(lambda _m, _i, out: outputs.append(out.detach()))
        for m in model.model
        if not isinstance(m, nn.Linear)
    ]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    return {
        "std": [a.std().item() for a in outputs],
        "saturation": [(a.abs() > SATURATION_LEVEL).float().mean().item() for a in outputs],
    }


class LayerStatsLog:
    """Accumulates layer statistics over training, for the checkpoint's
    "layer_stats" entry (a plain dict of lists -- see as_dict)."""

    def __init__(self, model: nn.Module) -> None:
        self.layers = layer_names(model)
        self.step: list[int] = []
        self.phase: list[str] = []
        self.grad_norm: dict[str, list[list[float]]] = {}
        self.weight_norm: list[list[float]] = []
        self.update_ratio: list[list[float]] = []
        self.act_std: list[list[float]] = []
        self.act_saturation: list[list[float]] = []

    def record(
        self,
        step: int,
        phase: str,
        model: nn.Module,
        losses: dict[str, torch.Tensor],
        x_activations: torch.Tensor,
    ) -> None:
        """Gradient, weight and activation stats at the current weights.
        Call update ratios in separately via add_update_ratio once the
        optimizer step has been taken."""
        self.step.append(step)
        self.phase.append(phase)
        for name, norms in per_term_gradient_norms(model, losses).items():
            self.grad_norm.setdefault(name, []).append(norms)
        self.weight_norm.append([layer.weight.norm().item() for layer in linear_layers(model)])
        acts = activation_stats(model, x_activations)
        self.act_std.append(acts["std"])
        self.act_saturation.append(acts["saturation"])

    def add_update_ratio(self, ratios: list[float]) -> None:
        self.update_ratio.append(ratios)

    def as_dict(self) -> dict:
        return {
            "layers": self.layers,
            "step": self.step,
            "phase": self.phase,
            "grad_norm": self.grad_norm,
            "weight_norm": self.weight_norm,
            "update_ratio": self.update_ratio,
            "act_std": self.act_std,
            "act_saturation": self.act_saturation,
        }
