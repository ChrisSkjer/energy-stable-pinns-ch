"""Differentiable (torch) initial conditions u0(x), for the hard IC constraint.

`--hard-ic` builds u(x, t) = u0(x) + g(t) * N(x, t) inside the PINN (see
src/pinn/model.py), so the PDE residual needs u0 and its derivatives by
autograd. src/common/initial_conditions.py stays numpy-only (the FEM env has
no torch) and remains the single source of truth for the IC parameters: this
module imports its constants and noise coefficients and only repeats the
formulas. tests/test_pinn_ic_torch.py checks both evaluate the same field.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from src.common import initial_conditions as ics


def _tanh_profile(sdf: torch.Tensor, epsilon: float) -> torch.Tensor:
    return -torch.tanh(sdf / (math.sqrt(2.0) * epsilon))


def _rect_sdf(px: torch.Tensor, py: torch.Tensor, half_w: float, half_h: float) -> torch.Tensor:
    qx = torch.abs(px) - half_w
    qy = torch.abs(py) - half_h
    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2 + 1e-30)
    # Where both q < 0 (inside) `outside` is ~1e-15 rather than 0; the 1e-30
    # only keeps sqrt's gradient finite there.
    inside = torch.clamp(torch.maximum(qx, qy), max=0.0)
    return outside + inside


def _cross(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    px = x[:, 0:1] - ics._CROSS_CENTER[0]
    py = x[:, 1:2] - ics._CROSS_CENTER[1]
    horizontal = _rect_sdf(px, py, ics._CROSS_ARM_HALF_L, ics._CROSS_ARM_HALF_W)
    vertical = _rect_sdf(px, py, ics._CROSS_ARM_HALF_W, ics._CROSS_ARM_HALF_L)
    return _tanh_profile(torch.minimum(horizontal, vertical), epsilon)


def _two_circles(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    # sqrt(r^2 + tiny) so the gradient at an exact circle centre is finite.
    sdfs = [
        torch.sqrt((x[:, 0:1] - cx) ** 2 + (x[:, 1:2] - cy) ** 2 + 1e-30) - r
        for (cx, cy), r in ics._CIRCLES
    ]
    sdf = sdfs[0]
    for other in sdfs[1:]:
        sdf = torch.minimum(sdf, other)
    return _tanh_profile(sdf, epsilon)


def _noise(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    a = coeffs["a"]
    modes = coeffs["modes"]
    cos_x = torch.cos(math.pi * x[:, 0:1] * modes)  # (N, n)
    cos_y = torch.cos(math.pi * x[:, 1:2] * modes)  # (N, n)
    field = ((cos_x @ a) * cos_y).sum(dim=1, keepdim=True)
    return ics._NOISE_MEAN + ics._NOISE_AMPLITUDE_2D * field


def _step(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    return -_tanh_profile(x[:, 0:1] - 0.5, epsilon)


def _two_intervals(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    sdfs = [torch.abs(x[:, 0:1] - c) - half_w for c, half_w in ics._INTERVALS]
    sdf = sdfs[0]
    for other in sdfs[1:]:
        sdf = torch.minimum(sdf, other)
    return _tanh_profile(sdf, epsilon)


def _noise_1d(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    field = torch.cos(math.pi * x[:, 0:1] * coeffs["modes"]) @ coeffs["a"].unsqueeze(1)
    return ics._NOISE_MEAN + ics._NOISE_AMPLITUDE * field


def _cosine(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    return ics.COSINE_AMPLITUDE * torch.cos(math.pi * ics.COSINE_MODE * x[:, 0:1])


def _cosine_large(x: torch.Tensor, epsilon: float, coeffs: dict) -> torch.Tensor:
    return -torch.cos(4.0 * math.pi * x[:, 0:1])


_Field = Callable[[torch.Tensor, float, dict], torch.Tensor]

TORCH_INITIAL_CONDITIONS: dict[str, _Field] = {
    "cross": _cross,
    "two_circles": _two_circles,
    "noise": _noise,
}

TORCH_INITIAL_CONDITIONS_1D: dict[str, _Field] = {
    "step": _step,
    "two_intervals": _two_intervals,
    "noise": _noise_1d,
    "cosine": _cosine,
    "cosine_large": _cosine_large,
}


class InitialConditionField(nn.Module):
    """u0 as a module: forward((N, n_space) spatial coordinates) -> (N, 1).

    Its tensors (noise coefficients) are non-persistent buffers: they follow
    .to(device) but stay out of the state_dict, so a PINN carrying this field
    saves exactly the same keys as one without it.

    Args:
        dim: 1 or 2 space dimensions.
        name: a key of INITIAL_CONDITIONS (dim 2) / INITIAL_CONDITIONS_1D (dim 1).
        epsilon: interface half-width (as for the numpy ICs).
        seed: IC seed, used only by "noise".
    """

    def __init__(self, dim: int, name: str, epsilon: float, seed: int = 0) -> None:
        super().__init__()
        registry = TORCH_INITIAL_CONDITIONS if dim == 2 else TORCH_INITIAL_CONDITIONS_1D
        if name not in registry:
            raise ValueError(f"unknown {dim}D initial condition {name!r}; choose from {sorted(registry)}")
        self.dim = dim
        self.name = name
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self._fn = registry[name]
        self._coeff_names: list[str] = []
        if name == "noise":
            if dim == 2:
                a = ics.noise_coefficients(seed)
                modes = torch.arange(a.shape[0], dtype=torch.float32)
            else:
                a = ics.noise_1d_coefficients(seed)
                modes = torch.arange(1, a.shape[0] + 1, dtype=torch.float32)
            self._add_buffer("a", torch.as_tensor(a, dtype=torch.float32))
            self._add_buffer("modes", modes)

    def _add_buffer(self, name: str, value: torch.Tensor) -> None:
        self.register_buffer(f"coeff_{name}", value, persistent=False)
        self._coeff_names.append(name)

    def forward(self, x_space: torch.Tensor) -> torch.Tensor:
        coeffs = {name: getattr(self, f"coeff_{name}").to(x_space.dtype) for name in self._coeff_names}
        return self._fn(x_space, self.epsilon, coeffs)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, name={self.name!r}, epsilon={self.epsilon}, seed={self.seed}"


def hard_ic_kwargs(train_args: dict) -> dict:
    """PINN(...) keyword arguments for the hard IC from a run's args
    (vars(argparse.Namespace), or a checkpoint's "args"). Empty for runs
    without --hard-ic, including checkpoints from before the flag existed,
    so train.py and evaluate.load_model build the same model."""
    if not train_args.get("hard_ic", False):
        return {}
    field = InitialConditionField(
        train_args.get("dim", 2), train_args["ic"], train_args["epsilon"], train_args.get("ic_seed", 0)
    )
    return {
        "hard_ic": field,
        "hard_ic_tau": train_args.get("hard_ic_tau", 0.0),
        "hard_ic_form": train_args.get("hard_ic_form", "add"),
    }
