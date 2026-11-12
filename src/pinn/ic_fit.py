"""A smooth, frozen MLP fit of u0 for the hard IC (--hard-ic-fit).

With the exact field (src/pinn/ic_torch.py) the hard IC u = u0 + g(t) N
carries u0's regularity to every t. The SDF-based ICs are not smooth enough
for that: outside a convex corner of the cross the SDF is the distance to the
corner point, so Lap(u0) ~ 1/r there, and the mu residual can never vanish.
A tanh MLP fitted to u0 is smooth by construction. It is fitted once, on its
own (a plain 2D regression, no PDE terms), and then frozen, so
u(x, 0) = M0(x) holds exactly and M0 differs from u0 only by its fitting error.

The weights are buffers, not parameters, so the main optimisers, the
parameter count and the layer statistics see only the PINN's own weights.
They are persistent, so they are saved in the PINN's state_dict under
"hard_ic.*" and evaluate.load_model restores them.

Fits are cached in ic_fits/ at the repo root (tracked in git, so a fresh
Colab clone has them too), one file per IC and fit settings (cache_key), so
a fit is done once and reused by every run with the same settings.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn


class FittedICField(nn.Module):
    """u0 as a frozen tanh MLP: forward((N, n_space) spatial coordinates) -> (N, 1).

    Inputs are mapped from [lower, upper] to [-1, 1], as in the PINN.
    A fresh instance has zero weights (u0 = 0) until fit_ic_field() fills
    them, or load_state_dict() restores them from a checkpoint; building
    it uses no random numbers.
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, hidden_layers: int = 4, hidden_width: int = 100) -> None:
        super().__init__()
        lower = torch.as_tensor(lower, dtype=torch.float32)
        upper = torch.as_tensor(upper, dtype=torch.float32)
        self.n_space = lower.numel()
        self.hidden_layers = hidden_layers
        self.hidden_width = hidden_width
        # Derived from the run args, so not part of the state_dict.
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        sizes = [self.n_space] + [hidden_width] * hidden_layers + [1]
        self.n_linear = len(sizes) - 1
        for k, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.register_buffer(f"weight{k}", torch.zeros(n_out, n_in))
            self.register_buffer(f"bias{k}", torch.zeros(n_out))

    def weights(self) -> list[torch.Tensor]:
        """[weight0, bias0, weight1, bias1, ...]"""
        return [getattr(self, f"{kind}{k}") for k in range(self.n_linear) for kind in ("weight", "bias")]

    def forward(self, x_space: torch.Tensor, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
        weights = self.weights() if weights is None else weights
        h = 2 * (x_space - self.lower) / (self.upper - self.lower) - 1.0
        for k in range(self.n_linear):
            h = h @ weights[2 * k].T + weights[2 * k + 1]
            if k < self.n_linear - 1:
                h = torch.tanh(h)
        return h

    def extra_repr(self) -> str:
        return f"n_space={self.n_space}, hidden_layers={self.hidden_layers}, hidden_width={self.hidden_width} (frozen)"


# Relative to the repo root, like results/pinn_models (train.py runs from there).
IC_FIT_CACHE_DIR = "ic_fits"


def cache_key(train_args: dict) -> dict:
    """Everything that determines a run's fitted M0: the IC and the fit settings."""
    return {
        "ic": train_args["ic"],
        "dim": train_args.get("dim", 2),
        "epsilon": float(train_args["epsilon"]),
        "ic_seed": train_args.get("ic_seed", 0),
        "x_max": float(train_args["x_max"]),
        "y_max": float(train_args["y_max"]) if train_args.get("dim", 2) == 2 else None,
        "layers": train_args["hard_ic_fit_layers"],
        "width": train_args["hard_ic_fit_width"],
        "points": train_args["hard_ic_fit_points"],
        "epochs": train_args["hard_ic_fit_epochs"],
        "lbfgs_steps": train_args["hard_ic_fit_lbfgs_steps"],
        "lr": float(train_args["lr"]),
        "seed": train_args["weight_seed"],
    }


def cache_path(cache_dir: str | Path, key: dict) -> Path:
    """ic_fits/<ic>_dim2_eps0.05_icseed42_4x100_n40000_e5000_lb500_lr0.001_seed0.pt
    (domain sizes only when not the unit square)."""
    name = f"{key['ic']}_dim{key['dim']}_eps{key['epsilon']:g}_icseed{key['ic_seed']}"
    if key["x_max"] != 1.0 or key["y_max"] not in (1.0, None):
        name += f"_x{key['x_max']:g}" + (f"_y{key['y_max']:g}" if key["y_max"] is not None else "")
    name += (f"_{key['layers']}x{key['width']}_n{key['points']}_e{key['epochs']}"
             f"_lb{key['lbfgs_steps']}_lr{key['lr']:g}_seed{key['seed']}.pt")
    return Path(cache_dir) / name


def load_or_fit_ic_field(
    field: FittedICField, target: nn.Module, key: dict, cache_dir: str | Path, refit: bool = False, **fit_kwargs
) -> dict:
    """Fill `field` from the cached fit for `key` in cache_dir if there is one
    (and refit is False), else fit it with fit_ic_field(field, target, **fit_kwargs)
    and save it there. Returns the fit statistics, with "cache" (the file) and
    "cached" (True if loaded)."""
    path = cache_path(cache_dir, key)
    if path.exists() and not refit:
        saved = torch.load(path, map_location=field.lower.device, weights_only=False)
        if saved["key"] != key:
            raise ValueError(f"{path} was fitted with {saved['key']}, not {key}")
        field.load_state_dict(saved["state_dict"])
        print(f"[ic fit] loaded {path} (rel L2 {saved['stats']['rel_l2']:.2e}, "
              f"eps^2 Lap max {saved['stats']['eps2_lap_fit_max']:.3g})")
        return {**saved["stats"], "cache": str(path), "cached": True}
    stats = fit_ic_field(field, target, **fit_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.cpu() for name, value in field.state_dict().items()}
    torch.save({"key": key, "state_dict": state, "stats": stats}, path)
    print(f"[ic fit] saved {path}")
    return {**stats, "cache": str(path), "cached": False}


def _sobol_points(n: int, lower: torch.Tensor, upper: torch.Tensor, seed: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(lower.numel(), scramble=True, seed=seed)
    return lower + (upper - lower) * engine.draw(n).to(lower.device)


def laplacian_of(field: nn.Module, x_space: torch.Tensor) -> torch.Tensor:
    """Lap field(x) per point, (N,), by autograd."""
    x = x_space.detach().clone().requires_grad_(True)
    u = field(x)
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return sum(torch.autograd.grad(grad[:, k].sum(), x, retain_graph=True)[0][:, k] for k in range(x.shape[1]))


def fit_ic_field(
    field: FittedICField,
    target: nn.Module,
    epsilon: float,
    n_points: int = 40000,
    adam_epochs: int = 5000,
    lbfgs_steps: int = 500,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 1000,
) -> dict:
    """Fit `field` to `target` (the exact u0, e.g. an InitialConditionField)
    by full-batch Adam and then L-BFGS on the normalised MSE
    mean((M0 - u0)^2) / mean(u0^2) over n_points scrambled Sobol points, and
    store the result in field's buffers.

    The weights are initialised (Xavier normal, zero biases) from a private
    generator, so the global RNG, and with it the PINN's own initial weights,
    are untouched.

    Returns statistics on 20000 held-out Sobol points: rel_l2 (relative L2
    error of M0 vs u0), max_abs_err, and eps^2 Lap M0 vs eps^2 Lap u0
    (99th percentile and max of the absolute value), plus the final training
    loss and the fitting time.
    """
    start = time.perf_counter()
    lower, upper = field.lower, field.upper
    device = lower.device
    gen = torch.Generator().manual_seed(seed)
    weights = []
    for w in field.weights():
        if w.dim() == 2:
            n_out, n_in = w.shape
            value = torch.randn(n_out, n_in, generator=gen) * math.sqrt(2.0 / (n_in + n_out))
        else:
            value = torch.zeros_like(w, device="cpu")
        weights.append(value.to(device).requires_grad_(True))

    x = _sobol_points(n_points, lower, upper, seed)
    with torch.no_grad():
        y = target(x)
    y_norm = torch.mean(y**2)

    def loss_fn() -> torch.Tensor:
        return torch.mean((field(x, weights) - y) ** 2) / y_norm

    adam = torch.optim.Adam(weights, lr=lr)
    for epoch in range(adam_epochs):
        adam.zero_grad()
        loss = loss_fn()
        loss.backward()
        adam.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[ic fit] adam {epoch + 1:5d}/{adam_epochs} loss {loss.item():.3e}")

    if lbfgs_steps:
        lbfgs = torch.optim.LBFGS(weights, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        for step in range(lbfgs_steps):
            loss = lbfgs.step(closure)
            if log_every and (step + 1) % max(1, log_every // 10) == 0:
                print(f"[ic fit] lbfgs {step + 1:4d}/{lbfgs_steps} loss {loss.item():.3e}")

    with torch.no_grad():
        for buffer, value in zip(field.weights(), weights):
            buffer.copy_(value)
        final_loss = loss_fn().item()

    x_held = _sobol_points(20000, lower, upper, seed + 1)
    with torch.no_grad():
        u_fit, u_exact = field(x_held), target(x_held)
        err = u_fit - u_exact
        rel_l2 = (torch.linalg.norm(err) / torch.linalg.norm(u_exact)).item()
        max_abs_err = err.abs().max().item()
    lap_fit = (epsilon**2 * laplacian_of(field, x_held)).abs().detach()
    lap_exact = (epsilon**2 * laplacian_of(target, x_held)).abs().detach()
    stats = {
        "final_loss": final_loss,
        "rel_l2": rel_l2,
        "max_abs_err": max_abs_err,
        "eps2_lap_fit_p99": torch.quantile(lap_fit, 0.99).item(),
        "eps2_lap_fit_max": lap_fit.max().item(),
        "eps2_lap_exact_p99": torch.quantile(lap_exact, 0.99).item(),
        "eps2_lap_exact_max": lap_exact.max().item(),
        "seconds": time.perf_counter() - start,
    }
    print(
        f"[ic fit] rel L2 {rel_l2:.2e}, max |err| {max_abs_err:.2e}; "
        f"|eps^2 Lap|: fit p99 {stats['eps2_lap_fit_p99']:.3g} max {stats['eps2_lap_fit_max']:.3g}, "
        f"exact p99 {stats['eps2_lap_exact_p99']:.3g} max {stats['eps2_lap_exact_max']:.3g} "
        f"({stats['seconds']:.1f} s)"
    )
    return stats
