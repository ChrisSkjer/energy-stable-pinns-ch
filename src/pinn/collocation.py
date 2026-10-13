"""Collocation-point sampling strategies for the Cahn-Hilliard PINN.

Each strategy is a small class with the same two-method interface, so
train.py can switch between them with one flag (--collocation) and the
experiments differ only in where the PDE residual is enforced:

    sampler.sample()             -> (n, 3) initial (x, y, t) points
    sampler.update(model, step)  -> new (n, 3) points, or None to keep the current ones

`update` takes the model even though no strategy here uses it yet: a
residual-based adaptive strategy (e.g. RAD, Wu et al. 2023, "A comprehensive
study of non-adaptive and residual-based adaptive sampling for PINNs") needs
it to evaluate the residual on a candidate pool, and slots in as one more
class in COLLOCATION_STRATEGIES without touching train.py.

Only the interior PDE points live here; the IC and BC points stay on the
fixed grids built in src/pinn/sampling.py.
"""

from __future__ import annotations

import torch
from torch import nn

Bounds = tuple[float, float, float]


def _scale(unit: torch.Tensor, lower: Bounds, upper: Bounds) -> torch.Tensor:
    """Map points from the unit cube [0, 1]^3 onto the box [lower, upper]."""
    lo = torch.tensor(lower, dtype=unit.dtype, device=unit.device)
    hi = torch.tensor(upper, dtype=unit.dtype, device=unit.device)
    return lo + (hi - lo) * unit


def default_grid_shape(n: int) -> tuple[int, int, int]:
    """Equal points per axis, as close to n in total as a cube allows
    (20000 -> 27^3 = 19683)."""
    side = max(1, round(n ** (1 / 3)))
    return side, side, side


class CollocationSampler:
    """Base class: a fixed set of points, never updated."""

    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    def update(self, model: nn.Module, step: int) -> torch.Tensor | None:
        return None


class UniformGridSampler(CollocationSampler):
    """Cell-centred regular grid over (x, y, t).

    Cell centres rather than linspace end points, so no PDE point sits
    exactly on t = 0 or on a wall, where the IC and BC points already are.
    Deterministic, so there is nothing to resample.
    """

    def __init__(
        self,
        lower: Bounds,
        upper: Bounds,
        shape: tuple[int, int, int],
        device: torch.device | str = "cpu",
    ):
        self.lower, self.upper, self.shape, self.device = lower, upper, tuple(shape), device

    def sample(self) -> torch.Tensor:
        axes = [(torch.arange(k, device=self.device, dtype=torch.float32) + 0.5) / k for k in self.shape]
        grid = torch.meshgrid(*axes, indexing="ij")
        unit = torch.stack([g.reshape(-1) for g in grid], dim=1)
        return _scale(unit, self.lower, self.upper)


class RandomSampler(CollocationSampler):
    """Independent uniform random points; redrawn every `resample_every`
    Adam epochs when that is > 0."""

    def __init__(
        self,
        lower: Bounds,
        upper: Bounds,
        n: int,
        seed: int = 0,
        resample_every: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.lower, self.upper, self.n, self.device = lower, upper, n, device
        self.resample_every = resample_every
        # CPU generator so the same seed gives the same points on CPU and GPU.
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self) -> torch.Tensor:
        unit = torch.rand(self.n, 3, generator=self.generator)
        return _scale(unit, self.lower, self.upper).to(self.device)

    def update(self, model: nn.Module, step: int) -> torch.Tensor | None:
        if self.resample_every and step > 0 and step % self.resample_every == 0:
            return self.sample()
        return None


class SobolSampler(CollocationSampler):
    """Scrambled Sobol (quasi-random) points: random-looking but spread more
    evenly than independent uniform draws (lower discrepancy).

    Resampling takes the *next* n points from the same sequence rather than
    re-seeding, so the union of all draws stays low-discrepancy too.
    """

    def __init__(
        self,
        lower: Bounds,
        upper: Bounds,
        n: int,
        seed: int = 0,
        resample_every: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.lower, self.upper, self.n, self.device = lower, upper, n, device
        self.resample_every = resample_every
        self.engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)

    def sample(self) -> torch.Tensor:
        unit = self.engine.draw(self.n, dtype=torch.float32)
        return _scale(unit, self.lower, self.upper).to(self.device)

    def update(self, model: nn.Module, step: int) -> torch.Tensor | None:
        if self.resample_every and step > 0 and step % self.resample_every == 0:
            return self.sample()
        return None


COLLOCATION_STRATEGIES = {
    "grid": UniformGridSampler,
    "random": RandomSampler,
    "sobol": SobolSampler,
}

# Strategies that have something new to draw, i.e. where --resample-every applies.
RESAMPLABLE = frozenset({"random", "sobol"})


def make_sampler(
    name: str,
    lower: Bounds,
    upper: Bounds,
    n: int,
    *,
    seed: int = 0,
    resample_every: int = 0,
    grid_shape: tuple[int, int, int] | None = None,
    device: torch.device | str = "cpu",
) -> CollocationSampler:
    """Build the collocation sampler called `name` (a COLLOCATION_STRATEGIES key).

    Args:
        name: "grid", "random", or "sobol".
        lower, upper: (x, y, t) box bounds.
        n: number of points (random, sobol); for "grid" only used to pick a
            default_grid_shape when grid_shape is None.
        seed: seed for the random/Sobol draw.
        resample_every: redraw every this many Adam epochs (0 = never).
            Not allowed for "grid".
        grid_shape: (NX, NY, NT) for "grid".
        device: device to place the points on.
    """
    if name not in COLLOCATION_STRATEGIES:
        raise ValueError(f"unknown collocation strategy {name!r}; choose from {sorted(COLLOCATION_STRATEGIES)}")
    if name == "grid":
        if resample_every:
            raise ValueError("a uniform grid has nothing to resample; use resample_every=0")
        return UniformGridSampler(lower, upper, grid_shape or default_grid_shape(n), device=device)
    return COLLOCATION_STRATEGIES[name](
        lower, upper, n, seed=seed, resample_every=resample_every, device=device
    )


def heldout_points(lower: Bounds, upper: Bounds, n: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Fixed set of fresh uniform random points for a held-out residual,
    drawn from its own generator so it never coincides with a training draw.
    Returned with requires_grad=True, ready for pde_residual_losses."""
    unit = torch.rand(n, 3, generator=torch.Generator().manual_seed(seed))
    return _scale(unit, lower, upper).to(device).requires_grad_(True)
