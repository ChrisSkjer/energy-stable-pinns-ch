"""Physical diagnostics (free energy, total mass) for a trained PINN checkpoint.

Unlike src/pinn/evaluate.py -- which is inference-only, explicitly wrapped in
torch.no_grad() -- this module needs d u / d(x, y) from autograd, because the
free energy's interfacial term is (eps^2/2)|grad u|^2 and at a small epsilon
(e.g. the default 0.01) finite differences on an evaluation-sized grid
under-report it by several percent. That is also why this is a separate CLI
step rather than a flag on evaluate.py: diagnostics wants a much denser grid
(400x400 by default) and many more time samples (101) than a handful of
field snapshots meant for pictures, so sharing one command would force every
cheap re-plot to pay a ~20s autograd pass or force diagnostics onto too
coarse a grid.

Writes diagnostics.csv with exactly the same three columns as
src/fem/cahn_hilliard.py's _DiagnosticsLogger (t, free_energy, total_mass),
so src/common/plotting.py::load_diagnostics reads both a FEM and a PINN run
with the same code.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from src.common.metrics import free_energy, total_mass, trapezoid_weights_2d
from src.pinn.evaluate import load_model
from src.pinn.losses import DEFAULT_EPSILON
from src.pinn.model import PINN
from src.pinn.run_paths import infer_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute free-energy and mass diagnostics for a trained Cahn-Hilliard PINN"
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--nx", type=int, default=400, help="Grid resolution in x (see module docstring for why "
        "this default is much denser than evaluate.py's)"
    )
    parser.add_argument("--ny", type=int, default=400, help="Grid resolution in y")
    parser.add_argument(
        "--nt", type=int, default=101, help="Number of time samples across [0, t-max]"
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Upper time bound to sample up to. Defaults to the checkpoint's own "
        "t_max (from training) -- never extrapolate past that.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Interface-width parameter for the energy functional. Defaults to "
        "the checkpoint's own --epsilon. Passing a value that doesn't match "
        "what the model was trained with silently invalidates the result.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' locally, 'cuda' on Colab.")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Grid points per autograd pass")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="diagnostics.csv path to write. Defaults to diagnostics.csv inside "
        "the checkpoint's own run folder (results/pinn_models/<run-name>/).",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(infer_run_dir(args.checkpoint_path), "diagnostics.csv")
    return args


def field_and_gradient(
    model: PINN,
    t: float,
    nx: int,
    ny: int,
    device: torch.device | str = "cpu",
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """u and its spatial gradient on an (nx, ny) grid at fixed t, via autograd.

    Args:
        model: a loaded PINN (see evaluate.load_model).
        t: the fixed time to evaluate at.
        nx, ny: grid resolution. Same grid convention as
            evaluate.evaluate_on_grid (np.linspace over the model's own
            lower_bound/upper_bound), so both tools agree on point locations.
        device: device to run the forward pass on.
        chunk_size: grid points per autograd call, to bound peak memory --
            the integral is a plain weighted sum, so chunking is exact.

    Returns:
        (u, grad_u): float64 numpy arrays of shape (nx, ny) and (nx, ny, 2).
        grad_u's last axis is (du/dx, du/dy) -- the du/dt column is dropped.

    Do NOT wrap this in torch.no_grad(), and do not rely on model.eval() to
    disable gradients -- it does not. requires_grad_(True) on the *input*
    tensor is what makes torch.autograd.grad work here.
    """
    x_lo, y_lo, _ = model.lower_bound.tolist()
    x_hi, y_hi, _ = model.upper_bound.tolist()
    X, Y = np.meshgrid(
        np.linspace(x_lo, x_hi, nx), np.linspace(y_lo, y_hi, ny), indexing="ij"
    )
    points = np.stack([X.reshape(-1), Y.reshape(-1), np.full(X.size, t)], axis=1)

    u_flat = np.empty(X.size, dtype=np.float64)
    grad_flat = np.empty((X.size, 2), dtype=np.float64)
    for start in range(0, X.size, chunk_size):
        block = torch.as_tensor(
            points[start : start + chunk_size], dtype=torch.float32, device=device
        ).requires_grad_(True)
        u = model(block)[:, 0:1]
        grad = torch.autograd.grad(u, block, grad_outputs=torch.ones_like(u))[0][:, :2]
        u_flat[start : start + chunk_size] = u.detach().reshape(-1).cpu().numpy()
        grad_flat[start : start + chunk_size] = grad.detach().cpu().numpy()

    return u_flat.reshape(nx, ny), grad_flat.reshape(nx, ny, 2)


def diagnostics_over_time(
    model: PINN,
    t_values: np.ndarray,
    nx: int,
    ny: int,
    epsilon: float,
    device: torch.device | str = "cpu",
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Free energy and total mass at each of t_values, via field_and_gradient.

    Returns:
        (t, free_energy, total_mass) arrays, each shape (len(t_values),).
    """
    t_values = np.asarray(t_values, dtype=np.float64)
    x_lo, y_lo, _ = model.lower_bound.tolist()
    x_hi, y_hi, _ = model.upper_bound.tolist()
    weight = trapezoid_weights_2d(nx, ny, (x_lo, x_hi), (y_lo, y_hi))

    energy = np.empty(t_values.size)
    mass = np.empty(t_values.size)
    for k, t in enumerate(t_values):
        u, grad_u = field_and_gradient(model, float(t), nx, ny, device, chunk_size)
        energy[k] = free_energy(u, grad_u, epsilon, weight)
        mass[k] = total_mass(u, weight)

    return t_values, energy, mass


def save_diagnostics(path: str, t: np.ndarray, energy: np.ndarray, mass: np.ndarray) -> None:
    """Write t,free_energy,total_mass -- byte-format-identical to
    src/fem/cahn_hilliard.py's _DiagnosticsLogger, so the two are
    interchangeable to src/common/plotting.py::load_diagnostics.

    Exactly three columns, always -- adding a fourth breaks load_diagnostics'
    three-value unpack. Written with newline="\\n" so this file (written on
    Windows) has the same line endings as a FEM run's CSV (written in WSL).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("t,free_energy,total_mass\n")
        for t_i, e_i, m_i in zip(t, energy, mass):
            fh.write(f"{t_i:.16e},{e_i:.16e},{m_i:.16e}\n")


if __name__ == "__main__":
    cli_args = parse_args()

    checkpoint = torch.load(cli_args.checkpoint_path, map_location="cpu")
    train_args = checkpoint["args"]
    epsilon = cli_args.epsilon if cli_args.epsilon is not None else train_args.get(
        "epsilon", DEFAULT_EPSILON
    )
    print(f"using epsilon = {epsilon:g}")

    model = load_model(cli_args.checkpoint_path, device=cli_args.device)
    t_max = cli_args.t_max if cli_args.t_max is not None else model.upper_bound[2].item()
    t_values = np.linspace(0.0, t_max, cli_args.nt)

    t_out, energy_out, mass_out = diagnostics_over_time(
        model,
        t_values=t_values,
        nx=cli_args.nx,
        ny=cli_args.ny,
        epsilon=epsilon,
        device=cli_args.device,
        chunk_size=cli_args.chunk_size,
    )

    save_diagnostics(cli_args.output, t_out, energy_out, mass_out)
    print(f"saved {cli_args.output}")
