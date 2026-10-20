# TL;DR

Short version of the [README](README.md).

**What it is:** A master's thesis project that tests whether PINNs solve the
Cahn-Hilliard equation better than a plain baseline PINN when they get (a) an
energy-dissipation penalty in the loss and (b) transfer learning across time
windows. All results are checked against a DOLFINx FEM solution.

## Research questions

1. Does the energy penalty improve energy monotonicity and mass conservation?
2. Does transfer learning across time windows speed up convergence?
3. How do the PINNs compare with FEM on accuracy, stability and cost?

## Two machines, two environments

- **Local, no GPU:**
  - The FEM solver needs WSL2 with conda (`fenicsx-env`), because DOLFINx only
    runs on Linux.
  - Everything else (PINN code, tests, plotting) runs in a plain Windows
    `.venv` with CPU torch.
- **Colab GPU:** used only for real PINN training, through
  [notebooks/train_pinn.ipynb](notebooks/train_pinn.ipynb). The only file that
  moves between the machines is the `.pt` checkpoint.

## Key commands

Run from the repo root, always with `-m`.

- FEM: `python -m src.fem.cahn_hilliard --output-dir results/<name>`. It
  refuses to overwrite an existing run unless you pass `--overwrite`.
- Smoke-test training: `python -m src.pinn.train --smoke-test --run-name demo_run`
- Analyze a run: `python scripts/analyze_run.py <run>`. This runs evaluate,
  diagnostics and plots in one go. Add `--fem-fields` / `--fem-diagnostics`
  for PINN-vs-FEM comparisons.
- Tests: `python -m pytest tests/`. The FEM test skips itself outside WSL.

## Training options worth knowing

- Collocation sampling: `--collocation random|sobol|grid`, plus `--resample-every`
- Held-out residual tracking: `--heldout-every`
- Adam has opt-in LR plateau scheduling and early stopping.
- L-BFGS stops early by default. Each `--lbfgs-steps` step runs up to 20
  inner iterations.

## Colab gotchas

- Run `git push` first, because Colab clones `main`.
- Choose the T4 GPU runtime before running any cells.
- Retrieve the model with base64 plus `scripts/pull_colab_model.py`.

## Status

- **Done:** FEM baseline, baseline PINN, post-training analysis and plots,
  metrics.
- **Not done:**
  - The energy penalty: `--energy-penalty` raises `NotImplementedError`,
    which blocks research question 1.
  - Transfer learning, which blocks research question 2.
