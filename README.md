# Energy-Stable Physics-Informed Neural Networks for the Cahn-Hilliard Equation

Master's thesis project investigating whether Physics-Informed Neural
Networks (PINNs) augmented with an energy-dissipation penalty term and
sequential transfer learning can solve the Cahn-Hilliard phase-field
equation with better physical fidelity and efficiency than a standard
baseline PINN, benchmarked against high-fidelity FEM solutions from
FEniCSx (DOLFINx).

## Research questions

1. **Energy penalty vs. baseline PINN** — does adding an energy-dissipation
   penalty term to the loss improve physical fidelity (energy monotonicity,
   mass conservation) over a standard PINN?
2. **Transfer learning for convergence speed** — does sequential/transfer
   learning across time windows improve training convergence and accuracy
   compared to training a single network over the full time horizon?
3. **PINN vs. FEM** — how do the baseline and enhanced PINNs compare against
   a DOLFINx FEM solution on accuracy, stability, and computational cost?

## Repo structure

```
.
├── src/
│   ├── pinn/       # PINN model, loss functions, training loop
│   ├── fem/        # DOLFINx Cahn-Hilliard solver (ground-truth baseline)
│   └── common/     # shared utils: metrics (relative L2, energy, mass), plotting
├── data/           # generated FEM reference solutions (gitignored, local only)
├── notebooks/      # exploration notebooks
├── results/        # figures, logs, benchmark tables (gitignored)
├── docs/           # thesis notes, literature extraction, workflow notes
├── tests/          # unit tests
├── environment.yml           # local (WSL2/conda) environment
├── requirements-colab.txt    # Colab (GPU) environment
```

This is a two-machine project:

- **Local (WSL2/conda, no GPU)** runs the FEM baseline (DOLFINx) and is used
  to author/smoke-test PINN code on CPU.
- **Google Colab (GPU)** runs the actual PINN training. Colab never touches
  FEM code or data — training is physics-only (PDE residual, IC/BC, and the
  energy penalty are computed from the network's own predictions).

## Local environment setup

This machine already has a conda environment named `fenicsx-env` with a
working DOLFINx install. Bring it up to date with the rest of the project's
dependencies:

```bash
conda env update -n fenicsx-env -f environment.yml --prune
conda activate fenicsx-env
```

(On a machine with no existing DOLFINx install, `conda env create -f
environment.yml` instead.)

**Note:** DOLFINx must be installed from `conda-forge`, not `pip`. Also note
that legacy FEniCS (the old `dolfin` package) tutorials and APIs are **not**
compatible with DOLFINx (`dolfinx`) — they are different projects despite the
similar name. Only follow DOLFINx-specific documentation.

## Running the FEM baseline

```bash
python src/fem/cahn_hilliard.py --output-dir results/my_run
```

Every field on `CahnHilliardConfig` (`src/fem/cahn_hilliard.py`) is exposed
as a matching CLI flag automatically (`nx` → `--nx`, `epsilon` → `--epsilon`,
`visualize` → `--visualize`/`--no-visualize`, etc.) — run `--help` to see the
full list. Two flags control where output goes, not the physics:

- `--output-dir PATH` — where the run writes `cahn_hilliard_diagnostics.csv`
  (t, free_energy, total_mass) and, if `--visualize` is set, `frames/*.png`.
  **Defaults to `data/` if you don't pass it.**
- `--overwrite` — required if `--output-dir` already contains a previous
  run's results. **Without a flag at all, the solver refuses to run and
  errors out** rather than silently clobbering the last run — this is what
  protects an overnight run from being erased by a later invocation that
  forgot to pick a new folder.

**Naming convention:** for a throwaway smoke test, omit `--output-dir`
entirely and pass `--overwrite` each time — you don't want to keep those.
For any run whose output you might actually reference (a figure, a number
for the thesis), give it a descriptive folder under `results/` that encodes
the config, e.g. `results/eps0.02_nx250_dt2e-7/`, and tag the commit that
produced it (`git tag -a fem-baseline-v1 -m "..."`) so the folder name and
the tag can point at each other later.

## Training on Colab

1. Clone the repo in a Colab cell and `pip install -r requirements-colab.txt`
   (no FEM data needed there).
2. Train the PINN (baseline or enhanced, via `--energy-penalty`).
3. Download the resulting checkpointed `.pt` weights back to the local
   machine to run inference and comparison against the local FEM output.

See [docs/colab_workflow.md](docs/colab_workflow.md) for the full walkthrough.

## Status

Work in progress. The FEM baseline (`src/fem/cahn_hilliard.py`) is working
end-to-end (mesh, weak form, adaptive time stepping, diagnostics, PNG
snapshots). PINN model, loss, and training modules are still skeletons.
