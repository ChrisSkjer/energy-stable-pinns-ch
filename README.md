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

DOLFINx does not run natively on Windows, only on Linux — so `conda
activate fenicsx-env` only works **inside a WSL2 shell**, never in a plain
Windows PowerShell or cmd prompt (including VS Code's *default* integrated
terminal, which is PowerShell unless you change it). If `conda activate`
fails with something like `conda: command not found` or `'conda' is not
recognized`, that's almost always the cause: you're not in WSL yet.

**Step 1 — get into a WSL shell.** Any of:
- Open a **Windows Terminal** tab/profile named "Ubuntu" (or whatever your
  WSL distro is called), or
- From PowerShell or cmd, just run `wsl`, or
- In VS Code: `Ctrl+Shift+P` → "WSL: Reopen Folder in WSL" (opens the whole
  editor, including its integrated terminal, inside WSL).

You know you're in the right place when the prompt looks like
`chris@yourmachine:/mnt/c/git/energy-stable-pinns-ch$`, not
`PS C:\git\energy-stable-pinns-ch>`.

**Step 2 — activate the environment** (now that you're in WSL):

```bash
conda env update -n fenicsx-env -f environment.yml --prune
conda activate fenicsx-env
```

(On a machine with no existing DOLFINx install, `conda env create -f
environment.yml` instead.)

**Step 3 — verify it actually worked:**

```bash
python -c "import dolfinx; print(dolfinx.__version__)"
```

If that prints a version number, you're set up correctly and can run FEM
commands as below. If it errors, the conda env doesn't have DOLFINx —
re-check Step 2, not the WSL step.

**Note:** DOLFINx must be installed from `conda-forge`, not `pip`. Also note
that legacy FEniCS (the old `dolfin` package) tutorials and APIs are **not**
compatible with DOLFINx (`dolfinx`) — they are different projects despite the
similar name. Only follow DOLFINx-specific documentation.

## Running the FEM baseline

Requires an activated `fenicsx-env` inside WSL2 (see setup above — you
should see `(fenicsx-env)` in your prompt before running this):

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

**Where to view the output:**

- `<output-dir>/cahn_hilliard_diagnostics.csv` — plot with pandas/matplotlib
  (`pd.read_csv(...).plot(x="t", y="free_energy")`), or open it directly in
  VS Code's built-in CSV viewer (click the file — it renders as a table/chart).
- `<output-dir>/frames/frame_*.png` — click any frame in VS Code's file
  explorer to preview it inline, no other tool needed. To turn a full frame
  sequence into an animation for the thesis:
  ```bash
  ffmpeg -framerate 10 -i frame_%06d.png -pix_fmt yuv420p output.mp4
  ```
  (run from inside the `frames/` folder; `ffmpeg` isn't in `environment.yml`
  yet — `conda install -n fenicsx-env -c conda-forge ffmpeg` if you want it.)

## Quick PINN demo (no setup — run it in Colab)

[notebooks/pinn_demo.ipynb](notebooks/pinn_demo.ipynb) is a small
self-contained PINN for getting a feel for how one trains and what it
produces. It is a sandbox, not thesis code: it imports nothing from `src/`,
uses only the standard loss terms (PDE residual + IC + BC), and has no energy
penalty or transfer learning. The last cell maps its pieces back onto the real
modules.

**Easiest way to run it: Colab.** The notebook imports only `math`, `time`,
`torch`, and `matplotlib` — all of which Colab's default runtime already has,
so there is nothing to clone, install, or `pip` first. Just open it and hit
run:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChrisSkjer/energy-stable-pinns-ch/blob/main/notebooks/pinn_demo.ipynb)

<https://colab.research.google.com/github/ChrisSkjer/energy-stable-pinns-ch/blob/main/notebooks/pinn_demo.ipynb>

That link opens whatever is on `main` in GitHub, so push before you expect a
change to show up there, and use *File → Save a copy in Drive* if you want to
keep your edits. Two Colab-only notes: the CPU runtime is enough (it is a
1D toy problem — GPU is optional, and if you do pick one, set
`device = torch.device("cuda")` in the setup cell), and the `../results/...`
save path in the last section does not exist there, so figures live only in
the session unless you save them to Drive.

**Alternative: locally on Windows, no WSL.** It needs only `torch` (CPU),
`matplotlib`, and `ipykernel`, so plain Windows Python works — no WSL, no
DOLFINx. One-time setup:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch matplotlib ipykernel --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

(`.venv/` is gitignored.) Then open the notebook in VS Code and pick `.venv`
in the kernel selector, top right — or `Ctrl+Shift+P` → "Python: Select
Interpreter" → `.venv` to make it this workspace's default.

Either way, the notebook runs top to bottom in about two minutes on a laptop
CPU:

- **1D heat equation** (~45 s) — has a closed-form solution, so the notebook
  reports a true relative L2 error; with the given settings it reaches ~1e-3.
  Start here, it is the part that tells you the machinery works.
- **1D Cahn-Hilliard** (~80 s) — the mixed (c, mu) formulation on a
  deliberately gentle configuration (eps = 0.05, t in [0, 0.05], smooth cosine
  IC), reporting mass drift and free-energy monotonicity instead of an error.
  Shrink `epsilon`, stretch the time window, or cut the epoch count and those
  two diagnostics are the first thing to degrade — the thesis' motivation in
  miniature. Both sections end with a list of things to try.

Figures render inline; save any you want to keep under `results/`.

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
