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
│   ├── pinn/       # model, sampling, losses, train/evaluate/plot CLI scripts
│   ├── fem/        # DOLFINx Cahn-Hilliard solver (ground-truth baseline)
│   └── common/     # shared utils: metrics (relative L2, energy, mass), plotting
├── data/           # generated FEM reference solutions (gitignored, local only)
├── notebooks/      # exploration notebooks
├── results/        # figures, logs, benchmark tables; results/pinn_models/<run>/
│                   # holds each training run's checkpoints + evaluation + plots
│                   # (all gitignored)
├── docs/           # thesis notes, literature extraction, workflow notes
├── tests/          # unit tests
├── environment.yml           # WSL2/conda environment (FEM + PINN authoring)
├── requirements-colab.txt    # Colab (GPU) environment (PINN training only)
```

This is a two-machine project:

- **Local (no GPU)** runs the FEM baseline (DOLFINx, WSL2/conda only) and is
  used to author/smoke-test/evaluate/plot PINN code on CPU (plain Windows
  `.venv`, no WSL needed).
- **Google Colab (GPU)** runs the actual, full-length PINN training. Colab
  never touches FEM code or data — training is physics-only (PDE residual,
  IC/BC, and the energy penalty are computed from the network's own
  predictions).

## Local setup

Two independent environments, depending on what you're running — most work
only needs the second one.

### 1. FEM baseline (`src/fem/`) — WSL2 + conda, DOLFINx

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

If that prints a version number, you're set up correctly. If it errors, the
conda env doesn't have DOLFINx — re-check Step 2, not the WSL step.

**Note:** DOLFINx must be installed from `conda-forge`, not `pip`. Also note
that legacy FEniCS (the old `dolfin` package) tutorials and APIs are **not**
compatible with DOLFINx (`dolfinx`) — they are different projects despite the
similar name. Only follow DOLFINx-specific documentation.

### 2. PINN code (`src/pinn/`, `src/common/`, `tests/`) — plain Windows venv, no WSL

Everything else in this repo — the PINN model/training/evaluation/plotting
code, the shared plotting/metrics utilities, the unit tests, and the demo
notebook — needs only CPU `torch`, `numpy`, and `matplotlib`, so it runs in
a plain Windows Python venv. One-time setup:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch matplotlib numpy ipykernel pytest --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

(`.venv/` is gitignored.) For notebooks, open one in VS Code and pick
`.venv` in the kernel selector, top right — or `Ctrl+Shift+P` → "Python:
Select Interpreter" → `.venv` to make it this workspace's default.

Every `python ...` command below assumes this venv is active and that
you're in the **repo root** — either activate it
(`.venv\Scripts\Activate.ps1`, then use plain `python`) or spell out
`.venv\Scripts\python.exe` each time, as the examples do.

## Running the FEM baseline

Requires the WSL2/conda setup above (`(fenicsx-env)` in your prompt):

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

## Quick PINN demo (no thesis code — a sandbox notebook)

[notebooks/pinn_demo.ipynb](notebooks/pinn_demo.ipynb) is a small
self-contained PINN for getting a feel for how one trains and what it
produces. It imports nothing from `src/`, uses only the standard loss terms
(PDE residual + IC + BC), and has no energy penalty or transfer learning.
The last cell maps its pieces back onto the real modules.

**Requirements:** none beyond `torch` + `matplotlib` — either the venv from
Local setup above, or Colab, which already has both preinstalled:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChrisSkjer/energy-stable-pinns-ch/blob/main/notebooks/pinn_demo.ipynb)

<https://colab.research.google.com/github/ChrisSkjer/energy-stable-pinns-ch/blob/main/notebooks/pinn_demo.ipynb>

That link opens whatever is on `main` in GitHub, so push before you expect a
change to show up there, and use *File → Save a copy in Drive* if you want to
keep your edits. On Colab, the CPU runtime is enough (it's a 1D toy
problem), and the `../results/...` save path in the last cell doesn't exist
there, so figures live only in the session unless you save them to Drive.

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

## Running the PINN training (`src/pinn/train.py`)

This is the real thesis training code — `src/pinn/model.py` (network),
`src/pinn/losses.py` (PDE/IC/BC/energy loss terms), and
`src/pinn/sampling.py` (domain point sampling) — as opposed to the
self-contained demo notebook above. **Requirements:** the venv from Local
setup above — only `torch` is actually exercised by this script.

**Run it as a module, not as a script.** `train.py` does
`from src.pinn.losses import ...`, which only resolves when `src` is
importable as a package — i.e. invoked with `-m` from the **repo root**, not
`python src/pinn/train.py`:

```powershell
.venv\Scripts\python.exe -m src.pinn.train --smoke-test --run-name demo_run
```

**`--smoke-test`** is the fast local sanity check: it shrinks the point
counts (100 collocation points, 10×10 IC grid, 10×10 BC grid per edge) and
epoch counts (5 Adam epochs + 2 L-BFGS steps) so the whole run — model
build, point sampling, Adam loop, L-BFGS loop, checkpoint save — finishes in
a couple of seconds on CPU. It proves the code path works end to end; it
says nothing about training quality, so don't read anything into the loss
value it produces.

**A real run** (on Colab GPU — see below), e.g.:

```bash
python -m src.pinn.train --run-name eps0.01_h64x4 --device cuda --epochs 20000 --checkpoint-every 500
```

Add `--energy-penalty --energy-weight <w>` to train the enhanced model
instead of the baseline (the energy-penalty loss term itself is not
implemented yet — see Status). Run `--help` for the full flag list:
domain bounds (`--x-max`/`--y-max`/`--t-max`), point counts
(`--n-collocation`/`--n-ic`/`--n-bc`/`--pde-at-t0`), network size
(`--hidden-layers`/`--hidden-width`), and optimizer settings
(`--lr`, `--lbfgs-steps`).

**Output — one folder per run.** `train.py` writes everything under
`results/pinn_models/<run-name>/` (`--run-name`; defaults to a
`YYYYMMDD_HHMMSS` timestamp if you don't pass one — always pass a
descriptive one for a run you intend to keep, same reasoning as the FEM
naming convention above):

```
results/pinn_models/<run-name>/
├── checkpoints/
│   ├── checkpoint_step000500.pt   # one per --checkpoint-every optimizer steps
│   └── checkpoint_step001000.pt
└── final.pt                       # weights once the full run finishes
```

Every `.pt` file (periodic or final) holds the model's `state_dict()`, the
CLI `args` it was trained with, and the loss `history` so far — see
`src/pinn/run_paths.py` for the exact layout, and the next section for
turning a checkpoint into field/loss plots. The run folder is printed to
stdout at the start of training (`run directory: ...`) so you don't have to
compute it yourself.

## Evaluating and plotting a trained checkpoint

**Quick look at a run — one command.** `scripts/analyze_run.py` runs all
three steps below in sequence, so the run name is typed once instead of
threaded through three separate flags. It changes no defaults; each step
writes exactly what it would when invoked by hand:

```powershell
.venv\Scripts\python.exe scripts\analyze_run.py demo_run
```

Useful flags: `--t 0.0 0.01 0.05` (snapshot times passed to `evaluate`),
`--device cuda`, and `--fem-diagnostics <path>` to overlay a FEM reference on
the energy/mass plots:

```powershell
.venv\Scripts\python.exe scripts\analyze_run.py demo_run --fem-diagnostics results/eps0.05_nx96_dt2e-4/cahn_hilliard_diagnostics.csv
```

**Or run the three by hand** — set `$run` once, paste all three. Do this when
you want flags the wrapper doesn't expose (`--nx`/`--ny` resolution,
`--epsilon`, `--output` overrides), or to re-run just one step:

```powershell
$run = "demo_run"  # <- change this to the run-name you want to look at

.venv\Scripts\python.exe -m src.pinn.evaluate --checkpoint-path results/pinn_models/$run/final.pt --t 0.0 0.001 0.005
.venv\Scripts\python.exe -m src.pinn.diagnostics --checkpoint-path results/pinn_models/$run/final.pt
.venv\Scripts\python.exe -m src.pinn.plot_results --evaluation results/pinn_models/$run/evaluation.npz --checkpoint results/pinn_models/$run/final.pt
```

Either way, everything lands in `results/pinn_models/<run-name>/plots/` — see
the per-script breakdown below for what each one produces and what flags exist
beyond the defaults used here.

Three more thin CLI scripts, run the same way as `train.py` above (`-m`, from
the repo root, same venv). All three default their output into the *same*
run folder as whichever checkpoint you point them at (via
`src/pinn/run_paths.py::infer_run_dir`), so a checkpoint, its evaluation, its
diagnostics, and its plots stay findable together without retyping the run
name each time.

**1. Evaluate** — `src/pinn/evaluate.py` loads a checkpoint (`final.pt`, or
any file under `checkpoints/`) and evaluates it on a grid at one or more
times, saving plain numpy arrays (no plotting here, and nothing downstream
needs torch):

```powershell
.venv\Scripts\python.exe -m src.pinn.evaluate --checkpoint-path results/pinn_models/demo_run/final.pt --t 0.0 0.001 0.005
```

Architecture and domain bounds are read back out of the checkpoint itself
(bundled there by `train.py`'s `save_checkpoint`), so only
evaluation-specific flags are needed: `--nx`/`--ny` (grid resolution), `--t`
(one or more snapshot times — must lie within `[0, t_max]` from training,
since the network was never trained past that), `--device`, `--output`
(`.npz` path — defaults to `evaluation.npz` inside that checkpoint's run
folder, override to place it elsewhere). **Output:**
`results/pinn_models/demo_run/evaluation.npz` holding `x`, `y` (each
`(nx, ny)`), `t`, and `u`, `mu` (each `(len(t), nx, ny)`).

**2. Diagnostics** — `src/pinn/diagnostics.py` loads a checkpoint and, via an
autograd pass (needed for the interfacial |∇u|² term — nothing below needs
this, so it's a separate step from evaluation), computes free energy and
total mass over time on a dense grid (400×400 by default, 101 time samples —
takes ~20s on CPU):

```powershell
.venv\Scripts\python.exe -m src.pinn.diagnostics --checkpoint-path results/pinn_models/demo_run/final.pt
```

**Output:** `results/pinn_models/demo_run/diagnostics.csv`, with the same
`t,free_energy,total_mass` columns as a FEM run's
`cahn_hilliard_diagnostics.csv` (see `src/fem/cahn_hilliard.py`), so both are
readable by the same `src/common/plotting.py::load_diagnostics`.

**3. Plot** — `src/pinn/plot_results.py` turns the `.npz`, a checkpoint's
loss history, and/or a diagnostics CSV into saved PNGs (and a text summary):

```powershell
.venv\Scripts\python.exe -m src.pinn.plot_results --evaluation results/pinn_models/demo_run/evaluation.npz --checkpoint results/pinn_models/demo_run/final.pt
```

`--evaluation`, `--checkpoint`, `--diagnostics`, and `--fem-diagnostics` are
each optional, but at least one is required — pass several to get everything
from one run into one place. If `--diagnostics` is omitted, a
`diagnostics.csv` inside the inferred run folder is picked up automatically
when present. **Output**, written under `--output-dir` (defaults to a
`plots/` subfolder in that same run folder, e.g.
`results/pinn_models/demo_run/plots/`; each saved path is also printed to
stdout):

- `field_u_<idx>_t<value>.png` / `field_mu_<idx>_t<value>.png` — one of each
  per saved time, via `src/common/plotting.py::plot_field`. The mu snapshots
  share one symmetric color scale across all times, from a robust quantile
  of `|mu|` rather than a fixed range (mu is unbounded, unlike u).
- `loss_history.png` — loss vs. optimizer step, from the checkpoint's
  bundled history (only if `--checkpoint` was passed).
- `run_summary.txt` — network size, domain, sampling, optimizer, loss
  weights, trainable parameter count, and each loss term's final value (only
  if `--checkpoint` was passed) — see
  `src/pinn/run_summary.py::format_run_summary`.
- `energy_dissipation.png` / `mass_conservation.png` — free energy and total
  mass vs. time (only if `--diagnostics` and/or `--fem-diagnostics` was
  passed or auto-detected); pass both to overlay PINN against a FEM
  reference (auto-clipped to the PINN's own time window).

Not produced yet: PINN-vs-FEM field comparison plots — blocked on
`src/fem/cahn_hilliard.py` not saving raw field arrays (only PNG frames and
the diagnostics CSV).

## Running tests

```powershell
.venv\Scripts\python.exe -m pytest tests/
```

From the repo root, same venv as above (`pytest` is included in the Local
setup pip install command). `tests/test_fem_cahn_hilliard.py` needs DOLFINx
and self-skips (via `pytest.importorskip`) when it isn't importable — so the
full suite runs clean in the plain Windows venv, with that one test reported
as skipped rather than failed. Run it for real from inside WSL's
`fenicsx-env` (which already includes `pytest`, see `environment.yml`) if
you need that test to actually execute.

## Training on Colab

1. Clone the repo in a Colab cell and `pip install -r requirements-colab.txt`
   (no FEM data needed there).
2. Train the PINN (baseline or enhanced, via `--energy-penalty`) using the
   `-m src.pinn.train` invocation above, with `--device cuda` and a
   descriptive `--run-name`.
3. Download the resulting `results/pinn_models/<run-name>/` folder back to
   the local machine (at minimum `final.pt`) to run evaluation/plotting and
   comparison against the local FEM output. If you drive the notebook from
   VS Code rather than the Colab web page, `files.download` won't work —
   use `scripts/pull_colab_model.py` instead (see the walkthrough).

See [docs/colab_workflow.md](docs/colab_workflow.md) for the full walkthrough.

## Status

Work in progress. The FEM baseline (`src/fem/cahn_hilliard.py`) is working
end-to-end (mesh, weak form, adaptive time stepping, diagnostics, PNG
snapshots). `src/pinn/` (model, losses, sampling, training loop) runs
end-to-end for the baseline PINN — validated so far via
`notebooks/pinn_CH_imp1.ipynb`, not yet via `src/pinn/train.py` itself on a
full-length run. `src/pinn/evaluate.py`, `src/pinn/diagnostics.py`, and
`src/pinn/plot_results.py` load a checkpoint back and produce field/loss/
energy/mass plots plus a run summary (see above). `src/common/metrics.py`'s
`free_energy` and `total_mass` (and their shared quadrature,
`trapezoid_weights_2d`) are implemented and drive `diagnostics.py`; still
stubs there: `relative_l2_error` and `mass_conservation_error`. Not
implemented yet: the energy-stability penalty (`energy_stability_loss` —
passing `--energy-penalty` currently raises `NotImplementedError`), and
PINN-vs-FEM comparison plots (blocked on raw FEM field arrays, which
`src/fem/cahn_hilliard.py` doesn't save yet — only PNG frames and scalar
diagnostics).
