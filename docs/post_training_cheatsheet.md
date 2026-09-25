# Post-training cheat sheet

Everything to run *after* `src/pinn/train.py` has produced a checkpoint:
evaluation, diagnostics, plots, and the PINN-vs-FEM comparison. Commands are
PowerShell, run **from the repo root**, with the Windows venv (see README
"Local setup"). Every module is invoked with `-m` — `python src/pinn/evaluate.py`
fails on the package-relative imports.

Set the run name once and reuse it:

```powershell
$run = "demo_run"          # <- the --run-name you trained with
$ck  = "results/pinn_models/$run/final.pt"
```

`$ck` can equally be a periodic snapshot, e.g.
`results/pinn_models/$run/checkpoints/checkpoint_step001000.pt` — useful when a
Colab session died before `final.pt` was written. Every step below writes back
into `results/pinn_models/$run/` regardless of which checkpoint file it was
pointed at (`run_paths.infer_run_dir`).

## 0. If the run was trained on Colab

The checkpoint has to be on the local machine first. When the notebook was
driven from VS Code against a Colab kernel, save the notebook (Ctrl+S) so the
base64 export cell's output is stored, then:

```powershell
.venv\Scripts\python.exe scripts\pull_colab_model.py
# or, for a different notebook:
.venv\Scripts\python.exe scripts\pull_colab_model.py notebooks\pinn_CH_imp1.ipynb
```

Other retrieval routes (Colab file browser, Drive mount) are in
[colab_workflow.md](colab_workflow.md) step 6.

## 1. The one-liner

`scripts/analyze_run.py` runs evaluate → diagnostics → plot in sequence with
the run name typed once. It changes no defaults.

```powershell
.venv\Scripts\python.exe scripts\analyze_run.py $run
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--t T [T ...]` | `0.0 0.001 0.005` | Snapshot times; must be inside the trained `[0, t_max]` |
| `--device` | `cpu` | `cuda` if you're running this on Colab |
| `--fem-diagnostics PATH` | none | Overlays a FEM reference on the energy/mass plots |
| `--fem-fields PATH` | none | PINN-vs-FEM comparison plots of u and mu, plus relative L2 error vs. t; evaluates on the FEM file's grid, so `--nx`/`--ny` match automatically |

```powershell
.venv\Scripts\python.exe scripts\analyze_run.py $run --fem-diagnostics results/eps0.05_nx96_dt2e-4/cahn_hilliard_diagnostics.csv
```

What it does **not** expose: `--nx`/`--ny` (set from `--fem-fields` when
given, otherwise evaluate's default), `--epsilon`, `--nt`, and `--output`
overrides. For any of those,
use the individual steps below.

## 2. The three steps by hand

```powershell
.venv\Scripts\python.exe -m src.pinn.evaluate     --checkpoint-path $ck --t 0.0 0.001 0.005
.venv\Scripts\python.exe -m src.pinn.diagnostics  --checkpoint-path $ck
.venv\Scripts\python.exe -m src.pinn.plot_results --evaluation results/pinn_models/$run/evaluation.npz --checkpoint $ck
```

### 2a. `src/pinn.evaluate` → `evaluation.npz`

Field snapshots on a grid. Inference only, no gradients. Architecture and
domain bounds come out of the checkpoint, so they're never passed here.

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint-path` | *(required)* | |
| `--t T [T ...]` | *(required)* | Must lie in `[0, t_max]` — the net was never trained past it |
| `--nx` / `--ny` | `100` / `100` | Must match the FEM run's `nx`/`ny` for `--fem-fields` comparisons |
| `--device` | `cpu` | |
| `--output` | `<run>/evaluation.npz` | |

Writes `x`, `y` `(nx, ny)` and `t`, `u`, `mu` `(len(t), nx, ny)`.

### 2b. `src/pinn.diagnostics` → `diagnostics.csv`

Free energy and total mass over time. Separate step because it needs an
autograd pass on a much denser grid — **~20 s on CPU**, so don't re-run it for
every re-plot.

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint-path` | *(required)* | |
| `--nx` / `--ny` | `400` / `400` | Dense on purpose: finite differences under-report the `|∇u|²` term at small epsilon |
| `--nt` | `101` | Time samples across `[0, t-max]` |
| `--t-max` | checkpoint's own | Never extrapolate past it |
| `--epsilon` | checkpoint's own | Passing a mismatched value silently invalidates the result |
| `--chunk-size` | `65536` | Lower it if the autograd pass runs out of memory |
| `--device` | `cpu` | |
| `--output` | `<run>/diagnostics.csv` | |

It prints `using epsilon = ...` — worth a glance to confirm it picked up what
you trained with.

Columns are `t,free_energy,total_mass`, byte-identical in format to a FEM
run's `cahn_hilliard_diagnostics.csv`, so both load through the same
`src/common/plotting.py::load_diagnostics`.

### 2c. `src/pinn.plot_results` → `plots/`

Pure orchestration, no computation. At least one of `--evaluation`,
`--checkpoint`, `--diagnostics`, `--fem-diagnostics` is required; pass several
to get everything in one folder.

| Flag | Produces |
| --- | --- |
| `--evaluation <npz>` | `field_u_<idx>_t<val>.png`, `field_mu_<idx>_t<val>.png` (one each per saved time) |
| `--checkpoint <pt>` | `loss_history.png` + `run_summary.txt` |
| `--diagnostics <csv>` | `energy_dissipation.png`, `mass_conservation.png` |
| `--fem-diagnostics <csv>` | FEM reference overlaid on those same two plots |
| `--fem-fields <npz>` | `comparison_<idx>_t<val>.png` + `comparison_mu_<idx>_t<val>.png` (relative L2 error in the diff panel's title), and `relative_l2_error.png` — requires `--evaluation` |
| `--output-dir` | defaults to `plots/` inside the inferred run folder |
| `--dpi` | `150` |

Two auto-detection rules worth remembering:

- `--diagnostics` omitted → a `diagnostics.csv` in the run folder is picked up
  automatically when present.
- `--fem-diagnostics` is **never** auto-detected. Always pass it explicitly.

`run_summary.txt` is the text artifact that's easy to forget exists: network
size, domain, sampling, optimizer, loss weights, trainable parameter count, and
each loss term's final value. It's only written when `--checkpoint` is passed.

## 3. PINN vs. FEM

Two separate comparisons, and they have different requirements.

**Energy/mass curves** — just needs the FEM run's CSV, no grid matching:

```powershell
.venv\Scripts\python.exe -m src.pinn.plot_results `
  --evaluation results/pinn_models/$run/evaluation.npz `
  --checkpoint $ck `
  --fem-diagnostics results/eps0.05_nx96_dt2e-4/cahn_hilliard_diagnostics.csv
```

The FEM curve is auto-clipped to the PINN's own `t_max` so the PINN curve isn't
squashed into a single pixel. If that leaves fewer than 2 FEM samples you'll get
a warning and an empty overlay — re-run the FEM solver with a shorter
`--t-final` (and a matching `--epsilon`).

**Field comparison** — needs `fem_fields.npz` and a matching grid:

```powershell
.venv\Scripts\python.exe -m src.pinn.plot_results `
  --evaluation results/pinn_models/$run/evaluation.npz `
  --fem-fields results/eps0.05_nx96_dt2e-4/fem_fields.npz
```

Requirements, all of which will bite otherwise:

- `--evaluation` must also be passed (the CLI errors out without it).
- The two `(x, y)` grids must be identical, or it raises — so
  `src.pinn.evaluate --nx/--ny` must equal the FEM run's `--nx/--ny` (both
  default to 100, so the defaults already line up).
- The FEM domain is the unit square, so the PINN must have been trained with
  `--x-max 1.0 --y-max 1.0` (the defaults).
- The FEM run needs `save_fields` on (default true) to have written
  `fem_fields.npz` in the first place.
- Each PINN time is matched to the *closest* FEM snapshot; a mismatch beyond
  `1e-9` emits a warning naming both times. FEM snapshot spacing is
  `--viz-every` steps.

Besides one comparison PNG per time, this writes `relative_l2_error.png`:
`metrics.relative_l2_error` (‖PINN − FEM‖₂ / ‖FEM‖₂) for u and mu against t,
on a log axis. Same number as in each comparison's diff-panel title. Reading it:

- It's scale-free — read it as a percentage of the FEM field's size. With u ≈
  ±1, 0.01 means an RMS error of about 0.01.
- Flat is good; a steady climb means the PINN error builds up over time.
  Where it crosses ~10% is roughly where the PINN stops being trustworthy.
- mu sitting well above u: phase layout right, chemical potential (higher
  derivatives) off.
- One point per `--t` value only — and since an unmatched time is compared
  against the *closest* FEM snapshot, extra `--t` values between snapshots
  give misleading points, not a smoother curve. For more points, re-run FEM
  with a smaller `--viz-every` and evaluate the PINN at those same times.

Generating the FEM reference is a WSL/`fenicsx-env` job (see README "Running
the FEM baseline"):

```bash
python src/fem/cahn_hilliard.py --output-dir results/eps0.05_nx96_dt2e-4 --epsilon 0.05 --nx 96 --ny 96
```

## Output map

After a full pass:

```
results/pinn_models/<run-name>/
├── checkpoints/checkpoint_step<N>.pt
├── final.pt
├── evaluation.npz                  # step 2a
├── diagnostics.csv                 # step 2b
└── plots/                          # step 2c
    ├── field_u_000_t0.png ...
    ├── field_mu_000_t0.png ...
    ├── loss_history.png
    ├── run_summary.txt
    ├── energy_dissipation.png
    ├── mass_conservation.png
    ├── comparison_000_t0.png, comparison_mu_000_t0.png ...   # only with --fem-fields
    └── relative_l2_error.png                                  # only with --fem-fields
```

## Gotchas

- **Repo root + `-m`.** `python src/pinn/evaluate.py` fails on imports.
- **`--t` past `t_max`.** Training defaults to `--t-max 5e-3`; asking for
  `--t 0.05` gives you extrapolation, not an error. The checkpoint's `t_max` is
  in `plots/run_summary.txt`.
- **`--epsilon` on diagnostics.** Leave it alone unless you know why you're
  overriding it. A mismatch produces a plausible-looking wrong energy curve.
- **Re-plotting is cheap, re-running diagnostics isn't.** `plot_results` reads
  `diagnostics.csv` from disk; only re-run step 2b when the checkpoint changes.
- **`--fem-diagnostics` is never auto-detected**, unlike the PINN's own
  `diagnostics.csv`.
- **A stale `evaluation.npz`.** It's overwritten in place per run folder, so a
  comparison plot against an old `.npz` looks fine and is wrong. Re-run 2a after
  pulling a newer checkpoint into the same run name.
