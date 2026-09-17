# Colab training workflow

PINN training runs on Google Colab (GPU). FEM never runs there, and Colab
never needs FEM data — the PDE residual, IC/BC terms, and the energy-stability
penalty are all computed from the network's own predictions, not from FEM
reference solutions.

## Steps

0. **Push local changes first.** Colab's clone in step 1 pulls whatever is
   currently on `main` on GitHub — not your working directory. If you've
   edited `src/pinn/` locally and haven't pushed, Colab trains the *old*
   code silently (no error, just stale results):

   ```powershell
   git push origin main
   ```

1. **New notebook, GPU runtime.** At <https://colab.research.google.com>,
   start a new notebook, then **Runtime → Change runtime type → Hardware
   accelerator → T4 GPU** (free tier) → Save. Do this before running any
   cells — changing it later restarts the runtime and wipes anything already
   in memory/disk.

2. **Clone the repo** in the first cell:

   ```python
   !git clone https://github.com/ChrisSkjer/energy-stable-pinns-ch.git
   %cd energy-stable-pinns-ch
   ```

   No authentication needed — the repo is public. If you pushed a branch
   other than `main` in step 0, add `-b <branch-name>` to the clone command.

3. **Install extra dependencies** (Colab already has `torch`, `numpy`,
   `matplotlib`):

   ```python
   !pip install -r requirements-colab.txt
   ```

4. **Verify the GPU actually attached** before training — a runtime can be
   set to "GPU" and still hand you a CPU box if the assignment silently
   failed:

   ```python
   import torch
   print(torch.__version__, "| cuda available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   ```

   If `cuda available` prints `False`, go back to step 1 — Runtime → Change
   runtime type — and confirm T4 GPU is actually selected, then rerun from
   the top.

5. **Smoke-test first, then train.** Confirm the code path works end to end
   (a few seconds) before committing a GPU session to a multi-hour run:

   ```python
   !python -m src.pinn.train --smoke-test --run-name smoke --device cuda
   ```

   Then the real run, e.g.:

   ```python
   !python -m src.pinn.train --run-name eps0.01_h64x4 --device cuda --epochs 20000 --checkpoint-every 500
   ```

   (`-m src.pinn.train`, not `python src/pinn/train.py` — the script imports
   `src.pinn.losses` etc. as a package, which only resolves when run with
   `-m` from the repo root.)

   Add `--energy-penalty` to train the enhanced model instead of the
   baseline. Always pass `--run-name` with something descriptive — omitting
   it falls back to a timestamp, which is harder to tell apart later once
   you have several checkpoints downloaded locally.

   Everything gets written under `results/pinn_models/<run-name>/`: periodic
   snapshots in `checkpoints/checkpoint_step<N>.pt` and the finished weights
   at `final.pt` (see `src/pinn/run_paths.py`).

6. **Retrieve the checkpoint** (Colab sessions can disconnect — free-tier
   idle timeout is ~90 minutes and max runtime is ~12 hours, so don't wait
   for `final.pt` on a long run; a periodic one under `checkpoints/` works
   with `evaluate.py` too). Download the run folder back to the local
   machine via one of:
   - the Colab file browser (folder icon in the left sidebar) — right-click
     `results/pinn_models/<run-name>/` → Download. For anything but a small
     run, zip it first (the browser downloads folders as many individual
     files otherwise):
     ```python
     !zip -r {run_name}.zip results/pinn_models/{run_name}
     from google.colab import files
     files.download(f"{run_name}.zip")
     ```
   - pushing it to Google Drive after mounting it:
     ```python
     from google.colab import drive
     drive.mount("/content/drive")
     !cp -r results/pinn_models/{run_name} "/content/drive/MyDrive/pinn_runs/{run_name}"
     ```

7. **Run comparisons locally.** Once the `.pt` weights are back on the local
   machine, run `src/pinn/evaluate.py` and `src/pinn/diagnostics.py` (energy
   dissipation and mass conservation over time, via `src/common/metrics.py`),
   then `src/pinn/plot_results.py` to turn those into plots -- see
   "Evaluating and plotting a trained checkpoint" in the README. For
   PINN-vs-FEM comparison, generate a FEM reference locally
   (`src/fem/cahn_hilliard.py`) at a matching `--epsilon`.

The `.pt` checkpoint is the only artifact that needs to travel between the
two machines.

## Alternative: the official `colab` CLI

Google publishes an official CLI,
[google-colab-cli](https://github.com/googlecolab/google-colab-cli), that can
drive this same clone → install → train → retrieve-checkpoint loop from a
local terminal instead of pasting cells into a notebook:

```bash
uv tool install google-colab-cli   # or: pip install google-colab-cli
colab auth                         # one-time OAuth2/ADC setup
colab run --gpu A100 train_wrapper.py --device cuda --epochs 20000 --checkpoint-every 500
```

`colab run` provisions a fresh VM, runs a local script with forwarded
arguments, pulls back output files, and tears the VM down automatically —
replacing steps 2-6 above in a single command. `colab new` / `colab install
-r requirements-colab.txt` / `colab exec` / `colab download` / `colab stop`
are also available for a more manual, step-by-step session.

Not adopted as the primary workflow above yet because:
- **Linux/macOS only** — there's no native Windows build, so it needs WSL on
  this machine.
- `colab run`/`colab exec` run a single local script, whereas
  `src/pinn/train.py` needs to be invoked as `-m src.pinn.train` (see step 5)
  for its package-relative imports to resolve. That hasn't been verified to
  work through this CLI yet.

Worth revisiting if training moves off manual notebook cells and onto a
scripted loop.

## Side note: the demo notebook needs none of this

[../notebooks/pinn_demo.ipynb](../notebooks/pinn_demo.ipynb) is self-contained
— it imports only `torch` and `matplotlib`, both already present in Colab's
default runtime — so it needs no clone and no `pip install`. Open it directly
from GitHub:

<https://colab.research.google.com/github/ChrisSkjer/energy-stable-pinns-ch/blob/main/notebooks/pinn_demo.ipynb>

The clone-and-install steps above are only for training the real `src/pinn/`
code.
