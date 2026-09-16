# Colab training workflow

PINN training runs on Google Colab (GPU). FEM never runs there, and Colab
never needs FEM data — the PDE residual, IC/BC terms, and the energy-stability
penalty are all computed from the network's own predictions, not from FEM
reference solutions.

## Steps

1. **Clone the repo** in a Colab cell:

   ```python
   !git clone https://github.com/ChrisSkjer/energy-stable-pinns-ch.git
   %cd energy-stable-pinns-ch
   ```

2. **Install extra dependencies** (Colab already has `torch`, `numpy`,
   `matplotlib`):

   ```python
   !pip install -r requirements-colab.txt
   ```

3. **Train**, e.g.:

   ```python
   !python -m src.pinn.train --device cuda --epochs 20000 --checkpoint-every 500
   ```

   (`-m src.pinn.train`, not `python src/pinn/train.py` — the script imports
   `src.pinn.losses` etc. as a package, which only resolves when run with
   `-m` from the repo root.)

   Add `--energy-penalty` to train the enhanced model instead of the baseline.

4. **Retrieve the checkpoint.** Training periodically writes a `.pt` weights
   file (Colab sessions can disconnect, so don't wait for the final epoch).
   Download it back to the local machine via:
   - the Colab file browser (right-click → Download), or
   - pushing it to Google Drive (`!cp checkpoint.pt /content/drive/MyDrive/...`)
     after mounting Drive.

5. **Run comparisons locally.** Once the `.pt` weights are back on the local
   machine, use `src/common/metrics.py` together with a FEM reference
   solution (generated locally via `src/fem/cahn_hilliard.py`) to compute
   relative L2 error, energy dissipation comparison, and mass conservation.

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
replacing steps 1-4 above in a single command. `colab new` / `colab install
-r requirements-colab.txt` / `colab exec` / `colab download` / `colab stop`
are also available for a more manual, step-by-step session.

Not adopted as the primary workflow above yet because:
- **Linux/macOS only** — there's no native Windows build, so it needs WSL on
  this machine.
- `colab run`/`colab exec` run a single local script, whereas
  `src/pinn/train.py` needs to be invoked as `-m src.pinn.train` (see step 3)
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
