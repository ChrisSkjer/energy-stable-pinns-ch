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
   !python src/pinn/train.py --device cuda --epochs 20000 --checkpoint-every 500
   ```

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
