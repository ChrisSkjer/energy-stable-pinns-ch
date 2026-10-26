#!/usr/bin/env bash
# FEM ground truth for every 1D initial condition (src.fem.cahn_hilliard --dim 1),
# matching the 1D PINN runs (eps 0.05, t in [0, 0.1], --ic-seed 42).
#
# Run in WSL from the repo root with fenicsx-env active:
#   bash scripts/run_fem_1d.sh            # refuses to touch existing folders
#   bash scripts/run_fem_1d.sh --overwrite
#
# nx=1000 (h = 1e-3, ~12x finer than the eps/4 rule) and dt=1e-5 are cheap in
# 1D; fields are saved every 10 steps (dt_out = 1e-4, 1001 snapshots).
# Writes results/eps0.05_nx1000_dt1e-5_t0.1_dim1_<ic>/: fem_fields.npz
# (x, t, u, mu), cahn_hilliard_diagnostics.csv and frames/space_time_{u,mu}.png.
set -euo pipefail

# The 1D system is tiny: multithreaded BLAS/MUMPS only fight each other
# (one run: ~4 min wall / 65 min CPU threaded vs ~50 s on one thread).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for ic in step two_intervals noise cosine; do
    out="results/eps0.05_nx1000_dt1e-5_t0.1_dim1_${ic}"
    echo "########## FEM 1D ic=${ic} -> ${out} ##########"
    python -m src.fem.cahn_hilliard --dim 1 --ic "${ic}" --seed 42 --epsilon 0.05 \
        --nx 1000 --dt 1e-5 --t-final 0.1 --viz-every 10 --log-every 10 \
        --output-dir "${out}" "$@"
done
