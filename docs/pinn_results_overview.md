# PINN results overview (2026-11-10)

Synthesis of all 113 trained models in `results/pinn_models/`. About 50 had no
saved FEM comparison (weight sweeps, grid/random/Sobol comparison); their errors
were computed from `evaluation.npz` against the FEM references. Runs from before
the IC normalisation fix (`_old_unnormalized_ic/`) are left out.

Unless stated, numbers are 2D cross, relative L2 vs FEM, given as
**u at t=0 / u at later times (mean of the four snapshots after t=0) / mu**.

**Every run uses weight seed 0 and sampling seed 0.** Differences under ~10%
are unresolved until seeds are repeated.

## Two corrections to earlier reasoning

- **"IC error predicts the total u error" is mostly arithmetic.** The space-time
  u error averages five snapshots, and on cross the t=0 snapshot is ~5x the
  others, so the total ≈ (t=0 error)/√5 regardless of later times. Rank runs by
  later-time u and mu instead.
- **An IC-loss plateau is not a safe stopping rule on its own.** In the one long
  cross run (e5000-lb1000p5) the IC fit improved while later-time u got 5x worse.

## What works

| Choice | Evidence |
|---|---|
| Weights `wmu` ≈ `icw` in 3000–10000 | mu at short budget: 6–11% at `wmu` 300, 3.8% at 3000, 2.9% at 10000 |
| L-BFGS 100–120 → 400 steps | Largest single lever. Cross mu ~2.0% → 1.1%. Noise u 13.7% → 3.2% (Adam also 1000 → 2000) |
| Fixed Sobol or random points | Sobol slightly ahead: later u 0.31% vs 0.48%, mu 1.45% vs 1.68% (one run each) |
| Reweighted adaptive PDE sampling in L-BFGS (`adlb24n500m2000rw100`) | Cross 0.96% / 0.18% / 1.10%. Noise u 3.20% → 1.95%, mu 4.9% → 3.4% |

On cross the adaptive run ties with the reweighted dense window and with 100k
L-BFGS points (later u ≈ 0.18%, mu 1.10–1.13%); the 100k run costs ~3x the
L-BFGS time. The reweighting is what makes adaptive sampling work: without it,
mu 1.44% instead of 1.23% and a worse IC fit.

## What does not work

| Choice | Evidence |
|---|---|
| Uniform grid with long L-BFGS | Later u 0.30% (lb100) → 2.6% (lb400): overfits the grid |
| Resampling during L-BFGS (`lbrs`) | Worse the more often: total u 2.1/1.3/1.1/0.9% for every 1/5/10/20 steps vs 0.77% fixed |
| Resampling during Adam (`-rs100`) | No effect (0.52 vs 0.52% Sobol, 0.58 vs 0.59% random) |
| IC pretraining | mu 3.2–3.3% vs 2.1% |
| Dense window without reweighting | Total u 1.3–1.6% vs 0.77% at lb120; more window points is worse |
| Adaptive IC sampling, with or without reweight | Flat on two_circles, worse later-time u on cross, worse overall on noise |
| Unweighted adaptive PDE sampling during Adam | Later u 0.32% vs 0.18% |
| Weights ≥ 30000 | u flat, mu worse (2.9 → 3.3 → 4.2%) |
| `icw` ≫ `wmu` | At `wmu` 300, `icw` 10000 doubles later u (1.06 → 2.05%) |
| More L-BFGS points at lb120 | Total u 0.83–0.87% vs 0.77% |
| Hard IC on cross and in 1D | Cross 20% u error; 1D solutions freeze at the IC (only stationary step improves) |
| Small-amplitude 1D ICs (cosine, noise) | ~99% error; weights, hard IC, net size, lr all fail |

## Not enough data

- **Seeds.** No run-to-run noise estimate. The three tied cross methods differ by
  <5%; two_circles adaptive variants all sit at 0.128–0.135%.
- **No plain baseline for the best runs.** No plain Sobol lb400 run at weights
  10000; the nearest is at 3000 (0.97% / 0.31% / 1.45%). The gain of
  adaptive/window methods over doing nothing extra is unmeasured.
- **Budget beyond lb400.** One pair (weights 3000, random, no held-out):
  two_circles improved (mu 0.86 → 0.44%), cross got worse later (0.24 → 1.30%).
- **Hard IC.** Helps 2D noise (u 1.79% vs 3.20%), fails on cross, but at
  different budgets (lb400 vs lb120). Blend form: one run.
- **PDE points at t=0.** One pair, mixed: IC much worse (1.97 vs 0.51%) but
  later u 0.14% and mu 1.09%, the best of any cross run.
- **Noise IC.** Only six runs at the current amplitude (3.5).
- **What predicts the error.** mu error follows final train `pde_u` (Spearman
  0.87–0.93 on cross). Later-time u among lb400 runs: nothing checked predicts it
  (≤ 0.17).
- **Never tested.** Energy penalty (0 runs), any net other than 5×100, any lr
  other than 1e-3, causal weighting, dissipation-based sampling, 2D beyond
  ε = 0.05 and t_max = 0.01.

## Suggested next step

Repeat the best cross setup and a plain Sobol lb400 baseline at weights 10000
over ~3 seeds. That settles whether the "tied" and "slightly better" results are
real and fills the missing baseline.
