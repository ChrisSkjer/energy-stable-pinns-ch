---
name: add-figure
description: Add figures from results/ (PINN runs or FEM runs) into the LaTeX research notes at Report/Research_Notes/main.tex. Use when the user asks to put a plot, field snapshot, comparison, loss curve, energy/mass plot or FEM frame from a run into their notes, e.g. "add the loss curve for the cross run to the notes" or "show u at t=0, 0.005, 0.01 side by side under the FEM section".
---

# Add a figure from results/ to the research notes

The notes are `Report/Research_Notes/main.tex`. Figures live under
`Report/Research_Notes/figures/` and are referenced relative to that folder's
parent (`figures/<run>/<file>`). `results/` is gitignored, so a figure must be
**copied** into `figures/`; never `\includegraphics` a path under `results/`.

## 1. Find the source files

- **PINN runs**: `results/pinn_models/<run>/plots/`. These were made by `plot_results`
  and are PDFs by default (PNGs if `--format png` was used). The files are:
  `loss_history`, `energy_dissipation`, `mass_conservation`, `relative_l2_error`,
  `field_u_<idx>_t<t>`, `field_mu_<idx>_t<t>`, `comparison_<idx>_t<t>` (u, PINN vs FEM),
  `comparison_mu_<idx>_t<t>`, `profile_<idx>_t<t>` (line cut), and `run_summary.txt`.
- **FEM runs**: `results/<fem_run>/frames/frame_NNNNNN.png`, plus
  `cahn_hilliard_diagnostics.csv` and `fem_fields.npz`.
- Run names are long (`eps0.05_h100x5_..._ic-cross`). If the user's description
  matches more than one run (e.g. "the cross run"), list the candidates and ask.
  Don't guess.
- If the plot the user wants doesn't exist yet (wrong time, field plots not made,
  no FEM comparison), generate it with the commands in `docs/post_training_cheatsheet.md`
  (`scripts/analyze_run.py` or `python -m src.pinn.plot_results ...` from the repo root
  with `.venv\Scripts\python.exe`). Use `--field-plots` for `field_*`, and
  `--fem-fields results/<fem_run>/fem_fields.npz` for comparisons/profiles/L2 error.
  Before running anything slow, tell the user what you're about to run.
- If the user asks for a plot that no existing script makes, write a small script
  in `scripts/` (look at `scripts/plot_transitions.py` for the style), save the
  output as a PDF, and say that you did.

## 2. Copy into figures/

Copy into `Report/Research_Notes/figures/<run>/`, using the same run folder name as
in `results/` so it's clear where each figure came from. Keep the original file
names. Prefer the PDF. Don't overwrite a file that's already there unless the
user asked to refresh it.

## 3. Insert the LaTeX

Put the figure where the user asked. If they named a section, add it at the end
of that section's text, before the next `\subsection`/`\section`. If they didn't
say where, ask. Skip commented-out (`%`) regions. Follow the notes' existing style:

Single figure:
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\linewidth]{figures/<run>/loss_history.pdf}
    \caption{<What it shows, plus the run's key settings: IC, epsilon, weights, epochs.>}
    \label{fig:<short-unique-name>}
\end{figure}
```

Time series (2 to 4 snapshots) as subfigures in one row:
```latex
\begin{figure}[H]
\centering
\begin{subfigure}[b]{0.32\linewidth}
  \includegraphics[width=\linewidth]{figures/<run>/field_u_000_t0.pdf}
  \caption{$t = 0$}
  \label{fig:<name>-t0}
\end{subfigure}
\hfill
... one subfigure per time ...
\caption{<Overall caption.>}
\label{fig:<name>}
\end{figure}
```

Typical widths: loss, energy and L2 plots `0.6\linewidth`; single field `0.5`;
comparisons (which are wide multi-panel figures) `0.8` to `1.0`; subfigures
`0.32` for three, `0.48` for two.

Rules:
- **Every label must be unique.** The older figures in the notes all reuse
  `fig:placeholder`/`fig:a`; don't copy that. Base the label on what the figure
  shows and the run (e.g. `fig:cross-pdeat-loss`), and grep `main.tex` to make
  sure the label isn't already used.
- Write a real caption, not `Caption`. Use `run_summary.txt` to get the settings
  right, and use the notes' macros (`\pfield`, `\Lterm{ic}`, `\varepsilon`).
- Reference the figure with `\Cref{fig:...}` if you add any text around it.
  Only add text if the user asks; the notes are in the user's own words.

## 4. Compile and check

From `Report/Research_Notes/`, run
`pdflatex -interaction=nonstopmode -halt-on-error main.tex` and grep `main.log`
for `^!` errors and `multiply defined` labels. Report the result, and say which
files were copied and where in `main.tex` the figure went (as a clickable link
with the line number).

Figures are sized for notes only. If the user says the figure is for the
thesis itself, raise the thesis figure preparation needed first: sizing to
the text width, font sizes, and moving titles into captions.
