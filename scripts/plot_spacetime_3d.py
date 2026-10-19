"""Interactive 3D space-time plot of a trained PINN: x, y horizontal, t upward.

Evaluates the network on an (nx, ny, nt) grid over its whole trained window
and writes a standalone HTML file (plotly) that can be rotated, zoomed and
panned in any browser -- no Python needed to view it. To keep the inside
visible, the solid phases are not drawn; the plot is built from layers that
can each be toggled by clicking their legend entry:

  interface   the u = 0 isosurface (semi-transparent), i.e. the phase
              boundary swept through time -- a "space-time tube"
  rings       the u = 0 contour at --rings evenly spaced times, coloured by
              t, so the motion of the interface reads off directly
  slices      semi-transparent u(x, y) heatmaps at the --slice-t times
              (default: first, middle and last)
  volume      fog volume rendering of the diffuse interface band
              (|u| small); only with --volume (roughly doubles the file
              size), and then hidden until clicked in the legend

Run from the repo root:
    .venv\\Scripts\\python.exe scripts/plot_spacetime_3d.py <run-name>
    .venv\\Scripts\\python.exe scripts/plot_spacetime_3d.py <run-name> --field mu --iso 0
    .venv\\Scripts\\python.exe scripts/plot_spacetime_3d.py <run-name> --slice-t 0 0.002 0.01 --open
    .venv\\Scripts\\python.exe scripts/plot_spacetime_3d.py <run-name> --volume --nx 60
    .venv\\Scripts\\python.exe scripts/plot_spacetime_3d.py --checkpoint path/to/checkpoint_step5000.pt

Output defaults to results/pinn_models/<run-name>/plots/spacetime_<field>.html.
Requires plotly (.venv\\Scripts\\python.exe -m pip install plotly).
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import contourpy
import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pinn.evaluate import evaluate_over_time, load_model  # noqa: E402
from src.pinn.run_paths import final_path_for, infer_run_dir  # noqa: E402

# Same blue -> neutral gray -> red as src/common/plotting.py::phase_colormap.
PHASE_COLORSCALE = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_name", nargs="?", default=None,
                        help="Run folder under results/pinn_models/ (uses its final.pt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit .pt checkpoint instead of <run_name>/final.pt")
    parser.add_argument("--field", choices=["u", "mu"], default="u")
    parser.add_argument("--iso", type=float, default=0.0,
                        help="Value of the isosurface and rings (u = 0 is the interface)")
    parser.add_argument("--nx", type=int, default=80, help="Grid points in x and y")
    parser.add_argument("--nt", type=int, default=41, help="Time levels in [0, t_max]")
    parser.add_argument("--t-max", type=float, default=None,
                        help="End of the plotted time window (default: the trained t_max)")
    parser.add_argument("--rings", type=int, default=11,
                        help="Number of interface contours stacked in time (0 to skip)")
    parser.add_argument("--slice-t", type=float, nargs="*", default=None,
                        help="Times of the heatmap slices (default: first, middle, last)")
    parser.add_argument("--slice-opacity", type=float, default=0.55)
    parser.add_argument("--iso-opacity", type=float, default=0.35)
    parser.add_argument("--t-aspect", type=float, default=1.2,
                        help="Height of the t axis relative to the x/y box")
    parser.add_argument("--volume", action="store_true",
                        help="Add the fog volume layer of the interface band")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .html path")
    parser.add_argument("--open", action="store_true", help="Open the HTML in the browser")
    args = parser.parse_args()
    if (args.run_name is None) == (args.checkpoint is None):
        parser.error("give exactly one of run_name or --checkpoint")
    return args


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def interface_rings(x, y, t, field, iso, n_rings) -> go.Scatter3d | None:
    """The iso-contour at n_rings times as one line trace, coloured by t."""
    if n_rings <= 0:
        return None
    xs, ys, ts = [], [], []
    for k in np.unique(np.linspace(0, len(t) - 1, n_rings).round().astype(int)):
        # contourpy wants z indexed [y, x]; the evaluation grid is [x, y].
        lines = contourpy.contour_generator(x, y, field[k].T).lines(iso)
        for line in lines:
            xs += [*line[:, 0], np.nan]
            ys += [*line[:, 1], np.nan]
            ts += [t[k]] * len(line) + [np.nan]
    if not xs:
        return None
    # NaN breaks the line between rings; its colour entry just needs to be finite.
    colors = np.nan_to_num(np.array(ts), nan=t[0])
    return go.Scatter3d(
        x=xs, y=ys, z=ts, mode="lines", name=f"rings ({n_rings} times)",
        line=dict(color=colors, colorscale="Viridis", cmin=t[0], cmax=t[-1], width=5,
                  colorbar=dict(title="t", x=1.12, len=0.6)),
        hoverinfo="x+y+z",
    )


def build_figure(x, y, t, field, args, field_label) -> go.Figure:
    nx, ny, nt = len(x), len(y), len(t)
    # (nt, nx, ny) -> flat arrays in matching [x, y, t] order for Isosurface/Volume.
    X3, Y3, T3 = np.meshgrid(x, y, t, indexing="ij")
    values = np.moveaxis(field, 0, -1)
    # float32 halves the HTML size; plotly embeds numpy arrays as binary.
    flat = {k: a.ravel().astype(np.float32)
            for k, a in dict(x=X3, y=Y3, z=T3, value=values).items()}

    is_u = args.field == "u"
    cmin, cmax = (-1.0, 1.0) if is_u else (float(field.min()), float(field.max()))
    colorscale = PHASE_COLORSCALE if is_u else "RdBu_r"

    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        **flat, isomin=args.iso, isomax=args.iso, surface_count=1,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[[0, "#8a8f98"], [1, "#8a8f98"]], showscale=False,
        opacity=args.iso_opacity, name=f"interface ({field_label} = {args.iso:g})",
        showlegend=True, lighting=dict(ambient=0.6, diffuse=0.6, specular=0.2, roughness=0.6),
        hoverinfo="skip",
    ))

    rings = interface_rings(x, y, t, field, args.iso, args.rings)
    if rings is not None:
        fig.add_trace(rings)

    slice_times = args.slice_t if args.slice_t is not None else [t[0], t[nt // 2], t[-1]]
    for i, ts in enumerate(slice_times):
        k = nearest_index(t, ts)
        fig.add_trace(go.Surface(
            x=x, y=y, z=np.full((ny, nx), t[k]), surfacecolor=field[k].T,
            colorscale=colorscale, cmin=cmin, cmax=cmax, opacity=args.slice_opacity,
            showscale=(i == 0), colorbar=dict(title=field_label, x=1.0, len=0.6),
            name=f"slice t = {t[k]:.4g}", showlegend=True, legendgroup="slices",
            hovertemplate=f"x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>{field_label}=%{{surfacecolor:.3f}}<extra>t={t[k]:.4g}</extra>",
        ))

    # Opaque only in a band around the iso value: the diffuse interface region.
    band = 0.6 * (cmax - cmin) / 2
    if args.volume:
        fig.add_trace(go.Volume(
            **flat, isomin=args.iso - band, isomax=args.iso + band, surface_count=12,
            opacity=0.12, colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="volume (interface band)", showlegend=True, visible="legendonly",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"{args.title} — {field_label}(x, y, t)",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="t",
            aspectmode="manual", aspectratio=dict(x=1, y=(y[-1] - y[0]) / (x[-1] - x[0]), z=args.t_aspect),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        legend=dict(x=0.0, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white",
    )
    return fig


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / final_path_for(args.run_name)
    if not checkpoint.exists():
        sys.exit(f"No checkpoint at {checkpoint}.")
    args.title = args.run_name or checkpoint.stem

    model = load_model(str(checkpoint), device=args.device)
    t_max = args.t_max if args.t_max is not None else float(model.upper_bound[2])
    t = np.linspace(float(model.lower_bound[2]), t_max, args.nt)
    ny = round(args.nx * float(model.upper_bound[1] - model.lower_bound[1])
               / float(model.upper_bound[0] - model.lower_bound[0]))
    X, Y, U, MU = evaluate_over_time(model, t, args.nx, ny, device=args.device)
    field = U if args.field == "u" else MU
    x, y = X[:, 0], Y[0, :]

    fig = build_figure(x, y, t, field, args, "u" if args.field == "u" else "μ")

    output = args.output or Path(infer_run_dir(str(checkpoint))) / "plots" / f"spacetime_{args.field}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)
    print(f"Wrote {output}")
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
