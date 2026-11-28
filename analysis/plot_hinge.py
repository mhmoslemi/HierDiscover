#!/usr/bin/env python3
"""
One step, RS-GRPO and frontier hinge side by side: advantage vs reward (full) + top-15% zoom.

Recomputes the hinge from RS-GRPO sibling advantages (same call as training)
with frontier = best valid reward before the step. The zoom panel draws a dotted
curve through each linked group's points, continued past the frontier.

Usage
    python3.11 -m analysis.plot_hinge runs/gpu_mode_gpt-oss-120b_1120-1258
    python3.11 -m analysis.plot_hinge RUN --step 5 --link 0 3     # blue + yellow
Writes <run>/adv_plots/hinge.png
"""

import argparse
import sys
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.plot_advantages import GROUP_COLORS, INK2, load_run, save, style_axis  # noqa: E402
from training.hinge import hinge_advantages  # noqa: E402

ZOOM_BG = "#fff3d6"
DOT, ZOOM_DOT = 14, 23.1         # zoom dots: 1.5 x 1.1 larger than the top panel
ENDPOINT_ARROW_STYLE = {
    0: dict(length=0.28, head_length=0.080),  # blue, in inches
    3: dict(length=0.26, head_length=0.074),  # yellow, in inches
}


def link_curve(x, y, x_end, n=200):
    """Smooth curve through (x, y), continued past the last point along its end slope."""
    x, i = np.unique(x, return_index=True)
    y = y[i]
    f = PchipInterpolator(x, y)
    assert np.allclose(f(x), y)
    xs = np.linspace(x[0], x_end, n)
    ys = np.where(xs <= x[-1], f(np.minimum(xs, x[-1])), y[-1] + f(x[-1], 1) * (xs - x[-1]))
    return xs, ys


def endpoint_hinge_arrow(fig, ax, pivot, endpoint, color, *, length, head_length):
    """Show the endpoint rotating about the frontier pivot."""
    pivot_inches, end = ax.transData.transform([pivot, endpoint]) / fig.dpi
    radial = end - pivot_inches
    pivot_distance = np.hypot(*radial)
    if pivot_distance <= length / 2:
        return

    # Keep the bend visible and limit the sweep when the dots are close to the pivot.
    radius = max(0.19 * pivot_distance, length / 2)
    # Offset the midpoint outward so the arrow stays separate from the endpoint dot.
    outward = radial / pivot_distance
    midpoint = end + 0.117 * outward
    center = midpoint - radius * outward
    # Work in inches so unequal axis scales preserve the circular path.
    start_angle = np.arctan2(radial[1], radial[0]) - length / (2 * radius)
    join_to_tip = 0.6 * head_length
    join_angle = start_angle + (length - join_to_tip) / radius
    angles = np.linspace(start_angle, join_angle, 100)
    shaft = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.array([-np.sin(join_angle), np.cos(join_angle)])
    tip = shaft[-1] + join_to_tip * tangent
    thickness_scale = 1.15
    wing = 0.36 * thickness_scale * head_length * np.array([-tangent[1], tangent[0]])
    base = tip - head_length * tangent
    tip_poly = [tip, base + wing, tip - 0.72 * head_length * tangent, base - wing]
    size = fig.get_size_inches()
    ax.add_line(Line2D(*((center + shaft) / size).T, transform=fig.transFigure, color=color,
                       lw=1.15 * thickness_scale, solid_capstyle="round", solid_joinstyle="round", zorder=2.5))
    ax.add_patch(Polygon((center + np.array(tip_poly)) / size, closed=True, transform=fig.transFigure,
                         fc=color, ec="none", zorder=2.5))


@plt.rc_context({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#e2e2e2",
})
def fig_hinge(steps, step, kappa, link, out):
    d = steps[step]
    r, v, g = d["reward"], d["valid"], d["group"]
    frontier = d["hinge_frontier"]
    if frontier is None:
        frontier = max(s["reward"][s["valid"]].max() for k, s in steps.items() if k < step)
    col = np.array([GROUP_COLORS[x % len(GROUP_COLORS)] for x in g])
    hinge = hinge_advantages(r, v, d["ttt"], frontier, kappa=kappa)

    fig, axs = plt.subplots(2, 2, figsize=(10.4, 7.4), dpi=200)
    fig.subplots_adjust(top=0.9, hspace=0.38, wspace=0.22)
    n = ceil(0.15 * len(r))
    lo = np.sort(r)[-n]
    hi = max(frontier, r.max())
    pad = 0.08 * (hi - lo)
    m = v & (r >= lo)
    for (top, zoom), (name, adv) in zip(axs.T, (("RS-GRPO", d["ttt"]), ("Frontier hinge", hinge))):
        top.scatter(r[v], adv[v], c=col[v], s=DOT, alpha=0.85, lw=0)
        top.scatter(r[~v], adv[~v], c=col[~v], s=22, marker="x", lw=1)
        style_axis(top, "Reward", "Advantage", name, zero=True)

        zoom.set_facecolor(ZOOM_BG)
        zoom.scatter(r[m], adv[m], c=col[m], s=ZOOM_DOT, alpha=0.9, lw=0, zorder=3)
        curves = []
        for grp in link:
            k = m & (g == grp)
            if k.sum() >= 2:
                c = GROUP_COLORS[grp % len(GROUP_COLORS)]
                curves.append((*link_curve(r[k], adv[k], hi + pad), c, grp))
                zoom.plot(*curves[-1][:2], color=c, ls=":", lw=1.0, zorder=2)
        zoom.set_xlim(lo - pad, hi + pad)
        style_axis(zoom, "Reward", "Advantage", zero=True)
        if name == "RS-GRPO":                       # hinge pivots each curve at the frontier
            zoom.set_ylim(zoom.get_ylim())
            for xs, ys, c, grp in curves:
                y0 = np.interp(frontier, xs, ys)
                zoom.plot(frontier, y0, "x", color="k", ms=5, mew=1.6, zorder=6)
                if grp in ENDPOINT_ARROW_STYLE:
                    endpoint_hinge_arrow(fig, zoom, (frontier, y0), (xs[0], ys[0]), c,
                                         **ENDPOINT_ARROW_STYLE[grp])
        for ax in (top, zoom):
            ax.axvline(frontier, color=INK2, lw=1, ls="--", zorder=1)

    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLORS[x % len(GROUP_COLORS)],
                          label=f"g{x}") for x in np.unique(g)]
    handles += [plt.Line2D([], [], marker="x", ls="", color="k", label="Invalid rollout"),
                plt.Line2D([], [], color=INK2, ls="--", label="Frontier"),
                Patch(fc=ZOOM_BG, ec="#d9c38f", label="Top 15% zoom")]
    fig.legend(handles=handles, ncol=len(handles), fontsize=7.5, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), columnspacing=1.2, handletextpad=0.3)
    save(fig, out / "hinge.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=0.7)
    ap.add_argument("--link", type=int, nargs="*", default=[0, 3],
                    help="groups joined by a dotted curve in the zoom (default: 0 blue, 3 yellow)")
    a = ap.parse_args()
    steps = load_run(a.run_dir)[3]
    fig_hinge(steps, a.step, a.kappa, a.link, Path(a.run_dir) / "adv_plots")
