#!/usr/bin/env python3
"""
Plot the applied advantages and available RS-GRPO and RS-SPO components.

Reads completed step summaries or legacy rollout metadata and writes PDFs to
<run>/adv_plots/. Linear hybrid components are scaled by their saved weights
by default. Use --unweighted to display raw component values.

Outputs
    1_evolution.pdf                how the advantages behave as steps move forward
    2_distribution_over_steps.pdf  per-step distribution (heatmap) of each advantage
    3_reward_to_advantage.pdf      reward -> advantage mapping, one curve cloud per step
    step_snapshots/step_XX.pdf     ONE step: advantage vs reward, available components,
                                   RS-SPO baseline + best-so-far raw score up to that step,
                                   same 3 advantage plots vs reward rank (symlog)

Usage
    python3.11 -m analysis.plot_advantages runs/gpu_mode_gpt-oss-120b_1120-1258
    python3.11 -m analysis.plot_advantages RUN --compare-steps 0 6 12 18 23
    python3.11 -m analysis.plot_advantages RUN --snapshots 0 12 23     # only these snapshots
    python3.11 -m analysis.plot_advantages RUN --no-snapshots

--max-step includes only checkpointed steps through that step number.
"""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ---- palette (dataviz reference instance, light surface) --------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_TTT, C_SPO, C_ALL = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua
C_BAD = "#d03b3b"
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ADVANTAGE_SERIES = (("wttt", C_TTT, "RS-GRPO"), ("wspo", C_SPO, "RS-SPO"),
                    ("adv", C_ALL, "overall"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "font.size": 9, "lines.linewidth": 2.0,
})


# ---- loading ----------------------------------------------------------------
def load_run(run, weighted=False, *, max_step=None):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    wt = float(cfg.get("hybrid_ttt_weight", 1.0))
    ws = float(cfg.get("hybrid_spo_weight", 1.0))
    steps, skipped = {}, []
    for directory in sorted(run.glob("step*")):
        match = re.fullmatch(r"step(\d+)", directory.name)
        if match is None or not directory.is_dir():
            continue
        step = int(match.group(1))
        if max_step is not None and step > max_step:
            continue
        summary_path = directory / f"{directory.name}.summary.json"
        try:
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            if max_step is not None and not summary.get("completed", False):
                skipped.append(step)
                continue
            metas = summary.get("advantage_plot_rollouts")
            if metas is None:
                metas = [json.loads(path.read_text()) for path in sorted(directory.glob("*.meta.json"))]
        except (OSError, json.JSONDecodeError):
            skipped.append(step)
            continue
        if (not metas or len(metas) != summary.get("planned_rollouts", len(metas))
                or any(meta.get("advantage") is None for meta in metas)):
            skipped.append(step)
            continue

        def f(key):
            return np.array([m.get(key) for m in metas], dtype=float)

        advantages = f("advantage")
        if not np.isfinite(advantages).any():
            skipped.append(step)
            continue
        mode = metas[0].get("advantage_mode") or cfg.get("advantage_mode", "hybrid")
        spo_info = metas[0].get("spo_rs") or {}
        linear_hybrid = (mode in {"hybrid", "hybrid-invariant"}
                         and not metas[0].get("legacy_invariant_rms")
                         and "hybrid_invariant_mixture_rms" not in spo_info)
        steps[step] = dict(
            group=f("group").astype(int), rollout=f("rollout").astype(int),
            reward=f("reward"), raw=f("raw_score"),
            valid=np.array([bool(m["valid"]) for m in metas]),
            ttt=f("ttt_advantage"), spo=f("spo_rs_advantage"),
            adv=advantages, v_t=f("spo_rs_v_t"),
            best_seen=summary.get("best_seen_raw_score"), maximize=summary.get("result_maximize"),
            linear_hybrid=linear_hybrid,
        )
        s = steps[step]
        if weighted and linear_hybrid:
            s["wttt"], s["wspo"] = wt * s["ttt"], ws * s["spo"]
        else:
            s["wttt"], s["wspo"] = s["ttt"], s["spo"]
    return cfg, wt, ws, steps, skipped


def check_weights(wt, ws, steps):
    errors = [np.abs(s["adv"] - (wt * s["ttt"] + ws * s["spo"]))
              for s in steps.values() if s["linear_hybrid"]]
    if errors:
        finite = finite_values(np.concatenate(errors))
        if finite.size:
            print(f"max |advantage - ({wt}*ttt + {ws}*spo)| over run = {finite.max():.2e}")


# ---- helpers ----------------------------------------------------------------
def style_axis(ax, xlabel=None, ylabel=None, title=None, zero=False):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if zero:
        ax.axhline(0, color=AXIS, lw=1, zorder=0)


def per_step(steps, fn):
    xs = sorted(steps)
    return np.array(xs), np.array([fn(steps[x]) for x in xs])


def finite_values(values):
    return values[np.isfinite(values)]


def statistic(values, fn):
    values = finite_values(values)
    return fn(values) if values.size else np.nan


def available_series(step_data):
    return [series for series in ADVANTAGE_SERIES
            if any(np.isfinite(step[series[0]]).any() for step in step_data)]


def corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return corr(pct_rank(a[mask]), pct_rank(b[mask]))


def pct_rank(x):
    """Percentile rank of each value among x (0 = lowest, 1 = highest; ties share a rank)."""
    x = np.asarray(x, dtype=float)
    ranks = np.full(x.shape, np.nan)
    finite = np.isfinite(x)
    values = x[finite]
    if values.size:
        _, indices, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks[finite] = (counts.cumsum() - 0.5 * counts)[indices] / len(values)
    return ranks


def mark_regimes(ax, steps):
    """Dotted vertical lines where the number of rollouts per step changes."""
    prev = None
    for x in sorted(steps):
        n = len(steps[x]["adv"])
        if prev is not None and n != prev:
            ax.axvline(x - 0.5, color=AXIS, lw=0.8, ls=":", zorder=0)
        prev = n


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(dir=path.parent, prefix=".plot-") as directory:
            pending = Path(directory) / path.name
            fig.savefig(pending, bbox_inches="tight")
            pending.replace(path)
    finally:
        plt.close(fig)
    print("wrote", path)


# ---- figure 1: evolution ----------------------------------------------------
def fig_evolution(steps, wt, ws, out):
    fig, axs = plt.subplots(3, 2, figsize=(13, 12))
    series = available_series(steps.values())

    ax = axs[0, 0]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: statistic(s[k], np.mean))
        ax.plot(x, y, color=c, label=lab, marker="o", ms=3)
    style_axis(ax, "step", "mean advantage",
               "Mean advantage per step", zero=True)
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[0, 1]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: statistic(s[k], np.std))
        ax.plot(x, y, color=c, label=lab, marker="o", ms=3)
    style_axis(ax, "step", "std of advantage",
               "Spread (std) of advantage per step")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[1, 0]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: statistic(s[k], lambda v: (v > 0).mean()))
        ax.plot(x, y, color=c, label=lab, marker="o", ms=3)
    ax.axhline(0.5, color=AXIS, lw=1, ls="--")
    ax.set_ylim(0, 1)
    style_axis(ax, "step", "fraction of rollouts", "Fraction with positive advantage")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[1, 1]
    if any((np.isfinite(s["ttt"]) & np.isfinite(s["spo"])).any() for s in steps.values()):
        x, y = per_step(steps, lambda s: corr(s["ttt"], s["spo"]))
        ax.plot(x, y, color=INK2, label="Pearson", marker="o", ms=3)
        x, y = per_step(steps, lambda s: spearman(s["ttt"], s["spo"]))
        ax.plot(x, y, color=MUTED, ls="--", label="Spearman", marker="o", ms=3)

        def sign_agreement(s):
            paired = np.isfinite(s["ttt"]) & np.isfinite(s["spo"])
            return (np.sign(s["ttt"][paired]) == np.sign(s["spo"][paired])).mean() if paired.any() else np.nan

        x, y = per_step(steps, sign_agreement)
        ax.plot(x, y, color=C_ALL, label="sign agreement (frac)", marker="o", ms=3)
        ax.set_ylim(-1.05, 1.05)
        style_axis(ax, "step", None, "Do RS-GRPO and RS-SPO agree?")
        ax.legend()
        mark_regimes(ax, steps)
    else:
        ax.set_visible(False)

    ax = axs[2, 0]
    x, y = per_step(steps, lambda s: s["reward"].mean())
    ax.plot(x, y, color=C_TTT, label="mean reward", marker="o", ms=3)
    x, y = per_step(steps, lambda s: s["reward"].max())
    ax.plot(x, y, color=C_SPO, label="best reward", marker="o", ms=3)
    x, y = per_step(steps, lambda s: s["valid"].mean())
    ax.plot(x, y, color=C_ALL, label="valid fraction", marker="o", ms=3)
    style_axis(ax, "step", None, "Context: reward and validity")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[2, 1]
    x, y = per_step(steps, lambda s: s["v_t"][0])
    if np.isfinite(y).any():
        ax.plot(x, y, color=C_SPO, marker="o", ms=3)
        style_axis(ax, "step", "RS-SPO value baseline v_t",
                   "Context: RS-SPO baseline (before the step's update)")
        mark_regimes(ax, steps)
    else:
        ax.set_visible(False)

    for ax in axs.flat:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    fig.suptitle("Advantage evolution across steps   "
                 "(dotted lines = rollouts-per-step changed)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out / "1_evolution.pdf")


# ---- figure 2: distribution heatmaps ----------------------------------------
def fig_distribution(steps, wt, ws, out):
    series = available_series(steps.values())
    xs = sorted(steps)
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 5.2),
                            sharey=True, squeeze=False)
    axs = axes[0]
    allv = finite_values(np.concatenate([steps[x][k] for x in xs for k, _, _ in series]))
    lo, hi = np.percentile(allv, [0.3, 99.7])
    if lo == hi:
        padding = max(0.5, abs(lo) * 0.01)
        lo, hi = lo - padding, hi + padding
    edges = np.linspace(lo, hi, 61)
    for ax, (k, _, lab) in zip(axs, series):
        H = np.full((len(edges) - 1, xs[-1] - xs[0] + 1), np.nan)
        for x in xs:
            v = finite_values(steps[x][k])
            if v.size:
                H[:, x - xs[0]] = np.histogram(np.clip(v, lo, hi), edges)[0] / len(v)
        im = ax.imshow(np.sqrt(H), origin="lower", aspect="auto", cmap=SEQ,
                       vmin=0, vmax=1,
                       extent=[xs[0] - 0.5, xs[-1] + 0.5, lo, hi],
                       interpolation="nearest")
        ax.plot(xs, [statistic(steps[x][k], np.mean) for x in xs], color=C_SPO, lw=1.8,
                marker="o", ms=3, label="mean")
        ax.plot(xs, [statistic(steps[x][k], np.median) for x in xs], color=C_ALL, lw=1.8,
                marker="x", ms=3, ls="--", label="median")
        ax.axhline(0, color=INK, lw=0.6, alpha=0.5)
        ax.grid(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        style_axis(ax, "step", None, lab)
    axs[0].set_ylabel("advantage value (ends clipped to 0.3/99.7 pct)")
    axs[0].legend(loc="upper left")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("sqrt(fraction of that step's rollouts in bin)")
    fig.suptitle("Distribution of each advantage, step by step",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, out / "2_distribution_over_steps.pdf")


# ---- figure 3: reward -> advantage per step ---------------------------------
def fig_reward_map(steps, wt, ws, sel, out):
    sel = [s for s in sel if s in steps]
    series = available_series(steps.values())
    fig, axes = plt.subplots(1, len(series), figsize=(5.4 * len(series), 5.3),
                            sharex=True, squeeze=False)
    axs = axes[0]
    norm = {s: i / max(1, len(sel) - 1) for i, s in enumerate(sel)}
    for ax, (k, _, lab) in zip(axs, series):
        for s in sel:
            d = steps[s]
            m = d["valid"]
            ax.scatter(d["reward"][m], d[k][m], s=10, alpha=0.55, lw=0,
                       color=SEQ(0.15 + 0.85 * norm[s]), label=f"step {s}")
            ax.scatter(d["reward"][~m], d[k][~m], s=14, marker="x", lw=0.8,
                       color=SEQ(0.15 + 0.85 * norm[s]), alpha=0.8)
        style_axis(ax, "reward (x = invalid)", None,
                   lab, zero=True)
    axs[0].set_ylabel("advantage")
    axs[0].legend(markerscale=2, ncol=2)
    fig.suptitle("Reward to advantage mapping per step (light = early, dark = late)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "3_reward_to_advantage.pdf")


# ---- per-step snapshot ------------------------------------------------------
def lower_is_better(steps):
    """Direction of the raw score, read from the data: reward falls as raw rises."""
    for step in steps.values():
        if isinstance(step["maximize"], bool):
            return not step["maximize"]
    raw = np.concatenate([s["raw"][s["valid"]] for s in steps.values()])
    rew = np.concatenate([s["reward"][s["valid"]] for s in steps.values()])
    return spearman(raw, rew) < 0


def best_so_far(steps):
    """Per step: best valid raw score seen in steps 0..step (direction-aware)."""
    minimize = lower_is_better(steps)
    xs, best, cur = sorted(steps), [], np.nan
    for x in xs:
        d = steps[x]
        if d["best_seen"] is not None and np.isfinite(d["best_seen"]):
            cur = d["best_seen"]
            best.append(cur)
            continue
        raw = d["raw"][d["valid"] & np.isfinite(d["raw"])]
        if raw.size:
            step_best = raw.min() if minimize else raw.max()
            if np.isnan(cur) or (step_best < cur if minimize else step_best > cur):
                cur = step_best
        best.append(cur)
    return np.array(xs), np.array(best), minimize


def fig_snapshot(step, steps, wt, ws, out):
    d = steps[step]
    ng = int(d["group"].max()) + 1
    gcol = np.array([GROUP_COLORS[g % len(GROUP_COLORS)] for g in d["group"]])
    v = d["valid"]
    series = available_series((d,))
    compare_components = len(series) == len(ADVANTAGE_SERIES)
    xv, yv = per_step(steps, lambda s: s["v_t"][0])
    baseline = (xv <= step) & np.isfinite(yv)
    has_baseline = baseline.any()

    # row 1: 4 panels, row 2: 3 panels (same width, white space between),
    # row 3: the first 3 row-1 panels re-plotted against reward rank (symlog y).
    H, rh = 13.9, 2.91                 # figure height / panel height, inches
    y1, y2, y3 = 9.86, 5.35, 0.84      # bottom edge of each row, inches
    top_count = len(series) + int(compare_components)
    bottom_count = 2 + int(has_baseline)
    columns = max(top_count, bottom_count)
    fig = plt.figure(figsize=(5 * columns, H))
    L, R, gap = 0.045, 0.94, 0.055
    w = (R - L - (columns - 1) * gap) / columns
    top = [fig.add_axes([L + i * (w + gap), y1 / H, w, rh / H]) for i in range(top_count)]
    bot = [fig.add_axes([x, y2 / H, w, rh / H]) for x in np.linspace(L, R - w, bottom_count)]
    rank = [fig.add_axes([L + i * (w + gap), y3 / H, w, rh / H]) for i in range(len(series))]

    # Advantage vs reward, followed by a comparison when both components exist.
    for ax, (k, _, lab) in zip(top, series):
        ax.scatter(d["reward"][v], d[k][v], c=gcol[v], s=22, alpha=0.8, lw=0)
        ax.scatter(d["reward"][~v], d[k][~v], c=gcol[~v], s=26, marker="x", lw=1)
        style_axis(ax, "reward (x = invalid)", "advantage",
                   f"{lab} vs reward", zero=True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLORS[g % len(GROUP_COLORS)],
                          label=f"g{g}") for g in np.unique(d["group"])]
    top[len(series) - 1].legend(handles=handles, ncol=2, fontsize=7.5, title="group",
                                title_fontsize=8, loc="upper left")

    if compare_components:
        ax = top[-1]
        sc = ax.scatter(d["wttt"][v], d["wspo"][v], c=d["adv"][v], cmap="viridis",
                        s=24, lw=0)
        ax.scatter(d["wttt"][~v], d["wspo"][~v], c=C_BAD, s=26, marker="x", lw=1,
                   label="invalid")
        fig.colorbar(sc, cax=ax.inset_axes([1.03, 0.0, 0.04, 1.0]),
                     label="overall advantage")
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(0, color=AXIS, lw=1)
        style_axis(ax, "RS-GRPO", "RS-SPO",
                   f"RS-GRPO vs RS-SPO   (corr {corr(d['ttt'], d['spo']):.2f})")
        if (~v).any():
            ax.legend()

    # bottom row: RS-SPO baseline so far, best so far (to this step), best so far (whole run)
    xs_all, best, minimize = best_so_far(steps)
    upto = xs_all <= step
    i = int(np.flatnonzero(upto)[-1])
    if has_baseline:
        ax = bot[0]
        ax.plot(xv[baseline], yv[baseline], color=C_SPO, marker="o", ms=3)
        if np.isfinite(d["v_t"][0]):
            ax.plot(step, d["v_t"][0], "o", color=C_SPO, ms=8)
        style_axis(ax, "step", "v_t (before the step's update)",
                   f"RS-SPO baseline through step {step}")

    ax = bot[-2]
    ax.plot(xs_all[upto], best[upto], color=C_TTT, marker="o", ms=3,
            drawstyle="steps-post")
    ax.plot(step, best[i], "o", color=C_TTT, ms=8)
    best_label = f"raw {best[i]:.5g}" if np.isfinite(best[i]) else "raw unavailable"
    style_axis(ax, "step", "best raw score so far",
               f"Best through step {step}   ({best_label})")

    ax = bot[-1]
    ax.plot(xs_all, best, color=C_TTT, alpha=0.35, drawstyle="steps-post")
    ax.plot(xs_all[upto], best[upto], color=C_TTT, drawstyle="steps-post")
    ax.axvline(step, color=INK2, lw=1, ls=":")
    ax.plot(step, best[i], "o", color=C_TTT, ms=8)
    best_label = f"raw {best[-1]:.5g}" if np.isfinite(best[-1]) else "raw unavailable"
    style_axis(ax, "step", "best raw score so far",
               f"Best across plotted steps   ({best_label})")
    for ax in bot:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    # row 3: same three plots, x = reward rank within the step, y = symlog advantage
    rk = pct_rank(d["reward"])
    for ax, (k, _, lab) in zip(rank, series):
        ax.scatter(rk[v], d[k][v], c=gcol[v], s=22, alpha=0.8, lw=0)
        ax.scatter(rk[~v], d[k][~v], c=gcol[~v], s=26, marker="x", lw=1)
        if np.nanmax(np.abs(d[k])) > 1.0:   # symlog only where the tail exceeds +-1
            ax.set_yscale("symlog", linthresh=1.0)
            lo, hi = ax.get_ylim()
            ticks = [q for q in (-8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
                     if lo <= q <= hi]
            if len(ticks) >= 3:
                ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(-0.03, 1.03)
        style_axis(ax, "reward rank within step (0 = worst, 1 = best; x = invalid)",
                   "advantage (symlog above +-1)", f"{lab} vs reward rank", zero=True)

    step_best = statistic(d["raw"][v], np.min if minimize else np.max)
    best_text = f"{step_best:.5g}" if np.isfinite(step_best) else "unavailable"
    baseline_text = f", SPO baseline v_t {d['v_t'][0]:.2f}" if np.isfinite(d["v_t"][0]) else ""
    fig.suptitle(
        f"Step {step}: {len(d['adv'])} rollouts, {ng} groups, valid {v.mean():.0%}, "
        f"best raw {best_text}, mean reward {d['reward'].mean():.2f}{baseline_text}",
        x=L, y=(y1 + rh + 0.8) / H, ha="left", fontsize=12,
        fontweight="bold")
    save(fig, out / "step_snapshots" / f"step_{step:02d}.pdf")


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="default: <run_dir>/adv_plots")
    ap.add_argument("--max-step", type=int, default=None,
                    help="include only checkpointed steps up to this step")
    ap.add_argument("--compare-steps", type=int, nargs="*", default=None,
                    help="steps overlaid in figure 3 (default: ~7 evenly spaced)")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None,
                    help="steps to snapshot (default: all completed steps)")
    ap.add_argument("--no-snapshots", action="store_true")
    ap.add_argument("--unweighted", action="store_true",
                    help="plot the raw advantages instead of the hybrid-weighted "
                         "ones (default: RS-GRPO x hybrid_ttt_weight, RS-SPO x "
                         "hybrid_spo_weight; labels are the same either way)")
    a = ap.parse_args()

    _, wt, ws, steps, skipped = load_run(a.run_dir, not a.unweighted, max_step=a.max_step)
    if not steps:
        raise SystemExit("no completed steps with advantage fields found")
    if skipped:
        print("skipped (incomplete) steps:", skipped)
    print(f"{len(steps)} steps loaded; weights ttt={wt}, spo={ws}")
    check_weights(wt, ws, steps)
    out = Path(a.out) if a.out else Path(a.run_dir) / (
        "adv_plots_unweighted" if a.unweighted else "adv_plots")

    fig_evolution(steps, wt, ws, out)
    fig_distribution(steps, wt, ws, out)
    xs = sorted(steps)
    sel = a.compare_steps or sorted({xs[int(i)] for i in
                                     np.linspace(0, len(xs) - 1, 7)})
    fig_reward_map(steps, wt, ws, sel, out)
    if not a.no_snapshots:
        for s in (a.snapshots if a.snapshots is not None else xs):
            if s in steps:
                fig_snapshot(s, steps, wt, ws, out)


if __name__ == "__main__":
    main()
