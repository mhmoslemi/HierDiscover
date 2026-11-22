#!/usr/bin/env python3
"""
Plot the RS-GRPO (a.k.a. TTT, meta key ttt_advantage) / RS-SPO / overall (hybrid) advantages saved in each rollout meta.

Reads  <run>/stepXX/*.meta.json   (fields: ttt_advantage, spo_rs_advantage,
advantage, reward, raw_score, valid, ...) and writes PNGs to <run>/adv_plots/.

    overall = hybrid_ttt_weight * ttt + hybrid_spo_weight * spo_rs     (config.json)
Plots show RS-GRPO (meta key ttt_advantage) and RS-SPO scaled by their hybrid weights, so they add up to the overall; --unweighted gives the raw values.

Outputs
    1_evolution.png                how the advantages behave as steps move forward
    2_distribution_over_steps.png  per-step distribution (heatmap) of each advantage
    3_reward_to_advantage.png      reward -> advantage mapping, one curve cloud per step
    4_across_steps_by_group.png    per-step, per-group mean advantage components
    step_snapshots/step_XX.png     everything about ONE step (rollout-level)

Usage
    python3.11 -m analysis.plot_advantages runs/gpu_mode_gpt-oss-120b_1120-1258
    python3.11 -m analysis.plot_advantages RUN --compare-steps 0 6 12 18 23
    python3.11 -m analysis.plot_advantages RUN --snapshots 0 12 23     # only these snapshots
    python3.11 -m analysis.plot_advantages RUN --no-snapshots

Steps that are still running (rollouts without advantage fields) are skipped.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
def load_run(run, weighted=False):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    wt = float(cfg.get("hybrid_ttt_weight", 1.0))
    ws = float(cfg.get("hybrid_spo_weight", 1.0))
    steps, skipped = {}, []
    for d in sorted(glob.glob(str(run / "step[0-9][0-9]"))):
        step = int(re.search(r"step(\d+)$", d).group(1))
        metas = [json.loads(Path(f).read_text())
                 for f in sorted(glob.glob(d + "/*.meta.json"))]
        if not metas or any("advantage" not in m or "ttt_advantage" not in m
                            for m in metas):
            skipped.append(step)
            continue
        f = lambda k, default=np.nan: np.array(  # noqa: E731
            [m.get(k) if m.get(k) is not None else default for m in metas],
            dtype=float)
        steps[step] = dict(
            group=f("group").astype(int), rollout=f("rollout").astype(int),
            reward=f("reward"), raw=f("raw_score"),
            valid=np.array([bool(m["valid"]) for m in metas]),
            ttt=f("ttt_advantage"), spo=f("spo_rs_advantage"),
            adv=f("advantage"), tokens=f("n_response_tokens"),
            parent_value=f("parent_value"), parent_raw=f("parent_raw_score"),
            v_t=f("spo_rs_v_t"), parent_group=f("parent_group").astype(int),
        )
        s = steps[step]
        if weighted:   # same labels, values scaled by the hybrid weights
            s["wttt"], s["wspo"] = wt * s["ttt"], ws * s["spo"]
        else:          # original, unweighted
            s["wttt"], s["wspo"] = s["ttt"], s["spo"]
        s["improve"] = s["reward"] - s["parent_value"]
    return cfg, wt, ws, steps, skipped


def check_weights(wt, ws, steps):
    err = max(np.abs(s["adv"] - (wt * s["ttt"] + ws * s["spo"])).max()
              for s in steps.values())
    print(f"max |advantage - ({wt}*ttt + {ws}*spo)| over run = {err:.2e}")


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


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return corr(ra, rb)


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
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---- figure 1: evolution ----------------------------------------------------
def fig_evolution(steps, wt, ws, out):
    fig, axs = plt.subplots(3, 2, figsize=(13, 12))
    series = [("wttt", C_TTT, "RS-GRPO"), ("wspo", C_SPO, "RS-SPO"),
              ("adv", C_ALL, "overall")]

    ax = axs[0, 0]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: s[k].mean())
        ax.plot(x, y, color=c, label=lab)
    style_axis(ax, "step", "mean advantage",
               "Mean advantage per step", zero=True)
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[0, 1]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: s[k].std())
        ax.plot(x, y, color=c, label=lab)
    style_axis(ax, "step", "std of advantage",
               "Spread (std) of advantage per step")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[1, 0]
    for k, c, lab in series:
        x, y = per_step(steps, lambda s, k=k: (s[k] > 0).mean())
        ax.plot(x, y, color=c, label=lab)
    ax.axhline(0.5, color=AXIS, lw=1, ls="--")
    ax.set_ylim(0, 1)
    style_axis(ax, "step", "fraction of rollouts", "Fraction with positive advantage")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[1, 1]
    x, y = per_step(steps, lambda s: corr(s["ttt"], s["spo"]))
    ax.plot(x, y, color=INK2, label="Pearson")
    x, y = per_step(steps, lambda s: spearman(s["ttt"], s["spo"]))
    ax.plot(x, y, color=MUTED, ls="--", label="Spearman")
    x, y = per_step(steps, lambda s: np.mean(np.sign(s["ttt"]) == np.sign(s["spo"])))
    ax.plot(x, y, color=C_ALL, label="sign agreement (frac)")
    ax.set_ylim(-0.05, 1.05)
    style_axis(ax, "step", None, "Do RS-GRPO and RS-SPO agree?")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[2, 0]
    x, y = per_step(steps, lambda s: s["reward"].mean())
    ax.plot(x, y, color=C_TTT, label="mean reward")
    x, y = per_step(steps, lambda s: s["reward"].max())
    ax.plot(x, y, color=C_SPO, label="best reward")
    x, y = per_step(steps, lambda s: s["valid"].mean())
    ax.plot(x, y, color=C_ALL, label="valid fraction")
    style_axis(ax, "step", None, "Context: reward and validity")
    ax.legend()
    mark_regimes(ax, steps)

    ax = axs[2, 1]
    x, y = per_step(steps, lambda s: s["v_t"][0])
    ax.plot(x, y, color=C_SPO)
    style_axis(ax, "step", "RS-SPO value baseline v_t",
               "Context: RS-SPO baseline (before the step's update)")
    mark_regimes(ax, steps)

    fig.suptitle("Advantage evolution across steps   "
                 "(dotted lines = rollouts-per-step changed)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out / "1_evolution.png")


# ---- figure 2: distribution heatmaps ----------------------------------------
def fig_distribution(steps, wt, ws, out):
    keys = [("wttt", "RS-GRPO"), ("wspo", "RS-SPO"), ("adv", "overall")]
    xs = sorted(steps)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    allv = np.concatenate([steps[x][k] for x in xs for k, _ in keys])
    lo, hi = np.percentile(allv, [0.3, 99.7])
    edges = np.linspace(lo, hi, 61)
    for ax, (k, lab) in zip(axs, keys):
        H = np.zeros((len(edges) - 1, len(xs)))
        for j, x in enumerate(xs):
            v = np.clip(steps[x][k], lo, hi - 1e-9)
            H[:, j] = np.histogram(v, edges)[0] / len(v)
        im = ax.imshow(np.sqrt(H), origin="lower", aspect="auto", cmap=SEQ,
                       extent=[xs[0] - 0.5, xs[-1] + 0.5, lo, hi],
                       interpolation="nearest")
        ax.plot(xs, [steps[x][k].mean() for x in xs], color=C_SPO, lw=1.8,
                label="mean")
        ax.plot(xs, [np.median(steps[x][k]) for x in xs], color=C_ALL, lw=1.8,
                ls="--", label="median")
        ax.axhline(0, color=INK, lw=0.6, alpha=0.5)
        ax.grid(False)
        style_axis(ax, "step", None, lab)
    axs[0].set_ylabel("advantage value (ends clipped to 0.3/99.7 pct)")
    axs[0].legend(loc="upper left")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("sqrt(fraction of that step's rollouts in bin)")
    fig.suptitle("Distribution of each advantage, step by step",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, out / "2_distribution_over_steps.png")


# ---- figure 3: reward -> advantage per step ---------------------------------
def fig_reward_map(steps, wt, ws, sel, out):
    sel = [s for s in sel if s in steps]
    keys = [("wttt", "RS-GRPO"), ("wspo", "RS-SPO"), ("adv", "overall")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True)
    norm = {s: i / max(1, len(sel) - 1) for i, s in enumerate(sel)}
    for ax, (k, lab) in zip(axs, keys):
        for s in sel:
            d = steps[s]
            m = d["valid"]
            ax.scatter(d["reward"][m], d[k][m], s=10, alpha=0.55, lw=0,
                       color=SEQ(0.15 + 0.85 * norm[s]), label=f"step {s}")
            ax.scatter(d["reward"][~m], d[k][~m], s=14, marker="x", lw=0.8,
                       color=SEQ(0.15 + 0.85 * norm[s]), alpha=0.8)
        style_axis(ax, "reward (higher = faster kernel; x = invalid)", None,
                   lab, zero=True)
    axs[0].set_ylabel("advantage")
    axs[0].legend(markerscale=2, ncol=2)
    fig.suptitle("Reward -> advantage mapping per step "
                 "(light = early, dark = late). "
                 "RS-GRPO is group-relative so its map shifts per group; "
                 "RS-SPO uses a run-wide baseline so its curve moves with time",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "3_reward_to_advantage.png")


# ---- figure 4: per-group components across steps ----------------------------
def fig_groups(steps, wt, ws, out):
    xs = sorted(steps)
    ng = max(int(steps[x]["group"].max()) for x in xs) + 1
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for ax, (k, lab) in zip(axs, [("wttt", "RS-GRPO"),
                                  ("wspo", "RS-SPO"), ("adv", "overall")]):
        for g in range(ng):
            y = [steps[x][k][steps[x]["group"] == g].mean() for x in xs]
            ax.plot(xs, y, color=GROUP_COLORS[g % 8], lw=1.4, alpha=0.9,
                    label=f"group {g}")
        style_axis(ax, "step", "group-mean advantage", f"{lab}: mean per group",
                   zero=True)
    axs[0].legend(ncol=2, fontsize=7.5)
    fig.suptitle("Per-group mean advantage across steps "
                 "(groups = PUCT parents; group ids are re-drawn each step, so "
                 "lines show spread between groups, not a persistent lineage)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / "4_across_steps_by_group.png")


# ---- per-step snapshot ------------------------------------------------------
def fig_snapshot(step, d, wt, ws, out):
    ng = int(d["group"].max()) + 1
    gcol = np.array([GROUP_COLORS[g % 8] for g in d["group"]])
    v = d["valid"]
    fig, axs = plt.subplots(3, 3, figsize=(15, 13))

    # row 1: advantage vs reward
    for ax, (k, lab, c) in zip(axs[0], [("wttt", "RS-GRPO", C_TTT),
                                        ("wspo", "RS-SPO", C_SPO),
                                        ("adv", "overall", C_ALL)]):
        ax.scatter(d["reward"][v], d[k][v], c=gcol[v], s=22, alpha=0.8, lw=0)
        ax.scatter(d["reward"][~v], d[k][~v], c=gcol[~v], s=26, marker="x", lw=1)
        style_axis(ax, "reward (x = invalid)", "advantage",
                   f"{lab} vs reward", zero=True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GROUP_COLORS[g % 8],
                          label=f"g{g}") for g in range(ng)]
    axs[0, 2].legend(handles=handles, ncol=2, fontsize=7.5, title="group",
                     title_fontsize=8)

    # row 2 col 1: TTT vs SPO
    ax = axs[1, 0]
    sc = ax.scatter(d["wttt"][v], d["wspo"][v], c=d["adv"][v], cmap="viridis",
                    s=24, lw=0)
    ax.scatter(d["wttt"][~v], d["wspo"][~v], c=C_BAD, s=26, marker="x", lw=1,
               label="invalid")
    fig.colorbar(sc, ax=ax, label="overall advantage", shrink=0.85)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    style_axis(ax, "RS-GRPO", "RS-SPO",
               f"RS-GRPO vs RS-SPO   (corr {corr(d['ttt'], d['spo']):.2f})")
    if (~v).any():
        ax.legend()

    # row 2 col 2: per-group strip of overall advantage
    ax = axs[1, 1]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.18, 0.18, len(d["group"]))
    ax.scatter(d["group"][v] + jit[v], d["adv"][v], c=gcol[v], s=18, alpha=0.8,
               lw=0)
    ax.scatter(d["group"][~v] + jit[~v], d["adv"][~v], c=gcol[~v], s=24,
               marker="x", lw=1)
    for g in range(ng):
        m = d["group"] == g
        ax.hlines(d["adv"][m].mean(), g - 0.3, g + 0.3, color=INK, lw=2)
    ax.set_xticks(range(ng))
    style_axis(ax, "group (bar = group mean)", "overall advantage",
               "Overall advantage inside each group", zero=True)

    # row 2 col 3: histograms
    ax = axs[1, 2]
    allv = np.concatenate([d["wttt"], d["wspo"], d["adv"]])
    bins = np.linspace(allv.min(), allv.max(), 40)
    for k, c, lab in [("wttt", C_TTT, "RS-GRPO"), ("wspo", C_SPO, "RS-SPO"),
                      ("adv", C_ALL, "overall")]:
        ax.hist(d[k], bins=bins, histtype="step", color=c, lw=2, label=lab)
    ax.axvline(0, color=AXIS, lw=1)
    style_axis(ax, "advantage", "# rollouts", "Histograms")
    ax.legend()

    # row 3 col 1: advantage vs improvement over parent
    ax = axs[2, 0]
    m = v & np.isfinite(d["improve"])
    ax.scatter(d["improve"][m], d["adv"][m], c=gcol[m], s=22, alpha=0.8, lw=0)
    ax.axvline(0, color=AXIS, lw=1)
    style_axis(ax, "reward - parent reward (valid rollouts)", "overall advantage",
               "Overall advantage vs improvement over parent", zero=True)

    # row 3 col 2: advantage vs response tokens
    ax = axs[2, 1]
    ax.scatter(d["tokens"][v], d["adv"][v], c=gcol[v], s=22, alpha=0.8, lw=0)
    ax.scatter(d["tokens"][~v], d["adv"][~v], c=gcol[~v], s=26, marker="x", lw=1)
    style_axis(ax, "response tokens", "overall advantage",
               f"Length vs advantage   (corr {corr(d['tokens'], d['adv']):.2f})",
               zero=True)

    # row 3 col 3: group means of each component
    ax = axs[2, 2]
    w = 0.27
    gx = np.arange(ng)
    for i, (k, c, lab) in enumerate([("wttt", C_TTT, "RS-GRPO"),
                                     ("wspo", C_SPO, "RS-SPO"),
                                     ("adv", C_ALL, "overall")]):
        ax.bar(gx + (i - 1) * w, [d[k][d["group"] == g].mean() for g in gx],
               w - 0.03, color=c, label=lab)
    ax.set_xticks(gx)
    style_axis(ax, "group", "group-mean advantage",
               "Group-mean of each component", zero=True)
    ax.legend()

    n = len(d["adv"])
    fig.suptitle(
        f"Step {step}: {n} rollouts, {ng} groups, valid {v.mean():.0%}, "
        f"best raw {np.nanmin(d['raw']):.0f} µs, mean reward {d['reward'].mean():.2f}, "
        f"SPO baseline v_t {d['v_t'][0]:.2f}",
        x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, out / "step_snapshots" / f"step_{step:02d}.png")


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="default: <run_dir>/adv_plots")
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

    cfg, wt, ws, steps, skipped = load_run(a.run_dir, not a.unweighted)
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
    fig_groups(steps, wt, ws, out)
    if not a.no_snapshots:
        for s in (a.snapshots if a.snapshots is not None else xs):
            if s in steps:
                fig_snapshot(s, steps[s], wt, ws, out)


if __name__ == "__main__":
    main()
