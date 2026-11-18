"use strict";
/* ---- Strategist tab ---------------------------------------------------------------------- */
let _stratView = "plan";
const ANAT = {
  task: ["Task + evaluator source", "--s1"], sota_hint: ["Prior-art hint (LaTeX)", "--s4"], parent_program: ["Parent score + parent program", "--s3"],
  rules: ["Rules + task requirements", "--s5"], prev_strategies: ["Previous strategies of this chain", "--s2"], contract: ["Output contract (strategy format)", "--s7"],
};

function slotOutcome(s) { return !s ? "none" : s.fallback ? "fallback" : s.attempts.length > 1 ? "retry" : "ok"; }

registerTab({
  id: "strategist", label: "Strategist", ico: "✎",
  async render(root) {
    const strat = await lazy("strat");
    const step = S.step, g = S.group, si = S.sidx;
    const slot = (st, gg, s) => strat.strategies[`${st}:${gg}:${s}`];
    const model = (CFG.strategy_model_name || "").split("/").filter(Boolean).pop();
    const label = [model, CFG.strategy_reasoning_effort].filter(Boolean).join(" · ");
    root.append(pageHead("Strategist: plans written before any code",
      `For each parent the frozen strategist${label ? ` (<b>${esc(label)}</b>)` : ""} writes <b>sequential</b> implementation plans. Each plan sees the previous ones and proposes a different approach. ` +
      `The coder receives the final <code>&lt;strategy&gt;</code> block or a direct-solve instruction when the slot has no usable plan. Fallbacks account for <b>${fmt.pct(Object.values(strat.strategies).filter(s => s.fallback).length / Object.values(strat.strategies).length, 0)}</b> of all slots in this run.`));

    // --- overview heat map: steps x 32 slots
    const hostO = h("div", { class: "chart", style: { overflowX: "auto" } });
    root.append(card("Every strategy slot of the run", "rows = steps, columns = parent·strategy (P0S0 … P7S3) · click to inspect", h("div", null, hostO,
      legend([{ color: cssVar("--good"), label: "accepted first try" }, { color: cssVar("--warn"), label: "accepted after retries" }, { color: cssVar("--crit"), label: "fallback (no usable plan)" }]))));
    const sts = ALL_STEPS.filter(s => slotsOf(s).length);
    heatmap(hostO, {
      rows: sts.map(s => "step " + s), cols: range(0, 32).map(i => `${i >> 2}.${i & 3}`), rowLabelW: 58, maxCell: 30, minCell: 14, cellH: 22,
      val: (i, j) => { const s = slot(sts[i], j >> 2, j & 3); return s ? s.attempts.length || 1 : null; },
      color: (v, i, j) => { const o = slotOutcome(slot(sts[i], j >> 2, j & 3)); return o === "fallback" ? cssVar("--crit") : o === "retry" ? cssVar("--warn") : cssVar("--good"); },
      text: (v, i, j) => { const s = slot(sts[i], j >> 2, j & 3); return s ? (s.fallback ? "✕" : s.attempts.length) : null; }, textColor: () => "#0b0b0b",
      tip: (i, j) => { const s = slot(sts[i], j >> 2, j & 3); return s ? `<b>step ${sts[i]} · parent ${j >> 2} · S${j & 3}</b><div>${s.attempts.length} attempt(s) ${s.attempts.map(a => a.ok ? "✓" : "✗").join(" ")}</div>` : null; },
      onClick: (i, j) => setState({ step: sts[i], group: j >> 2, sidx: j & 3 }),
    });

    // --- chain view for selected step/group
    const chainBox = h("div", { class: "grid g4" });
    const gSeg = seg(range(0, 8).map(x => ({ value: x, label: "parent " + x })), g, v => setState({ group: v }));
    root.append(h("div", { style: { marginTop: "14px" } }, card(`Chain of parent ${g} in step ${step}`, "S0 → S1 → S2 → S3 (each later strategy sees the earlier plans)", h("div", null, h("div", { class: "row-flex", style: { marginBottom: "10px" } }, gSeg), chainBox))));
    for (let s = 0; s < 4; s++) {
      const sl = slot(step, g, s);
      if (!sl) { chainBox.append(h("div", { class: "card flat" }, h("div", { class: "empty" }, "S" + s + ": not generated yet"))); continue; }
      const pips = h("div", { class: "row-flex", style: { gap: "4px", margin: "6px 0" } }, sl.attempts.map(a => {
        const p = h("span", { style: { width: "18px", height: "18px", borderRadius: "4px", background: a.ok ? cssVar("--good") : cssVar("--crit"), display: "inline-block", fontSize: "10px", textAlign: "center", color: "#0b0b0b", lineHeight: "18px" } }, a.ok ? "✓" : "✗");
        bindTip(p, () => `<b>attempt ${a.a}</b><div>${a.ok ? "accepted" : esc(a.issue || "rejected")}</div><div class="mut">response ${fmt.n(a.rc)} chars</div>`);
        return p;
      }));
      const c = h("div", { class: "card flat step-chip", style: { cursor: "pointer", borderColor: si === s ? cssVar("--accent") : "", borderWidth: si === s ? "2px" : "1px" }, onclick: () => setState({ sidx: s }) },
        h("div", { class: "row-flex", style: { justifyContent: "space-between" } }, h("b", null, h("span", { class: "dot", style: { background: S_COL(s) } }), "S" + s), sl.fallback ? pill("fallback", "bad") : sl.attempts.length > 1 ? pill(sl.attempts.length + " attempts", "warn") : pill("1st try", "good")),
        pips,
        h("div", { class: "sub" }, sl.fallback ? "coder had no plan" : `plan ${fmt.n(sl.block.length)} chars · reasoning ${fmt.tok(sl.think_chars)} chars`));
      chainBox.append(c);
    }

    // --- details of selected slot
    const sl = slot(step, g, si);
    const det = h("div");
    root.append(h("div", { style: { marginTop: "14px" } }, card(`Strategy S${si} · parent ${g} · step ${step}`, null, det)));
    if (!sl) { det.append(h("div", { class: "empty" }, "No strategy file for this slot yet.")); }
    else {
      det.append(h("div", { class: "row-flex", style: { marginBottom: "10px" } },
        seg([{ value: "plan", label: "Plan given to the coder" }, { value: "think", label: "Strategist reasoning" }, { value: "prompt", label: "Prompt anatomy" }, { value: "out", label: "What the coder did with it" }], _stratView, v => { _stratView = v; setState({}); })));
      if (_stratView === "plan") {
        if (sl.fallback) det.append(h("div", { class: "warnbox" }, "The coder received the direct-solve fallback. ", h("i", null, sl.block)));
        else det.append(textBlock(sl.block));
        det.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "This is the extracted <strategy> body - everything the coder sees from the strategist."));
      } else if (_stratView === "think") {
        det.append(h("p", { class: "sub" }, `The strategist reasoned for ${fmt.n(sl.think_chars)} characters before writing the plan (max 131,072 tokens). The coder never sees this. Excerpt:`));
        det.append(textBlock(sl.think_head + (sl.think_tail ? "\n\n[… " + fmt.n(sl.think_chars - sl.think_head.length - sl.think_tail.length) + " characters omitted …]\n\n" + sl.think_tail : "")));
      } else if (_stratView === "prompt") {
        const an = sl.anat;
        if (!an) det.append(h("div", { class: "empty" }, "No prompt available."));
        else {
          det.append(h("p", { class: "sub" }, `The strategist prompt of the last attempt: ${fmt.n(an.total)} characters. ${an.prev_strategies ? `It embeds ${an.prev_strategies} previous strategy block(s) from the same chain.` : "S0 has no previous strategies."}`));
          const bar = h("div", { class: "sbar", style: { height: "26px" } });
          an.pieces.forEach(([k, n]) => { const sp = h("span", { style: { width: (n / an.total * 100) + "%", background: cssVar(ANAT[k][1]) } }); bindTip(sp, () => `<b>${ANAT[k][0]}</b><div>${fmt.n(n)} chars (${fmt.pct(n / an.total, 1)})</div>`); bar.append(sp); });
          det.append(bar);
          det.append(legend(an.pieces.map(([k, n]) => ({ color: cssVar(ANAT[k][1]), label: `${ANAT[k][0]} · ${fmt.n(n)}` }))));
          det.append(h("p", { class: "sub", style: { marginTop: "8px" } }, "The prompt is dominated by the parent's source code. Later strategies carry the earlier plans, so S3 prompts are longer. The output contract repeatedly stresses closing the </strategy> tag - the most common failure is running out of tokens before it."));
          const per = sl.attempts.map(a => `attempt ${a.a}: prompt ${fmt.n(a.pc)} chars → response ${fmt.n(a.rc)} chars → ${a.ok ? "accepted" : esc(a.issue || "rejected")}`);
          det.append(h("div", { class: "eq" }, per.join("\n")));
        }
      } else {
        const rs = (RS[step] || []).filter(r => r.pg === g && r.sidx === si && r.planned !== false);
        if (!rs.length) det.append(h("div", { class: "empty" }, "No coder rollouts yet for this slot."));
        else {
          const v = rs.filter(r => r.valid);
          det.append(h("div", { class: "kv", html: `<div class="k">Programs</div><div class="v">${rs.length}</div><div class="k">Valid</div><div class="v">${v.length} (${fmt.pct(v.length / rs.length)})</div><div class="k">Beat parent</div><div class="v">${rs.filter(r => r.improved).length}</div><div class="k">Best bound</div><div class="v">${v.length ? fmt.raw(Math.min(...v.map(r => r.raw))) : "–"} (parent ${fmt.raw(rs[0].praw)})</div>` }));
          const t = table([
            { label: "#", get: r => String(r.rollout), num: true }, { label: PROB.metric, get: r => r.valid ? fmt.raw(r.raw) : "–", num: true },
            { label: "result", get: r => r.valid ? (r.improved ? pill("beats parent", "good") : pill("valid")) : pill(r.fail || "failed", "bad") }, { label: "tokens", get: r => fmt.tok(r.ntok), num: true }, { label: "advantage", get: r => fmt.f(r.adv, 3), num: true },
          ], rs.slice().sort((a, b) => (a.valid ? a.raw : 99) - (b.valid ? b.raw : 99)), { onRow: r => openRollout(r.id) });
          det.append(h("div", { class: "scroll", style: { marginTop: "10px" } }, t));
        }
      }
    }

    // --- aggregates across the run
    const by = k => STEPS.flatMap(step => range(0, CFG.groups_per_step || 8).map(g => slot(step, g, k)).filter(Boolean));
    const hostB = h("div", { class: "chart" });
    root.append(card("Format failures by chain position", "Share of slots that fell back and attempts per slot", h("div", null, hostB,
      legend([{ color: S_COL(7), label: "fallback rate" }, { color: S_COL(3), label: "attempts per slot (÷4)" }]))));
    barChart(hostB, { height: 230, cats: ["S0", "S1", "S2", "S3"], series: [{ name: "fallback rate", color: S_COL(7), vals: range(0, 4).map(k => mean(by(k).map(d => d.fallback ? 1 : 0))) }, { name: "attempts / slot (÷4)", color: S_COL(3), vals: range(0, 4).map(k => mean(by(k).map(d => d.attempts.length)) / 4) }], yFmt: v => v.toFixed(2), tip: (i, si2, v) => `<b>S${i}</b><div>${si2 === 0 ? "fallback " + fmt.pct(v) : "attempts per slot " + (v * 4).toFixed(2)}</div>` });
  }
});
