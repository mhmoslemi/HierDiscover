"use strict";
/* ---- Concepts tab: architecture, glossary, gotchas, file map ---------------------------------- */
const FILE_MAP = [
  ["config.json", "Resolved run configuration snapshot (stale for the JEV threshold - see stepNN.jev.json)."],
  ["setting.log · temirnal.log", "Startup settings; timestamped tee of stdout (HH:MM:SS only, no dates)."],
  ["result.txt", "One line per finished step: step best / best seen raw and reward."],
  ["search_tree.json", "Every program as a node {parent_id, child_ids, raw_problem_score}; failed programs have score 0."],
  ["stepNN/stepNN.parents.json", "The 8 selected parents with Q, P, bonus, score and the JEV verdicts."],
  ["stepNN/stepNN.jev.json", "Every GLiDE judgment, the scan order with accept/reject, threshold, and the candidate sources."],
  ["stepNN/stepNN.summary.json", "Counts, best scores, SPO² tracker, policy-update stats, timings. Its presence marks a finished step."],
  ["stepNN/stepNN.best.json", "Best program so far: code + construction + score."],
  ["stepNN/…_groupGG_rolloutRRR.txt", "Raw coder output: reasoning, </think>, then the python block."],
  ["…rolloutRRR.prompt.txt", "The fully rendered coder prompt (task, parent code, strategy, system line)."],
  ["…rolloutRRR.strategy.txt", "The strategist's full response (with reasoning) that produced the strategy of this rollout."],
  ["…rolloutRRR.meta.json", "Score, validity, failure kind, advantage, parent/PUCT snapshot, and strategy. Dict fields are Python-repr strings."],
  ["…_foldFF_strategySS.txt", "The accepted strategist response for the slot (or the fallback sentence)."],
  ["…_strategySS.{prompt.txt,meta.json}", "The strategist prompt and validation verdict for the slot."],
  ["vllm.log · strategy_vllm.log", "Engine logs of the coder and strategist pools."],
  ["plots.log · *.png/pdf/svg", "Per-step plots written by the analysis helpers."],
];

function architectureSvg(host) {
  registerChart(host, () => {
    const W = 1190, H = 700;
    host.innerHTML = "";
    const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-width:1190px" }, host);
    const defs = sv("defs", null, svg);
    ["a", "b"].forEach(k => { const m = sv("marker", { id: "ar" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs); sv("path", { d: "M0,0L10,5L0,10z", fill: k === "a" ? cssVar("--ink-3") : cssVar("--s4") }, m); });
    const col = { search: cssVar("--s4"), strat: cssVar("--s2"), coder: cssVar("--s1"), cpu: cssVar("--s3"), train: cssVar("--s8"), jev: cssVar("--s7") };
    const last = ST[STEPS[STEPS.length - 1]] || {};
    const nm = p => String(p || "").split("/").filter(Boolean).pop();
    const P = CFG.groups_per_step || 8, S = CFG.strategies_per_parent || 4, per = CFG.group_size || 64, programs = CFG.programs_per_strategy || Math.floor(per / S);
    const jevThr = ((last.jev || {}).threshold), mult = CFG.jev_candidate_multiplier || 3;
    /* centred box: soft fill in the stage colour, bold title, grey sub-lines; click = open that stage in the Pipeline tab */
    const box = (x, y, w, hh, title, sub, c, stage, tsize) => {
      const g = sv("g", { style: stage ? "cursor:pointer" : "" }, svg);
      sv("rect", { x, y, width: w, height: hh, rx: 7, fill: c, "fill-opacity": 0.11, stroke: c, "stroke-width": 1.6 }, g);
      const lines = String(sub || "").split("\n").filter(Boolean), top = y + hh / 2 - (lines.length * 7.5) - 2;
      const t = sv("text", { x: x + w / 2, y: top + 6, "text-anchor": "middle", style: `font-weight:700;font-size:${tsize || 15}px;fill:${cssVar("--ink")}` }, g); t.textContent = title;
      lines.forEach((ln, i) => { const s2 = sv("text", { x: x + w / 2, y: top + 25 + i * 14.5, "text-anchor": "middle", style: "font-size:11.5px;fill:" + cssVar("--ink-2") }, g); s2.textContent = ln; });
      if (stage) g.addEventListener("click", () => { _stageSel = stage; setState({ tab: "pipeline" }); });
      return g;
    };
    const arrow = (d, label, dashed, lx, ly, c, anchor) => {
      sv("path", { d, fill: "none", stroke: c || cssVar("--ink-3"), "stroke-width": 1.5, "stroke-dasharray": dashed ? "5 4" : "", "marker-end": `url(#ar${dashed ? "b" : "a"})` }, svg);
      if (label) { const t = sv("text", { x: lx, y: ly, "text-anchor": anchor || "middle", style: "font-size:11px;fill:" + (c || cssVar("--ink-3")) }, svg); t.textContent = label; }
    };
    const xs = [290, 520, 750, 980], cw = 190, cx = i => xs[i] + cw / 2;
    // headline
    const th = sv("text", { x: 730, y: 22, "text-anchor": "middle", style: "font-size:14px;fill:" + cssVar("--ink-2") }, svg);
    th.textContent = "Each new plan sees the earlier plans of the same parent, to encourage a different approach";
    // left column: archive -> PUCT -> JEV -> parent
    box(20, 44, 210, 84, "Parent p", "one of the " + P + " parents of the step\n(code, score" + (PROB.kernel ? "" : ", construction") + ")", col.search, "select");
    if (RUN.hasJev) box(20, 178, 210, 92, "JEV gate", "GLiDE-9B compares candidates\nwith the parents already picked;\nskip if P(same core idea) ≥ " + (jevThr != null ? jevThr : "threshold"), col.jev, "select");
    else box(20, 178, 210, 92, "Lineage blocking", "skip ancestors and descendants\nof every parent already picked\n(this run has no JEV gate)", col.search, "select");
    box(20, 318, 210, 84, "PUCT ranking", "Q + c·scale·P·√(1+T)/(1+n)\n" + (RUN.hasJev ? "window = top " + P + "×" + mult + " states" : "rank-based prior P"), col.search, "select");
    box(20, 448, 210, 70, "Archive", "scores + ancestry\ncap " + (CFG.max_buffer_size || 2000), col.search, "archive");
    arrow("M125,446V404", "", false); arrow("M125,316V272", "", false); arrow("M125,176V130", "", false);
    arrow("M232,86H286", "", false);
    // strategies and their planned programs
    for (let i = 0; i < S && i < 4; i++) {
      box(xs[i], 44, cw, 84, "Strategy " + (i + 1), "plan for parent p\nfrozen strategist", col.strat, "strategist");
      box(xs[i], 172, cw, 70, programs + " programs", nm(CFG.model_name || CFG.coder_model_name) + " + LoRA\n" + (CFG.coder_reasoning_effort ? "reasoning " + CFG.coder_reasoning_effort : ""), col.coder, "coder");
      arrow(`M${cx(i)},130V170`, "", false); arrow(`M${cx(i)},244V275`, "", false);
      if (i < S - 1 && i < 3) arrow(`M${xs[i] + cw + 2},86H${xs[i + 1] - 4}`, "", false);
    }
    // evaluation and archive update
    box(290, 275, 880, 54, (PROB.kernel ? "GPU evaluation" : "CPU evaluation streams results as programs arrive"), (PROB.kernel ? "kernel correctness tests + timing on the GPU · " : "isolated sandbox per program · ") + (PROB.rewardShort || "reward from the score") + " · failure ⇒ 0", col.cpu, "eval", 15);
    arrow("M730,331V482H232", "", false);
    { const t = sv("text", { x: 480, y: 470, "text-anchor": "middle", style: "font-size:11px;fill:" + col.search }, svg); t.textContent = "Verified programs update the archive"; }
    // learning lane
    sv("line", { x1: 20, x2: W - 20, y1: 598, y2: 598, stroke: cssVar("--line-2"), "stroke-dasharray": "2 4" }, svg);
    { const t = sv("text", { x: 20, y: 622, style: "font-size:12px;font-weight:700;fill:" + cssVar("--ink-2") }, svg); t.textContent = "Learning"; }
    { const t = sv("text", { x: 20, y: 638, style: "font-size:11px;fill:" + cssVar("--ink-3") }, svg); t.textContent = "all scored programs"; }
    box(290, 608, 250, 62, "SPO² advantage", "exp(β·r) against one run-wide\nbaseline v", col.train, "adv");
    box(590, 608, 250, 62, "LoRA update", "PPO-clip · AdamW · lr " + (CFG.learning_rate || "?") + "\n" + (CFG.num_training_gpus || "?") + " GPUs", col.train, "train");
    box(890, 608, 280, 62, "Updated adapter", "generates the coder programs\nof the next step", col.coder, "coder");
    arrow("M542,639H588", "", false); arrow("M842,639H888", "", false);
  });
}

registerTab({
  id: "concepts", label: "How it works", ico: "ⓘ",
  async render(root) {
    root.append(pageHead("How the whole system fits together",
      `Written from a read-through of the code base (<code>orchestrator.py</code>, <code>search/</code>, <code>training/</code>, <code>generation/</code>, <code>problems/ac_inequalities.py</code>) and checked against the files of this run. ` +
      `Click any box to open that stage in the <b>Pipeline</b> tab. Solid arrows happen inside one step; dashed arrows are the loops that tie steps together.`));
    const host = h("div", { class: "chart" });
    root.append(card("Architecture of one training step", "colour = which model/subsystem does the work", h("div", null, host,
      legend([{ color: cssVar("--s4"), label: "search / bookkeeping" }, { color: cssVar("--s7"), label: "GLiDE-9B judge" }, { color: cssVar("--s2"), label: "strategist (frozen)" }, { color: cssVar("--s1"), label: "coder (trained)" }, { color: cssVar("--s3"), label: PROB.kernel ? "GPU kernel evaluation" : "CPU sandboxes" }, { color: cssVar("--s8"), label: "trainer" }]))));
    architectureSvg(host);

    // the three loops
    root.append(h("div", { class: "grid g3", style: { marginTop: "14px" } },
      conceptCard("Search loop (across steps)", (RUN.hasJev ? "PUCT + JEV pick parents" : "PUCT picks parents (lineage-blocked)") + " → their children (valid, new, high reward) enter the archive → next step's parents. This is what drives the best-so-far curve and keeps the 8 parents of a step on different branches."),
      conceptCard("Coding pass", "Each plan gets the configured number of programs at the same coder reasoning effort. The evaluator scores each result."),
      conceptCard("Learning loop (across steps)", "SPO² turns rewards into advantages against one slowly-moving baseline; a single PPO-clipped LoRA step updates the coder; the new adapter samples step N+1. Mostly it learns to produce valid, well-formed programs.")));

    // stage deep dives
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Stage reference"));
    const wrap = h("div", { class: "cols2" });
    STAGES.map(stageView).forEach(sg => wrap.append(h("div", { class: "concept", style: { breakInside: "avoid" } },
      h("h2", null, `${sg.n}. ${sg.title}`), h("div", { html: sg.body }), h("div", { class: "eq" }, sg.eq), h("div", null, sg.refs.map(r => h("div", { class: "cite" }, r))))));
    root.append(wrap);

    // glossary
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Glossary"));
    root.append(card(null, null, h("div", { class: "kv", style: { gridTemplateColumns: "190px 1fr", gap: "8px 14px" }, html: glossary().map(([k, v]) => `<div class="k"><b style="color:var(--ink)">${esc(k)}</b></div><div class="v">${v}</div>`).join("") })));

    // gotchas
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "Things that are easy to get wrong (found while reading the code and the data)"));
    root.append(card(null, null, h("div", null, gotchas().map(([t, fl, b]) => h("div", { style: { padding: "8px 0", borderBottom: "1px solid var(--line)" } }, h("div", { style: { fontWeight: 600 } }, t), h("div", { html: b, style: { color: "var(--ink-2)", fontSize: "13px" } }))))));

    // file map
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "What each file in the run directory is"));
    root.append(card(null, null, h("div", { class: "kv", style: { gridTemplateColumns: "300px 1fr", gap: "6px 14px" }, html: FILE_MAP.filter(([k]) => RUN.hasJev || !/jev/.test(k)).map(([k, v]) => `<div class="k mono" style="color:var(--ink-2)">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("") })));
  }
});

function conceptCard(t, b) { return h("div", { class: "card" }, h("h2", null, t), h("p", { style: { color: "var(--ink-2)", margin: "4px 0 0" } }, b)); }
