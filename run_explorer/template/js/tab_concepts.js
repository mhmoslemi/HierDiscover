"use strict";
/* ---- Concepts tab: architecture, glossary, gotchas, file map ---------------------------------- */
const FILE_MAP = [
  ["config.json", "Resolved run configuration snapshot."],
  ["setting.log · temirnal.log", "Startup settings; timestamped tee of stdout (HH:MM:SS only, no dates)."],
  ["result.txt", "One line per finished step: step best / best seen raw and reward."],
  ["search_tree.json", "Every program as a node {parent_id, child_ids, raw_problem_score}; failed programs have score 0."],
  ["stepNN/stepNN.parents.json", "Selected parents and their logged search scores."],
  ["stepNN/stepNN.summary.json", "Counts, best scores, SPO² tracker, policy-update stats, timings. Its presence marks a finished step."],
  ["stepNN/stepNN.best.json", "Best program so far: code + construction + score."],
  ["stepNN/…_groupGG_rolloutRRR.txt", "Raw coder output: reasoning, </think>, then the python block."],
  ["…rolloutRRR.prompt.txt", "The fully rendered coder prompt (task, parent code, system message)."],
  ["…rolloutRRR.meta.json", "Score, validity, failure kind, advantage, and parent information. Dict fields are Python-repr strings."],
  ["vllm.log", "Coder engine logs."],
  ["plots.log · *.png/pdf/svg", "Per-step plots written by the analysis helpers."],
];

registerTab({
  id: "concepts", label: "How it works", ico: "ⓘ",
  async render(root) {
    root.append(pageHead("How the training pipeline fits together",
      "Open a stage to see the mechanism and the selected step’s data."));
    const stages = STAGES.map(stageView);
    const flow = h("div", { class: "flow" });
    stages.forEach((stage, index) => {
      if (index) flow.append(h("span", { class: "arrow", "aria-hidden": "true" }, "›"));
      flow.append(h("button", { class: "btn", onclick: () => {
        _stageSel = stage.id;
        setState({ tab: "pipeline" });
      } }, stage.title));
    });
    root.append(card("One training step", null, flow));

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
    root.append(card(null, null, h("div", null, GOTCHAS.map(([t, b]) => h("div", { style: { padding: "8px 0", borderBottom: "1px solid var(--line)" } }, h("div", { style: { fontWeight: 600 } }, t), h("div", { html: b, style: { color: "var(--ink-2)", fontSize: "13px" } }))))));

    // file map
    root.append(h("h2", { style: { margin: "22px 0 10px" } }, "What each file in the run directory is"));
    root.append(card(null, null, h("div", { class: "kv", style: { gridTemplateColumns: "300px 1fr", gap: "6px 14px" }, html: FILE_MAP.map(([k, v]) => `<div class="k mono" style="color:var(--ink-2)">${esc(k)}</div><div class="v">${esc(v)}</div>`).join("") })));
  }
});
