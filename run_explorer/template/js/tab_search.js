"use strict";
/* ---- Search tab: PUCT and UCT selection, genealogy tree, score landscape ---------- */
let _searchOpts = { landscape: "valid", treeColor: "raw", treeGrow: "all" };

/* signed symmetric-log axis for "gap to the reference value": 0 = equal to the reference, negative = better than it */
function gapOf(raw) { const g = raw - TARGET(), K = PROB.K; return Math.sign(g) * Math.log10(1 + Math.abs(g) / K); }
function gapInv(y) { return Math.sign(y) * (Math.pow(10, Math.abs(y)) - 1) * PROB.K; }
function gapTicks() { const K = PROB.K, ms = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5]; return [...ms.slice().reverse().map(m => -m * K), 0, ...ms.map(m => m * K)].map(g => Math.sign(g) * Math.log10(1 + Math.abs(g) / K)); }
function gapFmt(y) { const g = gapInv(y), a = Math.abs(g); if (a < 1e-12) return "0"; return (g < 0 ? "−" : "+") + (+a.toPrecision(1)); }
function gapDomain(ys) { const v = ys.filter(isFinite); if (!v.length) return [0, 1]; return [Math.min(...v) - 0.25, Math.max(...v) + 0.25]; }

registerTab({
  id: "search", label: "Search", ico: "⌥",
  async render(root) {
    const step = S.step;
    root.append(pageHead("Search: choosing which programs to extend",
      `<b>${RUN.selection}</b> scores the archive, then blocks the ancestors and descendants of each selected parent. If blocking leaves too few parents, the sampler fills the remaining slots in score order.`));
    selectionSection(root, step);
    treeSection(root);
    landscapeSection(root);
  }
});

/* ---------------------------------------------------------------- shared helpers ---------- */
function pucRows(step) {
  const ps = parentsOf(step), c0 = CFG.puct_c ?? 1;
  const c = P.c ?? c0;
  const rows = ps.map(p => {
    const bonus = (p.exploration_bonus ?? 0) * (c0 === 0 ? 1 : c / c0);
    return { ...p, bonus2: bonus, score2: (p.q_value ?? p.parent_value ?? 0) + bonus };
  });
  const order = rows.slice().sort((a, b) => b.score2 - a.score2).map(p => p.group);
  return { rows, order, c0 };
}

function selectionSection(root, step) {
  const ps = parentsOf(step);
  if (!ps.length) {
    root.append(card("Parent selection", null, h("div", { class: "empty" }, "No parents recorded.")));
    return;
  }
  const { c0 } = pucRows(step);
  const box = h("div");
  const slider = c0 === 0 ? null : paramSlider("Exploration constant c", 0, 20, 0.5,
    v => v.toFixed(1), () => P.c ?? c0, (v, row) => setParam({ c: v }, row));
  root.append(card(`${RUN.selection} parents at step ${step}`,
    "Recorded parent scores. The slider rescales their logged exploration bonuses.", h("div", null, slider, box)));

  const draw = () => {
    const { rows, order } = pucRows(step);
    box.replaceChildren(h("div", { class: "scroll" }, table([
      { label: "group", get: p => String(p.group), num: true },
      { label: "parent", get: p => h("button", { class: "btn sm mono", onclick: () => openNode(p.parent_id) }, fmt.id(p.parent_id)) },
      { label: "created", get: p => p.parent_is_seed ? "seed" : "step " + p.parent_timestep },
      { label: PROB.metric, get: p => fmt.raw(p.parent_raw_score), num: true },
      { label: "Q", get: p => fmt.f(p.q_value, 5), num: true },
      { label: "visits", get: p => String(p.visit_count), num: true },
      ...(RUN.selection === "PUCT" ? [{ label: "prior", get: p => fmt.f(p.prior, 5), num: true }] : []),
      { label: "bonus", get: p => fmt.f(p.bonus2, 4), num: true },
      { label: "score", get: p => fmt.f(p.score2, 5), num: true },
      { label: "score rank", get: p => String(order.indexOf(p.group) + 1), num: true },
    ], rows)));
  };
  draw();
  onParam(patch => { if ("c" in patch) draw(); });
}

/* ---------------------------------------------------------------- C. genealogy ------------- */
function genealogy(limit) {
  // nodes: selected parents + best-of-step + all their ancestors, optionally only those created up to step `limit`
  const keep = new Set(), selCount = {};
  const add = id => { const i = TREE.idx[id]; if (i == null) return; let k = i; while (k >= 0 && !keep.has(k)) { keep.add(k); k = TREE.nodes[k][0]; } };
  ALL_STEPS.filter(s => s <= limit).forEach(s => parentsOf(s).forEach(p => { add(p.parent_id); selCount[p.parent_id] = (selCount[p.parent_id] || 0) + 1; }));
  STEPS.filter(s => s <= limit).forEach(s => { const b = ST[s].best; if (b && b.state_id) add(b.state_id); });
  [...keep].forEach(i => { if (TREE.nodes[i][2] > limit) keep.delete(i); });
  const lastS = STEPS.filter(s => s <= limit).pop();
  const lastBest = lastS != null ? ST[lastS].best : null;
  const bestIdx = lastBest ? TREE.idx[lastBest.state_id] : null;
  const path = new Set(); { let k = bestIdx; while (k != null && k >= 0) { path.add(k); k = TREE.nodes[k][0]; } }
  const kids = {}; keep.forEach(i => { const p = TREE.nodes[i][0]; if (p >= 0 && keep.has(p)) (kids[p] = kids[p] || []).push(i); });
  const roots = [...keep].filter(i => TREE.nodes[i][0] < 0 || !keep.has(TREE.nodes[i][0]));
  let leafY = 0; const pos = {};
  const lay = i => {
    const ch = (kids[i] || []).sort((a, b) => (path.has(b) ? 1 : 0) - (path.has(a) ? 1 : 0) || TREE.nodes[a][2] - TREE.nodes[b][2]);
    if (!ch.length) pos[i] = leafY++; else { ch.forEach(lay); pos[i] = mean(ch.map(c => pos[c])); }
  };
  roots.forEach(lay);
  return { keep, kids, roots, pos, leaves: leafY, path, bestIdx, selCount };
}

function treeSection(root) {
  const grow = _searchOpts.treeGrow === "step";
  const limit = grow ? S.step : Infinity;
  const G = genealogy(limit);
  const nowSel = new Set(grow ? parentsOf(S.step).map(p => p.parent_id) : []);
  const host = h("div", { style: { overflow: "auto", maxHeight: "680px", border: "1px solid var(--line)", borderRadius: "10px", background: "var(--surface)" } });
  const ctl = h("div", { class: "row-flex", style: { marginBottom: "8px" } },
    seg([{ value: "raw", label: "color: upper bound" }, { value: "step", label: "color: step created" }], _searchOpts.treeColor, v => { _searchOpts.treeColor = v; drawTree(); }),
    seg([{ value: "all", label: "whole run" }, { value: "step", label: `grow: up to step ${S.step}` }], _searchOpts.treeGrow || "all", v => { _searchOpts.treeGrow = v; renderTab(); }),
    h("span", { class: "sub" }, `${G.keep.size} states: every state chosen as a parent, each step's best, and their ancestors (of ${fmt.n(TREE.ids.length)} programs in the full tree). Orange path leads to the best${grow ? " so far" : ""}${grow ? "; pulsing rings = parents chosen at the selected step - press ▶ play steps to watch the search grow" : ""}.`));
  root.append(card("Genealogy of the search", "left → right = step created · click a node to open its program", h("div", null, ctl, host)));
  const drawTree = () => {
    host.innerHTML = "";
    if (!G.keep.size) { host.append(h("div", { class: "empty" }, "nothing to show yet")); return; }
    const dx = 96, dy = 20, ml = 40, mt = 30;
    const maxLevel = Math.max(...[...G.keep].map(i => TREE.nodes[i][2] + 1));
    const W = ml + (maxLevel + 1) * dx + 40, H = mt + G.leaves * dy + 24;
    const svg = sv("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    host.append(svg);
    const X = i => ml + (TREE.nodes[i][2] + 1) * dx, Y = i => mt + G.pos[i] * dy;
    for (let l = 0; l <= maxLevel; l++) { const t = sv("text", { x: ml + l * dx, y: 16, "text-anchor": "middle", style: "font-size:11px;fill:" + cssVar("--ink-3") }, svg); t.textContent = l === 0 ? "seeds" : "step " + (l - 1); sv("line", { x1: ml + l * dx, x2: ml + l * dx, y1: 22, y2: H - 8, stroke: cssVar("--line"), "stroke-dasharray": "2 4" }, svg); }
    const nonSeed = [...G.keep].filter(i => TREE.nodes[i][0] >= 0).map(i => TREE.nodes[i][1]).filter(x => x > 0);
    const lo = nonSeed.length ? Math.min(...nonSeed) : 0, hi = nonSeed.length ? Math.max(...nonSeed) : 1;
    const colOf = i => { const n = TREE.nodes[i]; if (n[0] < 0) return cssVar("--ink-3"); if (_searchOpts.treeColor === "step") return seqRamp(n[2] / Math.max(1, maxLevel - 1)); return seqRamp(1 - clamp((n[1] - lo) / Math.max(1e-9, hi - lo), 0, 1)); };
    G.keep.forEach(i => { const p = TREE.nodes[i][0]; if (p >= 0 && G.keep.has(p)) {
      const x0 = X(p), y0 = Y(p), x1 = X(i), y1 = Y(i), mx = (x0 + x1) / 2, onP = G.path.has(i) && G.path.has(p);
      const e = sv("path", { d: `M${x0},${y0}C${mx},${y0} ${mx},${y1} ${x1},${y1}`, fill: "none", stroke: onP ? cssVar("--s2") : cssVar("--line-2"), "stroke-width": onP ? 2.6 : 1.2 }, svg);
      if (ANIM) { e.setAttribute("pathLength", "1"); e.setAttribute("class", "a-line"); }
    } });
    G.keep.forEach(i => {
      const n = TREE.nodes[i], id = TREE.ids[i], isSel = G.selCount[id], r = n[3] >= 0 ? R[n[3]] : null;
      if (nowSel.has(id)) sv("circle", { cx: X(i), cy: Y(i), r: 8, fill: "none", stroke: cssVar("--accent"), "stroke-width": 2, class: "ring-pulse" }, svg);
      const c = sv("circle", { cx: X(i), cy: Y(i), r: n[0] < 0 ? 6 : (isSel ? 6 : 4), fill: colOf(i), stroke: G.path.has(i) ? cssVar("--s2") : (isSel ? cssVar("--ink") : cssVar("--surface")), "stroke-width": G.path.has(i) ? 2.4 : (isSel ? 1.4 : 1), style: "cursor:pointer" }, svg);
      if (ANIM) { c.setAttribute("class", "a-fade"); c.style.animationDelay = (0.05 * (n[2] + 1)) + "s"; }
      c.addEventListener("mousemove", ev => tipShow(`<b>${n[0] < 0 ? "seed" : "step " + n[2]}</b> <span class="mut">${fmt.id(id)}</span><div>upper bound <b>${fmt.raw(n[1])}</b></div>${r ? `<div class="mut">parent group ${r.pg}</div>` : ""}${isSel ? `<div>chosen as parent ${isSel}×</div>` : ""}<div class="mut">${(TREE.kids[i] || []).length} programs derived (total)</div>`, ev));
      c.addEventListener("mouseleave", tipHide);
      c.addEventListener("click", () => openNode(id));
    });
    if (G.bestIdx != null && G.keep.has(G.bestIdx)) {
      const bt = sv("text", { x: X(G.bestIdx) + 10, y: Y(G.bestIdx) + 4, style: "font-size:11px;font-weight:600;fill:" + cssVar("--s2") }, svg); bt.textContent = "best " + fmt.f(TREE.nodes[G.bestIdx][1], 5);
      requestAnimationFrame(() => { host.scrollLeft = Math.max(0, X(G.bestIdx) - host.clientWidth + 140); host.scrollTop = Math.max(0, Y(G.bestIdx) - host.clientHeight / 2); });
    }
  };
  drawTree();
}

/* ---------------------------------------------------------------- D. landscape -------------- */
function landscapeSection(root) {
  const host = h("div", { class: "chart" });
  const ctl = h("div", { class: "row-flex", style: { marginBottom: "8px" } },
    seg([{ value: "valid", label: "all valid" }, { value: "improved", label: "beat their parent" }], _searchOpts.landscape, v => { _searchOpts.landscape = v; draw(); }),
    h("span", { class: "sub" }, `colour = parent group of the current step selection is emphasised: group ${S.group}`));
  root.append(card("Score landscape of every valid program", "x = step (jittered), y = gap to the reference value (log scale) · click a dot to open the program", h("div", null, ctl, host)));
  const draw = () => {
    const f = _searchOpts.landscape;
    const pts = R.filter(r => r.valid && r.raw > 0 && (f === "valid" || (f === "improved" && r.improved))).map(r => ({
      x: r.step + ((r.id * 2654435761 % 1000) / 1000 - 0.5) * 0.7, y: gapOf(r.raw), c: S_COL(r.pg), a: r.pg === S.group ? 0.85 : 0.35, r: r.pg === S.group ? 3.2 : 2.4, id: r.id, rr: r,
    }));
    scatterChart(host, {
      height: 380, points: pts, xLabel: "step", yLabel: "gap to reference (symlog)", margin: { l: 62 }, yTicks: gapTicks(), yFmt: gapFmt, xDomain: [-0.7, Math.max(...ALL_STEPS) + 0.7], yDomain: gapDomain(pts.map(p => p.y)),
      xTicks: ALL_STEPS, xFmt: v => String(Math.round(v)),
      tip: p => `<b>step ${p.rr.step} · parent ${p.rr.pg}</b><div>upper bound ${fmt.raw(p.rr.raw)}</div><div class="mut">${p.rr.improved ? "Beat parent" : "Valid"}</div>`,
      onPoint: p => openRollout(p.id), hlines: [],
    });
  };
  draw();
  root.append(legend(range(0, 8).map(g => ({ color: S_COL(g), label: "parent " + g }))));
}
