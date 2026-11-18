"use strict";
/* =====================================================================================
   knowledge.js - the written explanation of the pipeline, distilled from reading the code
   base (file:line references are to the repository the run was produced with).
   ===================================================================================== */
const STAGES = [
  {
    id: "select", n: 1, title: "Select parents", chip: ["search + GLiDE-9B", "mc-search", "PUCT ranking, JEV gate"],
    short: "8 parent programs",
    body: `
      <p>Every step starts by choosing <b>8 parent programs</b> (one per <i>group</i>) from the archive of everything the search has kept so far. The archive begins with 8 trivial <i>seed</i> programs ({{SEEDDESC}}).</p>
      <p><b>PUCT ranks</b> all archive states. <b>JEV</b> (the <code>PUCTJevSampler</code>) then walks down the top <b>3 × 8 = 24</b> states and rejects any candidate that a small judge model (<b>GLiDE-9B</b>) thinks is the <i>same core idea</i> as a parent already accepted in this step. JEV never changes a score - it is a pure diversity gate on PUCT's ranking.</p>
      <p>Step 0 is plain PUCT on the seeds (no comparisons possible). From step 1, accepted candidates cost 0+1+…+7 = 28 pairwise judgments when nothing is rejected.</p>`,
    eq: `score(s) = Q(s) + c · scale · P(s) · √(1+T) / (1+n(s))
Q(s)   = best reward among VALID children ever seen from s   (value(s) while n=0)
P(s)   = (N − rank(s)) / (N(N+1)/2)        rank-based prior, best state ⇒ 2/(N+1)
scale  = max(1e-6, max value − min value over non-seed states)
n(s)   = rollouts spent below s, T = total rollouts so far (= 512 · step)
JEV:   reject candidate if  P(same_core_idea | accepted parent) ≥ threshold   (0.9 in steps 0–3, 0.68 from step 4)`,
    refs: ["search/sampler.py:234-263 (PUCT score)", "search/sampler.py:274-308 (sample_states, lineage blocking)", "search/jev_sampler.py:198-287 (_diverse_picks)", "search/jev_worker.py:17-111 (GLiDE worker)", "orchestrator.py (train_step, parent selection and record_expansion)"],
    files: ["stepNN.parents.json", "stepNN.jev.json"],
  },
  {
    id: "strategist", n: 2, title: "Strategist chains", chip: ["Frozen strategist", "mc-strat", "Strategy generation with LoRA disabled"],
    short: "8 chains × 4 strategies",
    body: `
      <p>For each parent, a <b>reasoning model</b> writes a natural-language <b>implementation plan</b> (a <code>&lt;strategy&gt;</code> block with six sections: approach, math spec, procedure, search choices, compute budget, validation/return contract). No code is allowed; the coder only ever sees the extracted plan, never the strategist's thinking.</p>
      <p>The 4 strategies of a parent are generated <b>sequentially</b>: strategy <i>k</i> receives <code>&lt;previous_strategy_1..k&gt;</code> and is told to propose a <i>materially different</i> approach. The 8 parent chains run in parallel on two vLLM engines.</p>
      <p>Current runs generate <b>one response per slot</b>. Basic validation checks for a complete final plan. A rejected response gives the coder the direct-solve fallback. Older run records may show several attempts for a slot.</p>
      <p>The strategist is <b>never trained</b>. Its plans guide the coder, which is trained from the resulting program rewards.</p>`,
    eq: `chain(parent):  S0 → S1 | S0 → S2 | S0,S1 → S3 | S0,S1,S2
one generated response per slot; a complete final plan must pass validation
fallback slot ⇒ coder prompt contains the fixed "No usable strategy…" sentence`,
    refs: ["orchestrator.py (_record_strategy_response, _extract_final_strategy)", "generation/gen_workers.py (run_sequential_chains)", "problems/base.py (previous-strategy blocks)", "generation/strategy_prompts.py (LOCAL_STRATEGY_PROMPT)"],
    files: ["stepNN_groupGG_fold00_strategySS.{txt,prompt.txt,meta.json}"],
  },
  {
    id: "coder", n: 3, title: "Generate programs", chip: ["Coder + LoRA", "mc-coder", "One coding pass"],
    short: "fixed count per strategy",
    body: `
      <p>The <b>coder</b> receives the {{TASK}}, the parent's source and score, the <code>{{ARR}}</code> hint, and the strategist's plan when strategies are enabled.</p>
      <p>Every plan gets <code>programs_per_strategy</code> samples at the coder's configured reasoning effort. CPU evaluation can start as each program arrives.</p>`,
    eq: `planned programs = parents × strategies per parent × programs per strategy`,
    refs: ["orchestrator.py (train_step)", "{{PROBFILE}} (build_prompt)", "problems/base.py (strategy → coder prompt)"],
    files: ["stepNN_groupGG_rolloutRRR.{txt,prompt.txt,strategy.txt,meta.json}"],
  },
  {
    id: "eval", n: 4, title: "Sandbox evaluation", chip: ["CPU sandboxes", "mc-cpu", "isolated, 264 concurrent"],
    short: "512 scored programs",
    body: `
      <p>Each program runs in its own subprocess (pinned to one CPU, BLAS threads = 1, memory-limited, hard timeout 1900 s). It must return a non-empty finite list; the evaluator computes the AC1 upper bound.</p>
      <p><b>Reward = 1 / raw upper bound</b> (higher is better); every failure gets reward 0. Failure kinds: <code>code</code> (no code block / unclosed fence / runtime error / bad return), <code>constraint</code> (invalid score), <code>timeout</code>.</p>
      <p>The target upper bound 1.5027 only appears in the prompt; it does not enter the reward.</p>`,
    eq: `raw    = 2 · n · max(convolve(a, a, "full")) / (Σ a)²      a_i ← clip(a_i, 0, 1000),  Σa ≥ 0.01
reward = 1 / raw          (failure ⇒ 0)     seeds: raw = 2.0 ⇒ reward 0.5
target 1.5027  ⇔  reward ≈ 0.66547`,
    refs: ["problems/ac_inequalities.py:71-94 (raw score), :376-409 (score), :405-406 (reward = 1/raw)", "evaluation/sandbox.py:211", "orchestrator.py (train_step, sandbox_timeout_s)"],
    files: ["…rolloutRRR.meta.json (reward, raw_score, valid, failure_kind, msg, sandbox_stdout)"],
  },
  {
    id: "adv", n: 5, title: "SPO² advantage", chip: ["SPO-RS · β=2", "mc-train", "single-stream, risk-sensitive"],
    short: "exp(β·r) vs baseline v",
    body: `
      <p>Instead of GRPO-style group normalisation, SPO-RS keeps <b>one run-wide scalar baseline <code>v</code></b> (a persistent "value" of the exponentiated reward) shared by all 512 rollouts of a step. The reward is first passed through an exponential <b>risk-sensitive transform</b> <code>G = exp(β·r)</code>, which emphasises the upper tail (objective ≈ E[R] + β/2·Var R).</p>
      <p>Advantages are <b>not centred inside a group</b>. Because almost every program is either a failure (r=0) or a valid program with r≈{{VALIDR}}, the learning signal is essentially <i>valid vs. invalid</i>: failures get a negative advantage, valid programs a positive one, and the difference between a 1.5050 and 1.5048 solution is tiny (≈0.006).</p>
      <p>After the advantages are formed the baseline moves toward the new batch: <code>v ← (ρ·N_eff·v + ΣG)/(ρ·N_eff + M)</code>. The forgetting factor ρ depends on how far the policy moved (divergence D); in this run D≈4–5 nats so ρ sits at its floor 0.875.</p>`,
    eq: `G_i   = exp(β r_i)                       β = spo_rs_beta = 2.0 (fixed)
A_i   = (G_i / v − 1) / β                v = baseline BEFORE this step (one value for all rollouts)
break-even reward   r* = ln(v) / β
D     = mean over contexts of Σ_tokens (logπ_cur − logπ_prev)   (clamped ≥ 0)
ρ     = clip(2^(−D / d_half), ρ_min, ρ_max) = clip(2^(−D/0.06), 0.875, 0.96)
N_c   = ρ·N_eff + M                      v' = (ρ·N_eff·v + Σ G_i) / N_c      η = M / N_c      N_eff' = N_c`,
    refs: ["training/spo_rs.py:57-82 (advantages), :136-196 (tracker update)", "orchestrator.py (train_step, advantage assembly)", "orchestrator.py (main, tracker init from seed rewards: v0 = e = 2.718, N_eff = 64)", "_retire/bandit_strategist_spo_rs_method.txt:580 (\"SPO-RS, called SPO² in the slides\")"],
    files: ["…rolloutRRR.meta.json (advantage, beta, spo_rs)", "stepNN.summary.json → spo_rs_updates"],
  },
  {
    id: "train", n: 6, title: "LoRA policy update", chip: ["LoRA r=32 · 2 GPUs", "mc-train", "1 AdamW step per training step"],
    short: "PPO-clip, lr 4e-5",
    body: `
      <p>Only the <b>coder</b> is trained (LoRA rank 32, α 32 on all 7 projection modules; 233 M trainable of 27.1 B parameters). Behaviour log-probs come from vLLM; the trainer recomputes current log-probs with exact long-context kernels (sequences up to 262 k tokens).</p>
      <p>The loss is a per-token PPO-clip surrogate with <b>asymmetric clipping [0.8, 1.38]</b>, averaged per trajectory so long and short responses weigh the same, then summed with weight 1/N over all trained trajectories. It is <b>one epoch and one AdamW step per training step</b>, so the policy ratio ≈ 1 and the clip is almost inert (clipped fraction ≈ 0.1%). The KL-to-reference term and gradient clipping are effectively off.</p>
      <p>Training of the previous step overlaps with CPU evaluation of the current rollouts. Step N samples with the adapter from step N−1.</p>`,
    eq: `ρ_t  = exp(logπ_cur,t − logπ_behavior,t)                      (per token)
L_i  = − mean_t min( ρ_t · A_i ,  clip(ρ_t, 1−0.2, 1+0.38) · A_i )
Loss = Σ_i L_i / N                                                 N = all trained trajectories
AdamW: lr 4e-5, β=(0.9, 0.95), wd 0 — kl_penalty_coef=0.1 has no effect for SPO-RS (kl_coef forced to 0)`,
    refs: ["training/trainers.py (clipped_surrogate_loss, ProcessDistributedTrainer.train_spo_rs)", "orchestrator.py (train_step, release_for_training and SPO-RS overlap)"],
    files: ["stepNN.summary.json → policy_updates, training_seconds, fast_*"],
  },
  {
    id: "archive", n: 7, title: "Archive update", chip: ["sampler.update", "mc-search", "dedupe, top-r, top-k"],
    short: "+~63 states / step",
    body: `
      <p>Only <b>valid programs with code</b> are submitted to the archive. The sampler (1) updates each parent's <i>Q</i> (best valid child reward) before any pruning, (2) drops children whose <b>exact source</b> is already archived, (3) keeps the top <b>2</b> per (parent, strategy) — at most 8 per parent per step — and (4) keeps the top <b>8</b> children per direct parent over the whole run. Seeds are never evicted. The cap <code>max_buffer_size=2000</code> is not binding yet (≈ +64 states per step ⇒ reached near step 31).</p>
      <p><code>record_expansion(parent, 64)</code> adds 64 visits to the parent and all its ancestors; these visit counts <i>n</i> shrink PUCT's exploration bonus for heavily-used lineages.</p>
      <p>The "best so far" curve is tracked separately: <code>best_seen</code> uses every valid child (even ones later dedup-ed or pruned) and never regresses.</p>`,
    eq: `archive_size(step k) ≈ 8 seeds + 63·(k+1)        (71, 133, 197, … in this run)
Q(parent) ← max(Q, best valid child reward)       before dedupe
child.parents = [parent] + parent.parents          (nearest-first ancestor chain)`,
    refs: ["search/sampler.py:310-441 (update)", "search/sampler.py:443-449 (record_expansion)", "orchestrator.py (train_step, children submitted to the sampler)"],
    files: ["stepNN.summary.json → archive_size", "stepNN.best.json", "search_tree.json"],
  },
];

const GOTCHAS = [
  ["config.json is a stale snapshot", "jev", "It has no JEV threshold. The real value is stored per step in <code>stepNN.jev.json</code>: <b>0.9 for steps 0–3, 0.68 from step 4</b> (the run was resumed from a step-3 copy and the threshold changed; step 4's selection shows 0.68 → 0.9 → 0.68 across resumes)."],
  ["\"puct-jev\" is not a blended score", "jev", "PUCT ranks, JEV only vetoes. No JEV probability enters any score. The window is the top 3×8 = 24 PUCT states."],
  ["kl_penalty_coef and grad_clip are inert", "For SPO-RS the KL coefficient is forced to 0 (\"reference: off\"), and logged gradient norms are ~0.001–0.002, so the 1.0 clip never triggers."],
  ["Only the coder is trained", "The frozen strategist generates plans with LoRA disabled."],
  ["Advantages are not group-centred", "One run-wide baseline <i>v</i> serves all 512 rollouts, so learning is dominated by valid-vs-invalid."],
  ["rolloutRRR is arrival order", "The number in a rollout file name is not the strategy order. Group by <code>strategy_index</code> to see which programs came from each plan."],
  ["'fold00' is a legacy tag", "It is hard-coded in the file name layout and always reads fold00."],
  ["A rejected plan triggers the direct-solve fallback", "", "The coder receives the task and parent context with a direct-solve instruction. The slot still runs its planned coder rollouts."],
  ["search_tree.json lags one step and includes failures", "It is rebuilt post-hoc; every rollout is a node (a large share have score 0.0 = failed programs)."],
  ["Test expectation looks stale", "jev", "<code>tests/test_parent_picker_contract.py</code> asserts a same-core threshold of 0.80 while the code default is 0.68."],
];

const GLOSSARY = [
  ["Group", "One parent slot g ∈ [0,8) in a step: a parent program plus the 64 rollouts spawned from it (4 strategies × 16)."],
  ["Strategy / chain", "A natural-language plan from the strategist. 4 per parent, generated sequentially; later ones see earlier ones."],
  ["Attempt", "One generated strategy response. Current runs generate one response per slot."],
  ["Seed", "One of the 8 initial constant-sequence programs (upper bound 2.0, reward 0.5)."],
  ["Archive", "The set of valid, de-duplicated states the sampler can pick parents from (capped at 2000)."],
  ["PUCT", "Predictor + UCT: exploitation (best child reward Q) plus an exploration bonus scaled by a rank prior and shrunk by visit counts."],
  ["JEV", "The GLiDE-9B semantic diversity gate in <code>jev_sampler.py</code>. Judges Program A vs B as same_core_idea / meaningful_variant / different_approach. (The code never expands the acronym.)"],
  ["SPO² / SPO-RS", "Single-stream policy optimisation with a Risk-Sensitive exponential reward transform and a persistent baseline."],
  ["v, N_eff, η, ρ", "Baseline value, its effective sample count, its per-step update rate (M/N_c) and its forgetting factor."],
  ["Valid", "Program parsed, ran, returned a finite non-empty list and scored a finite positive bound. It does not need to beat the parent."],
  ["Upper bound (raw)", "2·n·max(a*a)/(Σa)² - the quantity to minimise (target 1.5027)."],
  ["Reward", "1 / raw (0 for failures); higher is better."],
];

/* GOTCHAS entries with the "jev" flag apply to runs with the diversity gate. */
function gotchas() {
  const out = GOTCHAS.map(g => g.length === 3 ? g : [g[0], "", g[1]]);
  const f = out.filter(g => g[1] === "" || (g[1] === "jev" && RUN.hasJev));
  if (!RUN.hasJev) f.unshift(["This run has no JEV diversity gate", "", "It was produced by an earlier version of the pipeline: parents are chosen by the plain <code>PUCTSampler</code> only (rank-based prior, best-child Q, lineage blocking). The GLiDE-9B judge, the 3×8 candidate window and the same-core threshold of the newer runs do not exist here."]);
  return f;
}
function glossary() {
  return GLOSSARY.filter(g => RUN.hasJev || g[0] !== "JEV").map(g => g[0] === "Seed" ? ["Seed", PROB.seedText ? "One of the 8 initial parents: " + PROB.seedText + "." : "One of the 8 initial trivial programs (" + fillText("{{SEEDDESC}}") + ", reward ≈ " + (PROB.seedReward ? PROB.seedReward.toFixed(2) : "?") + ")."]
    : g[0] === "Upper bound (raw)" ? ["Raw score (" + PROB.metric + ")", PROB.glossRaw] : g[0] === "Reward" ? ["Reward", PROB.rewardText || "1 / raw (0 for failures); higher is better."] : g[0] === "Valid" && PROB.validText ? ["Valid", PROB.validText] : g);
}
/* the per-stage text depends on the problem and on whether the run used the JEV gate */
function stageView(sg) {
  let v = Object.assign({}, sg);
  if (sg.id === "select" && !RUN.hasJev) {
    v.chip = ["PUCT sampler", "mc-search", "rank-based prior, lineage blocking"];
    v.body = `
      <p>Every step starts by choosing <b>8 parent programs</b> (one per <i>group</i>) from the archive of everything the search has kept so far. The archive begins with 8 trivial <i>seed</i> programs ({{SEEDDESC}}).</p>
      <p><b>PUCT ranks</b> all archive states. The sampler walks down the ranking, takes a state and then <b>blocks its whole lineage</b> (the state, all its ancestors and all its descendants - siblings and cousins stay allowed) so the 8 parents sit on different branches of the tree. If blocking leaves fewer than 8, the rest are topped up in score order.</p>
      <p><i>This run used the plain <code>PUCTSampler</code>: there is no semantic diversity gate (JEV / GLiDE-9B) in this earlier version of the pipeline.</i></p>`;
    v.eq = `score(s) = Q(s) + c · scale · P(s) · √(1+T) / (1+n(s))
Q(s)   = best reward among VALID children ever seen from s   (value(s) while n=0)
P(s)   = (N − rank(s)) / (N(N+1)/2)        rank-based prior, best state ⇒ 2/(N+1)
scale  = max(1e-6, max value − min value over non-seed states)
n(s)   = rollouts spent below s, T = total rollouts so far (= 512 · step)
pick:  best-scoring state not in the lineage of an already picked parent`;
    v.refs = ["search/sampler.py:234-263 (PUCT score)", "search/sampler.py:274-308 (sample_states, lineage blocking)", "orchestrator.py (train_step, parent selection and record_expansion)"];
    v.files = ["stepNN.parents.json"];
  }
  if (sg.id === "eval") { v.body = PROB.evalBody; v.eq = PROB.evalEq; v.refs = PROB.evalRefs; }
  v.body = fillText(v.body); v.eq = fillText(v.eq); v.refs = (v.refs || []).map(fillText);
  return v;
}
