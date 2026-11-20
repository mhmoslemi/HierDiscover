"use strict";
/* =====================================================================================
   knowledge.js - the written explanation of the pipeline, distilled from reading the code
   base (file:line references are to the repository the run was produced with).
   ===================================================================================== */
const STAGES = [
  {
    id: "select", n: 1, title: "Select parents", chip: ["Search", "mc-search", "Archive and lineage"],
    body: "<p>The sampler ranks the archive and excludes the ancestors and descendants of each pick. If too few parents remain, it fills the batch in score order.</p>",
    eq: "",
    refs: ["search/sampler.py (sample_states)", "orchestrator.py (train_step)"],
    files: ["stepNN.parents.json"],
  },
  {
    id: "coder", n: 2, title: "Generate programs", chip: ["Coder + LoRA", "mc-coder", "One coding pass"],
    body: "<p>The coder receives the {{TASK}}, the parent's source and score, and samples the configured number of rollouts per parent.</p>",
    eq: "planned programs = groups_per_step × group_size",
    refs: ["orchestrator.py (train_step)", "{{PROBFILE}} (build_prompt)"],
    files: ["stepNN_groupGG_rolloutRRR.{txt,prompt.txt,meta.json}"],
  },
  {
    id: "eval", n: 3, title: "Sandbox evaluation", chip: ["CPU sandboxes", "mc-cpu", "isolated, 264 concurrent"],
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
    id: "adv", n: 4, title: "SPO² advantage", chip: ["SPO-RS · β=2", "mc-train", "single-stream, risk-sensitive"],
    short: "exp(β·r) vs baseline v",
    body: `
      <p>Instead of GRPO-style group normalisation, SPO-RS keeps <b>one run-wide scalar baseline <code>v</code></b> (a persistent "value" of the exponentiated reward) shared by all 512 rollouts of a step. The reward is first passed through an exponential <b>risk-sensitive transform</b> <code>G = exp(β·r)</code>, which emphasises the upper tail (objective ≈ E[R] + β/2·Var R).</p>
      <p>Each program's advantage comes from its reward and the same run-wide baseline. Rewards above the break-even point give positive advantages. Rewards below it give negative advantages.</p>
      <p>After the advantages are formed the baseline moves toward the new batch: <code>v ← (ρ·N_eff·v + ΣG)/(ρ·N_eff + M)</code>. The forgetting factor ρ depends on how far the policy moved (divergence D); in this run D≈4–5 nats so ρ sits at its floor 0.875.</p>`,
    eq: `G_i   = exp(β r_i)                       β = spo_rs_beta = 2.0 (fixed)
A_i   = (G_i / v − 1) / β                v = baseline BEFORE this step (one value for all rollouts)
break-even reward   r* = ln(v) / β
D     = mean over contexts of Σ_tokens (logπ_cur − logπ_prev)   (clamped ≥ 0)
ρ     = clip(2^(−D / d_half), ρ_min, ρ_max) = clip(2^(−D/0.06), 0.875, 0.96)
N_c   = ρ·N_eff + M                      v' = (ρ·N_eff·v + Σ G_i) / N_c      η = M / N_c      N_eff' = N_c`,
    refs: ["training/spo_rs.py:57-82 (advantages), :136-196 (tracker update)", "orchestrator.py (train_step, advantage assembly)", "orchestrator.py (initial rewards and tracker startup)"],
    files: ["…rolloutRRR.meta.json (advantage, beta, spo_rs)", "stepNN.summary.json → spo_rs_updates"],
  },
  {
    id: "train", n: 5, title: "LoRA policy update", chip: ["LoRA r=32 · 2 GPUs", "mc-train", "1 AdamW step per training step"],
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
    id: "archive", n: 6, title: "Archive update", chip: ["sampler.update", "mc-search", "deduplication and per-parent pruning"],
    body: "<p>For valid programs, the sampler updates the parent's best child reward, deduplicates exact source, keeps the configured top children per parent, and applies the global cap. Seeds stay in the archive.</p>",
    eq: "Q(parent) ← max(Q, best valid child reward) before pruning",
    refs: ["search/sampler.py:310-441 (update)", "search/sampler.py:443-449 (record_expansion)", "orchestrator.py (train_step, children submitted to the sampler)"],
    files: ["stepNN.summary.json → archive_size", "stepNN.best.json", "search_tree.json"],
  },
];

const GOTCHAS = [
  ["kl_penalty_coef and grad_clip are inert", "For SPO-RS the KL coefficient is forced to 0 (\"reference: off\"), and logged gradient norms are ~0.001–0.002, so the 1.0 clip never triggers."],
  ["Advantages are not group-centred", "One run-wide baseline <i>v</i> serves all 512 rollouts, so learning is dominated by valid-vs-invalid."],
  ["search_tree.json lags one step and includes failures", "It is rebuilt post-hoc; every rollout is a node (a large share have score 0.0 = failed programs)."],
];

const GLOSSARY = [
  ["Group", "A parent and its configured batch of coder rollouts."],
  ["UCT", "Search scoring with a logarithmic exploration bonus based on visit counts."],
  ["Seed", "One of the 8 initial constant-sequence programs (upper bound 2.0, reward 0.5)."],
  ["Archive", "The set of valid, de-duplicated states the sampler can pick parents from (capped at 2000)."],
  ["PUCT", "Predictor + UCT: exploitation (best child reward Q) plus an exploration bonus scaled by a rank prior and shrunk by visit counts."],
  ["SPO² / SPO-RS", "Single-stream policy optimisation with a Risk-Sensitive exponential reward transform and a persistent baseline."],
  ["v, N_eff, η, ρ", "Baseline value, its effective sample count, its per-step update rate (M/N_c) and its forgetting factor."],
  ["Valid", "Program parsed, ran, returned a finite non-empty list and scored a finite positive bound. It does not need to beat the parent."],
  ["Upper bound (raw)", "2·n·max(a*a)/(Σa)² - the quantity to minimise (target 1.5027)."],
  ["Reward", "1 / raw (0 for failures); higher is better."],
];

function glossary() {
  return GLOSSARY.map(g => g[0] === "Seed" ? ["Seed", PROB.seedText || fillText("{{SEEDDESC}}")]
    : g[0] === "Upper bound (raw)" ? ["Raw score (" + PROB.metric + ")", PROB.glossRaw]
    : g[0] === "Reward" ? ["Reward", PROB.rewardText || "1 / raw (0 for failures); higher is better."]
    : g[0] === "Valid" && PROB.validText ? ["Valid", PROB.validText] : g);
}
function stageView(sg) {
  const v = { ...sg };
  if (sg.id === "select") {
    v.chip = [RUN.selection, "mc-search", "Lineage blocking"];
    const formulas = {
      UCT: "score = Q + c · scale · √(log(1+T) / (1+n))",
      PUCT: "score = Q + c · scale · P · √(1+T) / (1+n)",
    };
    v.eq = formulas[RUN.selection] || "See the recorded parent scores for this run.";
  }
  if (sg.id === "eval") { v.body = PROB.evalBody; v.eq = PROB.evalEq; v.refs = PROB.evalRefs; }
  v.body = fillText(v.body); v.eq = fillText(v.eq); v.refs = (v.refs || []).map(fillText);
  return v;
}
