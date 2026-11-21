from __future__ import annotations

import json
import math
import os
import secrets
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional

from generation.gen_workers import native_final_answer_scope, render_chat_prompt
from training.critic_step import CRITIC_CONTEXT, CRITIC_MODEL, load_tree, save_batch


# ---------------------------------------------------------------------------
# The GPT-OSS chat template formats this developer message.
# Placeholders are filled with str.replace, never str.format: the JSON braces
# in the prompt would break .format().
# ---------------------------------------------------------------------------
DEVELOPER_PROMPT = """## Role
You are an expert in reinforcement learning for language models (policy-gradient methods, GRPO and its risk-sensitive variants, credit assignment, advantage estimation) and in iterative scientific discovery, where a candidate solution is refined over many steps and progress is measured by an objective score. In this task you act as the critic inside an RL training loop. You turn raw scores into advantages that tell the policy which rollouts to reinforce and which to discourage.

## Setting
A policy LLM is solving a scientific discovery problem step by step. Its attempts form a search tree:

- Rollout: one attempt by the policy. It starts from a parent solution, produces a new candidate solution, and an external evaluator scores that candidate. Every rollout has a unique ID.
- Step: the training iteration. Step 0 contains the initial seeds. A later step can expand a parent from any earlier step.
- Parent: the solution a rollout started from.
- Child: a rollout generated from a given parent. A rollout you score now may become a parent in a later step.
- Siblings: rollouts with the same parent at the same step. A sibling group is the set of alternatives the policy produced from one starting point.
- SOTA: the best known score for this problem from outside the tree (literature or best known result).
- n/a: the rollout failed and has no valid score (crash, timeout, invalid output, constraint violation, or any other reason). You are not told why.

You see only numbers: rollout IDs, tree structure, raw scores, and SOTA. You never see the content of any solution.

Score direction: {SCORE_DIRECTION}
Below, "better" and "worse" always refer to this direction.

## Task
The user message contains the tree history, statistics computed from it, and a set of new rollouts to score, all from the same step. For each new rollout, output an advantage: a real number in [-10, 10]. A positive advantage reinforces the rollout and a negative one discourages it. The range is bounded, but you do not need to use all of it.

## Input
The user message is a JSON object. Every statistic in it was computed exactly by code, so use the values as given and do not recompute them. A null value means the statistic is undefined (for example, nothing valid to compare with).

Sign convention: every field whose name starts with "improvement_" is signed so that positive means better and negative means worse, whatever the score direction. Raw scores, bests, and means are in the original units.

- "problem": "score_direction" and "sota".
- "tree": the state of the tree before the current step.
  - "current_step": the step of the rollouts to score.
  - "running_best_before", "running_best_id": the best valid score in the tree before the current step, and the rollout that holds it.
  - "gap_to_sota": distance from running_best_before to SOTA. Positive means SOTA is still better; zero or negative means the tree has already matched or beaten SOTA.
  - "steps_since_running_best_improved": how many consecutive steps before the current one did not improve the running best.
  - "typical_step_spread": the median, over steps, of the standard deviation of valid scores within a step. A reference scale for what counts as a meaningful difference.
  - "steps": one row per step up to and including the current one, with "step", "n", "n_valid", "fail_rate", "best", "mean", "std", and "running_best" (best valid score up to and including that step).
- "groups": one entry per sibling group at the current step, with "parent_id", "parent_score", "parent_lineage" (scores along the path from the root to the parent, oldest first), "n", "n_valid", "n_failed", "best", "mean", "std", and "uniform" (true if the group has at least two rollouts and all of them have the same outcome).
- "to_score": one entry per rollout to score, with "id", "parent_id", and "score" ("n/a" for failures). Valid rollouts also have:
  - "rank_in_group": rank among valid siblings, 1 = best; ties share a rank.
  - "improvement_vs_parent": change from the parent score.
  - "gap_closed_frac": improvement_vs_parent divided by the distance from the parent to SOTA. Null if the parent has no score or already matches or beats SOTA.
  - "improvement_vs_group_mean": difference from the mean valid score of its group.
  - "improvement_vs_running_best": difference from running_best_before. Positive means it sets a new running best.
  - "improvement_vs_sota": difference from SOTA. Zero or positive means it matches or beats SOTA.
  - "z_in_step": standardized score among all valid rollouts at the current step.
- "history" (may be absent): every earlier rollout as [id, step, parent_id, score].

## Reasoning procedure
Work through these stages in your analysis before answering.

1. Orient. Read the provided statistics. For each sibling group, note the parent score and lineage, the siblings' outcomes, and any failures. For the tree, note the running best, the per-step trend, the gap to SOTA, the plateau length, and the failure rates.

2. Decide what counts as a meaningful difference, using typical_step_spread, the group std, and z_in_step.

3. Evaluate each rollout on these signals:
   a. Rank inside its sibling group.
   b. Change from its parent: improvement or regression, and its size relative to the remaining gap to SOTA (gap_closed_frac). If the parent already matches or beats SOTA, judge the change against the running best instead. If the parent has no score, rely on siblings and the tree.
   c. Standing in the whole tree: does it set a new running best? How does it compare with the best and mean of recent steps?
   d. Trajectory: does it break a plateau of several stagnant steps, continue the recent trend, or reverse it?

4. Assign values, then check every hard constraint below and fix any violation before answering.

## Hard constraints
1. Range: every advantage is in [-10, 10], with at most two decimal places.
2. Monotonic within a group: inside a sibling group, a better raw score never gets a lower advantage than a worse one, and equal scores get equal advantages. Across different groups this does not apply. A lower raw score can deserve a higher advantage if it was a bigger step forward from a weaker parent.
3. Failures rank last: an n/a rollout gets a strictly lower advantage than every valid sibling in its group.
4. Uniform groups: if a sibling group has at least two rollouts and all of them have the same outcome (all n/a, or all the same score), give all of them the same value, near -0.5. An all-n/a group may go lower if other groups at the same step succeeded.
5. Complete and exact: output exactly the rollout IDs in the to-score set, each one once. Do not output history rollouts.
6. Use only the numbers provided. Do not assume anything about solution content, and do not invent rollouts or scores.

## Principles
- Discovery is driven by the best outcome, not the average. Be risk-seeking: a genuine upside surprise deserves more magnitude than a downside of the same size. A valid but mediocre exploratory rollout should never be punished as hard as a failure.
- Value is relative. The same raw score can earn a positive advantage early in the tree and a negative one later, once the tree has moved past it.
- Weigh failures by context. An isolated failure among successful siblings is strongly negative. When failures are common across the whole step, the state is hard, so be milder, but still respect hard constraints 3 and 4.
- Be consistent. The same situation should get the same advantage across groups and across calls.

## Output
In the final channel, output only a JSON object that maps each rollout ID (string) to its advantage (number). No explanation, no markdown, no code fences.
Example: {"s3_r0": 6.5, "s3_r1": -2.0, "s3_r2": -8.0, "s3_r3": 1.5}

# Response Formats

## rollout_advantages

// Maps every rollout ID in the to-score set to its advantage in [-10, 10].
{RESPONSE_SCHEMA}"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rollout:
    id: str
    step: int
    parent_id: Optional[str]
    score: Optional[float]  # None = failed ("n/a")


def _as_rollouts(tree: Iterable[Any]) -> list[Rollout]:
    out = []
    for r in tree:
        if isinstance(r, Rollout):
            out.append(r)
            continue
        s = r.get("score")
        if s is None or isinstance(s, str):
            s = None
        else:
            s = float(s)
            if not math.isfinite(s):
                s = None
        pid = r.get("parent_id")
        out.append(Rollout(str(r["id"]), int(r["step"]), None if pid is None else str(pid), s))
    ids = [r.id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError("rollout ids must be unique")
    return out


def _r(x: Optional[float]) -> Optional[float]:
    """Round for display (6 significant digits). None stays None."""
    return None if x is None else float(f"{x:.6g}")


def _std(xs: list[float]) -> Optional[float]:
    return statistics.pstdev(xs) if len(xs) >= 2 else None


def _direction(higher_is_better: bool) -> str:
    return "higher is better" if higher_is_better else "lower is better"


# ---------------------------------------------------------------------------
# Statistics -> user message
# ---------------------------------------------------------------------------
def build_user_payload(tree, sota, higher_is_better=True, to_score=None, include_history=True):
    """Return (payload, nodes, to_score_ids). payload is the JSON user message."""
    rollouts = _as_rollouts(tree)
    nodes = {r.id: r for r in rollouts}
    sign = 1.0 if higher_is_better else -1.0

    def g(s):  # "goodness": larger is always better, whatever the direction
        return None if s is None else sign * s

    if to_score is None:
        cur = max(r.step for r in rollouts)
        ids = [r.id for r in rollouts if r.step == cur]
    else:
        ids = [str(i) for i in to_score]
        unknown = [i for i in ids if i not in nodes]
        if unknown:
            raise ValueError(f"unknown ids in to_score: {unknown}")
        steps = {nodes[i].step for i in ids}
        if len(steps) != 1:
            raise ValueError("all rollouts in to_score must be at the same step")
        cur = steps.pop()
    if not ids:
        raise ValueError("nothing to score")
    sota_g = g(float(sota)) if sota is not None else None

    # Per-step table, running best, plateau length
    by_step = defaultdict(list)
    for r in rollouts:
        if r.step <= cur:
            by_step[r.step].append(r)

    rows, stds = [], []
    run_best = run_best_before = last_improve = None
    for t in sorted(by_step):
        rs = by_step[t]
        valid = [r for r in rs if r.score is not None]
        scores = [r.score for r in valid]
        best = max(valid, key=lambda r: g(r.score)) if valid else None
        if t == cur:
            run_best_before = run_best
        if best is not None and (run_best is None or g(best.score) > g(run_best.score)):
            run_best = best
            if t < cur:
                last_improve = t
        sd = _std(scores)
        if sd is not None:
            stds.append(sd)
        rows.append({
            "step": t,
            "n": len(rs),
            "n_valid": len(valid),
            "fail_rate": _r(1 - len(valid) / len(rs)),
            "best": _r(best.score) if best else None,
            "mean": _r(statistics.fmean(scores)) if scores else None,
            "std": _r(sd),
            "running_best": _r(run_best.score) if run_best else None,
        })

    rb_g = g(run_best_before.score) if run_best_before else None
    tree_info = {
        "current_step": cur,
        "running_best_before": _r(run_best_before.score) if run_best_before else None,
        "running_best_id": run_best_before.id if run_best_before else None,
        "gap_to_sota": _r(sota_g - rb_g) if (rb_g is not None and sota_g is not None) else None,
        "steps_since_running_best_improved": (cur - 1 - last_improve) if last_improve is not None else None,
        "typical_step_spread": _r(statistics.median(stds)) if stds else None,
        "steps": rows,
    }

    # Sibling groups at the current step
    siblings = defaultdict(list)
    for r in by_step[cur]:
        siblings[r.parent_id].append(r)
    cur_g = [g(r.score) for r in by_step[cur] if r.score is not None]
    step_mean = statistics.fmean(cur_g) if cur_g else None
    step_sd = _std(cur_g)

    def parent_g(pid):
        p = nodes.get(pid) if pid is not None else None
        return g(p.score) if (p is not None and p.score is not None) else None

    def lineage(pid):
        path, seen = [], set()
        while pid is not None and pid in nodes and pid not in seen:
            seen.add(pid)
            p = nodes[pid]
            path.append(_r(p.score) if p.score is not None else "n/a")
            pid = p.parent_id
        return path[::-1]

    groups = []
    for pid in dict.fromkeys(nodes[i].parent_id for i in ids):
        members = siblings[pid]
        valid = [r for r in members if r.score is not None]
        scores = [r.score for r in valid]
        best = max(valid, key=lambda r: g(r.score)) if valid else None
        pg = parent_g(pid)
        groups.append({
            "parent_id": pid,
            "parent_score": _r(sign * pg) if pg is not None else None,
            "parent_lineage": lineage(pid),
            "n": len(members),
            "n_valid": len(valid),
            "n_failed": len(members) - len(valid),
            "best": _r(best.score) if best else None,
            "mean": _r(statistics.fmean(scores)) if scores else None,
            "std": _r(_std(scores)),
            "uniform": len(members) >= 2 and len({r.score for r in members}) == 1,
        })

    # Rollouts to score
    items = []
    for i in ids:
        r = nodes[i]
        e = {"id": r.id, "parent_id": r.parent_id,
             "score": _r(r.score) if r.score is not None else "n/a"}
        if r.score is not None:
            mine = g(r.score)
            sib = [g(s.score) for s in siblings[r.parent_id] if s.score is not None]
            pg = parent_g(r.parent_id)
            imp = mine - pg if pg is not None else None
            dist = sota_g - pg if (pg is not None and sota_g is not None) else None
            e.update({
                "rank_in_group": 1 + sum(x > mine for x in sib),
                "improvement_vs_parent": _r(imp),
                "gap_closed_frac": _r(imp / dist) if (imp is not None and dist and dist > 0) else None,
                "improvement_vs_group_mean": _r(mine - statistics.fmean(sib)),
                "improvement_vs_running_best": _r(mine - rb_g) if rb_g is not None else None,
                "improvement_vs_sota": _r(mine - sota_g) if sota_g is not None else None,
                "z_in_step": _r((mine - step_mean) / step_sd) if step_sd else None,
            })
        items.append(e)

    payload = {
        "problem": {"score_direction": _direction(higher_is_better), "sota": _r(sota)},
        "tree": tree_info,
        "groups": groups,
        "to_score": items,
    }
    if include_history:
        payload["history"] = [
            [r.id, r.step, r.parent_id, _r(r.score) if r.score is not None else "n/a"]
            for r in sorted(rollouts, key=lambda r: r.step) if r.step < cur
        ]
    return payload, nodes, ids


def build_developer_prompt(higher_is_better: bool, ids: list[str]) -> str:
    """Fill the placeholders. The schema lists the exact IDs for this call."""
    schema = {
        "type": "object",
        "properties": {i: {"type": "number", "minimum": -10, "maximum": 10} for i in ids},
        "required": list(ids),
        "additionalProperties": False,
    }
    return (DEVELOPER_PROMPT
            .replace("{SCORE_DIRECTION}", _direction(higher_is_better))
            .replace("{RESPONSE_SCHEMA}", json.dumps(schema, separators=(",", ":"))))


def _unique_json_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate id {key!r}")
        obj[key] = value
    return obj


def parse_advantages(text: Optional[str], ids: list[str]):
    """Require exactly the requested IDs and numbers with at most two decimal places."""
    if not text:
        return None, "no final-channel output"
    try:
        obj = json.loads(text, parse_float=Decimal, object_pairs_hook=_unique_json_keys)
    except (ValueError, ArithmeticError, RecursionError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(obj, dict) or set(obj) != set(ids):
        return None, "expected one JSON object with exactly the requested rollout IDs"
    out = {}
    for key, value in obj.items():
        if type(value) not in (int, Decimal) or not -10 <= value <= 10:
            return None, f"advantage for {key} must be a number in [-10, 10]"
        if value != Decimal(value).quantize(Decimal("0.01")):
            return None, f"advantage for {key} has more than two decimal places"
        out[key] = float(value)
    return out, None


def constraint_issue(advantages, nodes, higher_is_better):
    sign = 1 if higher_is_better else -1
    groups = defaultdict(list)
    for rollout_id in advantages:
        rollout = nodes[rollout_id]
        groups[rollout.parent_id].append(rollout)

    for siblings in groups.values():
        siblings.sort(key=lambda row: (row.score is None, -sign * (row.score or 0.0)))
        for better, worse in zip(siblings, siblings[1:]):
            a, b = advantages[better.id], advantages[worse.id]
            if better.score == worse.score:
                if a != b:
                    return f"tied siblings {better.id} and {worse.id} have different advantages"
            elif worse.score is None:
                if a <= b:
                    return f"failed rollout {worse.id} must rank below valid sibling {better.id}"
            elif a < b:
                return f"better sibling {better.id} ranks below {worse.id}"
    return None


def prepare_prompt(tokenizer, tree, sota, higher_is_better, *, to_score=None, include_history=True,
                   max_context=CRITIC_CONTEXT):
    payload, nodes, ids = build_user_payload(tree, sota, higher_is_better, to_score, include_history)
    messages = [{"role": "developer", "content": build_developer_prompt(higher_is_better, ids)},
                {"role": "user", "content": ""}]
    # Even a compact final JSON needs room when the optional history is long.
    compact_answer = json.dumps({i: -0.51 if nodes[i].score is None else -0.5 for i in ids},
                                separators=(",", ":"))
    answer_tokens = len(tokenizer.encode(
        f"<|channel|>final<|message|>{compact_answer}<|return|>", add_special_tokens=False))
    history = payload.pop("history", [])
    while True:
        if history:
            payload["history"] = history
        else:
            payload.pop("history", None)
        messages[1]["content"] = json.dumps(payload, ensure_ascii=False)
        prompt = tokenizer.encode(render_chat_prompt(tokenizer, messages, reasoning_effort="high"))
        if len(prompt) + answer_tokens <= max_context:
            return prompt, nodes, ids
        if not history:
            raise ValueError("critic prompt and required JSON exceed its context window")
        # The summaries and active parent lineages still use all history.
        history = history[(len(history) + 1) // 2:]


class AdvantageCritic:
    """Frozen GPT-OSS critic that keeps one vLLM engine across attempts."""

    def __init__(self, model=CRITIC_MODEL, llm=None, **llm_kwargs):
        from vllm import LLM, SamplingParams
        self._SamplingParams = SamplingParams
        self.llm = llm if llm is not None else LLM(model=model, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._stop = self.tokenizer.convert_tokens_to_ids(["<|return|>", "<|endoftext|>", "<|call|>"])
        self._end_tokens = {*self._stop, self.tokenizer.convert_tokens_to_ids("<|end|>")}

    def _params(self, seed):
        return self._SamplingParams(n=1, temperature=1.0, top_p=1.0, seed=seed,
                                    max_tokens=None, stop_token_ids=self._stop,
                                    skip_special_tokens=True)

    def score_prompt(self, prompt, nodes, ids, higher_is_better, *, seed=None):
        attempt = 0
        if seed is None:
            seed = secrets.randbits(32)
        while True:
            attempt += 1
            requests = self.llm.generate(
                [{"prompt_token_ids": prompt}], self._params((seed + attempt) % 2**32),
                use_tqdm=False)
            completion = requests[0].outputs[0] if requests and requests[0].outputs else None
            text = None
            if completion is not None:
                token_ids = completion.token_ids
                end = len(token_ids)
                while end and token_ids[end - 1] in self._end_tokens:
                    end -= 1
                text = native_final_answer_scope(self.tokenizer, token_ids[:end], require_final=True)
            advantages, issue = parse_advantages(text, ids)
            if issue is None:
                issue = constraint_issue(advantages, nodes, higher_is_better)
            if issue is None:
                return advantages, {"attempts": attempt}
            if completion is not None and completion.finish_reason == "length":
                issue += " (context window exhausted)"
            print(f"[llm-critic] rejected attempt {attempt}: {issue}; retrying", flush=True)

    def score(self, tree, sota, higher_is_better=True, to_score=None, include_history=True):
        prompt, nodes, ids = prepare_prompt(
            self.tokenizer, tree, sota, higher_is_better, to_score=to_score, include_history=include_history)
        return self.score_prompt(prompt, nodes, ids, higher_is_better)


def _score_saved_step(path, engine_options):
    from generation.gen_workers import (
        _prepare_flashinfer_comm_compat, _shutdown_vllm_engine, _start_vllm_parent_watchdog,
    )

    _start_vllm_parent_watchdog(int(os.environ["TTT_CRITIC_PARENT_PID"]))
    batch = json.loads(path.read_text(encoding="utf-8"))
    tree = load_tree(path, batch)
    higher_is_better = batch["higher_is_better"]
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    _prepare_flashinfer_comm_compat()
    critic = AdvantageCritic(**engine_options)
    try:
        prompt, nodes, ids = prepare_prompt(critic.tokenizer, tree, batch["sota"], higher_is_better)
        print(f"[llm-critic] ready to score {len(ids)} rollouts with {len(prompt)} prompt tokens", flush=True)
        seed = batch.get("seed")
        if seed is not None:
            seed = seed * 1_000_003 + batch["step"] * 1009
        advantages, info = critic.score_prompt(prompt, nodes, ids, higher_is_better, seed=seed)
        batch.update(advantages=advantages, **info)
        save_batch(path, batch)
        print(f"[llm-critic] accepted attempt {info['attempts']}; releasing the critic GPU memory", flush=True)
    finally:
        _shutdown_vllm_engine(critic.llm, None)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--score-step":
        _score_saved_step(Path(sys.argv[2]), json.loads(sys.argv[3]))
        sys.exit(0)

    demo_tree = [
        {"id": "root",  "step": 0, "parent_id": None,    "score": 0.50},
        {"id": "s1_r0", "step": 1, "parent_id": "root",  "score": 0.58},
        {"id": "s1_r1", "step": 1, "parent_id": "root",  "score": 0.66},
        {"id": "s1_r2", "step": 1, "parent_id": "root",  "score": None},
        {"id": "s1_r3", "step": 1, "parent_id": "root",  "score": 0.61},
        {"id": "s2_r0", "step": 2, "parent_id": "s1_r1", "score": 0.71},
        {"id": "s2_r1", "step": 2, "parent_id": "s1_r1", "score": 0.64},
        {"id": "s2_r2", "step": 2, "parent_id": "s1_r3", "score": 0.69},
        {"id": "s2_r3", "step": 2, "parent_id": "s1_r3", "score": "n/a"},
    ]

    if "--dry-run" in sys.argv:
        payload, _, ids = build_user_payload(demo_tree, sota=0.80, higher_is_better=True)
        print(json.dumps(payload, indent=1))
        sys.exit(0)

    critic = AdvantageCritic(model="openai/gpt-oss-120b", gpu_memory_utilization=0.90)
    adv, info = critic.score(demo_tree, sota=0.80, higher_is_better=True)
    print("advantages:", adv)
    print("attempts:", info["attempts"])
