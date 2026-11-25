"""Anchored adaptive RS-GRPO variants. Standalone: not wired into training yet.

  ta_rs_grpo_advantages        anchor as a virtual sibling with fixed weight lam (Ratchet-EMA anchor)
  virtual_sibling_advantages   (A/B) anchor appended as one ordinary sibling (all-time best anchor)
  frontier_aware_advantages    (C) eta * sibling credit + (1 - eta) * credit for beating the anchor
  ta_rs_grpo2_advantages       whole step: plain RS-GRPO times a floored frontier gate per child (all-time best anchor)
  ta_rs_grpo3_advantages       whole step: plain RS-GRPO times a parent weight from a KL-budget tilt (all-time best)

Per step t: score every group with the same anchor b_t, then call update_anchor once on the
whole batch to get b_{t+1} (gamma=0 gives the all-time best used by A/B and C). Start with
anchor=None (plain adaptive RS-GRPO) or a seed reward. All three share one failure rule.
Self-check: python -m training.ta_rs_grpo
"""

import numpy as np

from training.advantage import (
    ENTROPIC_KL_BUDGET, INVARIANT_LOGIT_CAP, _bisect_kl, capped_weighted_tilt,
    invariant_group_advantages)

DEFAULT_LAMBDA = 0.25
DEFAULT_GAMMA = 0.1
DEFAULT_ETA = 0.1
DEFAULT_GATE_FLOOR = 0.1
DEFAULT_RESOLUTION = 1e-3


def _check_anchor(anchor):
    if anchor is not None and not np.isfinite(anchor):
        raise ValueError("the anchor must be finite")


def update_anchor(anchor, rewards, valid, gamma=DEFAULT_GAMMA):
    """Ratchet-EMA: jump up to the batch's best valid reward, else relax toward it at rate gamma."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    _check_anchor(anchor)
    rewards = np.asarray(rewards, dtype=np.float64)
    best = rewards[np.asarray(valid, dtype=bool) & np.isfinite(rewards)]
    if best.size == 0:
        return anchor  # failed rollouts never move the anchor
    top = float(best.max())
    if anchor is None or top >= anchor:
        return top
    return (1.0 - gamma) * anchor + gamma * top


def _prepare(rewards, valid, anchor):
    """Validate one group and impute failed rewards; None means every advantage is zero."""
    r = np.array(rewards, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    if ok.shape != r.shape or r.ndim != 1:
        raise ValueError("rewards and valid must be aligned 1-D vectors")
    if not np.isfinite(r[ok]).all() or (anchor is not None and not np.isfinite(anchor)):
        raise ValueError("valid rewards and the anchor must be finite")
    anchors = [] if anchor is None else [float(anchor)]
    known = np.concatenate([r[ok], anchors])
    if r.size < 2 or known.size == 0:
        return None
    if not ok.all():
        if known.max() > known.min():
            r[~ok] = known.min() - known.std()
        else:
            r[ok], r[~ok], anchors = 1.0, 0.0, [1.0] * len(anchors)
    return r, (anchors[0] if anchors else None)


def _beta(theta, span):
    return theta / span if span else 0.0


def ta_rs_grpo_advantages(rewards, valid, anchor, lam=DEFAULT_LAMBDA):
    """Advantages for one parent's group; returns (advantages, beta in reward units)."""
    if not 0.0 <= lam < 0.5:
        raise ValueError("lam must be in [0, 0.5)")
    n = len(rewards)
    prepared = _prepare(rewards, valid, anchor)
    if prepared is None:
        return np.zeros(n), 0.0
    r, anchor = prepared
    anchors = [] if anchor is None else [anchor]

    weight = lam if anchors else 0.0
    values = np.concatenate([r, anchors])
    top = float(values.max())
    span = top - float(values.min())
    if span == 0.0:
        return np.zeros(n), 0.0
    scaled = (values - top) / span
    probs = np.full(values.size, (1.0 - weight) / n)
    if anchors:
        probs[n] = weight
    # theta = beta * span, so the cap theta <= L is the rule's beta * Delta <= L.
    theta, _, _ = capped_weighted_tilt(scaled, probs)
    logits = theta * scaled
    e = np.exp(logits[:n])
    denom = (1.0 - weight) * (e.sum() - e) / (n - 1)
    if anchors:
        denom += weight * np.exp(logits[n])
    return np.expm1(logits[:n] - np.log(denom)), theta / span


def virtual_sibling_advantages(rewards, valid, anchor):
    """(A/B) Adaptive RS-GRPO on [*children, anchor], keeping the children's advantages."""
    n = len(rewards)
    prepared = _prepare(rewards, valid, anchor)
    if prepared is None:
        return np.zeros(n), 0.0
    r, anchor = prepared
    values = r if anchor is None else np.append(r, anchor)
    advantages, theta, span, _ = invariant_group_advantages(values)
    return advantages[:n], _beta(theta, span)


def frontier_aware_advantages(rewards, valid, anchor, eta=DEFAULT_ETA, delta=0.0):
    """(C) eta * G + (1 - eta) * F; F reuses G's beta on the improvement [r - anchor - delta]_+."""
    n = len(rewards)
    prepared = _prepare(rewards, valid, anchor)
    if prepared is None:
        return np.zeros(n), 0.0
    r, anchor = prepared
    sibling, theta, span, _ = invariant_group_advantages(r)
    beta = _beta(theta, span)
    if anchor is None:
        return sibling, beta  # no frontier yet: every child counts as crossing it, so F = G
    gain = np.maximum(r - anchor - delta, 0.0)
    frontier = np.zeros(n)
    if gain.max() > gain.min():
        logits = beta * (gain - gain.max())
        e = np.exp(logits)
        frontier = np.expm1(logits - np.log((e.sum() - e) / (n - 1)))
    return eta * sibling + (1.0 - eta) * frontier, beta


def _step_arrays(rewards, valid, groups, anchor):
    """Validate one step's per-rollout inputs."""
    _check_anchor(anchor)
    rewards = np.asarray(rewards, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    groups = np.asarray(groups)
    if rewards.ndim != 1 or valid.shape != rewards.shape or groups.shape != rewards.shape:
        raise ValueError("rewards, valid and groups must be aligned 1-D vectors")
    if not np.isfinite(rewards[valid]).all():
        raise ValueError("valid rewards must be finite")
    return rewards, valid, groups


def _resolved_group_advantages(r, min_span):
    """invariant_group_advantages with the reward range floored at min_span; returns (advantages, theta, span)."""
    n = r.size
    top = float(r.max())
    span = max(top - float(r.min()), min_span)
    if span == 0.0:
        return np.zeros(n), 0.0, 0.0
    scaled = (r - top) / span
    theta, _, _ = capped_weighted_tilt(scaled, np.full(n, 1.0 / n))
    logits = theta * scaled
    w = np.exp(logits)
    return np.expm1(logits - np.log((w.sum() - w) / (n - 1))), theta, span


def ta_rs_grpo2_advantages(rewards, valid, groups, anchor, lam=DEFAULT_LAMBDA,
                           floor=DEFAULT_GATE_FLOOR, resolution=DEFAULT_RESOLUTION):
    """TA-RS-GRPO2 for one whole step: A_i = gate_i * plain RS-GRPO_i.

    beta and the range cap come from each group's children only, so no anchor or lam=0 gives
    plain RS-GRPO. The anchor only sets gate_i = E_-i / ((1-lam) E_-i + lam e^{beta (b - m)}),
    E_-i the sibling mean of e^{beta (r_j - m)}; no constant shift is added. The gate is mixed
    with a floor, floor + (1 - floor) * gate, so no parent is zeroed. Each group's range is at
    least resolution * the step's valid range, so float-level differences earn no credit. Gates
    are normalised to mean 1 over the step's gated children. Groups whose valid rewards all tie
    next to failures are scored on a 1/0 scale and stay ungated (gate 1).
    """
    if not 0.0 <= lam < 1.0 or not 0.0 <= floor < 1.0 or resolution < 0.0:
        raise ValueError("need 0 <= lam < 1, 0 <= floor < 1 and resolution >= 0")
    rewards, valid, groups = _step_arrays(rewards, valid, groups, anchor)
    step_valid = rewards[valid]
    min_span = resolution * float(np.ptp(step_valid)) if step_valid.size else 0.0
    sibling = np.zeros(rewards.size)
    log_gate = np.zeros(rewards.size)
    gated = np.zeros(rewards.size, dtype=bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        prepared = _prepare(rewards[idx], valid[idx], None)  # failures imputed from valid children only
        if prepared is None:
            continue
        r, ok = prepared[0], valid[idx]
        remapped = not ok.all() and np.ptp(rewards[idx][ok]) == 0.0  # _prepare rewrote rewards to 1/0
        sibling[idx], theta, span = _resolved_group_advantages(r, 0.0 if remapped else min_span)
        if remapped or anchor is None or lam == 0.0 or span == 0.0:
            continue
        e = np.exp(theta * (r - r.max()) / span)
        log_rest = np.log((e.sum() - e) / (r.size - 1))
        anchor_logit = theta * (anchor - r.max()) / span
        log_g = -np.logaddexp(np.log1p(-lam), np.log(lam) + anchor_logit - log_rest)
        if floor > 0.0:
            log_g = np.logaddexp(np.log(floor), np.log1p(-floor) + log_g)
        log_gate[idx] = log_g
        gated[idx] = True
    if gated.any():
        top = log_gate[gated].max()
        log_gate[gated] -= top + np.log(np.mean(np.exp(log_gate[gated] - top)))
    return np.exp(log_gate) * sibling


def _parent_weights(parent_rewards, anchor, lam):
    """Mean-1 weights sigmoid(tau (v - b) + log((1 - lam) / lam)), tau fitted like adaptive RS-GRPO's beta."""
    v = np.asarray(parent_rewards, dtype=np.float64)
    span = float(np.ptp(np.append(v, anchor)))
    if span == 0.0:
        return np.ones(v.size)
    offset = np.log((1.0 - lam) / lam)

    def log_weights(tau):
        lw = -np.logaddexp(0.0, -(tau * (v - anchor) + offset))  # log sigmoid
        return lw - lw.max()

    def kl(tau):
        p = np.exp(log_weights(tau))
        p /= p.sum()
        return float(np.sum(p * np.log(p * p.size)))

    tau = INVARIANT_LOGIT_CAP / span  # tau * range <= L, as for beta
    if kl(tau) > ENTROPIC_KL_BUDGET:
        tau = _bisect_kl(kl, ENTROPIC_KL_BUDGET, tau, 60)[0]
    w = np.exp(log_weights(tau))
    return w / w.mean()


def ta_rs_grpo3_advantages(rewards, valid, groups, parent_rewards, anchor, lam=DEFAULT_LAMBDA):
    """TA-RS-GRPO3 for one whole step: A_i = w_parent(i) * plain RS-GRPO_i.

    w_s = sigmoid(tau (v_s - b) + log((1 - lam) / lam)) for parent reward v_s, normalised to mean 1
    over the step's parents. tau is the largest value with tau * range({v_s} U {b}) <= L and
    KL(w / sum w || uniform) <= log 2, the same budget adaptive RS-GRPO uses for beta. No anchor or
    lam=0 gives plain RS-GRPO. Parents with unknown reward (NaN) get weight 1.
    """
    if not 0.0 <= lam < 1.0:
        raise ValueError("lam must be in [0, 1)")
    rewards, valid, groups = _step_arrays(rewards, valid, groups, anchor)
    parent_rewards = np.asarray(parent_rewards, dtype=np.float64)
    if parent_rewards.shape != rewards.shape:
        raise ValueError("parent_rewards must give each rollout its parent's reward")
    ids = np.unique(groups)
    sibling = np.zeros(rewards.size)
    parent = np.full(ids.size, np.nan)
    for k, g in enumerate(ids):
        idx = np.flatnonzero(groups == g)
        known = np.unique(parent_rewards[idx][np.isfinite(parent_rewards[idx])])
        if known.size > 1:
            raise ValueError("every rollout of a group must share one parent reward")
        parent[k] = known[0] if known.size else np.nan
        prepared = _prepare(rewards[idx], valid[idx], None)
        if prepared is not None:
            sibling[idx] = invariant_group_advantages(prepared[0])[0]
    weights = np.ones(ids.size)
    known = np.isfinite(parent)
    if anchor is not None and lam > 0.0 and known.any():
        weights[known] = _parent_weights(parent[known], anchor, lam)
    return weights[np.searchsorted(ids, groups)] * sibling


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    group = rng.normal(0.65, 0.05, 16)
    every = np.ones(16, dtype=bool)
    five = np.ones(5, dtype=bool)
    early, late = np.array([.62, .66, .60, .70, .64]), np.array([.63, .68, .61, .73, .62])

    # No anchor reproduces the existing range-capped adaptive RS-GRPO exactly, for all three.
    plain = invariant_group_advantages(group)[0]
    for fn in (ta_rs_grpo_advantages, virtual_sibling_advantages, frontier_aware_advantages):
        assert np.allclose(fn(group, every, None)[0], plain)
        # Affine invariance in rewards and anchor together.
        assert np.allclose(fn(group, every, 0.7)[0], fn(3.0 * group - 5.0, every, 3.0 * 0.7 - 5.0)[0])
        # A group with no valid rollout and no anchor carries no information.
        assert not fn([0.0, 0.0], [0, 0], None)[0].any()

    # TA: an equal group that beats the anchor gets about lam / (1 - lam) each.
    assert np.allclose(ta_rs_grpo_advantages(np.full(5, 0.7), five, 0.55)[0], 1.0 / 3.0, atol=1e-3)
    lag = ta_rs_grpo_advantages(late, five, 0.80)[0]
    assert lag.mean() < 0 and lag.argmax() == 3
    mixed = ta_rs_grpo_advantages([.6, .7, 0.0, .65], [1, 1, 0, 1], 0.68)[0]
    assert mixed.argmin() == 2
    dead = ta_rs_grpo_advantages([0.0, 0.0], [0, 0], 0.5)[0]
    assert (dead < 0).all() and np.allclose(dead, dead[0])

    # A/B: the worked examples from the proposal.
    assert np.allclose(virtual_sibling_advantages(np.full(5, 0.7), five, 0.55)[0], 0.25, atol=1e-3)
    assert np.allclose(virtual_sibling_advantages(np.full(5, 0.7), five, 0.80)[0], -0.68, atol=5e-3)
    assert abs(virtual_sibling_advantages(early, five, 0.55)[0][3] - 7.87) < 5e-3
    assert abs(virtual_sibling_advantages(late, five, 0.80)[0][3] - 0.17) < 5e-3

    # C: below the frontier keeps eta * G, above it is exactly G, an all-failed group is zero.
    G = invariant_group_advantages(late)[0]
    assert np.allclose(frontier_aware_advantages(late, five, 0.80)[0], DEFAULT_ETA * G)
    assert np.allclose(frontier_aware_advantages(late, five, 0.50)[0], G)
    assert not frontier_aware_advantages([0.0, 0.0], [0, 0], 0.5)[0].any()
    crossed = frontier_aware_advantages([.63, .68, .83, .62], np.ones(4, bool), 0.80)[0]
    assert crossed.argmax() == 2 and (np.delete(crossed, 2) < 0).all()

    # TA2: two groups of the same shape, one at the anchor and one far below it.
    pair = np.concatenate([group, group - 1.0])
    ids = np.repeat([0, 1], 16)
    both = np.ones(32, dtype=bool)
    plain2 = np.concatenate([plain, plain])
    assert np.allclose(ta_rs_grpo2_advantages(pair, both, ids, None), plain2)
    assert np.allclose(ta_rs_grpo2_advantages(pair, both, ids, group.max(), lam=0.0), plain2)
    t2 = ta_rs_grpo2_advantages(pair, both, ids, group.max())
    assert np.allclose(t2, ta_rs_grpo2_advantages(3.0 * pair - 5.0, both, ids, 3.0 * group.max() - 5.0))
    assert (np.sign(t2) == np.sign(plain2)).all()  # a gate rescales, it never flips sibling credit
    assert (np.abs(t2[16:]) < np.abs(t2[:16])).all()  # the stale group is down-weighted
    assert not ta_rs_grpo2_advantages([0.0, 0.0, .6, .7], [0, 0, 1, 1], [0, 0, 1, 1], 0.65)[:2].any()
    # The floor keeps the stale group alive: its gate is at least ~floor relative to the best gate.
    gates = t2 / plain2
    assert gates[16:].min() / gates[:16].max() > 0.05
    # Float-level differences inside a group earn ~no credit; a normal group in the step is unaffected.
    ties = np.concatenate([group, 0.6 + np.array([0, 1e-10, 2e-10, 3e-10])])
    tie_ids = np.repeat([0, 1], [16, 4])
    tied = ta_rs_grpo2_advantages(ties, np.ones(20, bool), tie_ids, group.max())
    assert np.abs(tied[16:]).max() < 1e-4 and np.abs(tied[:16]).max() > 0.5
    # A group whose valid rewards tie next to failures is scored 1/0 and left ungated.
    odd = np.concatenate([group, [0.6, 0.6, 0.0, 0.0]])
    odd_ok = np.concatenate([every, [1, 1, 0, 0]]).astype(bool)
    oa = ta_rs_grpo2_advantages(odd, odd_ok, tie_ids, 0.9)
    assert np.allclose(oa[16:], invariant_group_advantages(np.array([1.0, 1.0, 0.0, 0.0]))[0])
    assert np.allclose(oa, ta_rs_grpo2_advantages(3 * odd - 5, odd_ok, tie_ids, 3 * 0.9 - 5))
    for bad in (np.nan, np.inf):
        try:
            ta_rs_grpo2_advantages(pair, both, ids, bad)
            raise AssertionError("a non-finite anchor must raise")
        except ValueError:
            pass

    # TA3: one weight per parent from a KL-budget tilt of the parents' rewards toward the anchor.
    parents = np.repeat([0.66, 0.40], 16)
    assert np.allclose(ta_rs_grpo3_advantages(pair, both, ids, parents, None), plain2)
    assert np.allclose(ta_rs_grpo3_advantages(pair, both, ids, parents, 0.7, lam=0.0), plain2)
    t3 = ta_rs_grpo3_advantages(pair, both, ids, parents, 0.7)
    assert np.allclose(t3, ta_rs_grpo3_advantages(3 * pair - 5, both, ids, 3 * parents - 5, 3 * 0.7 - 5))
    w3 = t3 / plain2
    assert np.allclose(w3[:16], w3[0]) and np.allclose(w3[16:], w3[16]) and w3[0] > w3[16] > 0
    assert np.isclose((w3[0] + w3[16]) / 2, 1.0)
    many = np.linspace(0.1, 0.7, 8)
    pw = _parent_weights(many, 0.7, DEFAULT_LAMBDA)
    p = pw / pw.sum()
    assert np.sum(p * np.log(p * p.size)) <= ENTROPIC_KL_BUDGET + 1e-9 and (np.diff(pw) > 0).all()
    unknown = ta_rs_grpo3_advantages(pair, both, ids, np.repeat([0.66, np.nan], 16), 0.7)
    assert np.allclose(unknown, plain2)  # one known parent normalises to 1, the unknown one keeps 1

    assert update_anchor(None, [0.6, 0.4], [1, 1]) == 0.6
    assert update_anchor(0.5, [np.nan, 0.6], [1, 1]) == 0.6
    assert update_anchor(0.5, [0.6], [1]) == 0.6
    assert np.isclose(update_anchor(0.8, [0.7], [1], gamma=0.1), 0.79)
    assert update_anchor(0.8, [0.7], [1], gamma=0.0) == 0.8
    assert update_anchor(0.5, [0.9], [0]) == 0.5
    print("ta_rs_grpo self-check passed")
