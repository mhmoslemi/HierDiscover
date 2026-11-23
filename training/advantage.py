"""Adaptive-beta, leave-one-out entropic advantages for discovery training.

SPO-RS uses its separate run-wide value tracker in spo_rs.py.
"""

import math
import numpy as np


ENTROPIC_KL_BUDGET = math.log(2)
INVARIANT_LOGIT_CAP = 10.0


def _bisect_kl(kl_at, budget, hi, iterations):
    lo = 0.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if kl_at(mid) < budget:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _kl_to_uniform(beta: float, rewards: np.ndarray) -> float:
    """
    KL(q_\beta || uniform) where q_\beta(i) ->  exp(\beta * r_i).
    Numerically stabilized by subtracting r.max().
    """
    K = len(rewards)
    if K <= 1:
        return 0.0
    logK = math.log(K)
    logits = beta * (rewards - rewards.max())
    # log q_\beta
    log_Z = np.log(np.exp(logits).sum())
    log_q = logits - log_Z
    q = np.exp(log_q)
    # KL(q || uniform) = sum q * (log q + log K)
    return float((q * (log_q + logK)).sum())


def entropic_adaptive_advantages(
    rewards: np.ndarray,
    gamma: float = ENTROPIC_KL_BUDGET,
    beta_max: float = 1e6,
    n_bisect: int = 60,
    eps: float = 1e-12,
):
    """
    Compute leave-one-out entropic advantages for one group.

    rewards: shape (K,) — rewards of the K rollouts from the same parent.

    Returns:
      advantages: shape (K,) — what the policy gradient gets weighted by
      beta:       the temperature found by bisection

    All-equal-reward groups get a zero advantage vector (no gradient).
    """
    r = np.asarray(rewards, dtype=np.float64)
    K = r.shape[0]

    if K < 2 or float(r.max() - r.min()) < eps:
        return np.zeros_like(r), 0.0

    # Step 1: find \beta with KL(q_\beta || uniform) = gamma via bisection
    hi = 1.0

    # If even hi=1 has KL > gamma, then \beta is in (0, 1)
    if _kl_to_uniform(hi, r) < gamma:
        # Need to grow hi until KL exceeds gamma
        while hi < beta_max and _kl_to_uniform(hi, r) < gamma:
            hi *= 2.0
        if _kl_to_uniform(hi, r) < gamma:
            # Saturated; \beta = beta_max (effectively argmax)
            beta = hi
        else:
            beta = None
    else:
        beta = None

    if beta is None:
        _, beta = _bisect_kl(lambda value: _kl_to_uniform(value, r), gamma, hi, n_bisect)

    # Step 2: LOO entropic weights
    # w_n = e^{\beta(r_n - r_max)} / Z_{-n}
    # where Z_{-n} = (sum_m e^{\beta(r_m - r_max)} - e^{\beta(r_n - r_max)}) / (K-1)
    shift = r - r.max()
    e = np.exp(beta * shift)
    total = e.sum()
    Z_loo = (total - e) / (K - 1)
    w = e / (Z_loo + eps)
    advantages = w - 1.0

    return advantages, beta


def capped_weighted_tilt(scaled_rewards, probabilities):
    """Fit the tilt on centered, range-scaled rewards with a fixed logit cap."""
    def statistics(theta):
        logits = theta * scaled_rewards
        weights = probabilities * np.exp(logits)
        total = float(weights.sum())
        log_partition = math.log(total)
        kl = float(np.sum(weights * logits)) / total - log_partition
        return log_partition, max(0.0, kl)

    theta = INVARIANT_LOGIT_CAP
    log_partition, kl = statistics(theta)
    if kl > ENTROPIC_KL_BUDGET:
        theta, _ = _bisect_kl(lambda value: statistics(value)[1], ENTROPIC_KL_BUDGET, theta, 60)
        log_partition, kl = statistics(theta)
    return theta, log_partition, kl


def invariant_group_advantages(rewards):
    """Leave-one-out advantages using the uniform version of the capped tilt."""
    if rewards.size < 2:
        return np.zeros_like(rewards), 0.0, 0.0, 0.0
    center = float(rewards.max())
    span = center - float(rewards.min())
    if not math.isfinite(span):
        raise FloatingPointError("nonfinite invariant reward range")
    if span == 0.0:
        return np.zeros_like(rewards), 0.0, 0.0, 0.0
    scaled = (rewards - center) / span
    probabilities = np.full(rewards.size, 1.0 / rewards.size)
    theta, _, kl = capped_weighted_tilt(scaled, probabilities)
    logits = theta * scaled
    weights = np.exp(logits)
    sibling_mean = (weights.sum() - weights) / (rewards.size - 1)
    return np.expm1(logits - np.log(sibling_mean)), theta, span, kl


def decompose_advantages(log_rewards, group_size, log_history_mean):
    """Blend the sibling and historical baselines by their estimated errors."""
    rewards = np.exp(log_rewards).reshape(-1, group_size)
    group_count = rewards.shape[0]
    group_means = rewards.mean(axis=1)
    within = float(np.var(rewards, axis=1, ddof=1).mean())
    between = 0.0
    if group_count > 1:
        between = max(0.0, float(np.var(group_means, ddof=1)) - within / group_size)

    drift = 0.0
    if log_history_mean is None:
        history_mean = (rewards.sum() - rewards) / (rewards.size - 1)
    else:
        history_mean = math.exp(log_history_mean)
        drift = max(0.0, (float(group_means.mean()) - history_mean) ** 2
                    - between / group_count - within / rewards.size)

    historical_error = between + drift
    sibling_error = within / (group_size - 1)
    weight = historical_error / (historical_error + sibling_error) if historical_error else 0.0
    sibling_mean = (rewards.sum(axis=1, keepdims=True) - rewards) / (group_size - 1)
    baseline = weight * sibling_mean + (1.0 - weight) * history_mean
    sibling = np.expm1(log_rewards - np.log(sibling_mean.ravel()))
    applied = np.expm1(log_rewards - np.log(baseline.ravel()))
    return sibling, applied, weight
