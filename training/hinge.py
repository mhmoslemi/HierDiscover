"""Frontier hinge adjustment for a batch of sibling advantages."""

import math

import numpy as np
from numpy.typing import ArrayLike


def _frontier_scale(valid_rewards: np.ndarray, frontier_reward: float) -> float:
    median, q75, q95 = np.quantile(valid_rewards, (0.50, 0.75, 0.95))
    if q95 > q75:
        return float(q95 - q75)
    if q95 > median:
        return float(q95 - median)
    low = min(float(valid_rewards.min()), frontier_reward)
    high = max(float(valid_rewards.max()), frontier_reward)
    return high - low


def hinge_advantages(
    rewards: ArrayLike,
    valid: ArrayLike,
    group_advantages: ArrayLike,
    frontier_reward: float | None,
    *,
    kappa: float = 0.7,
) -> np.ndarray:
    """Apply the frontier hinge to one step of rollouts in batch order.

    Pass sibling advantages computed within each parent group. The valid
    rewards across all groups determine one shared percentile spread.
    The frontier is the best valid reward from earlier steps. Pass None
    before a frontier exists to use the original sibling advantages.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    advantages = np.asarray(group_advantages, dtype=np.float64)
    if rewards.ndim != 1 or valid.shape != rewards.shape or advantages.shape != rewards.shape:
        raise ValueError("rewards, validity mask, and group advantages must be aligned vectors")
    if not np.isfinite(advantages).all() or np.any(advantages < -1.0):
        raise ValueError("group advantages must be finite and at least -1")
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")

    valid_rewards = rewards[valid]
    if not np.isfinite(valid_rewards).all():
        raise ValueError("valid rollout rewards must be finite")
    if frontier_reward is None:
        return advantages.copy()
    if not math.isfinite(frontier_reward):
        raise ValueError("frontier reward must be finite")

    lever = np.full_like(rewards, -1.0)
    if valid_rewards.size:
        scale = _frontier_scale(valid_rewards, frontier_reward)
        if not math.isfinite(scale):
            raise FloatingPointError("frontier scale exceeds float64 range")
        if scale:
            with np.errstate(over="ignore"):
                gaps = valid_rewards - frontier_reward
                normalized = gaps / scale
                # Divide first where the reward difference itself overflowed.
                overflow = np.isinf(gaps)
                normalized[overflow] = valid_rewards[overflow] / scale - frontier_reward / scale
            lever[valid] = np.tanh(normalized)
        else:
            lever[valid] = 0.0

    adjusted = advantages.copy()
    affected = lever != 0.0
    # Work in log space to keep the product stable and preserve the -1 floor.
    with np.errstate(divide="ignore", over="raise"):
        log_ratio = np.log1p(advantages[affected]) + kappa * lever[affected]
        adjusted[affected] = np.expm1(log_ratio)
    return adjusted
