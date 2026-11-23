"""Run-wide reward history for fixed and adaptive SPO-RS advantages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from training.advantage import capped_weighted_tilt, decompose_advantages, invariant_group_advantages


def _validate_history_decay(d_half, rho_min, rho_max):
    if not math.isfinite(d_half) or d_half <= 0.0:
        raise ValueError("SPO-RS D_half must be finite and positive")
    if not 0.0 < rho_min <= rho_max < 1.0:
        raise ValueError("SPO-RS rho bounds must satisfy 0 < min <= max < 1")


def _history_retention(divergence, d_half, rho_min, rho_max):
    if math.isnan(divergence) or divergence < 0.0:
        raise ValueError("SPO-RS policy divergence must be nonnegative")
    raw_rho = 0.0 if math.isinf(divergence) else 2.0 ** (-divergence / d_half)
    return min(rho_max, max(rho_min, raw_rho))


class SPORSTracker:
    """Track one entropic value for the entire discovery run."""

    VERSION = 2

    def __init__(self, *, entropic_beta: float, d_half: float,
                 rho_min: float, rho_max: float):
        self.entropic_beta = float(entropic_beta)
        self.d_half = float(d_half)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self._validate_options()
        self._value: float | None = None
        self._effective_count = 0.0
        self._last_policy_adapter: str | None = None
        self._last_step = -1
        self._updates = 0

    def _validate_options(self) -> None:
        if not math.isfinite(self.entropic_beta) or self.entropic_beta <= 0.0:
            raise ValueError("SPO-RS entropic beta must be finite and positive")
        _validate_history_decay(self.d_half, self.rho_min, self.rho_max)

    def __len__(self) -> int:
        return int(self._value is not None)

    @property
    def initialized(self) -> bool:
        return self._value is not None

    @property
    def last_policy_adapter(self) -> str | None:
        return self._last_policy_adapter

    def baseline(self) -> float | None:
        return None if self._value is None else float(self._value)

    def transformed_rewards(self, rewards: Sequence[float]) -> np.ndarray:
        values = np.asarray(rewards, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("SPO-RS rewards must be a nonempty finite vector")
        transformed = np.exp(self.entropic_beta * values)
        if not np.isfinite(transformed).all() or np.any(transformed <= 0.0):
            raise FloatingPointError(
                "nonfinite SPO-RS exponential rewards; reduce spo_rs_beta")
        return transformed

    def normalized_advantages(
        self, transformed_rewards: Sequence[float]
    ) -> np.ndarray | None:
        """Use the saved pre-step value for every rollout in this step."""
        baseline = self.baseline()
        if baseline is None:
            return None
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if (values.ndim != 1 or values.size == 0
                or not np.isfinite(values).all() or np.any(values <= 0.0)):
            raise ValueError(
                "SPO-RS transformed rewards must be finite and positive")
        advantages = (values / baseline - 1.0) / self.entropic_beta
        if not np.isfinite(advantages).all():
            raise FloatingPointError("nonfinite SPO-RS normalized advantages")
        return advantages

    def rollout_advantages(
        self, rewards: Sequence[float]
    ) -> np.ndarray | None:
        """Return advantages against the immutable pre-step history."""
        return self.normalized_advantages(self.transformed_rewards(rewards))

    @staticmethod
    def consecutive_policy_divergence(
        current_scores: Sequence[Sequence[float] | None],
        previous_scores: Sequence[Sequence[float] | None],
        context_ids: Sequence[int],
        *, required_context_ids: Sequence[int] | None = None,
    ) -> tuple[float, int, dict[int, float]]:
        """Estimate mean KL(pi_t || pi_{t-1}) on current-step contexts.

        Current responses are sampled from ``pi_t``. Their trajectory
        log-likelihood difference is therefore a direct Monte Carlo estimate
        of the forward KL. Trajectories are first averaged within a selected
        parent and then the parents are weighted equally.
        """
        if not (len(current_scores) == len(previous_scores) == len(context_ids)):
            raise ValueError("SPO-RS policy scores and contexts must align")
        required_contexts = tuple(dict.fromkeys(
            int(x) for x in (
                context_ids if required_context_ids is None
                else required_context_ids)))
        by_context: dict[int, list[float]] = {
            context_id: [] for context_id in required_contexts
        }
        missing = 0
        for current_values, previous_values, context_id in zip(
                current_scores, previous_scores, context_ids):
            if current_values is None or previous_values is None:
                missing += 1
                continue
            current = np.asarray(current_values, dtype=np.float64)
            previous = np.asarray(previous_values, dtype=np.float64)
            if (current.ndim != 1 or current.size == 0
                    or current.shape != previous.shape
                    or not np.isfinite(current).all()
                    or not np.isfinite(previous).all()):
                missing += 1
                continue
            difference = current - previous
            by_context[int(context_id)].append(float(np.sum(difference, dtype=np.float64)))

        if any(not values for values in by_context.values()):
            return math.inf, missing, {}
        context_estimates = {
            context_id: float(math.fsum(values) / len(values))
            for context_id, values in by_context.items()
        }
        divergence = float(
            math.fsum(context_estimates.values()) / len(context_estimates))
        # A finite Monte Carlo estimate can be slightly negative even though
        # the population KL is nonnegative.
        return max(0.0, divergence), missing, context_estimates

    def update(self, transformed_rewards: Sequence[float], *,
               divergence: float, policy_adapter: str | Path,
               step: int) -> dict:
        """Update once, after this step's advantages have been formed."""
        values = np.asarray(transformed_rewards, dtype=np.float64)
        if (values.ndim != 1 or values.size == 0
                or not np.isfinite(values).all() or np.any(values <= 0.0)):
            raise ValueError(
                "SPO-RS transformed rewards must be finite and positive")
        current_sum = float(math.fsum(float(value) for value in values))
        current_count = int(values.size)
        current_mean = current_sum / current_count
        value_before = self._value
        count_before = float(self._effective_count)

        if value_before is None:
            rho = None
            eta = 1.0
            value_after = current_mean
            count_after = current_count / (1.0 - self.rho_min)
        else:
            divergence = float(divergence)
            rho = _history_retention(divergence, self.d_half, self.rho_min, self.rho_max)
            retained_count = rho * count_before
            count_after = retained_count + current_count
            value_after = (
                retained_count * value_before + current_sum
            ) / count_after
            eta = current_count / count_after

        if (not math.isfinite(value_after) or value_after <= 0.0
                or not math.isfinite(count_after) or count_after <= 0.0):
            raise FloatingPointError("nonfinite SPO-RS tracker update")
        self._value = float(value_after)
        self._effective_count = float(count_after)
        self._last_policy_adapter = Path(policy_adapter).name
        self._last_step = int(step)
        self._updates += 1
        return {
            "initialized": value_before is None,
            "value_before": value_before,
            "value_after": value_after,
            "effective_count_before": count_before,
            "effective_count_after": count_after,
            "group_mean": current_mean,
            "group_size": current_count,
            "divergence": (
                None if value_before is None or not math.isfinite(divergence)
                else float(divergence)),
            "divergence_missing": bool(
                value_before is not None and not math.isfinite(divergence)),
            "rho": rho,
            "eta": eta,
            "policy_adapter": self._last_policy_adapter,
            "step": self._last_step,
            "updates": self._updates,
        }

    def state_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "entropic_beta": self.entropic_beta,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "value": self._value,
            "effective_count": self._effective_count,
            "last_policy_adapter": self._last_policy_adapter,
            "last_step": self._last_step,
            "updates": self._updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != self.VERSION:
            raise ValueError("unsupported SPO-RS tracker state")
        expected = {
            "entropic_beta": self.entropic_beta,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
        }
        for name, value in expected.items():
            saved = float(state.get(name, math.nan))
            if not math.isclose(saved, value, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    f"SPO-RS tracker {name}={saved} does not match config {value}")

        value = state.get("value")
        effective_count = float(state.get("effective_count", 0.0))
        if value is None:
            if effective_count != 0.0:
                raise ValueError("uninitialized SPO-RS tracker has history")
            self._value = None
            self._effective_count = 0.0
        else:
            value = float(value)
            if (not math.isfinite(value) or value <= 0.0
                    or not math.isfinite(effective_count)
                    or effective_count <= 0.0):
                raise ValueError("invalid SPO-RS tracker state")
            self._value = value
            self._effective_count = effective_count
        adapter = state.get("last_policy_adapter")
        self._last_policy_adapter = None if adapter is None else Path(adapter).name
        self._last_step = int(state.get("last_step", -1))
        self._updates = int(state.get("updates", int(self._value is not None)))


def _reward_beta(theta, span):
    beta = theta / span if span else 0.0
    return beta if math.isfinite(beta) else None


@dataclass(frozen=True)
class InvariantBatch:
    effective_rewards: np.ndarray | None
    valid: np.ndarray
    historical: np.ndarray
    sibling: np.ndarray
    sibling_betas: list[float | None]
    applied: np.ndarray
    info: dict


class InvariantSPORSTracker:
    """Retain the effective rewards and weights needed to refit the tilt."""

    VERSION = 3
    consecutive_policy_divergence = staticmethod(SPORSTracker.consecutive_policy_divergence)

    def __init__(self, *, d_half, rho_min, rho_max, policy_adapter):
        _validate_history_decay(d_half, rho_min, rho_max)
        self.d_half = d_half
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.last_policy_adapter = Path(policy_adapter).name
        self.last_step = -1
        self._rewards = np.empty(0, dtype=np.float64)
        self._weights = np.empty(0, dtype=np.float64)
        self._valid = np.empty(0, dtype=bool)

    @property
    def initialized(self):
        # Policy KL bookkeeping starts before the first valid reward.
        return bool(self.last_policy_adapter)

    def __len__(self):
        return self._rewards.size

    def _impute_failures(self, rewards, valid):
        if valid.all():
            return rewards
        current = rewards[valid]
        history = self._rewards[self._valid]
        if not (history.size or current.size):
            return None

        minimum = min(float(values.min()) for values in (history, current) if values.size)
        reference = history if history.size else current
        weights = self._weights[self._valid] if history.size else None
        if weights is not None:
            weights = weights / weights.sum()
        low = float(reference.min())
        span = float(reference.max()) - low
        if not math.isfinite(span):
            raise FloatingPointError("nonfinite valid reward range")
        std = 0.0
        if span:
            scaled = (reference - low) / span
            mean = np.average(scaled, weights=weights)
            std = span * math.sqrt(np.average((scaled - mean) ** 2, weights=weights))
        failure_reward = minimum - std
        if not math.isfinite(failure_reward):
            raise FloatingPointError("nonfinite imputed failure reward")
        rewards[~valid] = failure_reward
        return rewards

    def calculate(self, rewards, valid, *, group_size, ttt_weight=None, spo_weight=1.0, decompose=False):
        """Calculate one complete batch against the history entering its step."""
        rewards = np.array(rewards, dtype=np.float64, copy=True)
        valid = np.asarray(valid, dtype=bool)
        if (rewards.ndim != 1 or rewards.shape != valid.shape or not rewards.size
                or group_size < 1 or rewards.size % group_size):
            raise ValueError("invariant SPO-RS needs an ordered batch of complete groups")
        if not np.isfinite(rewards[valid]).all():
            raise ValueError("invariant SPO-RS requires finite valid rewards")
        effective = self._impute_failures(rewards, valid)
        historical = np.zeros_like(rewards)
        theta = span = kl = 0.0
        log_partition = None
        if effective is not None:
            if not self._rewards.size:
                historical, theta, span, kl = invariant_group_advantages(effective)
                center = float(effective.max())
            else:
                center = float(self._rewards.max())
                span = max(center, float(effective.max())) - min(
                    float(self._rewards.min()), float(effective.min()))
                if not math.isfinite(span):
                    raise FloatingPointError("nonfinite invariant reward range")
                log_partition = 0.0
                if span:
                    probabilities = self._weights / self._weights.sum()
                    theta, log_partition, kl = capped_weighted_tilt(
                        (self._rewards - center) / span, probabilities)
                    historical = np.expm1(theta * ((effective - center) / span) - log_partition)

        sibling = np.zeros_like(rewards)
        sibling_betas = [0.0] * (rewards.size // group_size)
        info = {
            "theta": theta, "reward_span": span, "tilt_kl": kl,
            "beta": _reward_beta(theta, span), "log_partition": log_partition,
            "pooled_start": not self._rewards.size,
        }
        applied = historical
        if decompose:
            group_weight = 0.0
            if span and group_size > 1:
                logits = theta * ((effective - center) / span)
                sibling, applied, group_weight = decompose_advantages(
                    logits, group_size, log_partition)
                sibling_betas = [info["beta"]] * len(sibling_betas)
            info["decompose_group_weight"] = group_weight
        else:
            if effective is not None:
                for group, start in enumerate(range(0, rewards.size, group_size)):
                    group_slice = slice(start, start + group_size)
                    sibling[group_slice], sibling_theta, sibling_span, _ = (
                        invariant_group_advantages(effective[group_slice]))
                    sibling_betas[group] = _reward_beta(sibling_theta, sibling_span)
            if ttt_weight is not None:
                applied = ttt_weight * sibling + spo_weight * historical
                if not np.isfinite(applied).all():
                    raise FloatingPointError("nonfinite invariant hybrid advantages")
        return InvariantBatch(effective, valid, historical, sibling, sibling_betas, applied, info)

    def update(self, batch, *, divergence, policy_adapter, step):
        """Decay and append once after the complete batch has been scored."""
        if step != self.last_step + 1:
            raise ValueError("invariant SPO-RS history must advance one step at a time")
        divergence = float(divergence)
        rho = _history_retention(divergence, self.d_half, self.rho_min, self.rho_max)
        count_before = len(self)
        retained = self._weights * rho
        keep = retained > 0.0
        new_rewards = batch.effective_rewards
        if new_rewards is None:
            new_rewards = np.empty(0, dtype=np.float64)
        self._rewards = np.concatenate((self._rewards[keep], new_rewards))
        self._weights = np.concatenate((retained[keep], np.ones(new_rewards.size)))
        self._valid = np.concatenate((self._valid[keep], batch.valid[:new_rewards.size]))
        self.last_step = step
        self.last_policy_adapter = Path(policy_adapter).name
        return {
            **batch.info,
            "history_size_before": count_before,
            "history_size_after": len(self),
            "group_size": batch.valid.size,
            "divergence": divergence if math.isfinite(divergence) else None,
            "divergence_missing": math.isinf(divergence),
            "rho": rho,
            "policy_adapter": self.last_policy_adapter,
            "step": step,
        }

    def state_dict(self):
        return {
            "version": self.VERSION,
            "d_half": self.d_half,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "rewards": self._rewards.tolist(),
            "weights": self._weights.tolist(),
            "valid": self._valid.tolist(),
            "last_policy_adapter": self.last_policy_adapter,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get("version") != self.VERSION:
            raise ValueError("invariant SPO-RS requires its weighted history checkpoint")
        for key in ("d_half", "rho_min", "rho_max"):
            if float(state[key]) != getattr(self, key):
                raise ValueError(f"invariant SPO-RS checkpoint {key} differs from config")
        rewards = np.asarray(state["rewards"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        valid = np.asarray(state["valid"])
        adapter = state["last_policy_adapter"]
        last_step = int(state["last_step"])
        if (rewards.ndim != 1 or rewards.shape != weights.shape or rewards.shape != valid.shape
                or not np.isfinite(rewards).all()
                or not ((weights > 0.0) & (weights <= 1.0)).all()
                or (valid.size and valid.dtype != bool)
                or not isinstance(adapter, str) or not adapter or last_step < -1
                or (last_step == -1 and rewards.size)):
            raise ValueError("invalid invariant SPO-RS history checkpoint")
        self._rewards, self._weights, self._valid = rewards, weights, valid.astype(bool, copy=False)
        self.last_policy_adapter = Path(adapter).name
        self.last_step = last_step
