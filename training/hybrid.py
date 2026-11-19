"""Historical and sibling advantages with TTT-Discover's token loss."""

import numpy as np

from training.advantage import entropic_adaptive_advantages


DEFAULT_SIBLING_WEIGHT = 0.10


def hybrid_advantages(rewards, historical_advantages, *, sibling_weight=DEFAULT_SIBLING_WEIGHT):
    """Add a centered adaptive-softmax adjustment for one complete parent group."""
    rewards = np.asarray(rewards, dtype=np.float64)
    historical_advantages = np.asarray(historical_advantages, dtype=np.float64)
    if rewards.ndim != 1 or rewards.size == 0 or historical_advantages.shape != rewards.shape:
        raise ValueError("hybrid rewards and historical advantages must align in one nonempty group")
    _, sibling_beta = entropic_adaptive_advantages(rewards)
    weights = np.exp(sibling_beta * (rewards - rewards.max()))
    adjustment = sibling_weight * (weights / weights.sum() - 1.0 / rewards.size)
    return historical_advantages + adjustment, sibling_beta


def reference_logprob_mean(scored_responses):
    """Center the reference correction over all response tokens in batch order."""
    import torch

    # Each scorer returns only response positions, so the prompt mask is implicit.
    total = torch.zeros((), dtype=torch.float32)
    num_tokens = 0
    for response_length, behavior, reference in scored_responses:
        if behavior is None or reference is None:
            raise ValueError("hybrid reference KL requires saved sampler and base logprobs")
        old = torch.as_tensor(behavior, dtype=torch.float32)
        base = torch.as_tensor(reference, dtype=torch.float32)
        if old.shape != (response_length,) or base.shape != old.shape:
            raise ValueError("hybrid reference KL logprobs must align to the response tokens")
        total += (old - base).sum()
        num_tokens += response_length
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite hybrid batch reference KL")
    return float(total / num_tokens) if num_tokens else 0.0


def importance_sampling_loss(current, example, kl_coef, *, return_tensor_metrics=False):
    """Apply the frozen, reference-adjusted advantage with Tinker's token sum."""
    import torch

    current = current.float()
    old = example.get("old_policy_logprobs")
    if old is None or old.shape != current.shape:
        raise ValueError("hybrid loss requires aligned logprobs from the rollout sampler")
    old = old.detach().to(current)
    advantage = torch.as_tensor(example["advantage"], device=current.device, dtype=current.dtype).detach()
    with torch.no_grad():
        effective_advantage = advantage
        mean_delta = current.new_zeros(())
        if kl_coef:
            base = example.get("reference_logprobs")
            if base is None or base.shape != current.shape or example.get("reference_logprob_mean") is None:
                raise ValueError("hybrid loss requires base logprobs and their batch center")
            delta = old - base.to(current)
            mean_delta = delta.mean()
            effective_advantage = advantage + kl_coef * (example["reference_logprob_mean"] - delta)

    ratio = (current - old).exp()
    loss = -(ratio * effective_advantage).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite hybrid importance-sampling loss")
    with torch.no_grad():
        metrics = {
            "policy_loss": -(ratio * advantage).sum(),
            "kl_estimate": mean_delta,
            "ratio": ratio.mean(),
            "prefix_ratio_max": current.new_ones(()),
            "clipped": current.new_zeros(()),
        }
    if return_tensor_metrics:
        return loss, metrics
    return loss, {key: float(value.item()) for key, value in metrics.items()}
