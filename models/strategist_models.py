"""Checkpoint and native reasoning settings for the frozen strategist."""

STRATEGY_MODEL_NAME = "Qwen/Qwen3.8-27B"
STRATEGY_REASONING_EFFORT = "xhigh"
_NATIVE_CONTEXT = 262_144


def resolve_strategy_token_limits(max_seq_length="auto", max_new_tokens="auto"):
    """Resolve context and output ceilings for the pinned checkpoint."""
    if max_seq_length == "auto":
        max_seq_length = _NATIVE_CONTEXT
    if max_new_tokens == "auto":
        max_new_tokens = max_seq_length
    for key, value in (("strategy_max_seq_length", max_seq_length),
                       ("strategy_max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be auto or a positive integer")
    return max_seq_length, max_new_tokens
