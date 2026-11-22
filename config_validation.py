"""Validation for the self-contained problem YAML presets."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from pathlib import Path


LLM_CRITIC_MODE = "llm-critic"
SPO_RS_INVARIANT_MODE = "spo-rs-invariant"
HYBRID_INVARIANT_MODE = "hybrid-invariant"
INVARIANT_ADVANTAGE_MODES = (SPO_RS_INVARIANT_MODE, HYBRID_INVARIANT_MODE)
ADVANTAGE_MODES = ("entropic", "spo-rs", "hybrid", *INVARIANT_ADVANTAGE_MODES, LLM_CRITIC_MODE)
PARENT_SELECTION_MODES = ("puct", "uct")
LEGACY_CODER_CAP_KEY = "coder_thinking_budget"


def uses_hybrid_policy_loss(mode):
    return mode in ("hybrid", HYBRID_INVARIANT_MODE, LLM_CRITIC_MODE)


# These settings are consumed by the shared training, generation, search,
# runtimes. Every checked-in preset carries them so a
# selected YAML is self-contained, but that does not make problem-specific
# fields interchangeable.
COMMON_REQUIRED_KEYS = frozenset("""
problem target
fail_score coder_model_name backend load_in_4bit
lora_rank lora_alpha
lora_dropout target_modules
training_gpu_id available_gpu_ids reserve_last_gpu_for_evaluation
evaluation_gpu_id num_gpus gpu_ids sequential_generation
evaluation_shares_generation
generation_backend gen_micro_batch vllm_gpu_memory_utilization
vllm_enforce_eager vllm_enable_prefix_caching vllm_tensor_parallel_size
vllm_pipeline_parallel_size vllm_quantization
vllm_max_num_batched_tokens vllm_enable_expert_parallel vllm_sleep_level
vllm_staged_loading
num_steps groups_per_step group_size
num_seed_states learning_rate adam_beta1 adam_beta2 adam_epsilon weight_decay
kl_penalty_coef grad_clip
spo_rs_beta logprob_chunk
puct_c max_buffer_size topk_children_per_parent
temperature top_p top_k
deterministic seed
sandbox_timeout_s reward_workers max_saved_construction
""".split())

COMMON_OPTIONAL_KEYS = frozenset({
    "hardware_profile",
    "coder_reasoning_effort",
    "max_seq_length", "max_new_tokens",  # Otherwise derived from the coder.
    "advantage_mode", "isolate_eval",
    "fused_long_attention",
    "min_p", "uct", "parent_selection",
    "spo_rs_d_half", "spo_rs_rho_min", "spo_rs_rho_max",
    "spo_rs_clip_epsilon", "spo_rs_clip_epsilon_low",
    "spo_rs_clip_epsilon_high",
    "hybrid_ttt_weight", "hybrid_spo_weight",
    "hybrid_invariant_lambda", "hybrid_invariant_target_rms",
})

# Compatibility filter only: these keys cannot activate any runtime behavior.
# Archived configs may still contain them; fresh presets no longer do.
RETIRED_BATCH_GROWTH_KEYS = frozenset({
    "max_groups_per_step", "max_group_size", "growth_force_step",
    "growth_valid_yield", "growth_distinct_min", "growth_factor",
})

RETIRED_TRAINER_KEYS = frozenset({
    "fast", "training_layout", "train_examples_per_microbatch", "spo_rs_update_epochs",
    "print_responses", "thinking", "coder_template_kind", "coder_preserve_thinking",
    "sequence_clip_epsilon_low", "sequence_clip_epsilon_high",
})

CPU_PROBLEMS = frozenset({
    "circle_packing", "erdos", "erdos-c4", "ac1", "ac2", "denoising",
})

PROBLEM_REQUIRED_KEYS = {
    "circle_packing": frozenset({
        "num_circles", "degenerate_threshold", "eval_cpus",
    }),
    "erdos": frozenset({
        "budget_s", "eval_cpus",
    }),
    "erdos-c4": frozenset({
        "budget_s", "eval_cpus", "max_coeff_count", "benchmark_c4",
    }),
    "ac1": frozenset({"problem_type", "budget_s", "eval_cpus"}),
    "ac2": frozenset({"problem_type", "budget_s", "eval_cpus"}),
    "denoising": frozenset({"eval_seed", "eval_cpus"}),
    "gpu_mode": frozenset({
        "problem_type", "gpu_type", "gpu_lease_timeout_s",
        "triton_version", "task_yaml", "lib_dir", "kernel_log_chars",
        "kernel_timeout_s", "seed_from_reference",
    }),
}

PROBLEM_OPTIONAL_KEYS = {
    "circle_packing": frozenset(),
    "erdos": frozenset({"target_print", "seed_min_points", "seed_max_points"}),
    "erdos-c4": frozenset(),
    "ac1": frozenset({"target_print"}),
    "ac2": frozenset({"target_print"}),
    "denoising": frozenset(),
    "gpu_mode": frozenset({
        "kernel_lib_dir", "kernel_python", "mla_seed_runtime_us", "score_scale",
    }),
}

# A present field from this table is invalid outside its owner set, even for a
# partial external config. This catches exactly the copy/paste failure that put
# num_circles in every preset.
EXCLUSIVE_KEY_OWNERS = {
    "num_circles": frozenset({"circle_packing"}),
    "degenerate_threshold": frozenset({"circle_packing"}),
    "eval_seed": frozenset({"denoising"}),
    "seed_min_points": frozenset({"erdos"}),
    "seed_max_points": frozenset({"erdos"}),
    "budget_s": frozenset({"erdos", "erdos-c4", "ac1", "ac2"}),
    "eval_cpus": CPU_PROBLEMS,
    "max_coeff_count": frozenset({"erdos-c4"}),
    "benchmark_c4": frozenset({"erdos-c4"}),
    "problem_type": frozenset({"ac1", "ac2", "gpu_mode"}),
    "score_scale": frozenset({"gpu_mode"}),
    "gpu_type": frozenset({"gpu_mode"}),
    "gpu_lease_timeout_s": frozenset({"gpu_mode"}),
    "triton_version": frozenset({"gpu_mode"}),
    "task_yaml": frozenset({"gpu_mode"}),
    "lib_dir": frozenset({"gpu_mode"}),
    "kernel_lib_dir": frozenset({"gpu_mode"}),
    "kernel_python": frozenset({"gpu_mode"}),
    "kernel_log_chars": frozenset({"gpu_mode"}),
    "kernel_timeout_s": frozenset({"gpu_mode"}),
    "seed_from_reference": frozenset({"gpu_mode"}),
    "mla_seed_runtime_us": frozenset({"gpu_mode"}),
}


def _label(source) -> str:
    return str(source) if source is not None else "configuration"


def reject_retired_coder_cap(data: Mapping, *, source) -> None:
    """Reject old inputs that relied on the retired coder cap or its loss mask."""
    cap = data.get(LEGACY_CODER_CAP_KEY, 0)
    if type(cap) is not int or cap != 0 or data.get("forced_token_indices"):
        raise ValueError(
            f"{source}: the coder uses native reasoning effort. A legacy capped "
            "config or rollout cannot be used with this version.")


def retired_search_keys(data):
    """Recognize retired input fields for validation and old single-coder saves."""
    return {key for key in data if str(key).startswith(("strategy_", "jev_"))
            or key in {"strategies", "strategies_per_parent", "programs_per_strategy",
                       "strict_strategy_validation", "frozen_solver",
                       "frozen_programs_per_parent", "separate_strategy_inference_pool"}}


def _positive_number(data: Mapping, key: str, source) -> None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or value <= 0:
        raise ValueError(f"{_label(source)}: {key} must be a positive number")


def _positive_int(data: Mapping, key: str, source) -> None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{_label(source)}: {key} must be a positive integer")


def validate_hybrid_weight(key: str, value, *, source=None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{_label(source)}: {key} must be a finite nonnegative number")


def validate_hybrid_invariant_options(data, *, source=None) -> None:
    if "hybrid_invariant_lambda" in data:
        value = data["hybrid_invariant_lambda"]
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError(f"{_label(source)}: hybrid_invariant_lambda must be in [0, 1]")
    if "hybrid_invariant_target_rms" in data:
        value = data["hybrid_invariant_target_rms"]
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
            raise ValueError(f"{_label(source)}: hybrid_invariant_target_rms must be finite and positive")


def validate_problem_config(
    data: Mapping, *, source: str | Path | None = None,
    require_complete: bool = True,
) -> None:
    """Raise ValueError when a problem configuration violates its contract."""
    if not isinstance(data, Mapping) or not data:
        raise ValueError(f"{_label(source)}: config must be a non-empty mapping")

    obsolete = sorted(key for key in data if str(key).startswith(
        ("rank_", "x_grpo_", "cvar_", "binary_coder_")))
    if obsolete:
        raise ValueError(f"{_label(source)}: removed objective options: {obsolete}")

    retired_model_keys = data.keys() & {
        "model_name", "training_model_name", "coder_training_model_name",
    }
    if retired_model_keys:
        raise ValueError(
            f"{_label(source)}: use coder_model_name in place of {sorted(retired_model_keys)}")

    retired_trainer_keys = data.keys() & RETIRED_TRAINER_KEYS
    if retired_trainer_keys:
        raise ValueError(
            f"{_label(source)}: remove retired settings {sorted(retired_trainer_keys)}")

    retired_search = retired_search_keys(data)
    if retired_search:
        raise ValueError(f"{_label(source)}: removed search options: {sorted(retired_search)}")

    problem = str(data.get("problem", "")).strip().lower()
    if problem not in PROBLEM_REQUIRED_KEYS:
        raise ValueError(
            f"{_label(source)}: problem must be one of "
            f"{sorted(PROBLEM_REQUIRED_KEYS)}, got {problem!r}"
        )

    for key, owners in EXCLUSIVE_KEY_OWNERS.items():
        if key in data and problem not in owners:
            raise ValueError(
                f"{_label(source)}: {key} is not valid for problem={problem}"
            )

    required = COMMON_REQUIRED_KEYS | PROBLEM_REQUIRED_KEYS[problem]
    allowed = required | COMMON_OPTIONAL_KEYS | PROBLEM_OPTIONAL_KEYS[problem]
    if require_complete:
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"{_label(source)}: missing required keys: {sorted(missing)}"
            )
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"{_label(source)}: unknown or misplaced keys: {sorted(unknown)}"
            )

    if "hardware_profile" in data:
        from runtime.hardware_profiles import hardware_profile
        hardware_profile(data["hardware_profile"])
    if problem in ("ac1", "ac2") and data.get("problem_type") != problem:
        raise ValueError(
            f"{_label(source)}: {problem} requires problem_type={problem}"
        )
    if problem == "gpu_mode" and data.get("problem_type") not in {
        "trimul", "mla_decode_nvidia",
    }:
        raise ValueError(
            f"{_label(source)}: unsupported gpu_mode problem_type "
            f"{data.get('problem_type')!r}"
        )
    if (problem == "gpu_mode"
            and data.get("problem_type") == "mla_decode_nvidia"
            and "mla_seed_runtime_us" not in data):
        raise ValueError(
            f"{_label(source)}: mla_decode_nvidia requires the explicit "
            "mla_seed_runtime_us key (use null until measured on gpu_type)"
        )

    for key in ("num_steps", "groups_per_step", "group_size",
                "num_seed_states", "seed_min_points", "seed_max_points",
                "max_new_tokens", "max_seq_length"):
        if key in data:
            _positive_int(data, key, source)
    if (problem == "erdos"
            and data.get("seed_min_points", 40) > data.get("seed_max_points", 99)):
        raise ValueError(
            f"{_label(source)}: seed_min_points must be <= seed_max_points")
    if "parent_selection" in data:
        mode = data["parent_selection"]
        if not isinstance(mode, str) or mode.lower() not in PARENT_SELECTION_MODES:
            raise ValueError(
                f"{_label(source)}: parent_selection must be one of {PARENT_SELECTION_MODES}")
    if "top_k" in data:
        value = data["top_k"]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"{_label(source)}: top_k must be 0 because top-k filtering is disabled project-wide")
    if "min_p" in data:
        value = data["min_p"]
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not 0.0 <= float(value) <= 1.0):
            raise ValueError(f"{_label(source)}: min_p must be in [0, 1]")
    if ("advantage_mode" in data
            and data["advantage_mode"] not in ADVANTAGE_MODES):
        raise ValueError(
            f"{_label(source)}: advantage_mode must be one of {ADVANTAGE_MODES}")
    for key in ("hybrid_ttt_weight", "hybrid_spo_weight"):
        if key in data:
            validate_hybrid_weight(key, data[key], source=source)
    validate_hybrid_invariant_options(data, source=source)
    if "eval_cpus" in data:
        _positive_int(data, "eval_cpus", source)
    if "num_circles" in data:
        _positive_int(data, "num_circles", source)
    if "max_coeff_count" in data:
        _positive_int(data, "max_coeff_count", source)
        if int(data["max_coeff_count"]) < 3:
            raise ValueError(
                f"{_label(source)}: max_coeff_count must be at least 3"
            )
    for key in ("sandbox_timeout_s", "budget_s", "score_scale",
                "kernel_timeout_s", "benchmark_c4"):
        if key in data:
            _positive_number(data, key, source)

    if "top_p" in data and not 0 < float(data["top_p"]) <= 1:
        raise ValueError(f"{_label(source)}: top_p must be in (0, 1]")
    if ("coder_reasoning_effort" in data
            and str(data["coder_reasoning_effort"]).lower()
            not in {"low", "medium", "high", "xhigh"}):
        raise ValueError(
            f"{_label(source)}: coder_reasoning_effort must be low, medium, "
            "high, or xhigh")
    if "vllm_sleep_level" in data and int(data["vllm_sleep_level"]) not in (1, 2):
        raise ValueError(f"{_label(source)}: vllm_sleep_level must be 1 or 2")
    if "vllm_staged_loading" in data and not isinstance(data["vllm_staged_loading"], bool):
        raise ValueError(f"{_label(source)}: vllm_staged_loading must be true or false")
    for key in ("adam_beta1", "adam_beta2"):
        if key in data and not 0 <= float(data[key]) < 1:
            raise ValueError(f"{_label(source)}: {key} must be in [0, 1)")
    if "adam_epsilon" in data:
        _positive_number(data, "adam_epsilon", source)
    if "weight_decay" in data and float(data["weight_decay"]) < 0:
        raise ValueError(f"{_label(source)}: weight_decay must be >= 0")
    if "spo_rs_beta" in data:
        _positive_number(data, "spo_rs_beta", source)
    if "uct" in data and not isinstance(data["uct"], bool):
        raise ValueError(f"{_label(source)}: uct must be true or false")
    if "reward_workers" in data and int(data["reward_workers"]) < 0:
        raise ValueError(f"{_label(source)}: reward_workers must be >= 0")
    if (problem == "gpu_mode" and data.get("problem_type") != "trimul"
            and data.get("reward_workers") != 1):
        raise ValueError(
            f"{_label(source)}: gpu_mode requires reward_workers=1"
        )
    if problem in {"erdos", "erdos-c4", "ac1", "ac2"} and all(
        key in data for key in ("budget_s", "sandbox_timeout_s")
    ) and float(data["sandbox_timeout_s"]) <= float(data["budget_s"]):
        raise ValueError(
            f"{_label(source)}: sandbox_timeout_s must exceed budget_s so "
            "the candidate can return its best result before the hard timeout"
        )
    if problem == "gpu_mode" and not str(data.get("gpu_type", "")).strip():
        raise ValueError(f"{_label(source)}: gpu_mode requires gpu_type")
