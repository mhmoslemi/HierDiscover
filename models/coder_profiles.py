"""Coder model defaults for prompts and placement."""

import json
from pathlib import Path

_QWEN3_30B_A3B_MODEL_BASENAME = "qwen3-30b-a3b"
_GPT_OSS_120B_MODEL_BASENAME = "gpt-oss-120b"
_QWEN38_27B_MODEL_BASENAME = "qwen3.8-27b"


def _profile_default(merged, key, value, explicit_keys):
    """Apply an automatic model default unless configuration set the key."""
    if key not in explicit_keys:
        merged[key] = value


def _is_exact_qwen3_30b_a3b(model_name):
    """Match only the original hybrid-thinking A3B coder checkpoint.

    Match the exact checkpoint basename for both Hub IDs and local model paths.
    """
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower() == (_QWEN3_30B_A3B_MODEL_BASENAME)


def _model_basename(model_name):
    normalized = str(model_name or "").strip().rstrip("/").replace("\\", "/")
    return normalized.rsplit("/", 1)[-1].lower()


def _coder_model_profile(model_name):
    """Return the automatic runtime contract for explicitly supported coders."""
    basename = _model_basename(model_name)
    if basename == _GPT_OSS_120B_MODEL_BASENAME:
        return {
            "name": "gpt-oss-120b",
            "native_context": 131_072,
            "max_output": 131_072,
            "reasoning_effort": "high",
            "template_kind": "gpt-oss",
            # Modern Unsloth trains the published MXFP4 experts while keeping
            # them packed. This is the same base used by vLLM, without NF4
            # re-quantization or a second full-precision model copy.
            "training_4bit": False,
            "training_memory_fraction": 0.88,
            "vllm_runtime_reserve_gib": 5.0,
        }
    if basename == _QWEN38_27B_MODEL_BASENAME:
        return {
            "name": "qwen3.8-27b",
            "native_context": 262_144,
            "max_output": 262_144,
            "reasoning_effort": "low",
            "template_kind": "qwen3.8",
            "training_4bit": False,
            # A BF16 copy plus the LoRA state fits on each 95+ GiB card.  Use
            # one complete trainer per GPU so independent rollouts execute in
            # parallel; layer-sharding a single copy serializes the forward
            # and backward passes across cards.
            "training_memory_fraction": 0.88,
            # Qwen3.8 has a roughly 248K-token vocabulary. vLLM's exact
            # prompt-logprob path materializes an FP32 softmax for an entire
            # prefill chunk, so its transient projection is much larger than
            # the ordinary generation activation. Keep enough actual card
            # headroom instead of assigning that memory to the KV cache.
            "vllm_runtime_reserve_gib": 15.0,
        }
    return None


def _coder_vllm_persistent_workers(cfg):
    """Release coder engines when requested, or for GPT-OSS adapter training."""
    if getattr(cfg, "hardware_profile", "RTX6000") == "B200":
        return 0
    if bool(getattr(cfg, "reload_coder_between_phases", False)):
        return 0
    if (str(getattr(cfg, "coder_model_profile", "")) == "gpt-oss-120b" and not bool(getattr(cfg, "no_train", False))):
        return 0
    return None


def _resolve_coder_token_limits(merged):
    """Fill missing coder limits without changing explicit limits or model behavior.

    Known coder profiles run first. Other checkpoints supply their context
    through config.json; reading that small file does not load any weights or
    initialize CUDA. No RoPE scaling or context extension is enabled here.
    """
    keys = ("max_seq_length", "max_new_tokens")
    if all(key in merged for key in keys):
        return
    if "max_seq_length" not in merged:
        model_name = merged["coder_model_name"]
        try:
            model_path = Path(model_name).expanduser()
            if model_path.is_dir():
                config_path = model_path / "config.json"
            else:
                from huggingface_hub import hf_hub_download
                config_path = Path(hf_hub_download(model_name, "config.json"))
            metadata = json.loads(config_path.read_text())
            context = None
            for candidate in (metadata.get("text_config"), metadata):
                if not isinstance(candidate, dict):
                    continue
                for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
                    value = candidate.get(key)
                    if (isinstance(value, int) and not isinstance(value, bool) and value > 0):
                        context = value
                        break
                if context is not None:
                    break
            if context is None:
                raise ValueError("checkpoint config has no positive context length")
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve the coder context limit for {model_name!r}; "
                "make its config.json available or explicitly set max_seq_length. "
                f"Details: {exc}") from exc
        merged["max_seq_length"] = context
    # With no separate output cap, allow all context remaining after the prompt.
    merged.setdefault("max_new_tokens", int(merged["max_seq_length"]))
    print(f"[config] coder token limits: max_seq={merged['max_seq_length']}, "
          f"max_new={merged['max_new_tokens']}; output is capped by the "
          "remaining context after the prompt")


def _apply_coder_model_profile(merged, explicit_keys=frozenset()):
    """Wire reasoning, context, rollout, and exact-training behavior by model.

    Explicit YAML, saved configuration, and CLI values override the defaults.
    """
    profile = _coder_model_profile(merged["coder_model_name"])
    if profile is None:
        merged.pop("coder_model_profile", None)
        return

    native_context = int(profile["native_context"])
    merged["coder_model_profile"] = str(profile["name"])
    merged["coder_template_kind"] = str(profile["template_kind"])
    _profile_default(merged, "coder_reasoning_effort", str(profile["reasoning_effort"]), explicit_keys)
    merged["coder_preserve_thinking"] = bool(profile["template_kind"] == "qwen3.8")
    _profile_default(merged, "max_seq_length", native_context, explicit_keys)
    _profile_default(merged, "max_new_tokens", int(profile["max_output"]), explicit_keys)
    _profile_default(merged, "thinking", True, explicit_keys)

    # Quantization and placement describe only the differentiable copy. The
    # aligned GPT-OSS profile keeps the original checkpoint name on both sides.
    _profile_default(merged, "load_in_4bit", bool(profile["training_4bit"]), explicit_keys)
    _profile_default(
        merged, "training_memory_fraction",
        float(profile.get("training_memory_fraction", 0.80)), explicit_keys)

    # The model profile leaves memory outside vLLM's KV cache for CUDA/NCCL
    # and the bounded exact-token projection. The layout planner also accounts
    # for startup weight conversion before choosing the smallest fitting TP
    # group. Qwen3.8 needs a larger projection reserve than the other profiles.
    _profile_default(merged, "vllm_gpu_memory_utilization", 0.965, explicit_keys)
    _profile_default(merged, "vllm_quantization", "", explicit_keys)
    _profile_default(merged, "gen_micro_batch", "auto", explicit_keys)
    _profile_default(merged, "vllm_max_num_batched_tokens", "auto", explicit_keys)
    merged["vllm_runtime_reserve_override_gib"] = float(profile["vllm_runtime_reserve_gib"])
    _profile_default(merged, "vllm_staged_loading", True, explicit_keys)
    # Discard the obsolete hardware-name bypass, including resumed runs.
    # Admission now depends on the bounded-conversion worker contract in
    # gpu_runtime.py, rather than assuming a backend from the GPU's name.
    merged.pop("allow_gpt_oss_tp1", None)

    # Exact long-context training policy. This does not shorten, approximate,
    # or discard a trajectory. It selects exact memory-saving execution modes
    # before known-failing ordinary attempts and raises if all exact modes are
    # exhausted.
    merged["strict_exact_long_training"] = True
    _profile_default(merged, "fused_long_attention", True, explicit_keys)
    merged["long_training_checkpoint_min_tokens"] = 16_384
    merged["long_training_headroom_min_tokens"] = 32_768
    # CPU activation traffic is much slower than recomputation. Use it only
    # after an actual OOM proves it necessary; the persistent OOM profile then
    # starts comparable later trajectories in that exact fallback directly.
    merged["long_training_cpu_offload_min_tokens"] = None

    print(
        f"[model-profile] {profile['name']}: context limit="
        f"{merged['max_seq_length']}, max generated tokens="
        f"{merged['max_new_tokens']}, reasoning="
        f"{merged['coder_reasoning_effort']}, rollout target=maximum memory-safe replicas; "
        "explicit YAML/config values take precedence; "
        "exact long-context training enabled",
        flush=True,
    )


def _uses_sequence_level_policy_ratio(cfg):
    """Use the MoE-stable sequence ratio only for plain Qwen3-30B-A3B."""
    return (getattr(cfg, "advantage_mode", "") != "hybrid"
            and _is_exact_qwen3_30b_a3b(cfg.coder_model_name))
