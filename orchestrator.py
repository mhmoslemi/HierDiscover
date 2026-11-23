"""TTT-Discover multi-problem runner with test-time LoRA training.

Configuration: saved settings, current YAML rollout controls, then CLI.
Entropic training uses adaptive-beta, leave-one-out group advantages.
SPO-RS uses a historical run-wide baseline and clipped policy updates.
Hybrid and llm-critic use a token-summed importance-sampling update.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
import sys
import json
import math
import random
import re
import shutil
import threading
import time
from contextlib import ExitStack, nullcontext
from runtime import trimul_evaluation
from runtime.host_memory import HostMemoryUnavailable, HostMemoryTimeout
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from config_validation import (
    DECOMPOSE_MODE, DEFAULT_EVAL_PROGRESS_INTERVAL_S, HYBRID_INVARIANT_MODE, INVARIANT_ADVANTAGE_MODES,
    LLM_CRITIC_MODE, PARENT_SELECTION_MODES,
    SPO_RS_INVARIANT_MODE, reject_retired_coder_cap, uses_hybrid_policy_loss,
)
from generation.output_validation import (
    coder_output_issue, ttt_ac1_program, ttt_ac1_python_block, ttt_ac1_response_code,
)
# Same-name aliases retain entrypoint exports used by existing tools/tests.
from diagnostics.console import (
    _NoticeRoutingStream as _NoticeRoutingStream,
    _TerminalLogSink as _TerminalLogSink, _TerminalTeeStream as _TerminalTeeStream,
    _TimestampedLineStream as _TimestampedLineStream, _install_console_timestamps, _install_terminal_log,
    _route_dependency_notices, bind_setting_log, setting_log_only,
)
from diagnostics.evaluation_progress import CpuEvaluationProgress, write_progress_line
from models.coder_profiles import (
    _coder_model_profile as _coder_model_profile,
    _coder_vllm_persistent_workers, _model_basename as _model_basename,
)
from runtime.configuration import (
    _build_arg_parser as _build_arg_parser, _isolated_ac1_cpu_pairs, _isolated_evaluation_cpu_ids,
    _is_spo_rs_mode,
    _normalize_saved_advantage_config as _normalize_saved_advantage_config, _parse_gpu_ids,
    _resolve_training_backend as _resolve_training_backend,
    _resolve_training_memory_budgets as _resolve_training_memory_budgets,
    _uses_clipped_policy_loss, _uses_replayable_policy_update, _uses_reward_history,
    _validate_known_training_capacity as _validate_known_training_capacity, load_config,
)
from training.exact_overflow import (
    ExactLongContextTrainingOOM, _clear_exact_sharded_recovery,
    _load_exact_spo_rs_overflow, _read_exact_sharded_recovery, _restart_for_exact_resume,
    _resume_process_argv as _resume_process_argv, _save_exact_spo_rs_overflow as _save_exact_spo_rs_overflow,
    _write_exact_sharded_recovery, _validate_precomputed_exact_gradients,
)
from training.fast_distributed import _move_optimizer_state, _restore_optimizer_state_to_parameters
from training.trainers import (
    ProcessDistributedTrainer,
    _apply_precomputed_exact_gradients as _apply_precomputed_exact_gradients,
    _clipped_policy_options as _clipped_policy_options, _complete_spo_rs_overlap_examples,
    _initialize_behavior_logprobs as _initialize_behavior_logprobs,
    _prepare_entropic_overlap_group, _prepare_spo_rs_overlap_inputs,
    _valid_example_token_logprobs,
    clipped_surrogate_loss as clipped_surrogate_loss,
    compute_group_advantages, compute_token_logprobs as compute_token_logprobs,
)
from training import critic_step
from training.hybrid import hybrid_advantages, reference_logprob_mean


_ANSI_ORANGE = "\033[38;5;208m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"


def _pin_training_process(training_gpu_ids) -> None:
    """Expose the ordered training/model-parallel group before CUDA imports."""
    ids = _parse_gpu_ids(training_gpu_ids)
    if not ids:
        raise ValueError("training_gpu_ids must contain at least one GPU")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    visible = ",".join(str(gpu_id) for gpu_id in ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["TTT_TRAINING_GPU_ID"] = str(ids[0])
    os.environ["TTT_TRAINING_GPU_IDS"] = visible
    logical = ",".join(f"cuda:{idx}" for idx in range(len(ids)))
    print(f'[gpu] trainer model-parallel group: physical {ids} (logical {logical})')


def _attention_head_count(model) -> int:
    cfg = getattr(model, "config", None)
    for _ in range(3):
        if cfg is None:
            return 0
        for name in ("num_attention_heads", "n_head", "num_heads"):
            value = getattr(cfg, name, None)
            if value:
                return int(value)
        cfg = getattr(cfg, "text_config", None)
    return 0


# ======================================================================
# Generation
# ======================================================================
def _generate_batch(model, tokenizer, inputs, input_len, n_samples, cfg):
    """
    Generate n_samples completions for a SINGLE prompt in ONE batched
    model.generate() call (via num_return_sequences). Returns a list of
    (text, gen_token_ids).
    """
    import torch
    from generation.gen_workers import _canonical_generated_text

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    max_new_tokens = min(int(cfg.max_new_tokens), int(cfg.max_seq_length) - int(input_len))
    if max_new_tokens < 1:
        return [("", []) for _ in range(int(n_samples))]

    policy_sampling = {}
    sampling_top_k = getattr(cfg, "sampling_top_k", None)
    sampling_min_p = getattr(cfg, "sampling_min_p", None)
    if sampling_top_k is not None:
        policy_sampling["top_k"] = int(sampling_top_k)
    if sampling_min_p is not None:
        policy_sampling["min_p"] = float(sampling_min_p)
    if _uses_clipped_policy_loss(getattr(cfg, "advantage_mode", "entropic")):
        policy_sampling.update(top_k=0, repetition_penalty=1.0)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            **policy_sampling,
            pad_token_id=pad_id,
            num_return_sequences=n_samples,
        )
    results = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:].tolist()
        if eos_id is not None and eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id) + 1]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = _canonical_generated_text(tokenizer, gen_ids, decoded)
        results.append((text, gen_ids))
    return results


def generate_responses(model, tokenizer, prompt_text: str, group_size: int, cfg):
    """
    Generate `group_size` responses from a single prompt, batched.

    Try to generate all `group_size` at once. If OOMs, halve the
    per-call batch size and retry, accumulating until we have group_size
    responses. This keeps the algorithm identical (still group_size IID
    samples from the same policy) while using the GPU in parallel.

    Returns (list of (text, gen_token_ids), prompt_len_in_tokens).
    """
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    responses = []
    remaining = group_size
    # Start by trying the whole group in one call. Cap the first attempt at the
    # micro-batch size when set, so the single-GPU path honors the same per-call
    # ceiling as the multi-GPU workers; OOM halving still applies below it.
    mb = int(getattr(cfg, "gen_micro_batch", 0) or 0)
    batch = group_size if (mb <= 0 or mb > group_size) else mb

    while remaining > 0:
        n = min(batch, remaining)
        try:
            chunk = _generate_batch(model, tokenizer, inputs, input_len, n, cfg)
            responses.extend(chunk)
            remaining -= n
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                # Can't even do one — re-raise, nothing we can do
                raise
            batch = max(1, batch // 2)
            print(f"  [oom] halving generation batch size to {batch}")

    return responses, input_len


def generate_prompt_jobs(model, tokenizer, prompts_by_group, counts_by_group,
                         cfg, *, max_new_tokens=None, temperature=None,
                         top_p=None, top_k=None, min_p=None, cap_state=None):
    """Stream locally generated rollouts from cross-prompt HF batches."""
    if len(prompts_by_group) != len(counts_by_group):
        raise ValueError("counts_by_group must align with prompts_by_group")
    if _uses_clipped_policy_loss(getattr(cfg, "advantage_mode", "entropic")):
        # This local path owns its sampling arguments, including top_k=0.
        # Use the existing per-prompt OOM-aware micro-batcher instead of an
        # external HF helper that may inherit a top-k generation default.
        policy_cfg = SimpleNamespace(**vars(cfg))
        policy_cfg.temperature = 1.0
        policy_cfg.top_p = 1.0
        policy_cfg.sampling_top_k = 0
        policy_cfg.sampling_min_p = 0.0
        if max_new_tokens is not None:
            policy_cfg.max_new_tokens = int(max_new_tokens)
        for group_idx, (prompt, count) in enumerate(zip(prompts_by_group, counts_by_group)):
            if int(count) > 0:
                responses, _ = generate_responses(model, tokenizer, prompt, int(count), policy_cfg)
                yield group_idx, responses
        return

    from generation.gen_workers import _iter_hf_job_batches
    jobs = [
        (group_idx, prompt, int(count))
        for group_idx, (prompt, count)
        in enumerate(zip(prompts_by_group, counts_by_group))
        if int(count) > 0
    ]
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens),
        "temperature": float(temperature if temperature is not None else cfg.temperature),
        "top_p": float(top_p if top_p is not None else cfg.top_p),
        "micro_batch": int(getattr(cfg, 'gen_micro_batch', 0) or 0),
    }
    resolved_top_k = (top_k if top_k is not None else getattr(cfg, "sampling_top_k", None))
    resolved_min_p = (min_p if min_p is not None else getattr(cfg, "sampling_min_p", None))
    if resolved_top_k is not None:
        gen_kwargs["top_k"] = int(resolved_top_k)
    if resolved_min_p is not None:
        gen_kwargs["min_p"] = float(resolved_min_p)
    yield from _iter_hf_job_batches(
        model, tokenizer, jobs, model.device, int(cfg.max_seq_length),
        gen_kwargs, cap_state=cap_state, log_prefix="trainer rollout")


# ======================================================================
# LoRA adapter sync (main process -> generation workers)
# ======================================================================
def _as_float_list(seq, max_len: int = 4096):
    """
    Coerce a construction to a plain list of floats for the rollout meta.

    Returns None when absent, and refuses anything longer than max_len so a
    problem with a huge construction cannot bloat every meta file. 0 disables
    saving entirely.
    """
    if seq is None or max_len == 0:
        return None
    try:
        out = [float(x) for x in seq]
    except (TypeError, ValueError):
        return None
    if not out or (max_len > 0 and len(out) > max_len):
        return None
    return out


def _saved_evaluator_output(stdout):
    # Preserve the prompt's last 500 characters even on artifact-only resume.
    text = (stdout or "").strip()
    if len(text) > 2000:
        return text[:995] + "\n...\n" + text[-1000:]
    return text


def _evaluated_rollout_meta(pending_meta, result):
    """Add one completed CPU evaluation to its provisional rollout metadata."""
    return {
        **pending_meta,
        "status": "evaluated",
        "reward": float(result.reward),
        "raw_score": (
            float(result.raw_score) if result.raw_score is not None else None),
        "valid": bool(result.valid),
        "parsed": bool(result.parsed),
        "ran": bool(result.ran),
        "msg": result.msg,
        "failure_kind": result.failure_kind,
        "sandbox_stdout": _saved_evaluator_output(result.stdout),
        "poisson": result.poisson,
    }


def _ac1_archive_metadata(result, evaluation_order):
    from search.sampler import AC1_CONSTRUCTION_LIMITS

    construction = result.construction
    length = len(construction) if construction is not None else 0
    meta = {"evaluation_order": evaluation_order, "construction_length": length}
    low, high = AC1_CONSTRUCTION_LIMITS
    if result.valid and low <= length <= high:
        # Retain every admissible construction for exact replay, even beyond
        # the ordinary construction limit used for plotting/debug metadata.
        meta["construction"] = json.dumps(construction)
    return meta


def _rollout_advantage_fields(advantage, *, ttt_advantage, spo_rs_advantage,
                              ttt_sibling_max, ttt_adaptive_beta, parent_raw_score,
                              spo_rs_v_t):
    """Write the unweighted components and the applied advantage together."""
    return {
        "ttt_advantage": float(ttt_advantage),
        "spo_rs_advantage": float(spo_rs_advantage) if spo_rs_advantage is not None else None,
        "advantage": float(advantage),
        "ttt_sibling_max": float(ttt_sibling_max),
        "ttt_adaptive_beta": float(ttt_adaptive_beta) if ttt_adaptive_beta is not None else None,
        "parent_raw_score": float(parent_raw_score) if parent_raw_score is not None else None,
        "spo_rs_v_t": float(spo_rs_v_t) if spo_rs_v_t is not None else None,
    }


def _advance_live_best_raw(result, current, problem):
    if not bool(getattr(result, "valid", False)):
        return current, None
    try:
        value = float(result.raw_score)
    except (TypeError, ValueError):
        return current, None
    if not math.isfinite(value):
        return current, None
    if current is not None and not (
            value > current if problem.maximize else value < current):
        return current, None
    return value, f"best {problem.metric_name}: {value:.9f}"


def _adapter_dir(exp_dir, step_idx):
    from pathlib import Path
    return str(Path(exp_dir) / f"adapter_step{step_idx:03d}")


def _adapter_exists(exp_dir):
    from pathlib import Path
    p = Path(exp_dir)
    return any(p.glob("adapter_step*"))


def _save_adapter(model, exp_dir, step_idx, *, directory_name=None, base_policy=False):
    """
    Save the current LoRA adapter to disk so generation workers can load it.
    The caller writes a new step directory before advancing the checkpoint,
    then prunes superseded snapshots. This preserves crash-safe resume and a
    fresh path for LoRA caches without retaining the full adapter history.
    """
    out_dir = (str(Path(exp_dir) / str(directory_name))
               if directory_name is not None
               else _adapter_dir(exp_dir, step_idx))
    if getattr(model, "_ttt_gpt_oss_shared_lora", False):
        from models.gpt_oss_lora import save_shared_adapter
        save_shared_adapter(model, out_dir, base_policy=base_policy)
    else:
        model.save_pretrained(out_dir)
    return out_dir


def _ensure_initial_adapter_for_recovery(model, exp_dir):
    """Save the fresh zero-output LoRA when a copied step 0 lacks its snapshot."""
    path = Path(exp_dir) / "adapter_initial"
    if path.is_dir():
        return path, False
    return Path(_save_adapter(
        model, exp_dir, -1,
        directory_name="adapter_initial", base_policy=True)), True


def _prune_adapter_snapshots(exp_dir, keep_paths):
    """Delete completed adapter snapshots not required by the live state.

    A normal run retains only the checkpoint's current adapter. SPO-RS passes
    its preceding sampling adapter as a second keep path because the exact
    consecutive-policy KL at the next step requires both policies.
    """
    root = Path(exp_dir)
    keep_names = {Path(path).name for path in keep_paths if path is not None}
    removed = []
    for path in root.iterdir():
        is_step_adapter = bool(re.fullmatch(r"adapter_step\d+", path.name))
        if (not is_step_adapter and path.name != "adapter_initial"):
            continue
        if path.name in keep_names:
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
        except OSError as error:
            print(f'[warn] could not remove superseded adapter {path}: {error}', flush=True)
            continue
        removed.append(path.name)
    return removed


def _live_adapter_snapshots(exp_dir, current_adapter, spo_rs_tracker=None):
    """Return the minimal adapter set required to continue this run."""
    candidates = []
    if current_adapter is not None:
        candidates.append(Path(current_adapter))
    if spo_rs_tracker is not None:
        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is not None:
            candidates.append(Path(exp_dir) / previous_name)
    keep = []
    seen = set()
    for path in candidates:
        if path.name not in seen:
            keep.append(path)
            seen.add(path.name)
    return keep


def _load_adapter(model, adapter_dir, *, announce=True):
    """Load saved LoRA weights into the already-created trainable adapter."""
    if getattr(model, "_ttt_gpt_oss_shared_lora", False):
        from models.gpt_oss_lora import load_shared_adapter
        load_shared_adapter(model, adapter_dir)
        if announce:
            setting_log_only(f"[resume] loaded shared GPT-OSS LoRA adapter from {adapter_dir}")
        return

    import torch
    from peft import set_peft_model_state_dict

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"checkpoint adapter not found: {adapter_dir}")

    try:
        from peft.utils.save_and_load import load_peft_weights
        weights = load_peft_weights(str(adapter_dir), device=str(next(model.parameters()).device))
    except (ImportError, TypeError):
        safe_path = adapter_dir / "adapter_model.safetensors"
        bin_path = adapter_dir / "adapter_model.bin"
        if safe_path.is_file():
            from safetensors.torch import load_file
            weights = load_file(str(safe_path), device=str(next(model.parameters()).device))
        elif bin_path.is_file():
            try:
                weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)
            except TypeError:
                weights = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"no adapter weights found under {adapter_dir}")

    set_peft_model_state_dict(model, weights)
    if announce:
        setting_log_only(f"[resume] loaded LoRA adapter from {adapter_dir}")


def _save_training_checkpoint(exp_dir, next_step, adapter_path, sampler, optimizer, spo_rs_tracker=None):
    """Atomically save the state required for an exact next-step resume."""
    import torch

    target = Path(exp_dir) / "training_state.pt"
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "version": 2,
        "next_step": int(next_step),
        "adapter_dir": Path(adapter_path).name,
        "sampler": sampler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "spo_rs_tracker": (spo_rs_tracker.state_dict() if spo_rs_tracker is not None else None),
    }
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def _load_training_checkpoint(exp_dir, *, mmap=False):
    import torch

    path = Path(exp_dir) / "training_state.pt"
    if not path.is_file():
        return None
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False, **({"mmap": True} if mmap else {}))
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise ValueError(f"unsupported training checkpoint: {path}")
    # Version 1's next-batch fields are no longer required or consumed.
    # Fixed batch sizes come from the resolved saved config / CLI overrides.
    required = ("next_step", "adapter_dir", "sampler", "optimizer")
    for key in required:
        if key not in payload:
            raise ValueError(f"training checkpoint is missing {key!r}: {path}")
    saved_sampler = payload["sampler"]
    if not isinstance(saved_sampler, dict):
        raise ValueError(f"invalid sampler checkpoint: {path}")
    saved_mode = saved_sampler.get("selection_mode")
    if saved_mode is not None and saved_mode not in PARENT_SELECTION_MODES:
        raise ValueError(f"unsupported parent_selection {saved_mode!r} in checkpoint {path}")
    return payload


def _legacy_resume_info(exp_dir):
    """Locate the safe restart point for a pre-checkpoint run directory."""
    matches = []
    for path in Path(exp_dir).glob("adapter_step*"):
        try:
            matches.append((int(path.name.removeprefix("adapter_step")), path))
        except ValueError:
            continue
    if not matches:
        initial_adapter = Path(exp_dir) / "adapter_initial"
        if initial_adapter.is_dir():
            return 0, initial_adapter
        raise FileNotFoundError(
            "this older run has no training_state.pt and no adapter_step* "
            "directory; the trained policy cannot be resumed"
        )
    # Legacy adapters were written immediately before their numbered step.
    return max(matches, key=lambda item: item[0])


def _restore_legacy_archive(sampler, exp_dir, before_step, *, tokenizer=None):
    """Recover valid candidates from rollout files that predate checkpoints."""
    from evaluation.reward import extract_python_code
    from experiment_io import load_rollout_token_ids
    from search.sampler import State

    states = []
    total_rollouts = 0
    pattern = "step*/step*_group*_rollout*.meta.json"
    for meta_path in sorted(Path(exp_dir).glob(pattern)):
        try:
            meta = json.loads(meta_path.read_text())
            step = int(meta.get("step", -1))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if step < 0 or step >= before_step:
            continue
        total_rollouts += 1
        if not meta.get("valid"):
            continue
        text_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".txt")
        try:
            if sampler.ac1_ttt_rollouts:
                token_path = text_path.with_suffix(".tokens.zlib")
                if tokenizer is not None and token_path.is_file():
                    code = ttt_ac1_response_code(tokenizer, load_rollout_token_ids(token_path))
                else:
                    code = ttt_ac1_python_block(text_path.read_text(errors="replace"))
            else:
                code = extract_python_code(text_path.read_text(errors="replace"))
        except OSError:
            code = None
        if not code:
            continue
        try:
            reward = float(meta.get("reward", 0.0))
            raw = meta.get("raw_score")
            raw = float(raw) if raw is not None else None
            construction = (_ac1_saved_construction(meta) if sampler.ac1_archive
                            else meta.get("construction"))
        except (TypeError, ValueError):
            continue
        states.append(State.make(
            timestep=step, value=reward, code=code, raw_score=raw,
            construction=construction,
            observation=str(meta.get("sandbox_stdout") or ""),
            previous_raw_score=meta.get("parent_raw_score"),
            poisson=meta.get("poisson"),
            construction_length=(meta.get("construction_length")
                                 if sampler.ac1_archive else None),
        ))
    sampler.import_legacy_states(states, total_expansions=total_rollouts)
    return len(states), total_rollouts


def _restore_archive_observations(sampler, exp_dir):
    """Backfill evaluator output for checkpoints created before State carried it."""
    targets = {}
    for state in sampler._states:
        if not getattr(state, "observation", ""):
            targets.setdefault(str(state.id), []).append(state)
    for state in (sampler._best_raw_min, sampler._best_raw_max):
        if state is not None and not getattr(state, "observation", ""):
            targets.setdefault(str(state.id), []).append(state)
    if not targets:
        return 0

    restored = set()
    for meta_path in sorted(Path(exp_dir).glob(
            "step*/step*_group*_rollout*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        node_id = str(meta.get("node_id") or "")
        observation = meta.get("sandbox_stdout")
        if node_id not in targets or not isinstance(observation, str) or not observation:
            continue
        for state in targets.pop(node_id):
            state.observation = observation
        restored.add(node_id)
        if not targets:
            break
    return len(restored)


def _logged_step_rewards(exp_dir, step_idx):
    """Read one completed step's finite rewards for tracker migration."""
    step_dir = Path(exp_dir) / f"step{int(step_idx):02d}"
    rewards = []
    for meta_path in sorted(step_dir.glob(f"step{int(step_idx):02d}_group*_rollout*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
            reward = float(meta["reward"])
        except (OSError, KeyError, TypeError, ValueError,
                json.JSONDecodeError):
            continue
        if math.isfinite(reward):
            rewards.append(reward)
    return rewards


def _complete_unfinished_rollout_paths(exp_dir, step_idx, cfg):
    step_dir = Path(exp_dir) / f"step{step_idx:02d}"
    if ((step_dir / f"step{step_idx:02d}.summary.json").is_file()
            or not step_dir.is_dir()):
        return None
    paths = sorted(step_dir.glob(f"step{step_idx:02d}_group*_rollout*.meta.json"))
    parent_count = int(cfg.groups_per_step)
    if getattr(cfg, "ac1_ttt_rollouts", False) or getattr(cfg, "ac1_archive_seed", None) is not None:
        # Lineage blocking can yield fewer parents than the configured target.
        try:
            saved_parents = json.loads((step_dir / f"step{step_idx:02d}.parents.json").read_text())
            parent_count = len(saved_parents["parents"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
    expected = parent_count * int(cfg.group_size)
    if len(paths) > expected:
        raise RuntimeError(
            f"Unfinished step {step_idx} has {len(paths)} saved rollouts "
            f"for a batch of {expected}. This step cannot be replayed.")
    return paths if len(paths) == expected else None


def _ordered_rollout_artifacts(artifacts, group_size):
    def key(item):
        meta = item["meta"]
        return int(meta["group"]), int(meta["rollout"])

    artifacts.sort(key=key)
    if any(key(item) != divmod(index, group_size)
           for index, item in enumerate(artifacts)):
        return None
    return artifacts


def _recoverable_spo_rs_artifacts(exp_dir, step_idx, cfg):
    """Return a complete evaluated-but-uncheckpointed policy update.

    Rollout metadata is written only after reward evaluation, while the step
    summary is written only after the adapter and training checkpoint.  Their
    combination therefore forms a crash-safe boundary for replaying only the
    interrupted optimizer update without regenerating or reevaluating code.
    """
    step_idx = int(step_idx)
    critic = None
    if cfg.advantage_mode == LLM_CRITIC_MODE:
        critic = critic_step.load_batch(exp_dir, step_idx)
        if critic is None or "advantages" not in critic:
            return None
    paths = _complete_unfinished_rollout_paths(exp_dir, step_idx, cfg)
    if not paths:
        return None

    artifacts = []
    for meta_path in paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        reject_retired_coder_cap(meta, source=meta_path)
        critic_label_pending = critic is not None and "advantage" not in meta
        if critic is not None:
            advantage = critic["advantages"].get(meta.get("node_id"))
            if type(advantage) not in (int, float) or not -10 <= advantage <= 10:
                raise ValueError(f"missing or invalid accepted critic advantage for {meta_path}")
            if not critic_label_pending and meta["advantage"] != advantage:
                raise ValueError(f"rollout metadata disagrees with the accepted critic label in {meta_path}")
            meta["advantage"] = advantage
        required = (
            "step", "group", "rollout", "reward", "valid", "advantage",
            "advantage_mode", "n_response_tokens",
        )
        if any(name not in meta for name in required):
            return None
        if (int(meta["step"]) != step_idx
                or str(meta["advantage_mode"]).lower()
                != str(cfg.advantage_mode).lower()):
            return None
        try:
            reward = float(meta["reward"])
            advantage = float(meta["advantage"])
            response_tokens = int(meta["n_response_tokens"])
        except (TypeError, ValueError):
            return None
        if (not math.isfinite(reward) or not math.isfinite(advantage) or response_tokens < 0):
            return None
        base = meta_path.name.removesuffix(".meta.json")
        response_path = meta_path.with_name(base + ".txt")
        prompt_path = meta_path.with_name(base + ".prompt.txt")
        token_path = meta_path.with_name(base + ".tokens.zlib")
        behavior_path = meta_path.with_name(base + ".behavior_logprobs.zlib")
        if not response_path.is_file() or not prompt_path.is_file():
            return None
        if critic is not None and not token_path.is_file():
            raise FileNotFoundError(f"critic resume requires exact rollout tokens at {token_path}")
        artifacts.append({
            "meta_path": meta_path,
            "response_path": response_path,
            "prompt_path": prompt_path,
            "token_path": token_path if token_path.is_file() else None,
            "behavior_logprobs_path": (behavior_path if behavior_path.is_file() else None),
            "meta": meta,
        })
        if critic_label_pending:
            artifacts[-1]["critic_label_pending"] = True

    artifacts = _ordered_rollout_artifacts(artifacts, int(cfg.group_size))
    if critic is not None and (artifacts is None
            or [critic_step.rollout_row(item["meta"]) for item in artifacts] != critic["rollouts"]
            or set(critic["advantages"]) != {row["id"] for row in critic["rollouts"]}):
        raise ValueError(f"saved critic batch does not match the step {step_idx} rollout artifacts")
    return artifacts


def _recoverable_pending_spo_rs_artifacts(exp_dir, step_idx, cfg):
    """Return a completely generated SPO-RS step awaiting final evaluation.

    Generation writes response text, exact token IDs, prompts, and provisional
    metadata atomically as each rollout arrives. Rollout-policy logprobs may be
    added by the batched scorer after generation; exact token IDs let recovery
    reproduce them without sampling the policy again.
    Individually completed rewards may already be present for observability,
    but recovery deliberately recomputes them because advantages and the
    step-wide training reduction were not yet sealed.
    """
    step_idx = int(step_idx)
    paths = _complete_unfinished_rollout_paths(exp_dir, step_idx, cfg)
    if not paths:
        return None

    artifacts = []
    for meta_path in paths:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(meta, dict):
            return None
        reject_retired_coder_cap(meta, source=meta_path)
        if (int(meta.get("step", -1)) != step_idx
                or meta.get("status") not in {"evaluation_pending", "evaluated"}):
            return None
        if ((uses_hybrid_policy_loss(cfg.advantage_mode) or cfg.advantage_mode in INVARIANT_ADVANTAGE_MODES)
                and meta.get("advantage_mode") != cfg.advantage_mode):
            return None
        required = (
            "group", "parent_group", "rollout", "n_response_tokens",
        )
        if any(name not in meta for name in required):
            return None
        base = meta_path.name.removesuffix(".meta.json")
        response_path = meta_path.with_name(base + ".txt")
        prompt_path = meta_path.with_name(base + ".prompt.txt")
        token_path = meta_path.with_name(base + ".tokens.zlib")
        behavior_path = meta_path.with_name(base + ".behavior_logprobs.zlib")
        if not all(path.is_file() for path in (
                response_path, prompt_path, token_path)):
            return None
        artifacts.append({
            "meta_path": meta_path,
            "response_path": response_path,
            "prompt_path": prompt_path,
            "token_path": token_path,
            "behavior_logprobs_path": (
                behavior_path if behavior_path.is_file() else None),
            "meta": meta,
        })

    return _ordered_rollout_artifacts(artifacts, int(cfg.group_size))


def _literal_metadata_mapping(value):
    """Recover mappings stringified by the rollout JSON compatibility layer."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    import ast
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _literal_float_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        import ast
        try:
            values = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    else:
        return None
    try:
        result = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _ac1_saved_construction(meta):
    """Decode the saved coefficients without rounding integer values."""
    from search.sampler import AC1_CONSTRUCTION_LIMITS

    construction = meta.get("construction")
    if isinstance(construction, str):
        construction = json.loads(construction)
    length = meta.get("construction_length", 0)
    low, high = AC1_CONSTRUCTION_LIMITS
    if low <= length <= high and (construction is None or len(construction) != length):
        raise ValueError("saved AC1 rollout is missing its archive construction")
    return construction


def _metadata_is_code_failure(meta):
    """Mirror problems.base.is_code_failure from persisted reward fields."""
    kind = str(meta.get("failure_kind", "") or "").strip().lower()
    if kind:
        return kind == "code"
    if not bool(meta.get("parsed", False)):
        return True
    if bool(meta.get("ran", False)):
        return False
    return "timeout" not in str(meta.get("msg", "") or "").lower()


def _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_count):
    """Build a safe training sequence from a legacy text-only artifact.

    Old runs saved vLLM's decoded text but not ``candidate.token_ids``.  Such
    text is deliberately decoded with ``skip_special_tokens=True`` and BPE
    decode/encode is not generally invertible, so matching the recorded token
    count is neither necessary nor proof that the original IDs were recovered.
    Use the tokenizer's canonical representation of the *exact saved text* and
    restore one hidden terminal EOS when the recorded sequence was longer and
    an unambiguous, invisibly-decoded EOS is available.  Any remaining count
    difference is diagnostic only; padding it with guessed special tokens
    would fabricate a trajectory.

    New runs carry an exact ``.tokens.zlib`` sidecar and never use this path.
    """
    token_ids = list(tokenizer(response_text, add_special_tokens=False).input_ids)
    expected_count = int(expected_count)
    canonical_count = len(token_ids)
    restored_terminal = False

    configured = []

    def add_ids(value):
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else (value,)
        for item in values:
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
            if item >= 0 and item not in configured:
                configured.append(item)

    generation_config = getattr(model, "generation_config", None)
    add_ids(getattr(generation_config, "eos_token_id", None))
    add_ids(getattr(tokenizer, "eos_token_id", None))
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    try:
        tokenizer_eos = int(tokenizer_eos)
    except (TypeError, ValueError):
        tokenizer_eos = None
    chosen = (tokenizer_eos if tokenizer_eos in configured else configured[0] if len(configured) == 1 else None)
    if canonical_count < expected_count and chosen is not None:
        try:
            visible = tokenizer.decode([chosen], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except TypeError:
            visible = tokenizer.decode([chosen], skip_special_tokens=True)
        if not str(visible or ""):
            token_ids.append(chosen)
            restored_terminal = True

    recovered_count = len(token_ids)
    recovery = {
        "source": "canonical-text",
        "exact": False,
        "recorded_count": expected_count,
        "canonical_count": canonical_count,
        "recovered_count": recovered_count,
        "restored_terminal": restored_terminal,
        "unavailable_original_tokens": max(0, expected_count - recovered_count),
        "canonical_extra_tokens": max(0, recovered_count - expected_count),
    }
    return token_ids, recovery


def _prepare_recoverable_response_tokens(model, tokenizer, artifacts):
    """Validate/cache artifact token IDs before distributed replicas load."""
    from experiment_io import (load_rollout_logprobs, load_rollout_token_ids)
    from zlib import error as ZlibError

    summary = {
        "exact_sidecars": 0,
        "behavior_logprob_sidecars": 0,
        "canonical_text": 0,
        "restored_terminal_tokens": 0,
        "unavailable_original_tokens": 0,
        "canonical_extra_tokens": 0,
        "count_changed_responses": 0,
    }
    for item in artifacts:
        expected_count = int(item["meta"]["n_response_tokens"])
        token_path = item.get("token_path")
        if token_path is not None:
            try:
                token_ids = load_rollout_token_ids(token_path)
            except (OSError, ValueError, ZlibError) as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot read exact token sidecar "
                    f"{Path(token_path).name}: {error}") from error
            if len(token_ids) != expected_count:
                raise RuntimeError(
                    "evaluated-step recovery token sidecar length differs "
                    f"for {Path(token_path).name}: saved metadata "
                    f"{expected_count}, sidecar {len(token_ids)}")
            recovery = {
                "source": "exact-sidecar",
                "exact": True,
                "recorded_count": expected_count,
                "canonical_count": None,
                "recovered_count": len(token_ids),
                "restored_terminal": False,
                "unavailable_original_tokens": 0,
                "canonical_extra_tokens": 0,
            }
            summary["exact_sidecars"] += 1
        else:
            response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
            try:
                token_ids, recovery = (
                    _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_count))
            except ValueError as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot tokenize the saved "
                    "response text for "
                    f"{item['response_path'].name}: {error}") from error
            summary["canonical_text"] += 1
        item["response_token_ids"] = token_ids
        item["response_token_recovery"] = recovery
        behavior_path = item.get("behavior_logprobs_path")
        if behavior_path is not None:
            if not bool(recovery.get("exact", False)):
                raise RuntimeError(
                    "evaluated-step recovery found rollout-policy logprobs "
                    "without their exact token-ID sidecar for "
                    f"{Path(behavior_path).name}")
            try:
                behavior_logprobs = load_rollout_logprobs(behavior_path)
            except (OSError, ValueError, ZlibError) as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot read rollout-policy "
                    f"logprobs from {Path(behavior_path).name}: {error}") from error
            if len(behavior_logprobs) != len(token_ids):
                raise RuntimeError(
                    "evaluated-step recovery rollout-policy logprob length "
                    f"differs for {Path(behavior_path).name}: tokens "
                    f"{len(token_ids)}, logprobs {len(behavior_logprobs)}")
            item["behavior_logprobs"] = behavior_logprobs
            summary["behavior_logprob_sidecars"] += 1
        else:
            item["behavior_logprobs"] = None
        restored_terminal = bool(recovery["restored_terminal"])
        item["restored_terminal_token"] = restored_terminal
        summary["restored_terminal_tokens"] += int(restored_terminal)
        summary["unavailable_original_tokens"] += int(recovery["unavailable_original_tokens"])
        summary["canonical_extra_tokens"] += int(recovery["canonical_extra_tokens"])
        summary["count_changed_responses"] += int(len(token_ids) != expected_count)
    return summary


def _finalize_pending_spo_rs_artifacts(
        tokenizer, sampler, step_idx, cfg, exp_dir, problem, gen_pool,
        spo_rs_tracker, sampling_adapter_path, artifacts):
    """Evaluate and score a generated interrupted step, then seal its metadata.

    No policy sampling occurs here. Exact generated token IDs come from the
    durable sidecars; missing rollout-policy logprobs are reconstructed by
    exact teacher-forced scoring. CPU rewards are recomputed while vLLM scores
    those trajectories, after which the ordinary evaluated-step recovery
    performs the unchanged fast distributed adapter update.
    """
    from concurrent.futures import as_completed
    from copy import deepcopy

    from experiment_io import save_rollout, save_rollout_logprobs
    from generation.gen_workers import make_progress_bar
    from problems.base import ParentContext, RewardResult
    from search.sampler import State

    if getattr(problem, "problem_type", "") == "trimul":
        raise RuntimeError("TriMul recovery must use isolated evaluation")
    if cfg.generation_backend != "vllm":
        raise RuntimeError('generated-step SPO-RS recovery requires exact vLLM rollout artifacts')
    if gen_pool is None:
        raise RuntimeError(
            "generated-step SPO-RS recovery requires the "
            "configured vLLM pool for exact consecutive-policy scoring")
    if (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError('generated-step SPO-RS recovery requires the restored pre-step tracker')
    if sampling_adapter_path is None or not Path(sampling_adapter_path).is_dir():
        raise FileNotFoundError(
            "generated-step SPO-RS recovery cannot find its sampling adapter: "
            f"{sampling_adapter_path}")

    token_summary = _prepare_recoverable_response_tokens(None, tokenizer, artifacts)
    if token_summary["exact_sidecars"] != len(artifacts):
        raise RuntimeError("generated-step recovery refuses non-exact response token IDs")

    ac1_archive = sampler.ac1_archive
    parents = _sample_step_parents(sampler, cfg, exp_dir, step_idx, reuse_saved=True)
    if ac1_archive and step_idx == 0:
        _rebind_interrupted_step_zero_selection(sampler, exp_dir, parents)
    parents_path = (Path(exp_dir) / f"step{int(step_idx):02d}" / f"step{int(step_idx):02d}.parents.json")
    try:
        saved_parent_rows = json.loads(parents_path.read_text(encoding="utf-8"))["parents"]
    except (OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as error:
        raise RuntimeError("generated-step recovery cannot read its saved parent selection") \
            from error
    saved_parents = {int(row["group"]): row for row in saved_parent_rows}
    if set(saved_parents) != set(range(len(parents))):
        raise RuntimeError("generated-step recovery could not reconstruct every parent")
    for group, parent in enumerate(parents):
        row = saved_parents.get(group)
        if row is None:
            raise RuntimeError(f"generated-step recovery is missing parent group {group}")
        if (str(row.get("parent_id")) != str(parent.id)
                or int(row.get("parent_timestep", -1))
                != int(parent.timestep)
                or not math.isclose(float(row.get("parent_reward")), float(parent.value), rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(
                f"generated-step recovery parent {group} does not match the "
                "restored sampler checkpoint")

    parent_contexts = [
        ParentContext(
            code=parent.code,
            value=(parent.value if parent.value is not None else 0.0),
            raw_score=parent.raw_score,
            construction=parent.construction,
            observation=parent.observation,
            previous_raw_score=parent.previous_raw_score,
            poisson=parent.poisson,
        )
        for parent in parents
    ]

    cpu_eval_progress = None
    if cfg.evaluation_gpu_id is None:
        interval = getattr(cfg, "eval_progress_interval_s", DEFAULT_EVAL_PROGRESS_INTERVAL_S)
        cpu_eval_progress = CpuEvaluationProgress(step_idx, len(artifacts), interval=interval)
    evaluation_pool, evaluate_response, _ = _evaluation_pool(
        cfg, problem, len(artifacts), step_idx, gen_pool, recovering=True)

    def evaluate(item, *, on_sandbox_deadline=None):
        meta = item["meta"]
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        final_text, issue = _rollout_code_input(
            cfg, tokenizer, item["response_token_ids"], response_text,
            item["prompt_path"].read_text(encoding="utf-8", errors="replace"))
        meta["output_format_issue"] = issue
        if issue:
            return RewardResult(reward=float(problem.fail_score), msg=str(issue), failure_kind="code")
        answer_options = {} if final_text is None else {"final_answer_text": final_text}
        if on_sandbox_deadline is not None:
            answer_options["on_sandbox_deadline"] = on_sandbox_deadline
        parent_context = parent_contexts[int(meta["parent_group"])]
        return evaluate_response(response_text, parent_context, **answer_options)

    eval_bar = None
    eval_progress_lock = threading.Lock()
    live_best_state = sampler.best_raw_state(maximize=bool(problem.maximize))
    live_best_raw = (
        float(live_best_state.raw_score) if live_best_state is not None else None)
    evaluation_order = {}

    def evaluation_completed(future, index):
        nonlocal eval_bar, live_best_raw
        with eval_progress_lock:
            if ac1_archive:
                evaluation_order[index] = len(evaluation_order)
            if eval_bar is not None:
                eval_bar.update(1)
            elif cpu_eval_progress is None:
                eval_bar = make_progress_bar(len(artifacts), desc="recovering evaluations", initial=1)
            if not future.cancelled() and future.exception() is None:
                live_best_raw, best_line = _advance_live_best_raw(future.result(), live_best_raw, problem)
                if best_line:
                    write_progress_line(f"[step {step_idx}] {best_line}")

    results = [None] * len(artifacts)
    try:
        evaluation_futures = {}
        for index, item in enumerate(artifacts):
            if cpu_eval_progress is None:
                future = evaluation_pool.submit(evaluate, item)
            else:
                future = cpu_eval_progress.submit(evaluation_pool, evaluate, item)
            evaluation_futures[future] = index
            future.add_done_callback(lambda done, index=index: evaluation_completed(done, index))

        prompt_cache = {}
        score_pairs = []
        current_scores = []
        context_ids = []
        for item in artifacts:
            prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
            if prompt_text not in prompt_cache:
                prompt_cache[prompt_text] = tokenizer(prompt_text).input_ids
            score_pairs.append((
                prompt_cache[prompt_text], item["response_token_ids"]))
            current_scores.append(item["behavior_logprobs"])
            context_ids.append(int(item["meta"]["parent_group"]))

        if any(values is None for values in current_scores):
            rescored = gen_pool.score_token_logprobs(
                score_pairs, show_progress=True,
                adapter_path=str(sampling_adapter_path),
                require_complete=True)
            for item, existing, values in zip(
                    artifacts, current_scores, rescored):
                if existing is not None:
                    continue
                if values is None or len(values) != len(item["response_token_ids"]):
                    raise RuntimeError(
                        "generated-step recovery could not reconstruct every "
                        "rollout-policy logprob")
                item["behavior_logprobs"] = values
                behavior_path = item["meta_path"].with_name(
                    item["meta_path"].name.removesuffix(".meta.json")
                    + ".behavior_logprobs.zlib")
                save_rollout_logprobs(behavior_path, values)
                item["behavior_logprobs_path"] = behavior_path
            current_scores = [item["behavior_logprobs"] for item in artifacts]

        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is None:
            raise RuntimeError("generated-step recovery has no preceding SPO-RS adapter")
        previous_adapter_path = Path(exp_dir) / previous_name
        if not previous_adapter_path.is_dir():
            raise FileNotFoundError(
                "generated-step recovery cannot find the preceding policy "
                f"adapter: {previous_adapter_path}")
        if (previous_adapter_path.resolve() == Path(sampling_adapter_path).resolve()):
            previous_scores = current_scores
        else:
            try:
                previous_scores = gen_pool.score_token_logprobs(
                    score_pairs, show_progress=True,
                    adapter_path=str(previous_adapter_path),
                    require_complete=True)
            finally:
                if getattr(gen_pool, "sequential", False):
                    gen_pool.release()

        context_count = len(parents) if ac1_archive else int(cfg.groups_per_step)
        divergence, missing_scores, context_divergences = (
            spo_rs_tracker.consecutive_policy_divergence(
                current_scores, previous_scores, context_ids,
                required_context_ids=range(context_count)))
        if missing_scores or not math.isfinite(divergence):
            raise RuntimeError('generated-step recovery did not obtain every exact consecutive-policy score')

        for future in as_completed(evaluation_futures):
            index = evaluation_futures[future]
            result = future.result()
            results[index] = result
            item = artifacts[index]
            evaluated_meta = _evaluated_rollout_meta(item["meta"], result)
            save_rollout(
                exp_dir, step_idx, int(item["meta"]["group"]),
                int(item["meta"]["rollout"]), "", evaluated_meta,
                artifacts_already_saved=True)
            item["meta"] = evaluated_meta
    finally:
        evaluation_pool.shutdown(wait=True, cancel_futures=False)
        if cpu_eval_progress is not None:
            cpu_eval_progress.close()
        if eval_bar is not None:
            eval_bar.close()

    rewards = [float(result.reward) for result in results]
    prompt_indices = {}
    for item in artifacts:
        prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
        if prompt_text not in prompt_indices:
            prompt_indices[prompt_text] = len(prompt_indices)
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    advantages = spo_rs_tracker.normalized_advantages(transformed)
    if advantages is None:
        raise RuntimeError("generated-step recovery lost the pre-step SPO-RS baseline")
    preview_tracker = deepcopy(spo_rs_tracker)
    update = preview_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx)
    update.update({
        "groups": sorted({int(item["meta"]["group"]) for item in artifacts}),
        "prompt_jobs": list(range(len(prompt_indices))),
        "contexts": sorted(set(context_ids)),
        "context_divergences": context_divergences,
        "missing_policy_scores": int(missing_scores),
    })
    rollout_infos = [{
        **update,
        "transformed_reward": float(transformed[index]),
        "used_for_policy_update": True,
    } for index in range(len(artifacts))]
    group_size = int(cfg.group_size)
    ttt_groups = []
    for start in range(0, len(rewards), group_size):
        group_rewards = rewards[start:start + group_size]
        ttt_advantages, ttt_adaptive_beta = compute_group_advantages(group_rewards)
        ttt_groups.append((ttt_advantages, ttt_adaptive_beta, max(group_rewards)))

    save_ctor = bool(getattr(problem, "saves_construction", False))
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False) else type(sampler).__name__)
    step_dir = Path(exp_dir) / f"step{int(step_idx):02d}"
    for index, (item, result) in enumerate(zip(artifacts, results)):
        pending = item["meta"]
        group = int(pending["group"])
        parent_group = int(pending["parent_group"])
        rollout = int(pending["rollout"])
        parent = parents[parent_group]
        parent_row = saved_parents[parent_group]
        child = State.make(
            timestep=step_idx, value=float(result.reward),
            code=result.code or "", raw_score=result.raw_score,
            construction=result.construction, observation=result.stdout,
            poisson=result.poisson)
        archive_eligible = bool(result.valid and result.code)
        info = rollout_infos[index]
        ttt_advantages, ttt_adaptive_beta, ttt_sibling_max = ttt_groups[group]
        meta = {key: value for key, value in pending.items() if key != "status"}
        meta.update({
            "node_id": child.id,
            "parent_id": parent.id,
            "parent_timestep": int(parent.timestep),
            "sampler_type": sampler_type,
            "reward": float(result.reward),
            "raw_score": (float(result.raw_score) if result.raw_score is not None else None),
            "poisson": result.poisson,
            "valid": bool(result.valid),
            "parsed": bool(result.parsed),
            "ran": bool(result.ran),
            "msg": result.msg,
            "failure_kind": result.failure_kind,
            **_rollout_advantage_fields(
                advantages[index], ttt_advantage=ttt_advantages[rollout],
                spo_rs_advantage=advantages[index], ttt_sibling_max=ttt_sibling_max,
                ttt_adaptive_beta=ttt_adaptive_beta, parent_raw_score=parent.raw_score,
                spo_rs_v_t=info["value_before"]),
            "beta": float(cfg.spo_rs_beta),
            "advantage_mode": str(cfg.advantage_mode).lower(),
            "advantage_scale": float(cfg.spo_rs_beta),
            "sandbox_stdout": _saved_evaluator_output(result.stdout),
            "parent_value": (float(parent.value) if parent.value is not None else None),
            "parent_is_seed": parent.id in sampler._seed_ids,
            "parent_visit_count": int(parent_row.get("visit_count", 0)),
            "parent_q_value": parent_row.get("q_value"),
            "parent_prior": parent_row.get("prior"),
            "parent_exploration_bonus": parent_row.get("exploration_bonus"),
            "parent_selection_score": parent_row.get("selection_score"),
            "archive_eligible": archive_eligible,
            "construction": (_as_float_list(result.construction, cfg.max_saved_construction) if save_ctor else None),
            "parent_construction": (
                _as_float_list(parent.construction, cfg.max_saved_construction)
                if save_ctor else None),
            "seed": int(cfg.seed),
            "spo_rs": info,
        })
        if ac1_archive:
            meta.update(_ac1_archive_metadata(result, evaluation_order[index]))
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        save_rollout(exp_dir, step_idx, group, rollout, response_text, meta, artifacts_already_saved=True)

    print(
        f"[resume] step {step_idx}: reused {len(artifacts)} exact generated "
        "rollouts, recomputed their CPU rewards, and completed exact "
        "consecutive-policy scoring; no policy sampling was repeated",
        flush=True,
    )
    completed = _recoverable_spo_rs_artifacts(exp_dir, step_idx, cfg)
    if completed is None:
        raise RuntimeError("generated-step recovery failed to seal complete rollout metadata")
    cached = {item["meta_path"]: item for item in artifacts}
    return [{**cached[item["meta_path"]], **item} for item in completed]


def _rebind_saved_step_zero_parents(sampler, by_group, count):
    """Bind fresh seed UUIDs to the evaluated step-zero groups when possible."""
    seeds = [state for state in sampler._states
             if state.id in sampler._seed_ids]
    remaining = list(seeds)
    matches = []
    for group in range(int(count)):
        records = by_group.get(group, ())
        if not records:
            return None
        meta = records[0]["meta"]
        construction = meta.get("parent_construction")
        if isinstance(construction, str):
            try:
                construction = json.loads(construction)
            except json.JSONDecodeError:
                return None
        if construction is None:
            return None
        candidates = [
            state for state in remaining
            if int(state.timestep) == int(meta.get("parent_timestep", -1))
            and state.construction is not None
            and np.array_equal(
                np.asarray(state.construction), np.asarray(construction))
        ]
        if len(candidates) != 1:
            return None
        state = candidates[0]
        matches.append((state, str(meta["parent_id"])))
        remaining = [candidate for candidate in remaining if candidate is not state]
    saved_ids = [saved_id for _state, saved_id in matches]
    if len(set(saved_ids)) != len(saved_ids):
        return None
    for state, saved_id in matches:
        if state.id != saved_id:
            sampler.reassign_seed_id(state, saved_id)
    return [state for state, _saved_id in matches]


def _rebind_interrupted_step_zero_selection(sampler, exp_dir, parents):
    """Restore seed UUIDs saved after parent selection and before coder rollout."""
    path = Path(exp_dir) / "step00" / "step00.parents.json"
    if not path.exists():
        return
    rows = sorted(json.loads(path.read_text())["parents"],
                  key=lambda row: int(row["group"]))
    if len(rows) != len(parents):
        return
    if [str(row["parent_id"]) for row in rows] == [str(parent.id) for parent in parents]:
        return

    def same_number(saved, current):
        return saved is None and current is None or (
            saved is not None and current is not None
            and float(saved) == float(current))

    if (len({str(row["parent_id"]) for row in rows}) != len(rows)
            or any(int(row["group"]) != group
                   or not bool(row["parent_is_seed"]) or not parent.is_seed
                   or int(row["parent_timestep"]) != int(parent.timestep)
                   or not same_number(row.get("parent_reward"), parent.value)
                   or not same_number(row.get("parent_raw_score"), parent.raw_score)
                   for group, (row, parent) in enumerate(zip(rows, parents)))):
        return
    for row, parent in zip(rows, parents):
        sampler.reassign_seed_id(parent, str(row["parent_id"]))


def _sample_step_parents(sampler, cfg, exp_dir, step_idx, *, reuse_saved=False):
    if sampler.ac1_ttt_rollouts and reuse_saved:
        path = Path(exp_dir) / f"step{step_idx:02d}" / f"step{step_idx:02d}.parents.json"
        rows = sorted(json.loads(path.read_text(encoding="utf-8"))["parents"],
                      key=lambda row: int(row["group"]))
        return sampler.sample_states(int(cfg.groups_per_step), step_idx=step_idx, saved_parents=rows)
    return sampler.sample_states(int(cfg.groups_per_step), step_idx=step_idx)


def _rollout_code_input(cfg, tokenizer, token_ids, response_text, prompt_text):
    if getattr(cfg, "ac1_ttt_rollouts", False):
        block = ttt_ac1_response_code(tokenizer, token_ids)
        return block, None if block else "missing Python block"
    from generation.gen_workers import native_final_answer_scope

    final_text = native_final_answer_scope(tokenizer, token_ids, prompt_text=prompt_text)
    return final_text, coder_output_issue(response_text, final_answer_text=final_text)


def _invariant_batch(cfg, tracker, rows, *, combine=True):
    hybrid_options = {}
    if cfg.advantage_mode == DECOMPOSE_MODE:
        hybrid_options["decompose"] = True
    elif combine and cfg.advantage_mode == HYBRID_INVARIANT_MODE:
        hybrid_options = {
            "ttt_weight": cfg.hybrid_ttt_weight,
            "spo_weight": cfg.hybrid_spo_weight,
        }
    return tracker.calculate(
        [row["reward"] for row in rows], [row["valid"] for row in rows],
        group_size=int(cfg.group_size), **hybrid_options)


def _restore_saved_spo_rs_update(cfg, artifacts, spo_rs_tracker, sampling_adapter_path, step_idx):
    rows = [item["meta"] for item in artifacts]
    saved_advantages = np.asarray([float(row["advantage"]) for row in rows], dtype=np.float64)
    spo_infos = [_literal_metadata_mapping(row.get("spo_rs")) for row in rows]
    if any(not info or not bool(info.get("used_for_policy_update", False)) for info in spo_infos):
        raise RuntimeError(
            "evaluated-step recovery found a rollout that was not part of "
            "the saved SPO-RS policy update")
    spo_info = spo_infos[0]
    if int(spo_info.get("group_size", -1)) != len(artifacts):
        raise RuntimeError('evaluated-step recovery rollout count differs from the saved SPO-RS update')
    if Path(str(spo_info.get("policy_adapter", ""))).name != Path(sampling_adapter_path).name:
        raise RuntimeError('evaluated-step recovery sampling adapter differs from the saved SPO-RS update')
    divergence = spo_info.get("divergence")
    if divergence is None:
        if bool(spo_info.get("divergence_missing", False)):
            divergence = math.inf
        else:
            raise RuntimeError("evaluated-step recovery has no saved SPO-RS divergence")
    divergence = float(divergence)
    if math.isnan(divergence) or divergence < 0.0:
        raise RuntimeError("evaluated-step recovery found an invalid SPO-RS divergence")

    if cfg.advantage_mode in INVARIANT_ADVANTAGE_MODES:
        legacy_rms = (cfg.advantage_mode == HYBRID_INVARIANT_MODE
                      and "hybrid_invariant_mixture_rms" in spo_info)
        batch = _invariant_batch(cfg, spo_rs_tracker, rows, combine=not legacy_rms)
        expected = batch.applied
        if legacy_rms:
            # Reproduce steps whose labels were persisted with RMS normalization.
            mix = float(spo_info["hybrid_invariant_lambda"])
            expected = mix * batch.sibling + (1.0 - mix) * batch.historical
            peak = float(np.max(np.abs(expected)))
            if peak:
                scaled = expected / peak
                expected = (scaled / math.sqrt(np.mean(scaled ** 2))) * float(
                    spo_info["hybrid_invariant_target_rms"])
        if not np.allclose(expected, saved_advantages, rtol=0.0, atol=1e-10):
            raise RuntimeError("evaluated-step recovery advantages differ from the saved invariant batch")
        effective = batch.effective_rewards
        effective_rewards = [None] * len(rows) if effective is None else effective.tolist()
        if effective_rewards != [info["effective_reward"] for info in spo_infos]:
            raise RuntimeError("evaluated-step recovery rewards differ from the saved invariant batch")
        tracker_update = spo_rs_tracker.update(
            batch, divergence=divergence, policy_adapter=sampling_adapter_path, step=step_idx)
        if legacy_rms:
            for key in ("hybrid_invariant_lambda", "hybrid_invariant_target_rms", "hybrid_invariant_mixture_rms"):
                tracker_update[key] = spo_info[key]
        return spo_info, tracker_update, divergence

    rewards = [float(row["reward"]) for row in rows]
    transformed = spo_rs_tracker.transformed_rewards(rewards)
    recovered_advantages = spo_rs_tracker.normalized_advantages(transformed)
    if cfg.advantage_mode == "hybrid" and recovered_advantages is not None:
        group_size = int(cfg.group_size)
        for start in range(0, len(rewards), group_size):
            siblings = slice(start, start + group_size)
            recovered_advantages[siblings], _ = hybrid_advantages(
                rewards[siblings], recovered_advantages[siblings],
                ttt_weight=cfg.hybrid_ttt_weight,
                spo_weight=cfg.hybrid_spo_weight)
    saved_value_before = spo_info.get("value_before")
    if (saved_value_before is None
            or not math.isclose(float(saved_value_before),
                                float(spo_rs_tracker.baseline()),
                                rel_tol=0.0, abs_tol=1e-10)):
        raise RuntimeError('evaluated-step recovery does not match the saved pre-step SPO-RS value')
    for index, (info, transformed_reward) in enumerate(zip(spo_infos, transformed)):
        saved_transformed = info.get("transformed_reward")
        if (saved_transformed is None
                or not math.isclose(float(saved_transformed), float(transformed_reward), rel_tol=0.0, abs_tol=1e-10)):
            raise RuntimeError(
                "evaluated-step recovery transformed reward differs from "
                f"the saved SPO-RS record at rollout {index}")
    tracker_update = spo_rs_tracker.update(
        transformed, divergence=divergence,
        policy_adapter=sampling_adapter_path, step=step_idx)
    saved_after = spo_info.get("value_after")
    if (saved_after is not None
            and not math.isclose(float(saved_after), float(tracker_update["value_after"]), rel_tol=0.0, abs_tol=1e-10)):
        raise RuntimeError(
            "evaluated-step recovery produced a different SPO-RS tracker "
            "value; refusing an inexact update")

    if (recovered_advantages is None
            or recovered_advantages.shape != saved_advantages.shape
            or not np.allclose(recovered_advantages, saved_advantages, rtol=0.0, atol=1e-10)):
        raise RuntimeError(
            "evaluated-step recovery advantages differ from the saved SPO "
            "objective; refusing an inexact update")
    return spo_info, tracker_update, divergence


def _resume_spo_rs_step_from_artifacts(
        model, tokenizer, sampler, step_idx, cfg, exp_dir,
        problem, parallel_trainer, spo_rs_tracker, sampling_adapter_path,
        artifacts, evaluation_complete=None):
    """Finish one interrupted policy update from its durable evaluations."""
    import torch
    from experiment_io import (advantage_plot_rollout, load_rollout_logprobs,
                               save_best_state_snapshot, save_rollout)
    from evaluation.reward import extract_python_code
    from generation.gen_workers import native_final_answer_scope
    from search.sampler import State

    critic_mode = cfg.advantage_mode == LLM_CRITIC_MODE
    invariant_mode = cfg.advantage_mode in INVARIANT_ADVANTAGE_MODES
    training_enabled = not bool(getattr(cfg, "no_train", False))
    if parallel_trainer is None and (training_enabled or not (critic_mode or invariant_mode)):
        raise RuntimeError("evaluated-step recovery requires the process-distributed trainer")
    if not critic_mode and (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise RuntimeError("evaluated-step recovery requires the pre-step SPO-RS tracker")
    if sampling_adapter_path is None or not Path(sampling_adapter_path).is_dir():
        raise FileNotFoundError(f'evaluated-step recovery cannot find the sampling adapter: {sampling_adapter_path}')
    hybrid_mode = cfg.advantage_mode in ("hybrid", HYBRID_INVARIANT_MODE)
    hybrid_loss = uses_hybrid_policy_loss(cfg.advantage_mode)
    ac1_archive = sampler.ac1_archive
    if critic_mode:
        judged = critic_step.load_batch(exp_dir, step_idx)
        if judged["adapter"] != Path(sampling_adapter_path).name:
            raise RuntimeError("evaluated-step recovery sampling adapter differs from the saved critic batch")

    by_group = {}
    for item in artifacts:
        by_group.setdefault(int(item["meta"]["group"]), []).append(item)
    parents = _rebind_saved_step_zero_parents(
        sampler, by_group, int(cfg.groups_per_step)
    ) if step_idx == 0 and not ac1_archive else None
    if parents is None:
        parents = _sample_step_parents(sampler, cfg, exp_dir, step_idx, reuse_saved=True)
        if ac1_archive and step_idx == 0:
            _rebind_interrupted_step_zero_selection(sampler, exp_dir, parents)
    else:
        print(
            f"[resume] restored {len(parents)} saved step-zero seed parents; "
            "current parent-selection settings begin at step 1",
            flush=True,
        )
    archive_order = {}
    if set(by_group) != set(range(len(parents))):
        raise RuntimeError(
            "evaluated-step recovery could not reconstruct every selected "
            f"parent ({len(parents)}/{len(by_group)})")
    for group, parent in enumerate(parents):
        records = by_group.get(group, ())
        if not records:
            raise RuntimeError(f"evaluated-step recovery has no rollout group {group}")
        first = records[0]["meta"]
        try:
            saved_value = float(first["parent_value"])
            saved_timestep = int(first["parent_timestep"])
            saved_parent_id = str(first["parent_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f'evaluated-step recovery lacks parent identity for group {group}') from error
        current_value = float(parent.value)
        if (saved_timestep != int(parent.timestep)
                or not math.isclose(saved_value, current_value, rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(f'evaluated-step recovery parent {group} does not match the restored sampler state')
        if parent.id != saved_parent_id:
            if step_idx == 0 and parent.id in sampler._seed_ids:
                # Fresh seed State UUIDs are intentionally random. Reuse the
                # durable IDs already embedded in every rollout so the healed
                # checkpoint preserves the original lineage exactly.
                sampler.reassign_seed_id(parent, saved_parent_id)
            else:
                raise RuntimeError(
                    f"evaluated-step recovery parent ID {group} differs from "
                    "the restored sampler checkpoint")
        sampler.record_expansion(parent, count=int(cfg.group_size))

    if not critic_mode:
        spo_info, tracker_update, divergence = _restore_saved_spo_rs_update(
            cfg, artifacts, spo_rs_tracker, sampling_adapter_path, step_idx)

    prompt_cache = {}
    prompt_job_ids = {}
    all_examples = []
    all_children = []
    restored_terminal_tokens = 0
    canonical_recovery_responses = 0
    unavailable_original_tokens = 0
    canonical_extra_tokens = 0
    for item in artifacts:
        meta = item["meta"]
        response_text = item["response_path"].read_text(encoding="utf-8", errors="replace")
        prompt_text = item["prompt_path"].read_text(encoding="utf-8", errors="replace")
        expected_tokens = int(meta["n_response_tokens"])
        response_tokens = item.get("response_token_ids")
        if response_tokens is None:
            try:
                response_tokens, recovery = (
                    _reconstruct_saved_response_tokens(model, tokenizer, response_text, expected_tokens))
            except ValueError as error:
                raise RuntimeError(
                    "evaluated-step recovery cannot tokenize the saved "
                    "response text for "
                    f"{item['response_path'].name}: {error}") from error
        else:
            response_tokens = list(response_tokens)
            recovery = item.get("response_token_recovery") or {
                "source": "exact-sidecar",
                "exact": True,
                "recorded_count": expected_tokens,
                "canonical_count": None,
                "recovered_count": len(response_tokens),
                "restored_terminal": bool(item.get("restored_terminal_token")),
                "unavailable_original_tokens": 0,
                "canonical_extra_tokens": 0,
            }
            if (bool(recovery.get("exact", False)) and len(response_tokens) != expected_tokens):
                raise RuntimeError(
                    "evaluated-step recovery exact token cache changed for "
                    f"{item['response_path'].name}")
        restored_terminal = bool(recovery.get("restored_terminal", False))
        restored_terminal_tokens += int(restored_terminal)
        canonical_recovery_responses += int(recovery.get("source") == "canonical-text")
        unavailable_original_tokens += int(recovery.get('unavailable_original_tokens', 0) or 0)
        canonical_extra_tokens += int(recovery.get('canonical_extra_tokens', 0) or 0)
        if prompt_text not in prompt_cache:
            prompt_cache[prompt_text] = tokenizer(prompt_text, return_tensors="pt").input_ids
            prompt_job_ids[prompt_text] = len(prompt_job_ids)
        group = int(meta["group"])
        if response_tokens and training_enabled:
            behavior_values = item.get("behavior_logprobs")
            behavior_logprobs = (
                torch.tensor(behavior_values, dtype=torch.float32)
                if behavior_values is not None
                and len(behavior_values) == len(response_tokens)
                else None)
            example = {
                "prompt_ids": prompt_cache[prompt_text],
                "response_ids": torch.tensor([response_tokens], dtype=torch.long),
                "advantage": float(meta["advantage"]),
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": None,
                "reward_constant": False,
                "group_id": group,
                "rollout_index": int(meta["rollout"]),
                "prompt_job_id": prompt_job_ids[prompt_text],
                "sample_weight": 0.0,
                "_overlap_key": (group, int(meta["rollout"])),
            }
            if hybrid_loss:
                if not item["token_path"] or not _valid_example_token_logprobs(example, "behavior_logprobs"):
                    raise RuntimeError(f"{cfg.advantage_mode} recovery requires the saved sampler tokens and logprobs")
                if cfg.kl_penalty_coef:
                    reference_path = item["response_path"].with_suffix(".reference_logprobs.zlib")
                    example["reference_logprobs"] = torch.tensor(
                        load_rollout_logprobs(reference_path), dtype=torch.float32)
            if (cfg.advantage_mode != SPO_RS_INVARIANT_MODE or cfg.generation_backend != "vllm"
                    or _valid_example_token_logprobs(example, "behavior_logprobs")):
                all_examples.append(example)

        exact_tokens = bool(recovery.get("exact", False))
        if sampler.ac1_ttt_rollouts:
            code = (ttt_ac1_response_code(tokenizer, response_tokens) if exact_tokens
                    else ttt_ac1_python_block(response_text))
        else:
            # Re-encoded legacy text cannot prove where a native token occurred.
            final_text = (native_final_answer_scope(tokenizer, response_tokens, prompt_text=prompt_text)
                          if exact_tokens else None)
            code = extract_python_code(response_text, final_answer_text=final_text)

        if bool(meta.get("archive_eligible", False)):
            if not code:
                raise RuntimeError(
                    "evaluated-step recovery lost code from an archive-eligible "
                    f"rollout: {item['response_path'].name}")
            child = State.make(
                timestep=step_idx, value=float(meta["reward"]), code=code,
                raw_score=(float(meta["raw_score"]) if meta.get("raw_score") is not None else None),
                construction=(_ac1_saved_construction(meta) if ac1_archive
                              else _literal_float_list(meta.get("construction"))),
                observation=str(meta.get("sandbox_stdout") or ""),
                poisson=meta.get("poisson"),
                construction_length=meta.get("construction_length") if ac1_archive else None,
            )
            if meta.get("node_id"):
                child.id = str(meta["node_id"])
            parent_group = int(meta.get("parent_group", group))
            all_children.append((child, parents[parent_group]))
            if ac1_archive:
                archive_order[child.id] = int(meta["evaluation_order"])
        if item.get("critic_label_pending"):
            save_rollout(exp_dir, step_idx, group, int(meta["rollout"]), response_text,
                         meta, artifacts_already_saved=True)

    if not all_examples and not (hybrid_loss or invariant_mode):
        raise RuntimeError("evaluated-step recovery found no nonempty responses to train")
    if all_examples:
        trajectory_weight = 1.0 / len(all_examples)
        for example in all_examples:
            example["sample_weight"] = trajectory_weight
    hybrid_reference_mean = None
    if hybrid_loss and cfg.kl_penalty_coef and training_enabled:
        hybrid_reference_mean = reference_logprob_mean(
            (example["response_ids"].shape[1], example["behavior_logprobs"], example["reference_logprobs"])
            for example in all_examples)
        for example in all_examples:
            example["reference_logprob_mean"] = hybrid_reference_mean

    overflow_state = _load_exact_spo_rs_overflow(exp_dir, step_idx) if training_enabled else None
    recovery_examples = all_examples
    if overflow_state is not None:
        try:
            _validate_precomputed_exact_gradients(model, overflow_state, step_idx)
            deferred_keys = {(int(key[0]), int(key[1])) for key in overflow_state.get("deferred_keys", ())}
            remaining_keys = {
                (int(key[0]), int(key[1]))
                for key in overflow_state.get("remaining_keys", overflow_state.get("deferred_keys", ()))
            }
            marker_keys = {
                (int(key[0]), int(key[1]))
                for key in (_read_exact_sharded_recovery(exp_dir) or {}).get("deferred_keys", ())
            }
            if not deferred_keys or deferred_keys != marker_keys:
                raise RuntimeError('trajectory keys disagree between the marker and gradient checkpoint')
            if not remaining_keys.issubset(deferred_keys):
                raise RuntimeError(
                    "remaining overflow trajectories are not a subset of "
                    "the original deferred trajectories")
            example_by_key = {}
            for example in all_examples:
                key = tuple(example["_overlap_key"])
                if key in example_by_key:
                    raise RuntimeError(f"durable trajectory key {key} is duplicated")
                example_by_key[key] = example
            missing_keys = deferred_keys - set(example_by_key)
            if missing_keys:
                raise RuntimeError(f'deferred trajectories are absent from durable rollouts: {sorted(missing_keys)}')
            normal_count = int(overflow_state.get("normal_stats", {}).get("trained_examples", -1))
            if normal_count + len(deferred_keys) != len(all_examples):
                raise RuntimeError(
                    "trajectory accounting changed: retained="
                    f"{normal_count}, deferred={len(deferred_keys)}, "
                    f"recovered={len(all_examples)}")
            recovery_examples = [example_by_key[key] for key in sorted(remaining_keys)]
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            print(
                f"[warn] selective exact SPO-RS overflow state is not "
                f"reusable ({error}); replaying the evaluated batch on the process trainer",
                flush=True,
            )
            overflow_state = None
            recovery_examples = all_examples

    previous_best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    previous_best_value = float(previous_best_raw.raw_score) if previous_best_raw is not None else None
    if ac1_archive:
        all_children.sort(key=lambda entry: archive_order[entry[0].id])
    archive_stats = sampler.update(all_children)
    setting_log_only(
        f"[resume] step {step_idx}: reusing {len(artifacts)} saved rollout "
        "evaluations; generation and evaluation are skipped",
        flush=True,
    )
    if restored_terminal_tokens:
        print(
            f"[resume] step {step_idx}: restored the omitted terminal EOS in "
            f"{restored_terminal_tokens}/{len(artifacts)} saved responses",
            flush=True,
        )
    if canonical_recovery_responses:
        print(
            f"[resume] step {step_idx}: {canonical_recovery_responses} "
            "legacy text-only response(s) use canonical tokenizer IDs; "
            f"{unavailable_original_tokens} hidden/noncanonical original "
            f"token(s) were unavailable and {canonical_extra_tokens} "
            "canonical token(s) exceeded recorded counts",
            flush=True,
        )
    setting_log_only(
        f"[resume] step {step_idx}: restored {archive_stats['submitted']} "
        f"valid children and the exact saved {cfg.advantage_mode} advantages; resuming at "
        "adapter training",
        flush=True,
    )
    if overflow_state is not None:
        print(
            f"[recovery] step {step_idx}: reusing exact fast-path gradients "
            f"for {len(all_examples) - len(recovery_examples)}/"
            f"{len(all_examples)} trajectories; only "
            f"{len(recovery_examples)} oversized trajectory/trajectories "
            "will run on the process-distributed trainer",
            flush=True,
        )

    for group, records in sorted(by_group.items()):
        metas = [item["meta"] for item in records]
        _print_group_rollout_stats(
            group, [float(meta["reward"]) for meta in metas], [bool(meta["valid"]) for meta in metas],
            [int(meta["n_response_tokens"]) for meta in metas], beta=metas[0].get("beta"),
        )
    valid_count = sum(bool(item["meta"]["valid"]) for item in artifacts)
    setting_log_only(f"[step {step_idx}] reused {len(artifacts)} saved rollout .txt/.meta.json pairs", flush=True)
    elapsed = None
    if getattr(problem, "problem_type", "") == "trimul":
        evaluated_batch = trimul_evaluation.completed_batch(exp_dir, step_idx)
        if evaluated_batch is not None:
            elapsed = evaluated_batch["scoring"]["rollout_seconds"] + evaluated_batch["evaluation_seconds"]
    _print_rollout_totals(step_idx, valid_count, len(artifacts),
                          len(all_examples) if training_enabled else "disabled", len(all_children), elapsed)

    step_states = [child for child, _parent in all_children]
    step_best = _step_best_candidate(step_states, problem.maximize)
    best_seen = _print_best_ever_raw(sampler, step_idx, cfg, problem, previous_best_value)
    best_snapshot_path = save_best_state_snapshot(
        exp_dir, step_idx, best_seen,
        metric_name=str(problem.metric_name), maximize=bool(problem.maximize),
        construction=(
            _as_float_list(getattr(best_seen, "construction", None), cfg.max_saved_construction)
            if (best_seen is not None and getattr(problem, "saves_construction", False))
            else None),
    )
    if evaluation_complete is not None:
        evaluation_complete(step_idx)
    stats = {
        "best_seen_snapshot": (str(Path(best_snapshot_path).relative_to(exp_dir))
                               if best_snapshot_path is not None else None),
        "planned_rollouts": len(artifacts),
        "advantage_plot_rollouts": [advantage_plot_rollout(item["meta"]) for item in artifacts],
        "result_metric_name": str(problem.metric_name),
        "result_maximize": bool(problem.maximize),
        "step_best_raw_score": (
            float(step_best.raw_score)
            if step_best is not None and step_best.raw_score is not None else None),
        "step_best_reward": (float(step_best.value) if step_best is not None else None),
        "best_seen_raw_score": (
            float(best_seen.raw_score)
            if best_seen is not None and best_seen.raw_score is not None
            else None),
        "best_seen_reward": (float(best_seen.value) if best_seen is not None and best_seen.value is not None else None),
        "best_seen_step": (int(best_seen.timestep) if best_seen is not None else None),
        "valid_fraction": valid_count / len(artifacts),
        "code_valid_fraction": (
            1.0 - sum(_metadata_is_code_failure(item["meta"]) for item in artifacts) / len(artifacts)),
        "recovered_evaluated_step": True,
        "recovered_rollouts": len(artifacts),
    }
    if critic_mode:
        stats["critic_attempts"] = judged["attempts"]
    else:
        stats.update({
            "spo_rs_updates": [tracker_update],
            "spo_rs_tracker_size": len(spo_rs_tracker),
            "spo_rs_divergence": divergence if math.isfinite(divergence) else None,
            "spo_rs_context_divergences": spo_info.get("context_divergences", {}),
            "spo_rs_missing_policy_scores": int(spo_info.get("missing_policy_scores", 0) or 0),
            "spo_rs_sampling_adapter": Path(sampling_adapter_path).name,
        })
    if not training_enabled:
        stats["training_disabled"] = True
    if hybrid_loss:
        stats["hybrid_reference_kl_mean"] = hybrid_reference_mean
    if hybrid_mode and "hybrid_invariant_mixture_rms" not in spo_info:
        stats["hybrid_ttt_weight"] = cfg.hybrid_ttt_weight
        stats["hybrid_spo_weight"] = cfg.hybrid_spo_weight
    if overflow_state is not None:
        stats.update(parallel_trainer.recover_spo_rs_overflow(recovery_examples, cfg, step_idx, overflow_state))
    elif all_examples:
        stats.update(parallel_trainer.train_spo_rs(all_examples, cfg, step_idx))
    stats["critic_policy_updated" if critic_mode else "spo_rs_policy_updated"] = bool(all_examples)
    return stats


# ======================================================================
# One training step
#
# Generation is streamed and each rollout's program is evaluated on a CPU
# thread pool WHILE the GPUs keep generating.
#
# `reward_workers` is configured in YAML (0 = auto). Auto
# leaves ~one CPU core per GPU worker for the generation loop and divides the
# remainder by the CPU allocation of each candidate evaluation.
# AC1 uses one worker per CPU pair. Other isolated tasks use reward_workers
# as the number of processes per logical CPU.
# ======================================================================


def _resolve_reward_workers(cfg, problem, cpu_count=None) -> int:
    requested = int(getattr(cfg, "reward_workers", 0) or 0)
    if requested < 0:
        raise ValueError("reward_workers must be >= 0")
    if requested:
        return requested

    available = int(cpu_count if cpu_count is not None else (os.cpu_count() or 8))
    generation_workers = max(0, int(getattr(cfg, "num_gpus", 0) or 0))
    per_evaluation = max(1, int(getattr(problem, "eval_cpus", 1)))
    cpu_budget = max(1, available - generation_workers)
    return max(1, cpu_budget // per_evaluation)


def _automatic_sandbox_memory_limit_bytes(concurrent_sandboxes: int) -> int:
    """Return an automatic sandbox address-space cap of at most 1 GiB.

    This sets a per-program ceiling, not a promise that the whole evaluator
    pool fits beside resident models. HostMemoryAdmission separately reserves
    live RAM at each launch and shares that budget with transfers/activations.
    """
    concurrent_sandboxes = max(1, int(concurrent_sandboxes))
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total_bytes = page_size * physical_pages
    except (AttributeError, OSError, TypeError, ValueError):
        total_bytes = 8 * 1024**3
    shared_budget = max(1, int(total_bytes * 0.40))
    per_process = shared_budget // concurrent_sandboxes
    return max(512 * 1024**2, min(1024**3, per_process))


def _evaluation_pool(cfg, problem, rollout_count, step_idx, gen_pool,
                     parallel_trainer=None, *, recovering=False):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    from runtime.host_memory import HostMemoryAdmission
    host_admission = getattr(parallel_trainer, "host_memory_admission", None)
    if host_admission is None:
        host_admission = getattr(gen_pool, "host_memory_admission", None)
    if host_admission is None:
        import multiprocessing
        host_admission = HostMemoryAdmission(
            multiprocessing.get_context("spawn"), hardware_profile=getattr(cfg, "hardware_profile", "RTX6000"))
    isolated_eval = bool(getattr(cfg, "isolate_eval", False))
    paired = isolated_eval and getattr(cfg, "problem", None) == "ac1"
    isolated_cpu_slots = None
    isolated_cpu_count = 0
    isolated_processes_per_cpu = 0
    isolated_memory_limit_bytes = None
    if isolated_eval:
        allocations = (_isolated_ac1_cpu_pairs(cfg) if paired
                       else _isolated_evaluation_cpu_ids(cfg))
        isolated_cpu_count = len(allocations) * (2 if paired else 1)
        isolated_processes_per_cpu = 1 if paired else int(cfg.reward_workers)
        isolated_cpu_slots = Queue()
        # Give each CPU a slot before allowing any to host a second sandbox.
        for _layer in range(isolated_processes_per_cpu):
            for allocation in allocations:
                isolated_cpu_slots.put(allocation)
        isolated_capacity = len(allocations) * isolated_processes_per_cpu
        n_reward_workers = max(1, min(rollout_count, isolated_capacity))
        isolated_memory_limit_bytes = (_automatic_sandbox_memory_limit_bytes(n_reward_workers))
        if paired or not recovering:
            placement = ("one sandbox per exclusive CPU pair; " if paired else
                         f"up to {isolated_processes_per_cpu}/CPU from reward_workers; each "
                         "process tree is pinned to one CPU and ")
            capture = ("stdout streamed for prompt feedback, stderr capture is bounded"
                       if paired and getattr(cfg, "ac1_ttt_rollouts", False)
                       else "stdout+stderr capture is bounded")
            setting_log_only(
                f"[step {step_idx}] isolated evaluation pool: "
                f"{n_reward_workers} sandbox process(es) across "
                f"{isolated_cpu_count} CPU(s), {placement}excess candidates remain "
                "queued; live RAM admission is shared with model transfers and "
                f"activation offload; RAM reservation="
                f"{min(1024**3, isolated_memory_limit_bytes) / 1024**3:.2f} GiB/process; "
                f"automatic address-space ceiling="
                f"{isolated_memory_limit_bytes / 1024**3:.2f} GiB/process, "
                f"{capture}")
    else:
        n_reward_workers = _resolve_reward_workers(cfg, problem)
        if recovering:
            n_reward_workers = max(1, min(rollout_count, n_reward_workers))
        else:
            setting_log_only(f"[step {step_idx}] evaluation pool: {n_reward_workers} worker(s), "
                             f"{getattr(problem, 'eval_cpus', 1)} CPU(s) per candidate")

    def pin_worker_to_pair():
        # The scoring thread and the sandbox it launches share this mask.
        pair = set(isolated_cpu_slots.get_nowait())
        os.sched_setaffinity(0, pair)
        if os.sched_getaffinity(0) != pair:
            raise RuntimeError(f"AC1 could not pin the evaluation worker to CPUs {sorted(pair)}")

    reward_pool = ThreadPoolExecutor(
        max_workers=n_reward_workers, initializer=pin_worker_to_pair if paired else None)

    def evaluate(text, parent_ctx, **answer_options):
        if not isolated_eval:
            return problem.compute_reward(text, parent_ctx, cfg.sandbox_timeout_s, **answer_options)
        with host_admission.reservation(min(1024**3, isolated_memory_limit_bytes)):
            cpu_id = None if paired else isolated_cpu_slots.get()
            try:
                return problem.compute_reward(
                    text, parent_ctx, cfg.sandbox_timeout_s, cpu_id=cpu_id,
                    memory_limit_bytes=isolated_memory_limit_bytes, **answer_options)
            finally:
                if cpu_id is not None:
                    isolated_cpu_slots.put(cpu_id)

    return reward_pool, evaluate, {
        "evaluation_isolated": isolated_eval,
        "evaluation_workers": int(n_reward_workers),
        "evaluation_cpu_count": int(isolated_cpu_count),
        "evaluation_processes_per_cpu": int(isolated_processes_per_cpu),
    }


def _open_reward_pool(cfg, problem, rollout_count, step_idx, gen_pool,
                      parallel_trainer, *, cpu_eval_progress=None):
    from concurrent.futures import Future
    from problems.base import RewardResult

    reward_pool, evaluate, stats = _evaluation_pool(
        cfg, problem, rollout_count, step_idx, gen_pool, parallel_trainer)

    def submit(text, parent_ctx, final_text, format_issue):
        if format_issue is not None:
            future = Future()
            future.set_result(RewardResult(
                reward=float(problem.fail_score), msg=format_issue, failure_kind="code"))
            if cpu_eval_progress is not None:
                cpu_eval_progress.complete_without_worker()
            return future
        answer_options = {} if final_text is None else {"final_answer_text": final_text}
        if cpu_eval_progress is not None:
            return cpu_eval_progress.submit(reward_pool, evaluate, text, parent_ctx, **answer_options)
        return reward_pool.submit(evaluate, text, parent_ctx, **answer_options)

    return reward_pool, submit, stats


def _prepare_gpu_evaluation(backend, optimizer, cfg, problem, step_idx, count,
                            *, trainer_parked=False):
    import gc
    import torch

    if getattr(problem, "problem_type", "") == "trimul":
        raise RuntimeError("TriMul requires the trainer process to exit before evaluation")
    if trainer_parked:
        return False
    print(f"[step {step_idx}] releasing trainer memory before {count} benchmark(s); "
          "one benchmark at a time per evaluation GPU")
    try:
        torch.cuda.synchronize()
        _move_optimizer_state(optimizer, "cpu", hardware_profile=getattr(cfg, "hardware_profile", "RTX6000"))
        backend.offload_for_generation()
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        try:
            backend.restore_after_generation()
            _restore_optimizer_state_to_parameters(optimizer)
        except Exception:
            pass
        raise RuntimeError(
            "shared-GPU evaluation requires the trainer to offload before "
            "candidate benchmark evaluation, but this model/runtime "
            "cannot move the training state to CPU") from exc
    return True


def _restore_after_gpu_evaluation(backend, optimizer, step_idx):
    import gc
    import torch

    print(f"[step {step_idx}] restoring trainer after shared-GPU benchmark evaluation")
    gc.collect()
    torch.cuda.empty_cache()
    backend.restore_after_generation()
    _restore_optimizer_state_to_parameters(optimizer)
    backend.set_training_mode()


def _parent_context(parent):
    from problems.base import ParentContext

    return ParentContext(
        code=parent.code, value=parent.value if parent.value is not None else 0.0,
        raw_score=parent.raw_score, construction=parent.construction,
        observation=parent.observation, previous_raw_score=parent.previous_raw_score,
        poisson=parent.poisson,
    )


def _parent_rollout_fields(parent, pick_info, sampler_type):
    return {
        "parent_id": parent.id,
        "parent_timestep": int(parent.timestep),
        "sampler_type": sampler_type,
        "parent_raw_score": parent.raw_score,
        "parent_value": float(parent.value) if parent.value is not None else None,
        "parent_is_seed": parent.is_seed,
        "parent_visit_count": int(pick_info.get("n", 0)),
        "parent_q_value": float(pick_info["Q"]) if pick_info.get("Q") is not None else None,
        "parent_prior": float(pick_info["P"]) if pick_info.get("P") is not None else None,
        "parent_exploration_bonus": float(pick_info["bonus"]) if pick_info.get("bonus") is not None else None,
        "parent_selection_score": float(pick_info["score"]) if pick_info.get("score") is not None else None,
    }


def _step_best_candidate(states, maximize):
    raw_states = []
    for state in states:
        try:
            if math.isfinite(float(state.raw_score)):
                raw_states.append(state)
        except (TypeError, ValueError):
            continue
    if raw_states:
        choose = max if maximize else min
        return choose(raw_states, key=lambda state: float(state.raw_score))
    return max(states, key=lambda state: float(state.value) if state.value is not None else -math.inf,
               default=None)


def _print_group_rollout_stats(group, rewards, valids, response_tokens, beta=None):
    beta_label = f"  beta={beta:.9f}" if beta is not None else ""
    print(f"  group {group}: rewards min={min(rewards):.9f} "
          f"mean={np.mean(rewards):.9f} max={max(rewards):.9f}  "
          f"valid={sum(valids)}/{len(valids)}{beta_label}  "
          f"response tokens min={min(response_tokens) / 1000:.1f}k "
          f"mean={np.mean(response_tokens) / 1000:.1f}k max={max(response_tokens) / 1000:.1f}k")


def _print_rollout_totals(step, valid, total, training_examples, children, elapsed, *, overlapped=False):
    phase = ("rollout+eval+overlapped-backward wall time" if overlapped else "rollout+eval time")
    timing = f"{elapsed:.1f}s" if elapsed is not None else "unavailable"
    print(f"[step {step}] {phase}: {timing}  "
          f"training examples: {training_examples}  new children: {children}  valid={valid}/{total}",
          flush=True)


def _print_best_ever_raw(sampler, step_idx, cfg, problem, previous_best_value):
    best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    best_seen_result = best_raw if best_raw is not None else sampler.best_state()
    if best_raw is not None:
        direction = "higher is better" if problem.maximize else "lower is better"
        best_value = float(best_raw.raw_score)
        improved = (
            previous_best_value is None
            or (best_value > previous_best_value if problem.maximize else best_value < previous_best_value)
        )
        # ``target`` is part of the problem context shown to the model.
        # ``target_print`` is deliberately display-only, so changing the SOTA
        # comparison printed here cannot leak a new target into the prompt.
        target = getattr(cfg, "target_print", getattr(problem, "target", None))
        try:
            target = float(target) if target is not None else None
        except (TypeError, ValueError):
            target = None
        beats_sota = bool(
            improved and target is not None and math.isfinite(target)
            and (best_value > target if problem.maximize else best_value < target)
        )
        message = (
            f"[step {step_idx}] best-ever raw {problem.metric_name}: "
            f"{best_value:.9f} ({direction}; "
            f"reward={float(best_raw.value):.9f}, found step={best_raw.timestep})"
        )
        if beats_sota:
            margin = (best_value - target if problem.maximize else target - best_value)
            message += (f'  🏆 CONGRATULATIONS — NEW SOTA! target={target:.12f}, beaten by {margin:.12f} 🏆')
            color = "\033[32m"       # dark green
        elif improved:
            message += "  ★ NEW RUN BEST"
            color = "\033[92m"       # light green
        else:
            message += "  no improvement this step"
            color = "\033[93m"       # yellow
        print(f"{color}{message}\033[0m", flush=True)
    else:
        message = (f'[step {step_idx}] best-ever raw {problem.metric_name}: unavailable')
        print(f"\033[93m{message}\033[0m", flush=True)
    return best_seen_result


def train_step(backend, model, tokenizer, sampler, optimizer, step_idx: int,
               cfg, exp_dir, problem, gen_pool=None,
               parallel_trainer=None,
               spo_rs_tracker=None, sampling_adapter_path=None,
               evaluation_complete=None, pending_artifacts=None, park_trainer=None):
    import os
    import torch
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    from search.sampler import State
    from experiment_io import (advantage_plot_rollout, archive_unfinished_rollout_step,
                               load_rollout_logprobs, load_rollout_token_ids,
                               save_best_state_snapshot, save_parent_selections,
                               save_rollout, save_rollout_artifacts, save_rollout_logprobs)
    from problems.base import is_code_failure
    from generation.gen_workers import make_progress_bar, render_chat_prompt


    training_enabled = not bool(getattr(cfg, "no_train", False))
    if training_enabled and parallel_trainer is None:
        raise ValueError("training requires the process-distributed trainer")
    step_t0 = time.time()
    active_advantage_mode = str(getattr(cfg, "advantage_mode", "entropic")).lower()
    invariant_mode = active_advantage_mode in INVARIANT_ADVANTAGE_MODES
    hybrid_mode = active_advantage_mode in ("hybrid", HYBRID_INVARIANT_MODE)
    critic_mode = active_advantage_mode == LLM_CRITIC_MODE
    hybrid_loss = uses_hybrid_policy_loss(active_advantage_mode)
    history_mode = _uses_reward_history(active_advantage_mode)
    replayable_update = _uses_replayable_policy_update(active_advantage_mode)
    history_label = "hybrid" if hybrid_mode else "SPO-RS"
    if history_mode and (spo_rs_tracker is None or not spo_rs_tracker.initialized):
        raise ValueError("historical advantages require an initialized pre-step reward tracker")
    clipped_policy_mode = _uses_clipped_policy_loss(active_advantage_mode)
    # The historical tracker needs immutable consecutive policy versions.
    adapter_path = None
    if replayable_update:
        if sampling_adapter_path is None:
            adapter_path = _save_adapter(model, exp_dir, step_idx, directory_name="adapter_initial")
        else:
            adapter_path = str(Path(sampling_adapter_path))
            if not Path(adapter_path).is_dir():
                raise FileNotFoundError(f"sampling adapter not found: {adapter_path}")
    elif gen_pool is not None:
        adapter_path = _save_adapter(model, exp_dir, step_idx)

    is_trimul = getattr(problem, "problem_type", "") == "trimul"
    isolated_batch = trimul_evaluation.completed_batch(exp_dir, step_idx) if is_trimul else None
    ac1_archive = sampler.ac1_archive
    if sampler.ac1_ttt_rollouts and pending_artifacts is None:
        backup = archive_unfinished_rollout_step(exp_dir, step_idx)
        if backup is not None:
            setting_log_only(f"[resume] regenerating AC1 step {step_idx}; previous artifacts moved to {backup}")
    parents = _sample_step_parents(sampler, cfg, exp_dir, step_idx, reuse_saved=pending_artifacts is not None)
    if step_idx == 0 and (not sampler.ac1_ttt_rollouts or pending_artifacts is not None):
        _rebind_interrupted_step_zero_selection(sampler, exp_dir, parents)
    if isolated_batch is not None or pending_artifacts is not None:
        trimul_evaluation.check_parents(exp_dir, step_idx, parents)
    print(f"\n[step {step_idx}] parents picked: {len(parents)}")
    for i, info in enumerate(sampler.last_picks_info):
        tag = "seed" if info["is_seed"] else "expanded"
        prior_label = ("" if info.get("selection_mode") == "uct" else f"  P={info['P']:.9f}")
        print(f"  parent {i} [{tag}]  value={info['value']:.9f}  n={info['n']}  "
              f"Q={info['Q']:.9f}{prior_label}  "
              f"bonus={info['bonus']:.9f}  score={info['score']:.9f}")

    # Save the selection event immediately. Unlike the sampler checkpoint,
    # this survives later archive pruning and also exists if generation or
    # adapter training is interrupted.
    sampler_type = ("UCTSampler" if getattr(sampler, "use_uct", False) else type(sampler).__name__)
    save_parent_selections(exp_dir, step_idx, sampler_type, parents, sampler.last_picks_info, rollout_config=cfg)
    parent_fields = [_parent_rollout_fields(parent, sampler.last_picks_info[i]
                     if i < len(sampler.last_picks_info) else {}, sampler_type)
                     for i, parent in enumerate(parents)]
    if critic_mode:
        critic_context = critic_step.step_context(
            problem, (state for state in sampler._states if state.is_seed), adapter_path)
    setting_log_only(
        f"[step {step_idx}] saved {len(parents)} selected parent(s) before "
        f"generation/training", flush=True)

    # Only problems that declare it get their construction written to disk.
    save_ctor = (bool(getattr(problem, "saves_construction", False))
                 and int(getattr(cfg, "max_saved_construction", 0)) != 0)

    all_examples = []
    spo_rs_updates = []
    advantage_plot_rollouts = []
    entropic_overlap_state = None
    spo_rs_overlap_state = None
    all_children = []
    archive_order = {}
    saved_rollouts = 0
    # ----- BUILD PROMPTS (one per parent/group) -----
    parent_ctxs = []
    base_messages = []

    for parent in parents:
        sampler.record_expansion(parent, count=int(cfg.group_size))
        pc = _parent_context(parent)
        parent_ctxs.append(pc)
        base_messages.append(problem.build_prompt(pc))

    coder_effort = getattr(cfg, "coder_reasoning_effort", None)

    prompt_jobs = []
    for g, _parent in enumerate(parents):
        messages = base_messages[g]
        count = int(cfg.group_size)
        prompt_jobs.append({
            "parent_group": g,
            "messages": messages,
            "prompt_text": render_chat_prompt(tokenizer, messages, reasoning_effort=coder_effort),
            "coder_reasoning_effort": coder_effort,
            "count": count,
        })

    spo_rs_divergence = 0.0
    spo_rs_context_divergences = {}
    spo_rs_missing_policy_scores = 0
    spo_rs_previous_adapter_path = None
    hybrid_reference_mean = None
    if history_mode:
        previous_name = spo_rs_tracker.last_policy_adapter
        if previous_name is not None:
            spo_rs_previous_adapter_path = Path(exp_dir) / previous_name

    group_specs = []
    for parent_group in range(len(parents)):
        group_specs.append({
            "group_id": len(group_specs),
            "parent_group": parent_group,
            "context_id": parent_group,
        })
    num_groups = len(group_specs)
    total_rollouts = num_groups * cfg.group_size

    # ----- REWARD POOL (CPU), runs concurrently with generation -----
    # compute_reward delegates the heavy work to a subprocess sandbox, so the
    # launching thread mostly waits (GIL released) and many sandboxes run in
    # parallel across cores. THREAD-SAFETY REQUIREMENT: each compute_reward call
    # must use a unique temp file/dir and must not os.chdir or mutate shared
    # state; otherwise concurrent runs corrupt each other's rewards.
    cpu_eval_progress = None
    if cfg.evaluation_gpu_id is None:
        interval = getattr(cfg, "eval_progress_interval_s", DEFAULT_EVAL_PROGRESS_INTERVAL_S)
        cpu_eval_progress = CpuEvaluationProgress(step_idx, total_rollouts, interval=interval)
    reward_pool, submit_reward, evaluation_stats = _open_reward_pool(
        cfg, problem, total_rollouts, step_idx, gen_pool, parallel_trainer,
        cpu_eval_progress=cpu_eval_progress)
    # Large response/prompt files are persisted as soon as generation returns,
    # on a dedicated thread so disk I/O never blocks the generation scheduler.
    # The complete reward/advantage metadata replaces the provisional metadata
    # below after evaluation; the large text artifacts are written only once.
    rollout_io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rollout-save-step{step_idx}")
    rollout_io_futures = []

    # GPU evaluation shows a bar once a result arrives. CPU jobs report activity independently.
    eval_bar = None
    eval_progress_condition = threading.Condition()
    eval_progress_completed = 0
    eval_bar_closed = False
    live_best_state = sampler.best_raw_state(maximize=bool(problem.maximize))
    live_best_raw = (
        float(live_best_state.raw_score) if live_best_state is not None else None)

    def _record_evaluation_completion(_future, record=None):
        nonlocal eval_bar, eval_progress_completed, live_best_raw
        with eval_progress_condition:
            best_line = None
            if (record is not None and not _future.cancelled()
                    and _future.exception() is None):
                result = _future.result()
                live_best_raw, best_line = _advance_live_best_raw(result, live_best_raw, problem)
                evaluated_meta = _evaluated_rollout_meta(record["_pending_meta"], result)
                if critic_mode:
                    evaluated_meta["archive_eligible"] = bool(result.valid and result.code)
                    if save_ctor:
                        evaluated_meta["construction"] = _as_float_list(result.construction, cfg.max_saved_construction)
                if ac1_archive:
                    record["evaluation_order"] = eval_progress_completed
                    if critic_mode:
                        evaluated_meta.update(_ac1_archive_metadata(result, record["evaluation_order"]))
                save_future = rollout_io_pool.submit(
                    save_rollout,
                    exp_dir, step_idx, int(record["group_id"]),
                    int(record["rollout_index"]), record["text"],
                    evaluated_meta,
                    artifacts_already_saved=True,
                )
                rollout_io_futures.append(save_future)
            eval_progress_completed += 1
            try:
                if eval_bar is not None:
                    eval_bar.update(1)
                elif cpu_eval_progress is None and record is not None:
                    eval_bar = make_progress_bar(
                        total_rollouts, desc="evaluating", initial=eval_progress_completed)
            finally:
                eval_progress_condition.notify_all()
            if best_line:
                write_progress_line(f"[step {step_idx}] {best_line}")

    # Mutable records preserve streamed arrival order while vLLM reference
    # scoring fills its result concurrently with the CPU reward futures.
    group_responses = {g: [] for g in range(num_groups)}
    reward_futures = {g: [] for g in range(num_groups)}    # aligned RewardResult futures
    deferred_rollouts = []
    vllm_logprob_records = []
    queued_by_parent = {g: 0 for g in range(len(parents))}
    defer_gpu_evaluation = is_trimul or bool(getattr(cfg, "evaluation_shares_generation", False))

    def _submit_rollout(record, result=None):
        g = int(record["group_id"])
        group_responses[g].append(record)
        if result is None:
            fut = submit_reward(
                record["text"], parent_ctxs[record["parent_group"]], record["_final_answer_text"],
                record["output_format_issue"])
        else:
            fut = Future()
            fut.set_result(result)
        reward_futures[g].append(fut)
        record["_reward_future"] = fut
        fut.add_done_callback(
            lambda completed, rollout=record:
            _record_evaluation_completion(completed, rollout if result is None else None))
        return fut

    def _queue_rollout(job_idx, text, token_ids, behavior_logprobs=None):
        # Record malformed responses with the failure reward.
        token_ids = list(token_ids)
        if behavior_logprobs is not None:
            behavior_values = list(behavior_logprobs)
            behavior_logprobs = (
                behavior_values
                if len(behavior_values) == len(token_ids)
                and all(math.isfinite(float(value)) for value in behavior_values)
                else None)
        job = prompt_jobs[int(job_idx)]
        final_text, issue = _rollout_code_input(cfg, tokenizer, token_ids, text, job["prompt_text"])
        group_id = int(job["parent_group"])
        rollout_index = queued_by_parent[group_id]
        queued_by_parent[group_id] += 1
        record = {
            "job_idx": int(job_idx),
            "parent_group": group_id,
            "group_id": group_id,
            "context_id": group_id,
            "rollout_index": rollout_index,
            "text": text,
            "_final_answer_text": final_text,
            "token_ids": token_ids,
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": None,
            "output_format_issue": issue,
        }
        pending_meta = {
            "output_format_issue": record["output_format_issue"],
            "status": "evaluation_pending",
            "step": int(step_idx),
            "group": group_id,
            "parent_group": group_id,
            "rollout": rollout_index,
            "n_response_tokens": len(token_ids),
            "coder_reasoning_effort": job.get("coder_reasoning_effort"),
            "solver": "coder",
            "model_name": cfg.coder_model_name,
        }
        if hybrid_loss or invariant_mode:
            pending_meta["advantage_mode"] = active_advantage_mode
        if critic_mode:
            pending_meta.update(parent_fields[group_id],
                                node_id=critic_step.rollout_id(step_idx, group_id, rollout_index))
        record["_pending_meta"] = pending_meta
        if isolated_batch is None:
            rollout_io_futures.append(rollout_io_pool.submit(
                save_rollout_artifacts,
                exp_dir, step_idx, group_id, rollout_index, text,
                prompt_text=job["prompt_text"],
                response_token_ids=token_ids,
                behavior_logprobs=behavior_logprobs,
                pending_meta=pending_meta,
            ))
        if (training_enabled and cfg.generation_backend == "vllm" and token_ids):
            vllm_logprob_records.append(record)
        if defer_gpu_evaluation:
            deferred_rollouts.append(record)
        else:
            _submit_rollout(record)
        return record

    # ----- ROLLOUTS (streamed) + dispatch rewards as each rollout lands -----
    rollout_t0 = time.time()
    evaluation_trainer_offloaded = False
    coder_phase_complete = False
    try:
        try:
            if isolated_batch is not None or pending_artifacts is not None:
                saved_rows = (isolated_batch["candidates"] if isolated_batch is not None
                              else [item["meta"] for item in pending_artifacts])
                for row in saved_rows:
                    group = int(row["group"])
                    path, text, prompt = trimul_evaluation.read_rollout(exp_dir, step_idx, row)
                    index = queued_by_parent[group]
                    if int(row["rollout"]) != index:
                        raise RuntimeError(f"Unexpected saved TriMul rollout order: {path}")
                    if index == 0:
                        prompt_jobs[group]["prompt_text"] = prompt
                    elif prompt != prompt_jobs[group]["prompt_text"]:
                        raise RuntimeError(f"Inconsistent saved TriMul prompt: {path}")
                    behavior_path = path.with_suffix(".behavior_logprobs.zlib")
                    record = _queue_rollout(
                        group, text, load_rollout_token_ids(path.with_suffix(".tokens.zlib")),
                        load_rollout_logprobs(behavior_path) if behavior_path.is_file() else None)
                    if isolated_batch is not None:
                        if (record["_final_answer_text"] != row["final_answer"]
                                or record["output_format_issue"] != row["format_issue"]):
                            raise RuntimeError(f"Restored TriMul answer scope differs from {path}")
                        if row["has_reference_logprobs"]:
                            record["reference_logprobs"] = load_rollout_logprobs(
                                path.with_suffix(".reference_logprobs.zlib"))
                print(f"[step {step_idx}] restored {len(saved_rows)} saved coder rollouts", flush=True)
            elif gen_pool is not None:
                generation_options = {
                    "prompts_by_group": [job["prompt_text"] for job in prompt_jobs],
                    "group_size": cfg.group_size,
                    "counts_by_group": [job["count"] for job in prompt_jobs],
                    "adapter_path": adapter_path,
                    "max_new_tokens": cfg.max_new_tokens,
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "top_k": getattr(cfg, "sampling_top_k", None),
                    "min_p": getattr(cfg, "sampling_min_p", None),
                    "step_idx": step_idx,
                    "full_policy_sampling": hybrid_loss,
                }
                if cfg.generation_backend == "vllm":
                    generation_options["return_logprobs"] = training_enabled
                for job_idx, job_results in gen_pool.iter_group_jobs(**generation_options):
                    for result in job_results:
                        text, token_ids = result[:2]
                        behavior_logprobs = (result[2] if len(result) > 2 else None)
                        _queue_rollout(job_idx, text, token_ids, behavior_logprobs)
            else:
                # In-process generation uses cross-prompt micro-batches rather
                # than draining one parent at a time. Ordinary CPU verification
                # overlaps later batches; one-card GPU-mode verification waits.
                backend.set_inference_mode()
                cap_state = {"value": int(getattr(cfg, '_local_generation_cap', 0) or 0)}

                def _seed_local_generation(seed_step):
                    if cfg.deterministic:
                        torch.manual_seed((int(cfg.seed) * 1_000_003 + int(seed_step) * 1009 + 13) % (2**31 - 1))

                _seed_local_generation(step_idx)
                gen_bar = make_progress_bar(total_rollouts, desc="rollouts")
                try:
                    for job_idx, responses in generate_prompt_jobs(
                            model, tokenizer,
                            [job["prompt_text"] for job in prompt_jobs],
                            [job["count"] for job in prompt_jobs], cfg,
                            cap_state=cap_state):
                        for (text, token_ids) in responses:
                            _queue_rollout(job_idx, text, token_ids)
                        gen_bar.update(len(responses))
                    cfg._local_generation_cap = int(cap_state["value"])
                finally:
                    gen_bar.close()

            incomplete = {group: count for group, count in queued_by_parent.items()
                          if count != int(cfg.group_size)}
            if incomplete:
                raise RuntimeError(
                    f"expected {int(cfg.group_size)} planned rollouts per parent; got {incomplete}")
            if isolated_batch is None and pending_artifacts is None:
                print(f"[step {step_idx}] coder generation complete: {total_rollouts} rollouts", flush=True)

            # CPU reward processes were submitted as each rollout arrived and
            # continue running here. Keep vLLM awake and use that same interval
            # to score fixed base/reference token logprobs without the adapter.
            if isolated_batch is not None:
                scoring = isolated_batch["scoring"]
                spo_rs_divergence = scoring["divergence"]
                spo_rs_context_divergences = {int(key): value for key, value in scoring["context_divergences"].items()}
                spo_rs_missing_policy_scores = scoring["missing_policy_scores"]
            elif cfg.generation_backend == "vllm" and vllm_logprob_records:
                scoring_started = time.time()
                eval_done_before = sum(future.done() for futures in reward_futures.values() for future in futures)
                prompt_token_ids = {}
                score_pairs = []
                for record in vllm_logprob_records:
                    job_idx = record["job_idx"]
                    if job_idx not in prompt_token_ids:
                        prompt_token_ids[job_idx] = tokenizer(prompt_jobs[job_idx]["prompt_text"]).input_ids
                    score_pairs.append((
                        prompt_token_ids[job_idx], record["token_ids"]))
                if history_mode:
                    reference_scores = [None] * len(score_pairs)
                    current_policy_scores = [
                        (list(record["behavior_logprobs"])
                         if record["behavior_logprobs"] is not None
                         and len(record["behavior_logprobs"])
                         == len(record["token_ids"])
                         else None)
                        for record in vllm_logprob_records
                    ]
                    if any(value is None for value in current_policy_scores):
                        if hybrid_loss:
                            raise RuntimeError(f"{active_advantage_mode} training requires the original sampler logprobs for every response")
                        rescored_current = gen_pool.score_token_logprobs(
                            score_pairs, show_progress=True,
                            adapter_path=str(adapter_path),
                            require_complete=True)
                        current_policy_scores = [
                            existing if existing is not None else retry
                            for existing, retry in zip(current_policy_scores, rescored_current)
                        ]
                    for record, values in zip(vllm_logprob_records, current_policy_scores):
                        if values is None or len(values) != len(record["token_ids"]):
                            raise RuntimeError(
                                "SPO-RS could not obtain complete rollout-"
                                "policy logprobs before training")
                        if (record["behavior_logprobs"] is None
                                or len(record["behavior_logprobs"])
                                != len(record["token_ids"])):
                            record["behavior_logprobs"] = values
                            base = (
                                f"step{int(step_idx):02d}_"
                                f"group{int(record['group_id']):02d}_"
                                f"rollout{int(record['rollout_index']):03d}")
                            save_future = rollout_io_pool.submit(
                                save_rollout_logprobs,
                                Path(exp_dir) / f"step{int(step_idx):02d}"
                                / f"{base}.behavior_logprobs.zlib",
                                values,
                            )
                            rollout_io_futures.append(save_future)
                    previous_policy_scores = [None] * len(score_pairs)
                    same_policy = bool(
                        spo_rs_tracker.initialized
                        and spo_rs_previous_adapter_path is not None
                        and Path(adapter_path).resolve()
                        == spo_rs_previous_adapter_path.resolve())
                    if same_policy:
                        spo_rs_divergence = 0.0
                        spo_rs_context_divergences = {int(record["context_id"]): 0.0 for record in vllm_logprob_records}
                    elif (spo_rs_tracker.initialized):
                        if (spo_rs_previous_adapter_path is None or not spo_rs_previous_adapter_path.is_dir()):
                            raise FileNotFoundError(
                                "SPO-RS cannot compute the consecutive-policy "
                                "KL because its previous sampling adapter is "
                                f"missing: {spo_rs_previous_adapter_path}")
                        else:
                            previous_policy_scores = (
                                gen_pool.score_token_logprobs(
                                    score_pairs, show_progress=True,
                                    adapter_path=str(spo_rs_previous_adapter_path),
                                    require_complete=True))
                            (spo_rs_divergence,
                             spo_rs_missing_policy_scores,
                             spo_rs_context_divergences) = (
                                spo_rs_tracker.consecutive_policy_divergence(
                                    current_policy_scores,
                                    previous_policy_scores,
                                    [record["context_id"] for record in vllm_logprob_records],
                                    required_context_ids=[spec["context_id"] for spec in group_specs]))
                            if (spo_rs_missing_policy_scores or not math.isfinite(spo_rs_divergence)):
                                raise RuntimeError(
                                    "SPO-RS exact consecutive-policy scoring "
                                    "completed without a usable score for "
                                    f"{spo_rs_missing_policy_scores} rollout(s)")
                    if hybrid_loss and cfg.kl_penalty_coef:
                        reference_scores = gen_pool.score_token_logprobs(
                            score_pairs, show_progress=True, require_complete=True)
                elif critic_mode:
                    reference_scores = (gen_pool.score_token_logprobs(
                        score_pairs, show_progress=True, require_complete=True)
                        if cfg.kl_penalty_coef else [None] * len(score_pairs))
                else:
                    current_policy_scores = [None] * len(score_pairs)
                    try:
                        reference_scores = gen_pool.score_token_logprobs(score_pairs, show_progress=True)
                    except Exception as error:
                        reference_scores = [None] * len(score_pairs)
                        print(f"[warn] vLLM reference scoring failed ({error!r}); "
                              "missing values will be recomputed exactly by "
                              "the HF trainer", flush=True)
                for record, values in zip(vllm_logprob_records, reference_scores):
                    record["reference_logprobs"] = values
                    if hybrid_loss and values is not None and not is_trimul:
                        row = {"group": record["group_id"], "rollout": record["rollout_index"]}
                        path = trimul_evaluation.rollout_path(exp_dir, step_idx, row)
                        rollout_io_futures.append(rollout_io_pool.submit(
                            save_rollout_logprobs, path.with_suffix(".reference_logprobs.zlib"), values))
                behavior_count = sum(record["behavior_logprobs"] is not None for record in vllm_logprob_records)
                reference_count = sum(record["reference_logprobs"] is not None for record in vllm_logprob_records)
                reference_hf_fallbacks = (0 if replayable_update else len(vllm_logprob_records) - reference_count)
                eval_done_after = sum(future.done() for futures in reward_futures.values() for future in futures)
                reference_label = (
                    'off for SPO-RS' if clipped_policy_mode else f'{reference_count}/{len(vllm_logprob_records)}')
                if history_mode and not spo_rs_tracker.initialized:
                    policy_label = ", SPO-RS policy KL=initialization"
                elif history_mode and same_policy:
                    policy_label = ", SPO-RS policy KL=0 (same adapter)"
                elif history_mode:
                    divergence_label = (
                        "missing" if not math.isfinite(spo_rs_divergence)
                        else f"{spo_rs_divergence:.6f}")
                    policy_label = (
                        f", SPO-RS policy KL={divergence_label} "
                        f"on {len(spo_rs_context_divergences)} contexts, "
                        f"missing={spo_rs_missing_policy_scores}")
                else:
                    policy_label = ""
                setting_log_only(
                    f"[step {step_idx}] vLLM logprobs: rollout "
                    f"{behavior_count}/{len(vllm_logprob_records)}, "
                    f"reference {reference_label}{policy_label} in "
                    f"{time.time() - scoring_started:.1f}s "
                    f"(HF exact reference fallbacks: {reference_hf_fallbacks}; "
                    f"CPU evaluations completed during scoring: {eval_done_before}->{eval_done_after})",
                    flush=True)

            coder_phase_complete = True
        finally:
            if gen_pool is not None and getattr(gen_pool, "sequential", False):
                immediate_restore = bool(
                    coder_phase_complete and training_enabled and not critic_mode
                    and not defer_gpu_evaluation
                    and hasattr(gen_pool, "release_for_training"))
                if immediate_restore:
                    gen_pool.release_for_training(parallel_trainer)
                else:
                    gen_pool.release()

        if is_trimul and isolated_batch is None:
            for write in rollout_io_futures:
                write.result()
            candidates = []
            for record in deferred_rollouts:
                reference = record["reference_logprobs"]
                row = {
                    "group": record["group_id"], "rollout": record["rollout_index"],
                    "final_answer": record["_final_answer_text"],
                    "format_issue": record["output_format_issue"],
                    "has_reference_logprobs": reference is not None,
                }
                if reference is not None:
                    path = trimul_evaluation.rollout_path(exp_dir, step_idx, row)
                    save_rollout_logprobs(path.with_suffix(".reference_logprobs.zlib"), reference)
                candidates.append(row)
            trimul_evaluation.request_batch(
                exp_dir, step_idx, problem, cfg.sandbox_timeout_s, candidates,
                divergence=spo_rs_divergence, context_divergences=spo_rs_context_divergences,
                missing_policy_scores=spo_rs_missing_policy_scores, previous_best_raw=live_best_raw,
                rollout_seconds=time.time() - rollout_t0,
                **({"critic": critic_context} if critic_mode else {}))

        if isolated_batch is not None:
            for row, record in zip(isolated_batch["candidates"], deferred_rollouts):
                path = trimul_evaluation.rollout_path(exp_dir, step_idx, row)
                _submit_rollout(record, trimul_evaluation.load_result(
                    path, record["text"], record["_final_answer_text"]))
        elif deferred_rollouts:
            if critic_mode:
                park_trainer()
            evaluation_trainer_offloaded = _prepare_gpu_evaluation(
                backend, optimizer, cfg, problem, step_idx, len(deferred_rollouts),
                trainer_parked=(critic_mode or (not training_enabled and cfg.generation_backend == "vllm"
                                                and getattr(gen_pool, "sequential", False))))
            for record in deferred_rollouts:
                _submit_rollout(record)

        all_futs = [f for g in range(num_groups) for f in reward_futures[g]]
        try:
            if hybrid_loss and training_enabled:
                scored_records = [record for g in range(num_groups) for record in group_responses[g]
                                  if record["token_ids"]]
                if any(record["behavior_logprobs"] is None for record in scored_records):
                    raise RuntimeError(f"{active_advantage_mode} training requires the original sampler logprobs for every response")
                if cfg.kl_penalty_coef:
                    hybrid_reference_mean = reference_logprob_mean(
                        (len(record["token_ids"]), record["behavior_logprobs"], record["reference_logprobs"])
                        for record in scored_records)

            # The pre-step historical baseline is immutable. Hybrid training
            # can stream a group once all its siblings have rewards.
            spo_rs_overlap_eligible = bool(
                history_mode
                and not invariant_mode
                and training_enabled
                and not deferred_rollouts
                and not evaluation_trainer_offloaded
            )
            if spo_rs_overlap_eligible:
                staged_spo_rs_examples = []
                try:
                    staged_spo_rs_examples = (
                        _prepare_spo_rs_overlap_inputs(
                            tokenizer, group_responses, reward_futures, prompt_jobs,
                            reference_logprob_mean=hybrid_reference_mean))
                    if staged_spo_rs_examples:
                        pending_before = sum(not future.done() for future in all_futs)
                        pending_by_future = {
                            example["_overlap_reward_future"]: example
                            for example in staged_spo_rs_examples
                        }
                        if (len(pending_by_future) != len(staged_spo_rs_examples)):
                            raise RuntimeError(
                                "SPO-RS overlap found a reward future shared "
                                "by multiple rollout records")
                        spo_rs_overlap_state = (parallel_trainer.begin_spo_rs_overlap(cfg, step_idx))
                        print(
                            f"[step {step_idx}] {history_label} "
                            "evaluation/training overlap started; "
                            f"prepared {len(staged_spo_rs_examples)} exact "
                            "training inputs; "
                            f"{pending_before}/{len(all_futs)} CPU "
                            "evaluations were pending before trainer restore",
                            flush=True,
                        )
                        if hybrid_mode:
                            examples_by_group = {}
                            for example in staged_spo_rs_examples:
                                examples_by_group.setdefault(example["group_id"], []).append(example)
                            while examples_by_group:
                                ready_groups = [g for g in examples_by_group
                                                if all(f.done() for f in reward_futures[g])]
                                for group_id in ready_groups:
                                    ready_examples = examples_by_group.pop(group_id)
                                    _complete_spo_rs_overlap_examples(
                                        ready_examples, spo_rs_tracker,
                                        sibling_futures=reward_futures[group_id],
                                        ttt_weight=cfg.hybrid_ttt_weight,
                                        spo_weight=cfg.hybrid_spo_weight)
                                    parallel_trainer.queue_spo_rs_overlap(ready_examples)
                                if not ready_groups:
                                    wait([f for g in examples_by_group for f in reward_futures[g] if not f.done()],
                                         return_when=FIRST_COMPLETED)
                        else:
                            while pending_by_future:
                                ready_futures = [future for future in pending_by_future if future.done()]
                                if not ready_futures:
                                    wait(tuple(pending_by_future), return_when=FIRST_COMPLETED)
                                    continue
                                ready_examples = [pending_by_future.pop(future) for future in ready_futures]
                                _complete_spo_rs_overlap_examples(ready_examples, spo_rs_tracker)
                                parallel_trainer.queue_spo_rs_overlap(ready_examples)

                        if spo_rs_overlap_state is not None:
                            parallel_trainer.seal_spo_rs_overlap()
                        print(
                            f"[step {step_idx}] {history_label} overlap queued "
                            f"{spo_rs_overlap_state['queued_examples']} "
                            "examples to the shared "
                            f"{parallel_trainer.world_size}-GPU work queue "
                            f"(old logprobs "
                            f"{spo_rs_overlap_state['supplied_old']}/"
                            f"{spo_rs_overlap_state['queued_examples']}); "
                            "backward continues while rollout artifacts are "
                            "processed; optimizer step deferred",
                            flush=True,
                        )
                except Exception as error:
                    if spo_rs_overlap_state is not None:
                        parallel_trainer.finish_spo_rs_overlap(
                            cfg, step_idx,
                            spo_rs_overlap_state["queued_examples"],
                            apply_update=False)
                    elif not parallel_trainer._workers_idle:
                        raise
                    for responses in group_responses.values():
                        for record in responses:
                            record.pop("_spo_rs_overlap_example", None)
                    spo_rs_overlap_state = None
                    print(
                        f"[warn] {history_label} evaluation/training overlap was "
                        f"disabled for this step ({type(error).__name__}: "
                        f"{error}); falling back to the unchanged "
                        "post-evaluation update",
                        flush=True,
                    )

            # Entropic advantages need every reward in their own group, not the
            # rewards of later groups. Stream each completed group into the
            # process-per-GPU trainer while the remaining sandbox jobs continue
            # on CPU. All forwards see the frozen pre-update policy; gradients
            # are accumulated only, globally normalized after the final group,
            # and applied below the rollout persistence barrier.
            entropic_overlap_eligible = bool(
                active_advantage_mode == "entropic"
                and training_enabled
                and not deferred_rollouts
                and not evaluation_trainer_offloaded
            )
            if entropic_overlap_eligible:
                pending_groups = {int(group_id) for group_id in range(num_groups) if reward_futures[group_id]}
                overlap_normalization_examples = sum(
                    bool(record.get("token_ids"))
                    for responses in group_responses.values()
                    for record in responses)
                overlap_prompt_ids = {}
                try:
                    while pending_groups:
                        ready_groups = [
                            group_id for group_id in sorted(pending_groups)
                            if all(future.done() for future in reward_futures[group_id])
                        ]
                        if not ready_groups:
                            unfinished = [
                                future for group_id in pending_groups
                                for future in reward_futures[group_id]
                                if not future.done()
                            ]
                            if not unfinished:
                                continue
                            wait(unfinished, return_when=FIRST_COMPLETED)
                            continue

                        for group_id in ready_groups:
                            prepared = _prepare_entropic_overlap_group(
                                tokenizer, group_id,
                                group_responses[group_id],
                                reward_futures[group_id], prompt_jobs,
                                overlap_prompt_ids, num_groups)
                            if prepared:
                                if entropic_overlap_state is None:
                                    pending_before = sum(not future.done() for future in all_futs)
                                    entropic_overlap_state = (
                                        parallel_trainer.
                                        begin_entropic_overlap(cfg, step_idx, overlap_normalization_examples))
                                    print(
                                        f"[step {step_idx}] entropic "
                                        "evaluation/training overlap started; "
                                        f"{pending_before}/{len(all_futs)} CPU "
                                        "evaluations still pending",
                                        flush=True,
                                    )
                                queued = (parallel_trainer.queue_entropic_overlap(prepared))
                                remaining = sum(not future.done() for future in all_futs)
                                print(
                                    f"[step {step_idx}] entropic overlap: "
                                    f"group {group_id} reward-complete; queued "
                                    f"{queued} examples to all GPUs; "
                                    f"{remaining}/{len(all_futs)} CPU "
                                    "evaluations still pending",
                                    flush=True,
                                )
                            pending_groups.remove(group_id)

                    if entropic_overlap_state is not None:
                        parallel_trainer.seal_entropic_overlap()
                        print(
                            f"[step {step_idx}] entropic overlap queued "
                            f"{entropic_overlap_state['queued_examples']} "
                            "examples "
                            f"(old logprobs "
                            f"{entropic_overlap_state['supplied_old']}/"
                            f"{entropic_overlap_state['queued_examples']}, "
                            f"reference "
                            f"{entropic_overlap_state['supplied_reference']}/"
                            f"{entropic_overlap_state['queued_examples']}); "
                            "backward continues while rollout "
                            "artifacts are processed; optimizer step deferred",
                            flush=True,
                        )
                except Exception as error:
                    if entropic_overlap_state is not None:
                        parallel_trainer.finish_entropic_overlap(
                            cfg, step_idx,
                            entropic_overlap_state["queued_examples"],
                            apply_update=False)
                    elif not parallel_trainer._workers_idle:
                        # A restore/start failure can leave the distributed
                        # process group unusable; an ordinary retry on that same
                        # trainer would deadlock rather than provide a fallback.
                        raise
                    for responses in group_responses.values():
                        for record in responses:
                            record.pop("_entropic_overlap_example", None)
                    entropic_overlap_state = None
                    print(
                        f"[warn] entropic evaluation/training overlap was "
                        f"disabled for this step ({type(error).__name__}: "
                        f"{error}); falling back to the unchanged "
                        "post-evaluation update",
                        flush=True,
                    )

            # If training finished first, wait only for the remaining reward
            # tail. If rewards finished first, this returns immediately.  Do
            # not consume exceptions here: the aligned per-group result() call
            # below remains the single place that reports reward failures.
            wait(all_futs)
            with eval_progress_condition:
                eval_progress_condition.wait_for(lambda: eval_progress_completed >= len(all_futs))
        finally:
            if eval_bar is not None:
                eval_bar.close()
                eval_bar_closed = True
    finally:
        reward_pool.shutdown(wait=True)
        if cpu_eval_progress is not None:
            cpu_eval_progress.close()
        rollout_io_pool.shutdown(wait=True)
        if eval_bar is not None and not eval_bar_closed:
            eval_bar.close()
        if evaluation_trainer_offloaded:
            _restore_after_gpu_evaluation(backend, optimizer, step_idx)

    # Surface an unlikely disk failure before constructing the checkpoint.
    # In the normal path every write completed concurrently with generation or
    # evaluation, so this is only a non-blocking correctness barrier.
    for save_future in rollout_io_futures:
        save_future.result()
    if critic_mode:
        rows = []
        for group in range(num_groups):
            for record, future in zip(group_responses[group], reward_futures[group], strict=True):
                result = future.result()
                rows.append(critic_step.rollout_row({**record["_pending_meta"],
                                                    "valid": result.valid, "raw_score": result.raw_score}))
        critic_step.prepare_batch(exp_dir, step_idx, critic_context, rows)
        if isolated_batch is None:
            park_trainer()
        judged = critic_step.score_batch(
            exp_dir, step_idx, cfg, getattr(gen_pool, "host_memory_admission", None))
    if history_mode and not training_enabled:
        # With policy optimization disabled, consecutive policies are exactly
        # identical even though the selected parents may be new.
        spo_rs_divergence = 0.0
        spo_rs_context_divergences = {int(spec["context_id"]): 0.0 for spec in group_specs}

    # ----- SCORE + ADVANTAGE + SAVE + COLLECT TRAINING EXAMPLES -----
    step_valid_count = 0
    step_rollout_count = 0
    step_code_failure_count = 0
    prompt_ids_by_job = {}

    # All historical labels use the history entering the step. The invariant
    # variants also need the complete current batch for the cap and hybrid RMS.
    spo_rs_prepared = {}
    invariant_batch = None
    if history_mode:
        samples = []
        for group_spec in group_specs:
            group_id = int(group_spec["group_id"])
            responses = group_responses[group_id]
            futures = reward_futures[group_id]
            if len(responses) != len(futures):
                raise RuntimeError(
                    f"SPO-RS group {group_id} has {len(responses)} responses "
                    f"but {len(futures)} reward results")
            for rollout_index, record in enumerate(responses):
                job_idx = int(record["job_idx"])
                result = futures[rollout_index].result()
                samples.append({
                    "group": group_id,
                    "rollout": rollout_index,
                    "job": job_idx,
                    "record": record,
                    "reward": float(result.reward),
                    "valid": bool(result.valid),
                })
        if samples:
            if invariant_mode:
                invariant_batch = _invariant_batch(cfg, spo_rs_tracker, samples)
                history_input = invariant_batch
                normalized = invariant_batch.historical
            else:
                history_input = spo_rs_tracker.transformed_rewards([sample["reward"] for sample in samples])
                normalized = spo_rs_tracker.normalized_advantages(history_input)
                if normalized is None:
                    raise RuntimeError("SPO-RS lost its pre-step historical baseline")
            update = spo_rs_tracker.update(
                history_input,
                divergence=spo_rs_divergence,
                policy_adapter=adapter_path,
                step=step_idx,
            )
            update.update({
                "groups": sorted({sample["group"] for sample in samples}),
                "prompt_jobs": sorted({sample["job"] for sample in samples}),
                "contexts": sorted({int(sample["record"]["context_id"]) for sample in samples}),
                "context_divergences": spo_rs_context_divergences,
                "missing_policy_scores": int(spo_rs_missing_policy_scores),
            })
            spo_rs_updates.append(update)

            for sample_index, sample in enumerate(samples):
                rollout_key = (sample["group"], sample["rollout"])
                info = {**update, "used_for_policy_update": True}
                if invariant_mode:
                    effective_rewards = invariant_batch.effective_rewards
                    info["effective_reward"] = (
                        None if effective_rewards is None else float(effective_rewards[sample_index]))
                else:
                    info["transformed_reward"] = float(history_input[sample_index])
                spo_rs_prepared[rollout_key] = {
                    "advantage": float(normalized[sample_index]),
                    "trainable": True,
                    "info": info,
                }

            divergence_label = ("missing" if update["divergence"] is None else f"{update['divergence']:.6f}")
            if invariant_mode:
                print(f"[step {step_idx}] {active_advantage_mode} history: "
                      f"{update['history_size_before']}->{update['history_size_after']} rewards, "
                      f"theta={update['theta']:.6f} tilt_KL={update['tilt_kl']:.6f} "
                      f"D={divergence_label} rho={update['rho']:.6f}", flush=True)
            else:
                print(f"[step {step_idx}] SPO-RS global tracker: "
                      f"v={update['value_before']:.9f}->"
                      f"{update['value_after']:.9f} "
                      f"M={update['group_size']} D={divergence_label} "
                      f"rho={update['rho']:.6f} "
                      f"eta={update['eta']:.6f}", flush=True)

    for group_spec in group_specs:
        g = int(group_spec["group_id"])
        parent_group = int(group_spec["parent_group"])
        context_id = int(group_spec["context_id"])
        parent = parents[parent_group]
        responses = group_responses[g]          # streamed mutable rollout records
        futs = reward_futures[g]                 # aligned RewardResult futures

        rewards = []
        codes = []
        valids = []
        outs = []        # list of RewardResult
        for r_idx, record in enumerate(responses):
            res = futs[r_idx].result()           # already computed (or finishes now)
            rewards.append(res.reward)
            codes.append(res.code or '')
            valids.append(res.valid)
            outs.append(res)

        rewards_np = np.array(rewards, dtype=np.float64)
        if rewards_np.size == 0:
            print(f"  group {g}: no rollouts returned; no update for this group")
            continue
        adv_mode = active_advantage_mode
        spo_rs_advantages = None
        spo_rs_v_t = None
        spo_rs_rollout_info = [None] * len(responses)
        spo_rs_trainable = [True] * len(responses)
        if history_mode:
            prepared = [spo_rs_prepared[(g, r_idx)] for r_idx in range(len(responses))]
            spo_rs_advantages = np.asarray([item["advantage"] for item in prepared], dtype=np.float64)
            spo_rs_trainable = [item["trainable"] for item in prepared]
            spo_rs_rollout_info = [item["info"] for item in prepared]
            spo_rs_v_t = spo_rs_rollout_info[0].get("value_before")
        if invariant_mode:
            group_size = int(cfg.group_size)
            rows = slice(g * group_size, (g + 1) * group_size)
            advantages = invariant_batch.applied[rows]
            ttt_advantages = invariant_batch.sibling[rows]
            ttt_adaptive_beta = invariant_batch.sibling_betas[g]
        elif critic_mode:
            advantages = np.asarray([judged["advantages"][record["_pending_meta"]["node_id"]]
                                     for record in responses], dtype=np.float64)
        elif hybrid_mode:
            advantages, ttt_adaptive_beta, ttt_advantages = hybrid_advantages(
                rewards_np, spo_rs_advantages, ttt_weight=cfg.hybrid_ttt_weight,
                spo_weight=cfg.hybrid_spo_weight, return_ttt=True)
        else:
            ttt_advantages, ttt_adaptive_beta = compute_group_advantages(rewards_np)
            advantages = spo_rs_advantages if history_mode else ttt_advantages
        if invariant_mode:
            adv_scale = invariant_batch.info["beta"]
        else:
            adv_scale = (None if critic_mode else
                         float(cfg.spo_rs_beta) if history_mode else ttt_adaptive_beta)
        ttt_sibling_max = float(rewards_np.max())
        constant = (ttt_sibling_max - float(rewards_np.min()) < 1e-12)

        step_valid_count += sum(valids)
        step_rollout_count += len(valids)
        step_code_failure_count += sum(is_code_failure(res) for res in outs)

        _print_group_rollout_stats(g, rewards_np, valids,
                                   [len(record["token_ids"]) for record in responses], beta=adv_scale)

        # Save every rollout (response + meta) to disk for debugging
        for r_idx, record in enumerate(responses):
            text = record["text"]
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            res = outs[r_idx]
            job = prompt_jobs[job_idx]
            artifact_rollout_index = int(record.get("rollout_index", r_idx))
            if artifact_rollout_index != r_idx:
                raise RuntimeError(
                    f"rollout persistence order changed for group {g}: "
                    f"generated index {artifact_rollout_index}, final index "
                    f"{r_idx}")
            # Allocate a durable ID even for invalid/duplicate candidates. A
            # valid candidate uses this exact State object in sampler.update,
            # so a child selected in a later step links back to this rollout.
            child = State.make(
                timestep=step_idx,
                value=rewards[r_idx],
                code=res.code or "",
                raw_score=res.raw_score,
                construction=res.construction,
                observation=res.stdout,
                poisson=res.poisson,
            )
            if critic_mode:
                child.id = record["_pending_meta"]["node_id"]
            archive_eligible = bool(valids[r_idx] and codes[r_idx])
            if archive_eligible:
                all_children.append((child, parent))
                if ac1_archive:
                    archive_order[child.id] = record["evaluation_order"]
            spo_rs_advantage = spo_rs_advantages[r_idx] if history_mode else None
            meta = {
                "output_format_issue": record["output_format_issue"],
                "step": step_idx,
                "group": g,
                "parent_group": parent_group,
                "rollout": r_idx,
                "node_id": child.id,
                **parent_fields[parent_group],
                "reward": float(rewards[r_idx]),
                "raw_score": (float(res.raw_score) if res.raw_score is not None else None),
                "poisson": res.poisson,
                "solver": "coder",
                "model_name": cfg.coder_model_name,
                "valid": bool(valids[r_idx]),
                "parsed": bool(res.parsed),
                "ran": bool(res.ran),
                "msg": res.msg,
                "failure_kind": res.failure_kind,
                **({"advantage": float(advantages[r_idx])} if critic_mode else _rollout_advantage_fields(
                    advantages[r_idx], ttt_advantage=ttt_advantages[r_idx],
                    spo_rs_advantage=spo_rs_advantage,
                    ttt_sibling_max=ttt_sibling_max, ttt_adaptive_beta=ttt_adaptive_beta,
                    parent_raw_score=parent.raw_score, spo_rs_v_t=spo_rs_v_t)),
                "beta": adv_scale,
                "advantage_mode": adv_mode,
                "advantage_scale": adv_scale if adv_scale is not None and math.isfinite(adv_scale) else None,
                "n_response_tokens": len(token_ids),
                "coder_reasoning_effort": job.get("coder_reasoning_effort"),
                "sandbox_stdout": _saved_evaluator_output(res.stdout),
                "archive_eligible": archive_eligible,
                # The solution itself, and the one it started from. Neither is
                # recoverable afterwards: `construction` lives only in the
                # in-memory sampler State, and a mid-run rollout's parent array
                # is gone by the time anyone wants to plot it. Saving the result
                # means reproducing a figure needs no re-execution at all, which
                # also sidesteps programs that are stochastic or wall-clock
                # bounded and therefore cannot replay identically.
                "construction": (_as_float_list(getattr(res, "construction", None), cfg.max_saved_construction)
                                 if save_ctor else None),
                "parent_construction": (_as_float_list(
                    getattr(parent, "construction", None),
                    cfg.max_saved_construction) if save_ctor else None),
                "seed": int(cfg.seed),
            }
            if ac1_archive:
                meta.update(_ac1_archive_metadata(res, record["evaluation_order"]))
            if history_mode:
                meta["spo_rs"] = spo_rs_rollout_info[r_idx]
            if invariant_mode:
                meta["status"] = "evaluated"
            if hybrid_mode:
                meta["hybrid_sibling_beta"] = ttt_adaptive_beta
                meta["hybrid_spo_adjustment"] = float(cfg.hybrid_spo_weight * spo_rs_advantage)
            save_rollout(exp_dir, step_idx, g, artifact_rollout_index,
                         text, meta,
                         prompt_text=job["prompt_text"],
                         artifacts_already_saved=True)
            advantage_plot_rollouts.append(advantage_plot_rollout(meta))
            saved_rollouts += 1

        # Entropic training skips constant groups. SPO-RS compares every
        # rollout with its historical baseline, so tied groups can still learn.
        if constant and not replayable_update:
            continue

        for r_idx, (record, adv) in enumerate(zip(responses, advantages)):
            token_ids = record["token_ids"]
            job_idx = record["job_idx"]
            if not training_enabled:
                continue
            if history_mode and not spo_rs_trainable[r_idx]:
                continue
            if len(token_ids) == 0:
                continue
            prepared = (record.get("_entropic_overlap_example") or record.get("_spo_rs_overlap_example"))
            res = outs[r_idx]
            if prepared is not None:
                prepared_advantage = prepared.get("advantage")
                # Streamed and full-batch reductions can differ by float64 roundoff.
                if (prepared_advantage is not None and not math.isclose(
                        float(prepared_advantage), float(adv),
                        rel_tol=1e-12, abs_tol=1e-12)):
                    prepared["_overlap_label_mismatch"] = True
                prepared.update({
                    "advantage": float(adv),
                    "reward_constant": constant,
                    "sample_weight": 1.0 / (num_groups * len(responses)),
                })
                all_examples.append(prepared)
                continue
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.to(model.device)
            response_ids = torch.tensor([token_ids], device=model.device)
            behavior_values = record["behavior_logprobs"]
            reference_values = record["reference_logprobs"]
            behavior_logprobs = (
                torch.tensor(behavior_values, dtype=torch.float32, device=model.device)
                if behavior_values is not None
                and len(behavior_values) == len(token_ids) else None)
            reference_logprobs = (
                torch.tensor(reference_values, dtype=torch.float32, device=model.device)
                if reference_values is not None
                and len(reference_values) == len(token_ids) else None)
            example = {
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": response_ids,
                "advantage": float(adv),
                "behavior_logprobs": behavior_logprobs,
                "reference_logprobs": reference_logprobs,
                "reward_constant": constant,
                "group_id": g,
                "rollout_index": r_idx,
                "prompt_job_id": job_idx,
                "sample_weight": 1.0 / (num_groups * len(responses)),
            }
            if hybrid_loss:
                example["reference_logprob_mean"] = hybrid_reference_mean
                example["_overlap_key"] = (g, r_idx)
            all_examples.append(example)

    # Persistence barrier: every response/prompt/meta file is on disk before the
    # optimizer can mutate the adapter. Streamed entropic and SPO-RS training
    # may already have accumulated exact gradients during
    # CPU evaluation, but their single update is deliberately deferred until
    # this barrier.
    # stepXX.summary.json remains the completion marker and is written only
    # after the adapter/checkpoint.
    save_suffix = ("with training disabled" if not training_enabled else "before adapter update")
    setting_log_only(
        f'[step {step_idx}] saved {saved_rollouts} rollout .txt/.meta.json pairs {save_suffix}', flush=True)

    valid_fraction = (step_valid_count / step_rollout_count if step_rollout_count else 0.0)
    code_valid_fraction = (1.0 - step_code_failure_count / step_rollout_count if step_rollout_count else 1.0)

    # Preserve the standard objectives' constant-reward-group filtering.
    if not replayable_update:
        all_examples = [ex for ex in all_examples if not ex["reward_constant"]]

    if clipped_policy_mode and cfg.generation_backend == "vllm":
        # SPO-RS requires the frozen behavior probability. A malformed vLLM
        # payload cannot be reconstructed safely after the adapter changes and
        # must never force a full-context HF cache pass. Rollouts remain saved;
        # only an unusable training record is omitted.
        before = len(all_examples)
        all_examples = [
            example for example in all_examples
            if _valid_example_token_logprobs(example, "behavior_logprobs")
        ]
        unusable = before - len(all_examples)
        if unusable:
            print(f"[step {step_idx}] vLLM omitted behavior logprobs for "
                  f"{unusable} rollout(s); saved them but excluded only those "
                  "records from the clipped-policy update", flush=True)

    if replayable_update and all_examples:
        trajectory_weight = 1.0 / len(all_examples)
        for example in all_examples:
            example["sample_weight"] = trajectory_weight


    rollout_time = time.time() - rollout_t0
    if isolated_batch is not None:
        rollout_time += isolated_batch["scoring"]["rollout_seconds"] + isolated_batch["evaluation_seconds"]
    _print_rollout_totals(
        step_idx, step_valid_count, step_rollout_count,
        len(all_examples) if training_enabled else "disabled", len(all_children), rollout_time,
        overlapped=(entropic_overlap_state is not None or spo_rs_overlap_state is not None),
    )

    # Keep the previous raw optimum so the status line can distinguish a new
    # run best from a step that merely repeats the best-ever value.
    previous_best_raw = sampler.best_raw_state(maximize=bool(problem.maximize))
    previous_best_value = (
        float(previous_best_raw.raw_score) if previous_best_raw is not None else None)

    # Update archive
    if ac1_archive:
        all_children.sort(key=lambda entry: archive_order[entry[0].id])
    sampler.update(all_children)

    # Keep the best valid candidate produced by this exact step separate from
    # the cumulative best. ``all_children`` still contains candidates removed
    # later by deduplication or archive caps, so the per-step result reflects
    # what was actually evaluated rather than only what survived the archive.
    step_states = [child for child, _parent in all_children]
    step_best_result = _step_best_candidate(step_states, problem.maximize)
    step_best_raw_score = (
        float(step_best_result.raw_score)
        if step_best_result is not None and step_best_result.raw_score is not None else None)

    best_seen_result = _print_best_ever_raw(sampler, step_idx, cfg, problem, previous_best_value)

    best_snapshot_path = save_best_state_snapshot(
        exp_dir, step_idx, best_seen_result,
        metric_name=str(problem.metric_name), maximize=bool(problem.maximize),
        construction=(
            _as_float_list(getattr(best_seen_result, "construction", None), cfg.max_saved_construction)
            if (best_seen_result is not None and getattr(problem, "saves_construction", False))
            else None),
    )
    if evaluation_complete is not None:
        evaluation_complete(step_idx)

    if entropic_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(math.isfinite(float(example["advantage"])) for example in all_examples)
        advantages_match = not any(example.get("_overlap_label_mismatch", False) for example in all_examples)
        expected_keys = entropic_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys or not advantages_complete or not advantages_match):
            parallel_trainer.finish_entropic_overlap(
                cfg, step_idx,
                entropic_overlap_state["queued_examples"],
                apply_update=False)
            print(
                f"[warn] discarded streamed entropic gradients because the "
                f"final training set changed ({len(actual_keys)}/"
                f"{len(expected_keys)} records); using the unchanged "
                "post-evaluation update",
                flush=True,
            )
            entropic_overlap_state = None

    if spo_rs_overlap_state is not None:
        actual_keys = {
            example.get("_overlap_key") for example in all_examples
            if example.get("_overlap_key") is not None
        }
        advantages_complete = all(
            example.get("advantage") is not None
            and math.isfinite(float(example["advantage"]))
            for example in all_examples)
        advantages_match = not any(example.get("_overlap_label_mismatch", False) for example in all_examples)
        expected_keys = spo_rs_overlap_state["scheduled_keys"]
        if (actual_keys != expected_keys or not advantages_complete or not advantages_match):
            parallel_trainer.finish_spo_rs_overlap(
                cfg, step_idx,
                spo_rs_overlap_state["queued_examples"],
                apply_update=False)
            print(
                f"[warn] discarded streamed SPO-RS gradients because the "
                f"final training set changed ({len(actual_keys)}/"
                f"{len(expected_keys)} records); using the unchanged "
                "post-evaluation update",
                flush=True,
            )
            spo_rs_overlap_state = None

    step_stats = {
        "best_seen_snapshot": (str(Path(best_snapshot_path).relative_to(exp_dir))
                               if best_snapshot_path is not None else None),
        "planned_rollouts": total_rollouts,
        "advantage_plot_rollouts": advantage_plot_rollouts,
        "result_metric_name": str(problem.metric_name),
        "result_maximize": bool(problem.maximize),
        "step_best_raw_score": step_best_raw_score,
        "step_best_reward": (
            float(step_best_result.value)
            if (step_best_result is not None and step_best_result.value is not None) else None),
        "best_seen_raw_score": (
            float(best_seen_result.raw_score)
            if (best_seen_result is not None and best_seen_result.raw_score is not None)
            else None),
        "best_seen_reward": (
            float(best_seen_result.value)
            if (best_seen_result is not None and best_seen_result.value is not None)
            else None),
        "best_seen_step": (int(best_seen_result.timestep) if best_seen_result is not None else None),
        "valid_fraction": float(valid_fraction),
        "code_valid_fraction": float(code_valid_fraction),
        **evaluation_stats,
    }
    if history_mode:
        step_stats["spo_rs_updates"] = spo_rs_updates
        step_stats["spo_rs_tracker_size"] = (0 if spo_rs_tracker is None else len(spo_rs_tracker))
        step_stats["spo_rs_divergence"] = (None if not math.isfinite(spo_rs_divergence) else float(spo_rs_divergence))
        step_stats["spo_rs_context_divergences"] = (spo_rs_context_divergences)
        step_stats["spo_rs_missing_policy_scores"] = int(spo_rs_missing_policy_scores)
        step_stats["spo_rs_sampling_adapter"] = Path(adapter_path).name
        step_stats["spo_rs_policy_updated"] = False
    if hybrid_loss:
        step_stats["hybrid_reference_kl_mean"] = hybrid_reference_mean
    if hybrid_mode:
        step_stats["hybrid_ttt_weight"] = cfg.hybrid_ttt_weight
        step_stats["hybrid_spo_weight"] = cfg.hybrid_spo_weight
    if critic_mode:
        step_stats["critic_attempts"] = judged["attempts"]
        step_stats["critic_policy_updated"] = False


    if not training_enabled:
        step_stats["training_disabled"] = True
        step_stats["training_seconds"] = 0.0
        disabled_reason = str(getattr(cfg, "_training_disabled_reason", "--no-train"))
        print(f"[step {step_idx}] training disabled ({disabled_reason}); "
              "skipping training-only logprobs, backward, and optimizer update",
              flush=True)
        return step_stats

    if not all_examples:
        print(f"[step {step_idx}] no usable training examples")
        return step_stats

    if replayable_update:
        if spo_rs_overlap_state is not None:
            step_stats.update(parallel_trainer.finish_spo_rs_overlap(
                cfg, step_idx, len(all_examples), apply_update=True))
            spo_rs_overlap_state = None
        else:
            step_stats.update(parallel_trainer.train_spo_rs(all_examples, cfg, step_idx))
        step_stats["critic_policy_updated" if critic_mode else "spo_rs_policy_updated"] = True
        return step_stats

    # ----- TRAIN STEP -----
    print(f'[step {step_idx}] starting adapter training; rollout artifacts are already on disk', flush=True)
    if entropic_overlap_state is not None:
        step_stats.update(parallel_trainer.finish_entropic_overlap(
            cfg, step_idx, len(all_examples), apply_update=True))
        entropic_overlap_state = None
    else:
        step_stats.update(parallel_trainer.train_policy(all_examples, cfg, step_idx))
    best = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
    if best is not None:
        raw = f" raw={best.raw_score:.9f}" if best.raw_score is not None else ""
        print(f"[step {step_idx}] best so far: value={best.value:.9f}{raw}  "
              f"(step total {time.time() - step_t0:.1f}s, "
              f"archive={sampler.archive_size()})")
    return step_stats


# ======================================================================
# Main
# ======================================================================
def main():
    with ExitStack() as trainer_startup:
        return _main(trainer_startup)


def _main(trainer_startup):
    _install_console_timestamps()
    terminal_log = _install_terminal_log()
    from builtins import print as console_print
    from diagnostics.startup_logs import StartupLog, print_dashboard
    startup_log = StartupLog()
    with startup_log.capture():
        cfg, merged = load_config()
        from runtime.gpu_runtime import GPU_PROBLEM_NAMES
        is_trimul = (str(cfg.problem).lower() in GPU_PROBLEM_NAMES
                     and str(getattr(cfg, "problem_type", "trimul")).lower() == "trimul")
        if is_trimul and not os.environ.get("TTT_SUPERVISOR_EVAL_FILE"):
            script = Path(__file__).resolve()
            supervisor = script.parent / "runtime" / "run_supervisor.py"
            os.execv(sys.executable, [sys.executable, str(supervisor), "--",
                                      sys.executable, str(script), *sys.argv[1:]])
        # This must precede every import path that can initialize CUDA. Worker
        # and evaluation children set their own physical groups before import.
        _pin_training_process(cfg.training_gpu_ids)
        if cfg.evaluation_gpu_id is not None:
            os.environ["TTT_EVALUATION_GPU_ID"] = str(cfg.evaluation_gpu_id)

        # Save the reusable base configuration before runtime-only adjustments.
        from experiment_io import (append_step_result, make_experiment_dir,
                                   restart_changed_rollout_step,
                                   save_final_summary, save_step_summary)
        resume_dir = merged.pop("_resume_dir", None)
        restart_step = 0
        if resume_dir:
            bind_setting_log(Path(resume_dir) / "setting.log")
            # Read only the cursor via memory-mapped CPU storage before model
            # loading, so stale overflow recovery cannot consume a new batch.
            checkpoint = _load_training_checkpoint(resume_dir, mmap=True)
            if checkpoint is not None:
                restart_step = int(checkpoint["next_step"])
            else:
                try:
                    restart_step, _ = _legacy_resume_info(resume_dir)
                except FileNotFoundError:
                    restart_step = 0
            del checkpoint
            if restart_changed_rollout_step(cfg, resume_dir, restart_step):
                # Re-resolve layout without the old step's overflow marker.
                cfg, merged = load_config()
                merged.pop("_resume_dir", None)
        exp_dir = make_experiment_dir(cfg, resume_dir=resume_dir, config_dict=merged)
        supervisor_run_file = os.environ.get("TTT_SUPERVISOR_RUN_FILE")
        if supervisor_run_file:
            run_file = Path(supervisor_run_file)
            run_file_tmp = run_file.with_suffix(".tmp")
            run_file_tmp.write_text(str(Path(exp_dir).resolve()), encoding="utf-8")
            run_file_tmp.replace(run_file)
        setting_log_path = Path(exp_dir).resolve() / "setting.log"
        startup_log.bind(setting_log_path)
        if not resume_dir:
            bind_setting_log(setting_log_path)
    # Keep the full resolved settings in the file; show their grouped dashboard
    # once below. Restore normal print before loading/training begins.
    print = startup_log.print
    terminal_log_path = str(Path(exp_dir).resolve() / "temirnal.log")
    terminal_log.bind(terminal_log_path)
    if is_trimul and resume_dir and restart_step < cfg.num_steps:
        from runtime.run_supervisor import request_evaluation
        try:
            trimul_evaluation.completed_batch(exp_dir, restart_step)
        except trimul_evaluation.EvaluationHandoff as handoff:
            request_evaluation(handoff.path)
            return
    if cfg.advantage_mode == LLM_CRITIC_MODE and resume_dir and restart_step < cfg.num_steps:
        judged = critic_step.load_batch(exp_dir, restart_step)
        if judged is not None and "advantages" not in judged:
            critic_step.score_batch(exp_dir, restart_step, cfg)
    if cfg.generation_backend == "vllm":
        from runtime.host_memory import validate_memory_handoff_runtime
        runtime = validate_memory_handoff_runtime(__file__)
        with (Path(exp_dir) / "runtime.json").open("w", encoding="utf-8") as handle:
            json.dump(runtime, handle, indent=2)
        runtime_message = (f"[runtime] checkout={runtime['checkout']}; memory handoff: {runtime['memory_handoff']}")
        setting_log_only(runtime_message, flush=True)
    print(f"[logs] terminal output: {terminal_log_path}", flush=True)
    result_log_path = Path(exp_dir).resolve() / "result.txt"
    result_log_path.touch(exist_ok=True)
    print(f"[logs] step results: {result_log_path}", flush=True)
    vllm_log_path = None
    dependency_log_path = None
    if cfg.generation_backend == "vllm":
        vllm_log_path = str((Path(exp_dir).resolve() / "vllm.log"))
        with open(vllm_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(f'\n=== TTT vLLM log parent_pid={os.getpid()} run_dir={Path(exp_dir).resolve()} ===\n')
        print(f"[logs] vLLM output: {vllm_log_path}", flush=True)
        dependency_log_path = vllm_log_path
    else:
        dependency_log_path = str(Path(exp_dir).resolve() / "dependency_warnings.log")
        with open(dependency_log_path, "a", encoding="utf-8") as log_handle:
            log_handle.write(
                f"\n=== TTT dependency warnings parent_pid={os.getpid()} "
                f"run_dir={Path(exp_dir).resolve()} ===\n")
        print(f"[logs] dependency warnings: {dependency_log_path}", flush=True)

    # One effective seed for every generation stream. None => not seeded, which
    # is the original behaviour. Set once here and threaded through unchanged.
    run_seed = cfg.seed if cfg.deterministic else None
    print(f"[init] deterministic = {cfg.deterministic}" + (f" (seed {cfg.seed})" if cfg.deterministic else ""))

    if not cfg.no_train:
        cfg.training_replica_device = 0

    # Build the problem from the merged config (the registry reads problem-only
    # knobs like num_circles / problem_type / budget_s / score_scale from here).
    from problems.registry import get_problem
    problem_config = merged
    if cfg.isolate_eval and cfg.problem != "ac1":
        # Prompts must describe the sandbox's real allocation.
        problem_config = {**merged, "eval_cpus": 1}
    problem = get_problem(cfg.problem, problem_config)
    if getattr(problem, "problem_type", "") == "trimul":
        from problems.gpu_mode import verify_benchmark_environment
        for gpu_id in problem.kernel_gpu_ids:
            verify_benchmark_environment(
                problem.kernel_python, problem.gpu_type, problem.triton_version,
                gpu_id=gpu_id)

    startup_log.begin_summary()
    print("=" * 70)
    print("Single-Model Discovery")
    print("=" * 70)
    problem_type = getattr(cfg, "problem_type", "")
    print(f"Problem:            {cfg.problem}" + (f" ({problem_type})" if problem_type else ""))
    print(f"Entrypoint:         {getattr(problem, 'entrypoint', '?')}")
    print(f"Metric:             {getattr(problem, 'metric_name', '?')} "
          f"({'maximize' if getattr(problem, 'maximize', True) else 'minimize'})")
    print(f"Coder model:        {cfg.coder_model_name}")
    print(f"Training backend:   {cfg.backend}")
    print(f"Generation backend: {cfg.generation_backend}")
    print(f"Hardware profile:   {cfg.hardware_profile}")
    print(f"Training GPUs:      physical {cfg.training_gpu_ids}")
    print("Training layout:    " + ("disabled (--no-train)" if cfg.no_train else "process-distributed LoRA"))
    if not cfg.no_train:
        print("Training scheduler: adaptive process-per-GPU")
    print(f"Generation GPUs:    {cfg.gpu_ids or 'in-process'}")
    evaluation_gpus = (cfg.gpu_ids if getattr(problem, "problem_type", "") == "trimul"
                       else cfg.evaluation_gpu_id)
    print(f"Evaluation GPU:     {evaluation_gpus if evaluation_gpus is not None else 'none'}")
    if cfg.isolate_eval and cfg.problem == "ac1":
        evaluation_mode = "isolated, one sandbox per exclusive CPU pair"
    elif cfg.isolate_eval:
        evaluation_mode = f"isolated, {cfg.reward_workers} processes per logical CPU"
    else:
        evaluation_mode = "default"
    print(f"CPU evaluation:    {evaluation_mode}")
    if cfg.generation_backend == "vllm":
        print(f"vLLM parallelism:   TP={cfg.vllm_tensor_parallel_size or 'auto'} "
              f"PP={cfg.vllm_pipeline_parallel_size}")
    print(f"Configured target:  {cfg.target}")
    print(f"Print/SOTA target:  {getattr(cfg, 'target_print', cfg.target)}")
    print(f"Steps:              {cfg.num_steps}")
    selection = cfg.parent_selection
    selection_detail = f"c={cfg.puct_c}"
    print(f"Search selection:   {selection.upper()} ({selection_detail})")
    configured_advantage_mode = getattr(cfg, "advantage_mode", "entropic")
    hybrid_mode = configured_advantage_mode in ("hybrid", HYBRID_INVARIANT_MODE)
    invariant_mode = configured_advantage_mode in INVARIANT_ADVANTAGE_MODES
    critic_mode = configured_advantage_mode == LLM_CRITIC_MODE
    hybrid_loss = uses_hybrid_policy_loss(configured_advantage_mode)
    history_mode = _uses_reward_history(configured_advantage_mode)
    replayable_update = _uses_replayable_policy_update(configured_advantage_mode)
    print(f"Groups per step:    {cfg.groups_per_step}")
    print(f"Group size:         {cfg.group_size}")
    print(f"Total rollouts/step: {cfg.groups_per_step * cfg.group_size}")
    print(f"LR:                 {cfg.learning_rate}")
    if _is_spo_rs_mode(configured_advantage_mode):
        print("Reference KL:        off (SPO-RS PPO objective)")
    else:
        print(f"KL coef:            {cfg.kl_penalty_coef}")
    print(f"Advantage mode:     {getattr(cfg, 'advantage_mode', 'entropic')}")
    if hybrid_loss:
        print("Policy loss:        TTT token-summed importance sampling")
    if history_mode:
        if invariant_mode:
            from training.advantage import ENTROPIC_KL_BUDGET, INVARIANT_LOGIT_CAP
            print(f"SPO-RS beta:        adaptive (KL budget {ENTROPIC_KL_BUDGET:.6f}, "
                  f"logit cap {INVARIANT_LOGIT_CAP:g})")
        else:
            print(f"SPO-RS beta:        fixed {cfg.spo_rs_beta}")
        print(f"SPO-RS D_half:      {cfg.spo_rs_d_half}")
        print(f"SPO-RS rho min/max: {cfg.spo_rs_rho_min} / {cfg.spo_rs_rho_max}")
        if hybrid_mode:
            print(f"Hybrid TTT weight:  {cfg.hybrid_ttt_weight:g}")
            print(f"Hybrid SPO-RS weight: {cfg.hybrid_spo_weight:g}")
        elif _is_spo_rs_mode(configured_advantage_mode):
            print(f"SPO-RS clip eps low/high: "
                  f"{cfg.spo_rs_clip_epsilon_low} / "
                  f"{cfg.spo_rs_clip_epsilon_high} "
                  f"(bounds {1.0 - cfg.spo_rs_clip_epsilon_low:.4f} / "
                  f"{1.0 + cfg.spo_rs_clip_epsilon_high:.4f})")
    print(f"Max new tokens:     {cfg.max_new_tokens}")
    if getattr(cfg, "coder_reasoning_effort", None) is not None:
        print(f"Coder reasoning:    {cfg.coder_reasoning_effort}")
    print(
        f"Coder sampling:     temperature={cfg.temperature}, "
        f"top_p={cfg.top_p}, "
        f"top_k={getattr(cfg, 'sampling_top_k', None)}, "
        f"min_p={getattr(cfg, 'sampling_min_p', None)}")
    print(f"Max seq length:     {cfg.max_seq_length}")
    if getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b":
        fused_attention_label = ('GPT-OSS sink-aware Unsloth path (generic Flash-SDPA prohibited)')
    else:
        fused_attention_label = ("on" if cfg.fused_long_attention else "off")
    print(f"Fused long attention: {fused_attention_label}")
    print(f'LoRA:               rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}')
    print(f"Deterministic:      {cfg.deterministic}")
    if not cfg.no_train:
        print(f"Training memory cap: {100.0 * float(cfg.training_memory_fraction):g}% "
              "per GPU (96%->98%->99% staged exact long-rollout rescue)")
    print(f"Logprob chunk:      {cfg.logprob_chunk or 'off (single shot)'}")
    print(f"Seed:               {cfg.seed}")
    print(f"Sandbox timeout:    {cfg.sandbox_timeout_s}s")
    print("=" * 70)

    # ---- experiment dir ----
    startup_log.end_summary()
    action = "resuming in" if resume_dir else "writing all rollouts to"
    print(f"[init] {action}: {exp_dir}")

    # ---- seed states (problem-defined) ----
    seeds = problem.seed_states()
    print(f"[init] problem produced {len(seeds)} seed state(s)")
    setting_log_only("Settings: setting.log   |   Terminal: temirnal.log   |   Results: result.txt")
    print = console_print
    if restart_step == 0:
        print_dashboard(startup_log.fields, exp_dir, resuming=bool(resume_dir))

    # ---- backend + model ----
    cfg._startup_step = restart_step
    parallel_trainer = None
    if not cfg.no_train:
        parallel_trainer = ProcessDistributedTrainer(
            cfg, exp_dir, dependency_log_path=dependency_log_path)
        trainer_startup.enter_context(parallel_trainer.loading_replicas())
    # Load backend FIRST so Unsloth can patch transformers if used.
    with startup_log.capture(loading=True), _route_dependency_notices(dependency_log_path):
        from models.model_backend import load_backend
        backend = load_backend(cfg.backend, cfg)
        model, tokenizer = backend.load()
    effective_4bit = bool(getattr(cfg, "effective_load_in_4bit", cfg.load_in_4bit))
    if getattr(cfg, "coder_model_profile", "") == "gpt-oss-120b":
        training_precision = "original packed MXFP4 (no NF4 re-quantization)"
    else:
        training_precision = ("BitsAndBytes 4-bit" if effective_4bit else "checkpoint/default precision")
    setting_log_only(f"[precision] training copy: {training_precision}")

    if cfg.generation_backend == "vllm":
        from runtime.gpu_runtime import validate_attention_heads
        validate_attention_heads(
            _attention_head_count(model), int(cfg.vllm_tensor_parallel_size), cfg.coder_model_name)

    import torch  # safe to import now
    import random
    random.seed(cfg.seed)
    if run_seed is not None:
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

    # Load policy weights before constructing the optimizer, so both describe
    # the same completed step.
    resume_payload = None
    legacy_resume = None
    interrupted_spo_rs_artifacts = None
    start_step = 0
    current_policy_adapter_path = None
    recover_update = critic_mode or invariant_mode or (
        history_mode and not ((is_trimul or hybrid_mode) and cfg.no_train))
    if resume_dir:
        resume_payload = _load_training_checkpoint(exp_dir, mmap=True)
        if resume_payload is not None:
            start_step = int(resume_payload["next_step"])
            adapter_path = Path(exp_dir) / resume_payload["adapter_dir"]
            _load_adapter(model, adapter_path)
            current_policy_adapter_path = adapter_path
            setting_log_only(f"[resume] training checkpoint found; next step is {start_step}")
            if resume_payload["version"] == 1:
                print('[resume] legacy batch-growth state ignored; using fixed configured batch sizes')
            if recover_update:
                interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, start_step, cfg))
                if (critic_mode and interrupted_spo_rs_artifacts is None
                        and critic_step.load_batch(exp_dir, start_step) is not None):
                    raise RuntimeError(f"cannot restore rollouts for the saved critic batch at step {start_step}")
                if interrupted_spo_rs_artifacts is not None:
                    token_recovery = (
                        _prepare_recoverable_response_tokens(model, tokenizer, interrupted_spo_rs_artifacts))
                    setting_log_only(
                        "[resume] preflight prepared the interrupted "
                        f"step-{start_step} responses: "
                        f"{token_recovery['exact_sidecars']} exact token "
                        "sidecar(s), "
                        f"{token_recovery['canonical_text']} legacy "
                        "text reconstruction(s), "
                        f"{token_recovery['behavior_logprob_sidecars']} "
                        "rollout-policy logprob sidecar(s), "
                        f"{token_recovery['restored_terminal_tokens']} "
                        "terminal EOS token(s) restored",
                        flush=True,
                    )
        else:
            if invariant_mode and _adapter_exists(exp_dir):
                raise FileNotFoundError("resuming an invariant run requires its training_state.pt")
            if recover_update:
                interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, 0, cfg))
            initial_adapter = Path(exp_dir) / "adapter_initial"
            recreated_initial_adapter = False
            if interrupted_spo_rs_artifacts is not None:
                initial_adapter, recreated_initial_adapter = (
                    _ensure_initial_adapter_for_recovery(model, exp_dir))
            if interrupted_spo_rs_artifacts is not None:
                start_step = 0
                adapter_path = initial_adapter
                _load_adapter(model, adapter_path)
                token_recovery = _prepare_recoverable_response_tokens(model, tokenizer, interrupted_spo_rs_artifacts)
                current_policy_adapter_path = adapter_path
                print(
                    "[resume] found a fully evaluated interrupted SPO-RS "
                    f"step 0; {'recreated' if recreated_initial_adapter else 'restored'} "
                    "adapter_initial and will skip rollout "
                    "generation/evaluation",
                    flush=True,
                )
                setting_log_only(
                    "[resume] preflight prepared all saved responses: "
                    f"{token_recovery['exact_sidecars']} exact token "
                    "sidecar(s), "
                    f"{token_recovery['canonical_text']} legacy text "
                    "reconstruction(s), "
                    f"{token_recovery['behavior_logprob_sidecars']} "
                    "rollout-policy logprob sidecar(s), "
                    f"{token_recovery['restored_terminal_tokens']} terminal "
                    "EOS token(s) restored",
                    flush=True,
                )
            elif not invariant_mode:
                legacy_resume = _legacy_resume_info(exp_dir)
                start_step, adapter_path = legacy_resume
                _load_adapter(model, adapter_path)
                current_policy_adapter_path = adapter_path
                print(
                    f"[resume] legacy run (no training_state.pt): restarting "
                    f"step {start_step} from {adapter_path.name}. The archive "
                    "will be reconstructed, but old PUCT/optimizer statistics "
                    "are unavailable."
                )

    optimizer_class = torch.optim.AdamW
    optimizer_options = {}
    if getattr(model, "_ttt_gpt_oss_shared_lora", False):
        from training.optimizer import HostMomentAdamW
        optimizer_class = HostMomentAdamW
        optimizer_options["hardware_profile"] = cfg.hardware_profile
    optimizer = optimizer_class(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
        **optimizer_options,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        setting_log_only("[resume] restored optimizer state")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    setting_log_only(f'[init] trainable params: {trainable:,} / total {total:,} ({100 * trainable / total:.2f}%)')

    # ---- sampler ----
    from search.sampler import make_sampler
    ac1_archive_seed = getattr(cfg, "ac1_archive_seed", None)
    ac1_ttt_rollouts = getattr(cfg, "ac1_ttt_rollouts", False)
    restored_selection = (resume_payload["sampler"].get("selection_mode", cfg.parent_selection)
                          if resume_payload is not None else cfg.parent_selection)
    if restored_selection != cfg.parent_selection:
        raise ValueError("saved sampler differs from the requested parent picker")
    sampler = make_sampler(
        parent_selection=restored_selection,
        num_seeds=len(seeds) if seeds else cfg.num_seed_states,
        puct_c=cfg.puct_c,
        max_buffer_size=cfg.max_buffer_size,
        topk_children=cfg.topk_children_per_parent,
        seed_value=0.0,
        seed_states=seeds,
        ac1_archive_seed=ac1_archive_seed,
        refresh_ac1_seed=(problem.refresh_ac1_seed if ac1_ttt_rollouts or ac1_archive_seed is not None else None),
        ac1_ttt_rollouts=ac1_ttt_rollouts,
    )
    if resume_payload is not None:
        sampler.load_state_dict(resume_payload["sampler"])
        setting_log_only(f'[resume] restored exact {sampler.selection_mode.upper()} archive and search statistics')
    elif legacy_resume is not None:
        n_states, n_rollouts = _restore_legacy_archive(
            sampler, exp_dir, before_step=start_step, tokenizer=tokenizer)
        print(f'[resume] reconstructed {n_states} valid archived candidates from {n_rollouts} earlier rollouts')
    if resume_dir and getattr(problem, "name", "") == "gpu_mode":
        restored_observations = _restore_archive_observations(sampler, exp_dir)
        if restored_observations:
            print(f"[resume] restored evaluator output for "
                  f"{restored_observations} archived GPU candidate(s)")
    setting_log_only(f"[init] sampler archive size = {sampler.archive_size()}", flush=True)

    spo_rs_tracker = None
    if invariant_mode:
        from training.spo_rs import InvariantSPORSTracker
        if current_policy_adapter_path is None:
            current_policy_adapter_path = Path(_save_adapter(
                model, exp_dir, -1, directory_name="adapter_initial", base_policy=True))
        spo_rs_tracker = InvariantSPORSTracker(
            d_half=cfg.spo_rs_d_half, rho_min=cfg.spo_rs_rho_min, rho_max=cfg.spo_rs_rho_max,
            policy_adapter=current_policy_adapter_path)
        if resume_payload is not None:
            spo_rs_tracker.load_state_dict(resume_payload.get("spo_rs_tracker"))
            if spo_rs_tracker.last_step + 1 != start_step:
                raise ValueError("invariant reward history does not match the checkpoint step")
            setting_log_only("[resume] restored the weighted SPO-RS history")
        else:
            print("[init] invariant SPO-RS starts with pooled leave-one-out advantages")
    elif history_mode:
        from training.spo_rs import SPORSTracker
        spo_rs_tracker = SPORSTracker(
            entropic_beta=cfg.spo_rs_beta,
            d_half=cfg.spo_rs_d_half,
            rho_min=cfg.spo_rs_rho_min,
            rho_max=cfg.spo_rs_rho_max,
        )
        tracker_state = (resume_payload.get("spo_rs_tracker") if resume_payload is not None else None)
        tracker_version = tracker_state.get("version") if tracker_state is not None else None
        if tracker_version == SPORSTracker.VERSION:
            spo_rs_tracker.load_state_dict(tracker_state)
            setting_log_only("[resume] restored the run-wide SPO-RS tracker")
        elif tracker_version not in (None, 1):
            raise ValueError("unsupported SPO-RS tracker state")
        elif start_step:
            # Version 1 attached history to rendered prompts. It cannot be
            # merged into the requested global statistic, so initialize the
            # new tracker once from the most recent completed rollout batch.
            last_step = start_step - 1
            rewards = _logged_step_rewards(exp_dir, last_step)
            if not rewards:
                raise ValueError(
                    "cannot initialize the run-wide SPO-RS tracker: no finite "
                    f"rewards were logged for completed step {last_step}")
            prior_sampling_adapter = Path(exp_dir) / (
                "adapter_step000" if last_step == 0
                else f"adapter_step{last_step - 1:03d}")
            if not prior_sampling_adapter.is_dir():
                raise FileNotFoundError(
                    "cannot initialize the run-wide SPO-RS tracker: the "
                    f"sampling adapter for step {last_step} is missing at "
                    f"{prior_sampling_adapter}")
            transformed = spo_rs_tracker.transformed_rewards(rewards)
            spo_rs_tracker.update(
                transformed, divergence=0.0,
                policy_adapter=prior_sampling_adapter,
                step=last_step)
            source = ("legacy prompt trackers" if tracker_state is not None else "logged rollout history")
            print(f"[resume] replaced {source} with one run-wide SPO-RS "
                  f"tracker initialized from {len(rewards)} rewards in step "
                  f"{last_step}")
        else:
            # Step 0 needs a baseline that exists before its responses are
            # sampled. The problem provides initial rewards on its scoring scale.
            seed_rewards = []
            for initial_reward in problem.initial_spo_rs_rewards(seeds):
                try:
                    reward = float(initial_reward)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(reward):
                    seed_rewards.append(reward)
            if not seed_rewards:
                raise ValueError(
                    "a fresh SPO-RS run requires at least one finite "
                    "initial reward to initialize its pre-step baseline")
            initial_adapter = Path(_save_adapter(
                model, exp_dir, -1, directory_name="adapter_initial",
                base_policy=True))
            transformed = spo_rs_tracker.transformed_rewards(seed_rewards)
            initialization = spo_rs_tracker.update(
                transformed, divergence=0.0,
                policy_adapter=initial_adapter, step=-1)
            current_policy_adapter_path = initial_adapter
            print("[init] SPO-RS global tracker initialized before step 0 "
                  f"from {len(seed_rewards)} initial rewards: "
                  f"v={initialization['value_after']:.9f}, "
                  f"N_eff={initialization['effective_count_after']:.2f}; "
                  "step 0 policy update enabled", flush=True)

    if (is_trimul or hybrid_loss or invariant_mode) and resume_payload is None:
        if current_policy_adapter_path is None:
            current_policy_adapter_path = Path(_save_adapter(
                model, exp_dir, start_step, directory_name="adapter_initial",
                base_policy=True))
        _save_training_checkpoint(exp_dir, start_step, current_policy_adapter_path,
                                  sampler, optimizer, spo_rs_tracker)

    del resume_payload
    from runtime.host_memory import reclaim_unused_host_memory
    reclaim_unused_host_memory(torch, force=True)

    removed_adapters = _prune_adapter_snapshots(
        exp_dir,
        _live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker),
    )
    if removed_adapters:
        setting_log_only(f"[checkpoint] removed {len(removed_adapters)} superseded "
                         "adapter snapshot(s) while restoring the rolling checkpoint", flush=True)

    if parallel_trainer is not None:
        parallel_trainer.finish_startup(backend, model, tokenizer, optimizer)
    trainer_startup.close()

    # ---- generation pool ----
    gen_pool = None
    # vLLM forms exact TP/PP engines over every rollout card. Because those
    # cards also hold either training replicas or model shards, the two runtimes
    # alternate residency.
    use_gen_pool = bool(cfg.num_gpus) and (
        cfg.generation_backend == "vllm"
        or (cfg.num_gpus > 1 and cfg.num_training_gpus == 1))
    if use_gen_pool:
        from generation.gen_workers import (GenerationPool, HybridHFGenerationPool,
                                 PhasedVLLMGenerationPool, rollout_engine_options, worker_seed)
        gpu_ids = _parse_gpu_ids(cfg.gpu_ids)
        pool_options = dict(
            **rollout_engine_options(cfg),
            model_name=cfg.coder_model_name,
            num_workers=cfg.num_gpus,
            gpu_ids=gpu_ids,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=effective_4bit,
            seed=run_seed,
            gen_micro_batch=cfg.gen_micro_batch,
            backend=cfg.generation_backend,
            lora_rank=(cfg.lora_rank),
            vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
            vllm_pipeline_parallel_size=cfg.vllm_pipeline_parallel_size,
            vllm_sleep_level=cfg.vllm_sleep_level,
            # A sleeping GPT-OSS EngineCore retains enough CUDA/NCCL state to
            # starve the packed-MXFP4 trainer's fused dequant workspace.  A
            # fresh rollout pool is faster overall than training every step
            # through the OOM-driven unfused fallback.
            # Shared benchmark GPUs must be released by generation.
            vllm_persistent_workers=(0 if cfg.evaluation_shares_generation
                                     else _coder_vllm_persistent_workers(cfg)),
            # Training modes consume exact chosen-token prompt logprobs after
            # generation. Give that vocabulary projection explicit memory
            # headroom; this changes scheduling only, never the scores.
            vllm_token_scoring=not bool(cfg.no_train),
            vllm_staged_loading=cfg.vllm_staged_loading,
            vllm_log_path=vllm_log_path,
        )
        if getattr(model, "_ttt_gpt_oss_shared_lora", False):
            from generation.gen_workers import gpt_oss_coder_engine_options
            pool_options["vllm_engine_options"] = gpt_oss_coder_engine_options(
                cfg.groups_per_step * cfg.group_size,
                use_cuda_graphs=cfg.hardware_profile == "B200_full_ram",
            )
        if cfg.generation_backend == "vllm":
            from runtime.host_memory import HostMemoryAdmission
            import multiprocessing
            pool_options["host_memory_admission"] = (
                getattr(parallel_trainer, "host_memory_admission", None)
                or HostMemoryAdmission(multiprocessing.get_context("spawn"), hardware_profile=cfg.hardware_profile))
            trainer_offloaded = False

            def _offload_trainer_for_generation():
                nonlocal trainer_offloaded
                disk_directory = getattr(cfg, "trainer_offload_dir", None)
                frozen_to_disk = bool(disk_directory)
                if (parallel_trainer is not None and parallel_trainer.is_offloaded):
                    return
                if parallel_trainer is None and trainer_offloaded:
                    return
                from runtime.host_memory import (
                    GIB, estimated_module_bytes, estimated_optimizer_bytes,
                    host_memory_info, reclaim_unused_host_memory, RAM_WAIT_TIMEOUT_S,
                )
                from runtime.incremental_offload import can_incrementally_offload, offload_optimizer
                incremental = can_incrementally_offload(model, cfg, disk_directory)

                # Reclaim cached activation-offload blocks on *every* trainer
                # process before allocating complete CPU model replicas. This
                # is a phase-boundary operation: it leaves live tensors and
                # gradients untouched and runs only once per rollout step.
                if parallel_trainer is not None:
                    cleanups = parallel_trainer.reclaim_host_memory()
                    required_bytes = parallel_trainer.estimated_host_offload_bytes(
                        frozen_to_disk=frozen_to_disk)
                else:
                    torch.cuda.synchronize()
                    cleanups = [reclaim_unused_host_memory(
                        torch,
                        cuda_devices=tuple(range(torch.cuda.device_count())),
                        force=True,
                    )]
                    required_bytes = int(
                        1.05 * (estimated_module_bytes(model, frozen_to_disk=frozen_to_disk)
                                + estimated_optimizer_bytes(optimizer))
                        + GIB
                    )

                reclaimed_bytes = sum(max(
                    0,
                    int(cleanup.get("pinned_cached_before", 0))
                    - int(cleanup.get("pinned_cached_after", 0)),
                ) for cleanup in cleanups)
                memory = host_memory_info()
                if reclaimed_bytes >= GIB:
                    available = ("unknown" if memory is None else f"{memory['available'] / GIB:.1f} GiB")
                    setting_log_only(
                        f"[host-memory] released "
                        f"{reclaimed_bytes / GIB:.1f} GiB of inactive pinned "
                        f"trainer cache; available RAM={available}",
                        flush=True,
                    )

                # Under host pressure, discard the sleeping coder backup so
                # the trainer can park safely. The coder reloads on next use.
                reserve_bytes = None
                if memory is not None:
                    if frozen_to_disk or cfg.hardware_profile == "B200_full_ram":
                        from runtime.host_memory import host_memory_reserve_bytes
                        reserve_bytes = host_memory_reserve_bytes(memory, cfg.hardware_profile)
                    else:
                        reserve_bytes = 5 * GIB
                    target_available = required_bytes + reserve_bytes
                    discard = getattr(gen_pool, "discard_sleeping_host_residency", None)
                    if memory["available"] < target_available and callable(discard) and discard():
                        memory = host_memory_info() or memory
                        print(
                            "[host-memory] low-RAM guard discarded the sleeping "
                            "coder vLLM weight backup; it will reload on its next use",
                            flush=True,
                        )
                    # Incremental anonymous mappings remain resident after
                    # each chunk. If the complete replicas do not fit, do not
                    # begin a transfer that the host will later SIGKILL.
                    if (incremental and not disk_directory
                            and memory["available"] < target_available):
                        import shutil
                        automatic_disk_directory = (
                            Path(exp_dir) / ".trainer_offload"
                        )
                        disk_free = shutil.disk_usage(
                            automatic_disk_directory.parent).free
                        if disk_free >= required_bytes:
                            automatic_disk_directory.mkdir(
                                parents=True, exist_ok=True)
                            disk_directory = str(automatic_disk_directory)
                            frozen_to_disk = True
                            incremental = False
                            if parallel_trainer is not None:
                                required_bytes = (
                                    parallel_trainer.estimated_host_offload_bytes(
                                        frozen_to_disk=True)
                                )
                            else:
                                required_bytes = int(
                                    1.05 * (
                                        estimated_module_bytes(
                                            model, frozen_to_disk=True)
                                        + estimated_optimizer_bytes(optimizer)
                                    ) + GIB
                                )
                            print(
                                "[host-memory] complete trainer replicas do "
                                "not fit safely in RAM; streaming frozen "
                                "weights to the run-local offload directory "
                                "for this phase",
                                flush=True,
                            )
                        else:
                            # Full admission waits before allocating anything;
                            # it is slower than chunks but cannot overcommit.
                            incremental = False
                            print(
                                "[host-memory] insufficient RAM and disk for "
                                "safe trainer parking; waiting for the full "
                                "CPU allocation before starting the transfer",
                                flush=True,
                            )
                replica_label = (
                    "inactive training model"
                    if parallel_trainer is None
                    else f"all {parallel_trainer.world_size} trainer replicas"
                )
                destination = (f"disk ({disk_directory}); only adapters/optimizer stay in RAM"
                               if disk_directory else "CPU")
                setting_log_only(
                    f"[gpu] offloading {replica_label} to {destination} before "
                    "shared-GPU vLLM generation", flush=True)
                try:
                    host_admission = pool_options["host_memory_admission"]
                    transfer_options = ({"host_admission": host_admission,
                                         "deadline": time.monotonic() + RAM_WAIT_TIMEOUT_S}
                                        if incremental else {})
                    admission = (host_admission.prioritize_transfers() if incremental else host_admission.reservation(
                        required_bytes, stage="trainer CPU offload", priority=True,
                        reserve_bytes=reserve_bytes))
                    with admission:
                        if parallel_trainer is not None:
                            parallel_trainer.offload_for_generation(
                                purge_host_cache=False, disk_directory=disk_directory,
                                **transfer_options)
                        else:
                            if incremental:
                                offload_optimizer(optimizer, host_admission,
                                                  deadline=transfer_options["deadline"])
                            else:
                                _move_optimizer_state(optimizer, "cpu", hardware_profile=cfg.hardware_profile)
                            backend.offload_for_generation(disk_directory=disk_directory, **transfer_options)
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                    trainer_offloaded = True
                except Exception as exc:
                    try:
                        if parallel_trainer is not None:
                            parallel_trainer.restore_after_generation()
                        else:
                            backend.restore_after_generation()
                            _restore_optimizer_state_to_parameters(optimizer)
                    except Exception:
                        pass
                    if isinstance(exc, HostMemoryUnavailable):
                        pool_options["host_memory_admission"].stop_waiting()
                        raise
                    raise RuntimeError(
                        "the training model could not be offloaded for the "
                        "shared vLLM rollout phase; use generation_backend=hf "
                        "for live-model generation on this runtime") from exc

            def _restore_trainer_after_generation():
                nonlocal trainer_offloaded
                if parallel_trainer is None:
                    setting_log_only("[gpu] keeping the inactive model offloaded")
                elif parallel_trainer.is_offloaded:
                    setting_log_only("[gpu] keeping trainer replicas offloaded until the adapter update", flush=True)
                else:
                    trainer_offloaded = False

            gen_pool = PhasedVLLMGenerationPool(
                before_start=_offload_trainer_for_generation,
                after_stop=_restore_trainer_after_generation,
                **pool_options,
            )
            setting_log_only(
                f"[init] phase-shared vLLM pool configured across all "
                f"rollout GPUs {gpu_ids}", flush=True)
        else:
            # Do not load a duplicate base model beside the trainer. Rank zero
            # is the live HF/Unsloth training model; only the remaining cards
            # need worker processes.
            remote_options = dict(pool_options)
            remote_options["num_workers"] = len(gpu_ids) - 1
            remote_options["gpu_ids"] = gpu_ids[1:]
            remote_pool = GenerationPool(**remote_options)
            local_cap = {"value": 0}

            def _local_hf_rollouts(prompts_by_group, counts_by_group,
                                   adapter_path,
                                   max_new_tokens, temperature, top_p,
                                   step_idx, top_k=None, min_p=None):
                backend.set_inference_mode()
                if run_seed is not None:
                    local_seed = worker_seed(run_seed, step_idx, 0)
                    torch.manual_seed(local_seed)
                    torch.cuda.manual_seed_all(local_seed)
                try:
                    adapter_context = (backend.disable_adapter() if adapter_path is None else nullcontext())
                    with adapter_context:
                        yield from generate_prompt_jobs(
                            model, tokenizer, prompts_by_group,
                            counts_by_group, cfg,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p,
                            top_k=top_k, min_p=min_p,
                            cap_state=local_cap)
                finally:
                    backend.set_training_mode()

            gen_pool = HybridHFGenerationPool(remote_pool=remote_pool, local_iter=_local_hf_rollouts)
            print(f"[init] hybrid HF rollout pool: live trainer on "
                  f"physical GPU {gpu_ids[0]} plus persistent workers "
                  f"{gpu_ids[1:]}")
        if cfg.gen_micro_batch and cfg.gen_micro_batch > 0:
            limit_name = ("max_num_seqs" if cfg.generation_backend == "vllm" else "micro-batch")
            limit_scope = ("/engine" if cfg.generation_backend == "vllm" else "/GPU")
            setting_log_only(
                f"[init] generation pool ready "
                f"({limit_name} {cfg.gen_micro_batch}{limit_scope})",
                flush=True)
        else:
            setting_log_only("[init] generation pool ready", flush=True)
    else:
        if cfg.num_training_gpus > 1:
            print(f"[init] model-parallel HF generation across training GPUs "
                  f"{cfg.training_gpu_ids}; prompts remain cross-batched")
        else:
            print("[init] single-GPU generation (no worker pool)")

    if parallel_trainer is not None:
        # Exact overflow may need the same sleeping-backup reclamation as generation.
        parallel_trainer._generation_pools = (gen_pool,)

    # ---- main loop ----
    from experiment_io import StepPlotter
    step_plotter = StepPlotter(exp_dir, cfg.problem)
    cumulative_training_seconds = 0.0
    for completed_step in range(start_step):
        summary_path = (Path(exp_dir) / f"step{completed_step:02d}" / f"step{completed_step:02d}.summary.json")
        try:
            completed_summary = json.loads(summary_path.read_text())
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if "result_metric_name" in completed_summary:
            append_step_result(exp_dir, completed_step, completed_summary)
        cumulative_training_seconds += float(completed_summary.get("training_seconds", 0.0) or 0.0)
    if cumulative_training_seconds:
        print(f'[resume] prior cumulative training time: {cumulative_training_seconds:.1f}s')
    restart_for_exact_recovery = False
    evaluation_handoff = None
    active_step = None
    try:
        if start_step >= cfg.num_steps:
            print(f"[resume] run already reached requested num_steps={cfg.num_steps}")
        for step in range(start_step, cfg.num_steps):
            active_step = int(step)
            step_cfg = cfg
            setting_log_only(
                f"[step {step}] batch: G={cfg.groups_per_step} K={cfg.group_size} "
                f"({cfg.groups_per_step * cfg.group_size} coder rollouts)", flush=True)

            if resume_dir and step == start_step and recover_update:
                pending_spo_rs_artifacts = None
                if interrupted_spo_rs_artifacts is None:
                    interrupted_spo_rs_artifacts = (_recoverable_spo_rs_artifacts(exp_dir, step, step_cfg))
                if (interrupted_spo_rs_artifacts is None
                        and not (is_trimul and trimul_evaluation.batch_path(exp_dir, step).is_file())):
                    pending_spo_rs_artifacts = (_recoverable_pending_spo_rs_artifacts(exp_dir, step, step_cfg))
            else:
                interrupted_spo_rs_artifacts = None
                pending_spo_rs_artifacts = None

            finalize_pending_spo_rs = pending_spo_rs_artifacts is not None and not (
                is_trimul or hybrid_loss or invariant_mode)
            if interrupted_spo_rs_artifacts is not None:
                stats = _resume_spo_rs_step_from_artifacts(
                    model, tokenizer, sampler, step,
                    step_cfg, exp_dir, problem, parallel_trainer,
                    spo_rs_tracker, current_policy_adapter_path,
                    interrupted_spo_rs_artifacts,
                    evaluation_complete=step_plotter.submit)
            elif finalize_pending_spo_rs:
                print(
                    f"[resume] found {len(pending_spo_rs_artifacts)} exact "
                    f"saved trajectories for interrupted SPO-RS step {step}; "
                    "reusing the saved generation batch",
                    flush=True,
                )
                completed_spo_rs_artifacts = (
                    _finalize_pending_spo_rs_artifacts(
                        tokenizer, sampler, step, step_cfg, exp_dir, problem,
                        gen_pool, spo_rs_tracker,
                        current_policy_adapter_path,
                        pending_spo_rs_artifacts))
                stats = _resume_spo_rs_step_from_artifacts(
                    model, tokenizer, sampler, step,
                    step_cfg, exp_dir, problem, parallel_trainer,
                    spo_rs_tracker, current_policy_adapter_path,
                    completed_spo_rs_artifacts,
                    evaluation_complete=step_plotter.submit)
            else:
                stats = train_step(
                    backend, model, tokenizer, sampler, optimizer, step,
                    step_cfg, exp_dir, problem, gen_pool,
                    parallel_trainer=parallel_trainer,
                    spo_rs_tracker=spo_rs_tracker,
                    sampling_adapter_path=(
                        current_policy_adapter_path
                        if replayable_update else None),
                    evaluation_complete=step_plotter.submit,
                    pending_artifacts=pending_spo_rs_artifacts,
                    park_trainer=_offload_trainer_for_generation if critic_mode else None)

            stats = stats or {}
            step_training_seconds = float(stats.get("training_seconds", 0.0) or 0.0)
            cumulative_training_seconds += step_training_seconds
            stats["training_seconds"] = step_training_seconds
            stats["cumulative_training_seconds"] = cumulative_training_seconds
            print(f"[step {step}] TOTAL TRAINING TIME: "
                  f"{step_training_seconds:.1f}s  "
                  f"(run cumulative {cumulative_training_seconds:.1f}s)",
                  flush=True)

            # Version the adapter first, then atomically advance the state
            # pointer. A crash during either write leaves the previous pair
            # valid.
            if history_mode:
                if stats.get("spo_rs_policy_updated", False):
                    adapter_path = _save_adapter(model, exp_dir, step)
                else:
                    adapter_path = (Path(exp_dir) / stats["spo_rs_sampling_adapter"])
            elif critic_mode and not stats.get("critic_policy_updated", False):
                adapter_path = current_policy_adapter_path
            else:
                adapter_path = _save_adapter(model, exp_dir, step)
            checkpoint_path = _save_training_checkpoint(
                exp_dir, step + 1, adapter_path, sampler, optimizer,
                spo_rs_tracker=spo_rs_tracker,
            )
            step_summary = {
                "step": step,
                "completed": True,
                "next_step": step + 1,
                "adapter_dir": Path(adapter_path).name,
                "checkpoint": Path(checkpoint_path).name,
                "archive_size": sampler.archive_size(),
                "groups_per_step": int(cfg.groups_per_step),
                "group_size": int(cfg.group_size),
                **(stats or {}),
            }
            save_step_summary(exp_dir, step, step_summary)
            step_plotter.submit_advantages(step)
            if is_trimul:
                batch_path = trimul_evaluation.batch_path(exp_dir, step)
                batch_path.unlink(missing_ok=True)
            if is_trimul or hybrid_loss:
                for sidecar in (Path(exp_dir) / f"step{step:02d}").glob("*.reference_logprobs.zlib"):
                    sidecar.unlink(missing_ok=True)
            append_step_result(exp_dir, step, step_summary)
            current_policy_adapter_path = Path(adapter_path)
            removed_adapters = _prune_adapter_snapshots(
                exp_dir,
                _live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker),
            )
            if removed_adapters:
                retained = (
                    "two rolling adapters for consecutive-policy KL"
                    if (spo_rs_tracker is not None
                        and len(_live_adapter_snapshots(exp_dir, current_policy_adapter_path, spo_rs_tracker)) > 1)
                    else Path(adapter_path).name
                )
                setting_log_only(f"[checkpoint] removed {len(removed_adapters)} "
                                 f"superseded adapter snapshot(s); retained {retained}", flush=True)
            print(f"[checkpoint] completed step {step}; resume at step {step + 1}")
            if (stats.get("exact_overflow_inprocess", False)
                    or getattr(cfg, "_exact_inprocess_recovery_step", None) == step):
                # Keep partial gradients until the *full* optimizer/sampler
                # checkpoint is durable. A crash before this point can resume
                # the same update; a successful step needs no process restart.
                _clear_exact_sharded_recovery(exp_dir)
                if hasattr(cfg, "_exact_inprocess_recovery_step"):
                    delattr(cfg, "_exact_inprocess_recovery_step")
            replica_capture_step = getattr(cfg, "_exact_replica_capture_step", None)
            if replica_capture_step is not None:
                if int(replica_capture_step) != int(step):
                    raise RuntimeError(
                        "exact replica-capture marker expected step "
                        f"{int(replica_capture_step)}, but completed step "
                        f"{step}")
                _clear_exact_sharded_recovery(exp_dir)
                print(
                    f"[recovery] legacy exact-OOM update for step {step} "
                    "completed on the fast replicas; recovery marker removed",
                    flush=True,
                )
                delattr(cfg, "_exact_replica_capture_step")
            replica_recovery_step = getattr(cfg, "_exact_replica_recovery_step", None)
            if replica_recovery_step is not None:
                if int(replica_recovery_step) != int(step):
                    raise RuntimeError(
                        "exact replica recovery marker expected step "
                        f"{int(replica_recovery_step)}, but completed step "
                        f"{step}")
                _clear_exact_sharded_recovery(exp_dir)
                print(
                    f"[recovery] RAM-bounded replicated overflow update for "
                    f"step {step} completed; recovery state removed",
                    flush=True,
                )
                delattr(cfg, "_exact_replica_recovery_step")
    except trimul_evaluation.EvaluationHandoff as handoff:
        evaluation_handoff = handoff.path
        setting_log_only(f"[step {active_step}] saved TriMul candidates; shutting down all model processes", flush=True)
    except ExactLongContextTrainingOOM as error:
        if active_step is None or not replayable_update:
            raise
        if _recoverable_spo_rs_artifacts(exp_dir, active_step, cfg) is None:
            raise RuntimeError(
                "exact overflow recovery requires the complete saved rollout and evaluation batch"
            ) from error
        marker = _write_exact_sharded_recovery(exp_dir, active_step, error)
        recovery = _read_exact_sharded_recovery(exp_dir) or {}
        retrying_same_step = active_step in (
            getattr(cfg, "_exact_replica_capture_step", None),
            getattr(cfg, "_exact_replica_recovery_step", None),
            getattr(cfg, "_exact_inprocess_recovery_step", None),
        )
        if recovery.get("replica_offload_exhausted", False) or retrying_same_step:
            raise RuntimeError(
                f"Exact {configured_advantage_mode} recovery exhausted the process trainer. "
                f"The unfinished step is saved in {exp_dir}."
            ) from error
        restart_for_exact_recovery = True
        print(
            f"[recovery] step {active_step}: restarting the process trainer "
            f"from saved evaluations and gradients where available ({marker.name})",
            flush=True,
        )
    finally:
        # Plotting uses no GPUs and never delays generation of the next step.
        # Finish queued figures on normal completion; cancel pending work when
        # the training loop exits with an exception or interruption.
        step_plotter.close(wait=sys.exc_info()[0] is None)
        if gen_pool is not None:
            setting_log_only("[shutdown] stopping generation pool ...")
            gen_pool.shutdown()
        if parallel_trainer is not None:
            print("[shutdown] stopping process-distributed trainer ...")
            parallel_trainer.shutdown(for_restart=restart_for_exact_recovery)

    if evaluation_handoff is not None:
        from runtime.run_supervisor import request_evaluation
        request_evaluation(evaluation_handoff)
        return
    if restart_for_exact_recovery:
        _restart_for_exact_resume(exp_dir)

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    # Use the problem-native best-ever record, not merely the best state that
    # survived archive deduplication/pruning.
    best = (sampler.best_raw_state(maximize=bool(problem.maximize)) or sampler.best_state())
    if best is not None:
        raw = f"  (raw {getattr(problem, 'metric_name', 'metric')} = {best.raw_score:.9f})" \
            if best.raw_score is not None else ""
        print(f"Best reward (higher=better): {best.value:.9f}{raw}")
        print(f"Found at step:     {best.timestep}")
        best_code = best.code
        if ac1_ttt_rollouts:
            best_code = ttt_ac1_program(best_code) or best_code
        print(f"\n--- best code ---\n{best_code}\n--- end ---")
        save_final_summary(exp_dir, best.value, best_code, best.timestep,
                           best_construction=(_as_float_list(
                               getattr(best, "construction", None),
                               cfg.max_saved_construction)
                               if getattr(problem, "saves_construction", False)
                               else None),
                           best_raw_score=(float(best.raw_score) if best.raw_score is not None else None))
    else:
        print("No valid solution was ever produced.")
        save_final_summary(exp_dir, None, None, None)
    print(f"\nAll outputs saved under: {exp_dir}")


if __name__ == "__main__":
    try:
        main()
    except HostMemoryTimeout as error:
        print(f"[stop] {error}\n[resume] Saved checkpoints and rollout artifacts are retained. "
              "Restart with --resume and the run directory printed above.", flush=True)
        raise SystemExit(75) from None
