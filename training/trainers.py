"""Policy likelihood, loss, and process-distributed LoRA training."""

from contextlib import ExitStack, contextmanager, nullcontext
import math
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np

from config_validation import uses_ttt_policy_loss
from diagnostics.console import setting_log_only
from runtime.configuration import (
    SPO_RS_CLIP_EPSILON_DEFAULT, SPO_RS_CLIP_EPSILON_HIGH_DEFAULT,
    SPO_RS_CLIP_EPSILON_LOW_DEFAULT,
    _parse_gpu_ids, _uses_replayable_policy_update, _uses_reward_history,
)
from training.exact_overflow import (
    ExactLongContextTrainingOOM, _clip_sharded_grad_norm_,
    _commit_exact_spo_rs_overflow_batch, _load_exact_spo_rs_overflow,
    _read_exact_sharded_recovery, _save_exact_spo_rs_overflow, _write_exact_sharded_recovery,
    _write_exact_spo_rs_overflow_state, _validate_precomputed_exact_gradients,
)
from training.fast_distributed import _move_optimizer_state, _restore_optimizer_state_to_parameters
from training.hybrid import DEFAULT_SPO_WEIGHT, DEFAULT_TTT_WEIGHT, hybrid_advantages


_FUSED_LONG_SINGLETON_MIN_TOKENS = 8192


def compute_group_advantages(rewards_np):
    """Return entropic group advantages and their adaptive beta.

    SPO-RS obtains advantages from its frozen historical tracker instead.
    """
    from training.advantage import entropic_adaptive_advantages
    return entropic_adaptive_advantages(rewards_np)


def _causal_lm_decoder_and_head(model):
    """Return the decoder and output head without bypassing injected LoRA."""
    causal_lm = model
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            causal_lm = getter()
        except (AttributeError, TypeError):
            return None
    head_getter = getattr(causal_lm, "get_output_embeddings", None)
    if not callable(head_getter):
        return None
    head = head_getter()
    prefix = str(getattr(causal_lm, "base_model_prefix", "") or "")
    decoder = getattr(causal_lm, prefix, None) if prefix else None
    if decoder is None:
        for name in ("model", "transformer", "backbone"):
            candidate = getattr(causal_lm, name, None)
            if candidate is not None and candidate is not causal_lm:
                decoder = candidate
                break
    if decoder is None or decoder is causal_lm or head is None:
        return None
    return decoder, head


def _score_hidden_token_chunks(hidden_states, targets, output_head, *, chunk, with_grad):
    """Project hidden states in checkpointed vocabulary-sized chunks.

    CausalLM.forward normally materializes logits for every token at once.
    For a 32k sequence and a 150k vocabulary that tensor alone is enormous.
    This exact path retains only the selected log-probabilities and
    recomputes each short output-head chunk during backward.
    """
    import torch
    import torch.nn.functional as F

    length = int(targets.shape[0])
    step = int(chunk or 0)
    if step < 1:
        step = min(length, 256)

    def project(hidden_chunk, target_chunk):
        logits = output_head(hidden_chunk).float()
        log_probs = F.log_softmax(logits, dim=-1)
        # Examples may be built while the trainer is offloaded to CPU. With
        # model sharding, the output head can also run on a different GPU from
        # the input embeddings. Align indices with the actual projected scores
        # here, including when checkpoint recomputes this chunk in backward.
        target_chunk = target_chunk.to(device=log_probs.device)
        return log_probs.gather(1, target_chunk.unsqueeze(-1)).squeeze(-1)

    chosen_parts = []
    # Split backward assembles gradients once; slicing scatters a full sequence per tile.
    for hidden_chunk, target_chunk in zip(
            hidden_states.split(step, dim=0), targets.split(step, dim=0)):
        if with_grad and hidden_chunk.requires_grad:
            from torch.utils.checkpoint import checkpoint
            result = checkpoint(project, hidden_chunk, target_chunk, use_reentrant=False)
        else:
            result = project(hidden_chunk, target_chunk)
        chosen_parts.append(result)
    return torch.cat(chosen_parts, dim=0)


def _decoder_token_logprobs(model, input_ids, attention_mask,
                            response_ranges, response_targets, *, chunk,
                            with_grad):
    """Memory-bounded exact scoring, or None for an unsupported model shape."""
    components = _causal_lm_decoder_and_head(model)
    if components is None:
        return None
    decoder, output_head = components
    outputs = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        try:
            hidden = outputs[0]
        except (IndexError, KeyError, TypeError):
            return None
    scored = []
    for row, ((start, end), targets) in enumerate(zip(response_ranges, response_targets)):
        scored.append(_score_hidden_token_chunks(
            hidden[row, start:end, :], targets, output_head,
            chunk=chunk, with_grad=with_grad))
    return scored


def _shared_prefix_token_logprobs(model, examples, *, chunk, with_grad):
    """Score response branches while evaluating their common prompt once.

    The packed sequence contains one prompt followed by isolated response
    branches.  Each branch resets its logical positions after the prompt and
    the registered attention backend lets it see only the prompt and its own
    causal history.  The prompt remains in the autograd graph, so gradients
    from every response accumulate through it exactly; no detached-KV
    approximation is used.
    """
    import torch

    examples = list(examples)
    if len(examples) < 2:
        return None
    components = _causal_lm_decoder_and_head(model)
    if components is None:
        return None
    decoder, output_head = components
    decoder_config = getattr(decoder, "config", None)
    if str(getattr(decoder_config, "_attn_implementation", "")) != ("ttt_blockwise_attention"):
        return None
    # Hybrid Gated-DeltaNet layers carry a recurrent state along the sequence.
    # A custom mask isolates the full-attention layers, but cannot reset that
    # recurrent state between concatenated response branches. Score hybrid
    # models as ordinary padded batch rows so every rollout has independent
    # state, matching the original one-rollout forward exactly.
    layer_types = tuple(getattr(decoder_config, "layer_types", ()) or ())
    if "linear_attention" in layer_types:
        return None

    prompt = examples[0]["prompt_ids"]
    if prompt.ndim != 2 or int(prompt.shape[0]) != 1:
        return None
    for example in examples[1:]:
        candidate = example["prompt_ids"]
        if (candidate.shape != prompt.shape
                or candidate.device != prompt.device
                or candidate.dtype != prompt.dtype
                or not torch.equal(candidate, prompt)):
            return None

    prompt_length = int(prompt.shape[1])
    if prompt_length < 1:
        return None
    device = prompt.device
    input_parts = [prompt[0]]
    segment_parts = [torch.zeros(prompt_length, dtype=torch.long, device=device)]
    position_parts = [torch.arange(prompt_length, dtype=torch.long, device=device)]
    response_ranges = []
    attention_branches = []
    cursor = prompt_length
    for branch_index, example in enumerate(examples, start=1):
        response = example["response_ids"]
        if (response.ndim != 2 or int(response.shape[0]) != 1 or int(response.shape[1]) < 1):
            return None
        branch_input = response[0, :-1]
        branch_length = int(branch_input.shape[0])
        start = cursor
        end = start + branch_length
        response_ranges.append((start, end))
        if branch_length:
            input_parts.append(branch_input)
            segment_parts.append(torch.full((branch_length,), branch_index, dtype=torch.long, device=device))
            position_parts.append(torch.arange(
                prompt_length, prompt_length + branch_length,
                dtype=torch.long, device=device))
            attention_branches.append((start, end))
            cursor = end

    packed_ids = torch.cat(input_parts, dim=0).unsqueeze(0)
    segments = torch.cat(segment_parts, dim=0)
    positions = torch.cat(position_parts, dim=0)
    compact_mask = torch.stack((segments, positions), dim=0).unsqueeze(0)
    from models.model_backend import register_shared_prefix_attention_mask
    register_shared_prefix_attention_mask(compact_mask, prompt_length, attention_branches)

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        outputs = decoder(
            input_ids=packed_ids,
            attention_mask=compact_mask,
            position_ids=positions.unsqueeze(0),
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            try:
                hidden = outputs[0]
            except (IndexError, KeyError, TypeError):
                return None
        prompt_last = hidden[0, prompt_length - 1:prompt_length, :]
        scored = []
        for example, (start, end) in zip(examples, response_ranges):
            prediction_hidden = torch.cat((prompt_last, hidden[0, start:end, :]), dim=0)
            targets = example["response_ids"][0]
            scored.append(_score_hidden_token_chunks(
                prediction_hidden, targets, output_head,
                chunk=chunk, with_grad=with_grad))
    return scored


def compute_token_logprobs(model, prompt_ids, response_ids, with_grad: bool,
                           chunk: int = 0):
    """
    Per-token log-probabilities of the response under the model.

    prompt_ids:   (1, P) tensor
    response_ids: (1, R) tensor
    Output:       (R,) tensor of token logprobs

    The decoder runs once for exact causal attention. Its output head and
    log-softmax run in checkpointed `chunk` slices so full-sequence vocabulary
    logits are never retained. Unsupported model wrappers use the compatible
    full-forward fallback.
    """
    import torch
    import torch.nn.functional as F

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    P = prompt_ids.shape[1]
    R = response_ids.shape[1]
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        bounded = _decoder_token_logprobs(
            model, full_ids, None,
            [(int(P) - 1, int(P) - 1 + int(R))],
            [response_ids[0]], chunk=chunk, with_grad=with_grad)
        if bounded is not None:
            return bounded[0]

        out = model(full_ids)
        logits = out.logits  # (1, T, V)
        # Predict response token at position P+k from logits at position P+k-1.
        pred_logits = logits[:, P - 1 : P - 1 + R, :]  # (1, R, V)
        targets = response_ids.to(device=pred_logits.device)

        if chunk and 0 < chunk < R:
            parts = []
            for s in range(0, R, chunk):
                e = min(s + chunk, R)
                lp = F.log_softmax(pred_logits[:, s:e, :].float(), dim=-1)
                g = lp.gather(2, targets[:, s:e].unsqueeze(-1)).squeeze(-1)
                parts.append(g)          # keep only (1, e-s); lp freed next iter
            gathered = torch.cat(parts, dim=1)  # (1, R)
        else:
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return gathered.squeeze(0)


def compute_batched_token_logprobs(
        model, examples, with_grad: bool, chunk: int = 0, *,
        pad_token_id: int = 0):
    """Compute response log-probabilities for a padded example microbatch.

    Padding is appended after each complete prompt/response sequence and masked,
    so every real token retains the same causal positions as the old one-example
    path.  A singleton deliberately uses compute_token_logprobs() directly,
    preserving the previous behavior for long examples that the planner cannot
    safely combine with another sequence.
    """
    import torch
    import torch.nn.functional as F

    examples = list(examples)
    if not examples:
        return []
    if len(examples) == 1:
        example = examples[0]
        return [compute_token_logprobs(
            model, example["prompt_ids"], example["response_ids"],
            with_grad=with_grad, chunk=chunk)]

    shared = _shared_prefix_token_logprobs(model, examples, chunk=chunk, with_grad=with_grad)
    if shared is not None:
        return shared

    prompts = [example["prompt_ids"] for example in examples]
    responses = [example["response_ids"] for example in examples]
    for prompt_ids, response_ids in zip(prompts, responses):
        if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or response_ids.ndim != 2 or response_ids.shape[0] != 1):
            raise ValueError("training examples must contain (1, length) token tensors")
        if prompt_ids.shape[1] < 1 or response_ids.shape[1] < 1:
            raise ValueError("training prompt/response tensors must be nonempty")

    device = prompts[0].device
    dtype = prompts[0].dtype
    totals = [int(prompt.shape[1] + response.shape[1]) for prompt, response in zip(prompts, responses)]
    max_total = max(totals)
    input_ids = torch.full((len(examples), max_total), int(pad_token_id), dtype=dtype, device=device)
    attention_mask = torch.zeros((len(examples), max_total), dtype=torch.long, device=device)
    for row, (prompt_ids, response_ids, total) in enumerate(zip(prompts, responses, totals)):
        if prompt_ids.device != device or response_ids.device != device:
            raise ValueError("one training microbatch spans multiple devices")
        full_ids = torch.cat((prompt_ids, response_ids), dim=1)
        input_ids[row, :total] = full_ids[0]
        attention_mask[row, :total] = 1

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        response_ranges = []
        response_targets = []
        for prompt_ids, response_ids in zip(prompts, responses):
            prompt_len = int(prompt_ids.shape[1])
            response_len = int(response_ids.shape[1])
            response_ranges.append((prompt_len - 1, prompt_len - 1 + response_len))
            response_targets.append(response_ids[0])
        bounded = _decoder_token_logprobs(
            model, input_ids, attention_mask, response_ranges,
            response_targets, chunk=chunk, with_grad=with_grad)
        if bounded is not None:
            return bounded

        gathered_examples = []
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        for row, (prompt_ids, response_ids) in enumerate(zip(prompts, responses)):
            prompt_len = int(prompt_ids.shape[1])
            response_len = int(response_ids.shape[1])
            pred_logits = logits[row, prompt_len - 1:prompt_len - 1 + response_len, :]
            targets = response_ids[0].to(device=pred_logits.device)
            step = (int(chunk) if chunk and 0 < int(chunk) < response_len else response_len)
            parts = []
            for start in range(0, response_len, step):
                end = min(start + step, response_len)
                log_probs = F.log_softmax(pred_logits[start:end, :].float(), dim=-1)
                parts.append(log_probs.gather(1, targets[start:end].unsqueeze(-1)).squeeze(-1))
            gathered_examples.append(torch.cat(parts, dim=0))
    return gathered_examples


def _requires_fused_long_singleton(cfg, sequence_tokens):
    """Whether one long trajectory should use unmasked fused attention.

    Multi-example batches require padding or the custom shared-prefix mask and
    therefore cannot enter the exact full-sequence Flash-SDPA path.  Keep the
    configured microbatch cap for ordinary trajectories, but isolate genuinely
    long ones when the user explicitly enabled fused long attention.
    """
    return bool(
        getattr(cfg, "fused_long_attention", False)
        and int(sequence_tokens) >= _FUSED_LONG_SINGLETON_MIN_TOKENS
    )


def _is_cuda_oom(error):
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    return (isinstance(error, RuntimeError) and "out of memory" in str(error).lower() and "cuda" in str(error).lower())


def _is_attention_kernel_unavailable(error):
    """Recognize SDPA failures caused by excluding quadratic math attention."""
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return (
        "no available kernel" in message
        or "no viable backend for scaled_dot_product_attention" in message
        or "no suitable kernel" in message
        or message.strip() == "invalid backend"
    )


def _split_training_batches(batches):
    """Halve every non-singleton while preserving order and partitioning."""
    split = []
    changed = False
    for batch in batches:
        batch = list(batch)
        if len(batch) <= 1:
            split.append(batch)
            continue
        middle = (len(batch) + 1) // 2
        split.extend((batch[:middle], batch[middle:]))
        changed = True
    return split, changed


def _split_training_batches_to_work_cap(batches, work_cap):
    """Reuse a learned safe workload size without hard-coding batch counts.

    The OOM controller's work estimate accounts for ordinary padded batches
    and packed shared-prefix batches.  Recursively halving only batches above
    the learned cap preserves every example and the original order while
    avoiding a known-to-fail large attempt on later, similarly sized work.
    """
    work_cap = max(1, int(work_cap))
    split = []
    changed = False

    def append(batch):
        nonlocal changed
        batch = list(batch)
        if (len(batch) <= 1 or _gradient_checkpointing_work_units([batch]) <= work_cap):
            split.append(batch)
            return
        middle = (len(batch) + 1) // 2
        changed = True
        append(batch[:middle])
        append(batch[middle:])

    for batch in batches:
        if batch:
            append(batch)
    return split, changed


def _oom_profile_work_floor(work):
    """Lower edge of the workload neighborhood covered by an OOM result."""
    work = max(1, int(work))
    # Lengths of sibling rollouts are rarely identical.  A narrow 12.5%
    # neighborhood lets the next comparable branch reuse the proven mode,
    # while materially shorter work still gets a chance to use the fast path.
    return max(1, work - max(1, work // 8))


def _remember_oom_profile_minimum(profile, key, value):
    value = max(1, int(value))
    previous = profile.get(key)
    profile[key] = value if previous is None else min(int(previous), value)


def _gradient_checkpointing_work_units(batches):
    """Estimate the largest independently executed batch's activation load."""
    largest = 0
    for batch in batches:
        batch = list(batch)
        if not batch:
            continue
        prompt_job_id = batch[0].get("prompt_job_id")
        shared_prefix = bool(
            len(batch) > 1
            and prompt_job_id is not None
            and all(bool(example.get("_shared_prefix_packing_allowed", True)) for example in batch)
            and all(
                example.get("prompt_job_id") == prompt_job_id
                and tuple(example["prompt_ids"].shape)
                == tuple(batch[0]["prompt_ids"].shape)
                for example in batch[1:]
            )
        )
        if shared_prefix:
            work = int(batch[0]["prompt_ids"].shape[1]) + sum(
                max(0, int(example["response_ids"].shape[1]) - 1)
                for example in batch
            )
        else:
            work = max(
                int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1])
                for example in batch
            ) * len(batch)
        largest = max(largest, work)
    return largest


def _run_oom_resilient_backward(
        model, batches, attempt, *, device_label, expand_memory=None,
        start_checkpointed=False, start_with_headroom=False,
        headroom_preallocated=False,
        start_activation_offload=False,
        activation_offload_semaphore=None, defer_exact_oom=False,
        allow_activation_offload=True, split_failed_batches=True):
    """Retry with checkpointing, smaller batches, and memory fallbacks.

    Each attempt starts the local replica's gradients from zero, so an OOM
    during backward cannot double-count a partially accumulated microbatch.
    Ordinary workloads first run without transformer gradient checkpointing.
    A caller that has identified an exact fused long singleton can start with
    checkpointing and reserved GPU headroom, avoiding known-failing attempts.
    Otherwise an OOM retries with checkpointing before batches are split.
    The learned safe work cap, checkpointing threshold, CPU-offload threshold,
    and reserved-headroom threshold are retained on the persistent model, so
    comparable later batches start directly in the proven execution mode.
    Once batches reach one example, fast training uses reserved card headroom
    before the much slower saved-tensor CPU offload. A truly untrainable
    outlier is quarantined only after every exact fallback has failed.  The
    strict SPO-RS replica path may instead return that singleton in the third
    result so its gradient can be computed by the exact multi-GPU overflow
    lane without discarding gradients already completed on other replicas.
    """
    import gc
    import traceback
    import torch
    from runtime.host_memory import (ProcessRSSPeakSampler,
                             host_cache_reclaim_supported,
                             process_rss_bytes,
                             reclaim_unused_host_memory)
    from models.model_backend import set_gradient_checkpointing

    # Saved-activation offload is safe in the replicated SPO-RS path only when
    # every process shares a bounded node-wide admission gate. Without that
    # gate, several 100K+ token graphs can consume host RAM concurrently and
    # OOM-kill the VM. The model-sharded fallback deliberately supplies no
    # gate and therefore continues to use GPU checkpointing/headroom only.
    if defer_exact_oom and activation_offload_semaphore is None:
        allow_activation_offload = False

    def offload_gate_can_admit(work):
        if activation_offload_semaphore is None:
            return True
        checker = getattr(activation_offload_semaphore, "can_admit", None)
        return bool(checker(work)) if callable(checker) else True

    def offload_admission(work):
        if activation_offload_semaphore is None:
            return nullcontext()
        reservation = getattr(activation_offload_semaphore, "reservation", None)
        if callable(reservation):
            return reservation(work)
        return activation_offload_semaphore

    active = [list(batch) for batch in batches if batch]
    quarantined = []
    profile = getattr(model, "_ttt_oom_execution_profile", None)
    if not isinstance(profile, dict):
        profile = {}
        setattr(model, "_ttt_oom_execution_profile", profile)

    cached_work_cap = profile.get("safe_work_cap")
    reused_split = False
    if split_failed_batches and cached_work_cap is not None:
        active, reused_split = _split_training_batches_to_work_cap(active, cached_work_cap)

    strict_exact_training = bool(getattr(model, "_ttt_strict_exact_long_training", False))
    activation_offload = False
    memory_expanded = False
    expanded_fraction = None
    split_backoff = False
    offload_trigger_work = None
    headroom_trigger_work = None
    checkpoint_threshold = getattr(model, "_ttt_gradient_checkpointing_min_work", None)
    checkpointing_available = set_gradient_checkpointing(model, False)
    checkpointing = False

    def expand_to_next_memory_ceiling():
        """Request one strictly larger exact-training allocator ceiling."""
        nonlocal expanded_fraction
        if expand_memory is None:
            return None
        candidate = expand_memory()
        if candidate is None:
            return None
        candidate = float(candidate)
        if (expanded_fraction is not None and candidate <= float(expanded_fraction) + 1e-9):
            return None
        expanded_fraction = candidate
        return candidate

    def select_checkpointing():
        nonlocal checkpointing, checkpointing_available
        work = _gradient_checkpointing_work_units(active)
        wanted = bool(
            checkpointing_available
            and (bool(start_checkpointed) or (checkpoint_threshold is not None and work >= int(checkpoint_threshold)))
        )
        if wanted != checkpointing:
            if not set_gradient_checkpointing(model, wanted):
                checkpointing_available = False
                wanted = False
            checkpointing = wanted
        return work

    active_work = select_checkpointing()
    configured_headroom_min_work = (
        None if headroom_preallocated else
        getattr(model, "_ttt_reserved_headroom_min_work", None))
    headroom_min_work = profile.get("headroom_min_work", configured_headroom_min_work)
    if ((bool(start_with_headroom) or (headroom_min_work is not None and active_work >= int(headroom_min_work)))
            and expand_memory is not None):
        memory_expanded = expand_to_next_memory_ceiling() is not None
    configured_offload_min_work = getattr(model, "_ttt_activation_offload_min_work", None)
    offload_min_work = profile.get("activation_offload_min_work", configured_offload_min_work)
    if (allow_activation_offload
            and (bool(start_activation_offload)
                 or (offload_min_work is not None and active_work >= int(offload_min_work)))
            and offload_gate_can_admit(active_work)
            and hasattr(torch.autograd.graph, "save_on_cpu")):
        activation_offload = True

    reused_modes = []
    if reused_split:
        reused_modes.append(f"work cap {int(cached_work_cap)}")
    if checkpointing:
        reused_modes.append("gradient checkpointing")
    if activation_offload:
        reused_modes.append("CPU activation offload")
    if memory_expanded:
        reused_modes.append("reserved GPU headroom")
    if reused_modes:
        signature = tuple(reused_modes)
        if getattr(model, "_ttt_reported_oom_profile", None) != signature:
            modes = ", ".join(reused_modes)
            if start_checkpointed or start_with_headroom:
                setting_log_only(f"[train-memory] {device_label}: starting long-sequence mode ({modes})", flush=True)
            else:
                print(f"[train-oom] {device_label}: reusing learned safe mode ({modes})", flush=True)
            setattr(model, "_ttt_reported_oom_profile", signature)

    cuda_devices = sorted({
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda" and parameter.device.index is not None
    })
    while active:
        model.zero_grad(set_to_none=True)
        try:
            # Pinned saved tensors make the GPU<->CPU replay substantially
            # faster, but PyTorch caches the released host blocks separately
            # from the CUDA device allocator.  Use pinned storage only when
            # this runtime exposes a way to return that cache to the OS.
            admission = (offload_admission(active_work) if activation_offload else nullcontext())
            # Keep the admission token until inactive pinned blocks and libc
            # arenas have been returned. Otherwise the next trainer can begin
            # a large offload while this process still owns the prior batch's
            # host cache, recreating the node-wide RAM spike.
            with admission:
                offload_rss_before = (process_rss_bytes() if activation_offload else None)
                rss_sampler = (ProcessRSSPeakSampler() if activation_offload else nullcontext())
                offload_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=host_cache_reclaim_supported(torch))
                    if activation_offload
                    and hasattr(torch.autograd.graph, "save_on_cpu")
                    else nullcontext()
                )
                try:
                    with rss_sampler as sampled_rss:
                        with offload_context:
                            result = attempt(active)
                except BaseException as error:
                    # Clean the failed graph before releasing the shared
                    # admission slot. Otherwise another rank can begin a CPU
                    # offload while this exception still retains saved
                    # tensors, recreating the node-wide host-RAM spike.
                    if activation_offload:
                        traceback.clear_frames(error.__traceback__)
                        model.zero_grad(set_to_none=True)
                        gc.collect()
                        for cuda_device in cuda_devices:
                            with torch.cuda.device(cuda_device):
                                torch.cuda.empty_cache()
                        reclaim_unused_host_memory(torch, cuda_devices=cuda_devices, force=True)
                    raise
                else:
                    if activation_offload:
                        cleanup = reclaim_unused_host_memory(torch, cuda_devices=cuda_devices, force=True)
                        # Forced cleanup records the pinned allocator and RSS
                        # immediately before releasing the saved activations.
                        # Publish the larger measurement while this rank still
                        # owns its reservation; waiting ranks then recompute
                        # their token-weighted request before they can enter.
                        observer = getattr(activation_offload_semaphore, "observe", None)
                        if callable(observer):
                            footprint = int(cleanup.get("pinned_owned_before", 0) or 0)
                            rss_at_cleanup = cleanup.get("rss_before")
                            if (offload_rss_before is not None and rss_at_cleanup is not None):
                                footprint = max(footprint, max(0, int(rss_at_cleanup) - int(offload_rss_before)))
                            footprint = max(footprint, int(getattr(sampled_rss, "footprint_bytes", 0) or 0))
                            observer(active_work, footprint)
            if split_backoff and active:
                successful_work = _gradient_checkpointing_work_units(active)
                _remember_oom_profile_minimum(profile, "safe_work_cap", successful_work)
            if offload_trigger_work is not None:
                _remember_oom_profile_minimum(
                    profile, "activation_offload_min_work",
                    _oom_profile_work_floor(offload_trigger_work))
            if headroom_trigger_work is not None:
                _remember_oom_profile_minimum(
                    profile, "headroom_min_work",
                    _oom_profile_work_floor(headroom_trigger_work))
            set_gradient_checkpointing(model, False)
            return result, active, quarantined
        except BaseException as error:
            cuda_oom = _is_cuda_oom(error)
            kernel_unavailable = _is_attention_kernel_unavailable(error)
            if not cuda_oom and not kernel_unavailable:
                set_gradient_checkpointing(model, False)
                raise
            traceback.clear_frames(error.__traceback__)
            model.zero_grad(set_to_none=True)
            gc.collect()
            for cuda_device in cuda_devices:
                with torch.cuda.device(cuda_device):
                    torch.cuda.empty_cache()
            if cuda_oom and not checkpointing and checkpointing_available:
                learned_checkpoint_floor = _oom_profile_work_floor(active_work)
                checkpoint_threshold = (
                    learned_checkpoint_floor
                    if checkpoint_threshold is None else
                    min(int(checkpoint_threshold), learned_checkpoint_floor)
                )
                setattr(model, "_ttt_gradient_checkpointing_min_work", int(checkpoint_threshold))
                if set_gradient_checkpointing(model, True):
                    checkpointing = True
                    setting_log_only(
                        f"[train-oom] {device_label}: retrying the same "
                        "workload with gradient checkpointing",
                        flush=True,
                    )
                    continue
                checkpointing_available = False
            if not split_failed_batches and any(len(batch) > 1 for batch in active):
                # The caller commits the children separately. Retrying a
                # split list here could discard earlier successful backward.
                set_gradient_checkpointing(model, False)
                raise
            smaller, changed = _split_training_batches(active)
            if changed:
                active = smaller
                split_backoff = True
                active_work = select_checkpointing()
                reason = ("fused attention rejected the padded batch" if kernel_unavailable else "CUDA OOM")
                print(f"[train-oom] {device_label}: {reason}; retrying with maximum "
                      f"microbatch {max(len(batch) for batch in active)}",
                      flush=True)
                continue
            if cuda_oom and expand_memory is not None:
                previous_fraction = expanded_fraction
                next_fraction = expand_to_next_memory_ceiling()
                if next_fraction is not None:
                    memory_expanded = True
                    headroom_trigger_work = active_work
                    retry_label = ("singleton OOM" if previous_fraction is None else "singleton still OOM")
                    print(
                        f"[train-oom] {device_label}: {retry_label}; retrying "
                        "the exact update with "
                        f"reserved GPU headroom (allocator cap "
                        f"{100.0 * next_fraction:.1f}%)",
                        flush=True,
                    )
                    continue
            if (cuda_oom and allow_activation_offload
                    and not activation_offload
                    and offload_gate_can_admit(active_work)
                    and hasattr(torch.autograd.graph, "save_on_cpu")):
                activation_offload = True
                offload_trigger_work = active_work
                setting_log_only(
                    f"[train-oom] {device_label}: singleton still OOM after "
                    "GPU headroom; retrying with saved activations offloaded "
                    "to CPU", flush=True)
                continue

            victim_index = max(
                range(len(active)),
                key=lambda index: int(
                    active[index][0]["prompt_ids"].shape[1]
                    + active[index][0]["response_ids"].shape[1]),
            )
            victim_batch = active.pop(victim_index)
            victim = victim_batch[0]
            victim_tokens = int(victim["prompt_ids"].shape[1] + victim["response_ids"].shape[1])
            if strict_exact_training:
                if defer_exact_oom:
                    quarantined.append(victim)
                    setting_log_only(
                        f"[train-overflow] {device_label}: deferring one "
                        f"{victim_tokens}-token rollout to the exact "
                        "multi-GPU overflow lane; completed replica "
                        "gradients remain valid",
                        flush=True,
                    )
                    # Continue with any other singleton batches in this local
                    # workload. ``attempt`` starts from zero after every OOM,
                    # while gradients from earlier work-queue items live in
                    # the caller's separate accumulators.
                    active_work = select_checkpointing()
                    continue
                set_gradient_checkpointing(model, False)
                model.zero_grad(set_to_none=True)
                if kernel_unavailable:
                    failure = ("no compatible exact linear-memory attention kernel")
                elif allow_activation_offload:
                    failure = (
                        "insufficient memory after checkpointing, reserved "
                        "headroom, and CPU activation offload")
                else:
                    failure = (
                        "insufficient memory after checkpointing and all "
                        "reserved-headroom stages; CPU activation offload "
                        "was disabled for this GPU-only overflow pass")
                raise ExactLongContextTrainingOOM(
                    f"exact long-context training failed for a "
                    f"{victim_tokens}-token rollout on {device_label}: "
                    f"{failure}. The configured coder profile forbids "
                    "truncating or silently excluding this trajectory.") from error
            quarantined.append(victim)
            # Once the reserved-headroom path was required, keep activations
            # offloaded while checking the remaining long singletons. This
            # avoids repeating one guaranteed-to-fail ordinary attempt for
            # every neighboring rollout.
            activation_offload = memory_expanded
            if kernel_unavailable:
                reason = "has no compatible linear-memory attention kernel"
            else:
                reason = "cannot fit even with activation offload"
            print(f"[train-oom] {device_label}: one {victim_tokens}-token "
                  f"rollout {reason}; excluding only that rollout from this "
                  "adapter update", flush=True)
            active_work = select_checkpointing()
    model.zero_grad(set_to_none=True)
    set_gradient_checkpointing(model, False)
    return None, [], quarantined


def _valid_example_token_logprobs(example, key):
    """Whether a cached vector is finite and aligned to the response."""
    import torch

    value = example.get(key)
    response_len = int(example["response_ids"].shape[1])
    return bool(
        torch.is_tensor(value)
        and value.ndim == 1
        and value.shape[0] == response_len
        and torch.isfinite(value).all()
    )


def _initialize_behavior_logprobs(examples):
    """Cache frozen rollout scores and return trajectories needing HF scoring."""
    missing = []
    for example in examples:
        if _valid_example_token_logprobs(example, "behavior_logprobs"):
            example["old_policy_logprobs"] = example["behavior_logprobs"].detach()
        else:
            example["old_policy_logprobs"] = None
            missing.append(example)
    return missing


def clipped_surrogate_loss(
        current_logprobs, old_logprobs, reference_logprob,
        advantage, *, clip_epsilon=SPO_RS_CLIP_EPSILON_DEFAULT,
        clip_epsilon_low=None, clip_epsilon_high=None, kl_coef=0.0,
        return_tensor_metrics=False):
    """One trajectory's loss; callers average equally within each parent.

    The per-token ratio exp(current_lp - old_lp) is differentiable. Old
    log-probabilities and the scalar group advantage are frozen. Reference
    regularization uses the same centered correction as entropic mode:
        d = log pi_theta - log pi_ref
        A_KL = kl_coef * (mean(d) - d).
    Its policy-gradient surrogate is kept separate from the clipped PPO surrogate
    so SPO-RS advantages and asymmetric clipping retain their existing behavior.
    Clipping is asymmetric when requested: [1 - epsilon_low,
    1 + epsilon_high]. The legacy clip_epsilon remains the fallback for both.

    Accumulation and exponentiation use float64, and nonfinite ratios fail
    explicitly rather than silently clamping the objective.
    """
    import torch

    symmetric_epsilon = float(clip_epsilon)
    epsilon_low = (symmetric_epsilon if clip_epsilon_low is None else float(clip_epsilon_low))
    epsilon_high = (symmetric_epsilon if clip_epsilon_high is None else float(clip_epsilon_high))
    if (not math.isfinite(epsilon_low) or not 0.0 < epsilon_low < 1.0
            or not math.isfinite(epsilon_high)
            or not 0.0 < epsilon_high < 1.0):
        raise ValueError("clip epsilon low/high must be finite and in (0, 1)")
    if not math.isfinite(float(kl_coef)) or kl_coef < 0.0:
        raise ValueError("KL coefficient must be finite and nonnegative")
    cur = current_logprobs.to(dtype=torch.float64)
    old = torch.as_tensor(old_logprobs, dtype=cur.dtype, device=cur.device).detach()
    if cur.ndim != 1 or old.shape != cur.shape or cur.numel() == 0:
        raise ValueError("current and old logprobs must be matching nonempty vectors")
    ref = (None if reference_logprob is None else
           torch.as_tensor(reference_logprob, dtype=cur.dtype, device=cur.device).detach())
    adv = torch.as_tensor(advantage, dtype=cur.dtype, device=cur.device).detach()
    if adv.numel() != 1 or not torch.isfinite(adv).all():
        raise ValueError("policy advantage must be a finite scalar")
    if not torch.isfinite(cur).all() or not torch.isfinite(old).all():
        raise ValueError("policy logprobs must be finite")

    policy_ratios = (cur - old).exp()
    clipped_policy_ratios = policy_ratios.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    policy_loss = -torch.minimum(policy_ratios * adv, clipped_policy_ratios * adv).mean()
    ratio = policy_ratios.mean()
    clipped_fraction = (policy_ratios.detach() != clipped_policy_ratios.detach()).double().mean()
    policy_reference_delta = cur.new_zeros(())
    kl_policy_loss = cur.new_zeros(())
    if kl_coef:
        if reference_logprob is None:
            raise ValueError("reference logprobs are required for nonzero KL coefficient")
        if ref.shape != cur.shape or not torch.isfinite(ref).all():
            raise ValueError("reference logprobs must be a finite vector matching current logprobs")
        logp_difference = (cur - ref).detach()
        policy_reference_delta = logp_difference.mean()
        kl_advantage = kl_coef * (policy_reference_delta - (cur - ref))
        kl_policy_loss = -(policy_ratios.detach() * kl_advantage.detach() * cur).mean()

    prefix_ratio_max = cur.new_ones(())
    loss = policy_loss + kl_policy_loss
    if not torch.isfinite(loss).all() or not torch.isfinite(ratio).all():
        raise FloatingPointError(
            "nonfinite clipped objective/trajectory ratio; reduce learning "
            "rate or update epochs and inspect model logprobs")
    tensor_metrics = {
        "policy_loss": policy_loss.detach(),
        # Retain the metric key for checkpoint/log compatibility. Its value now
        # matches entropic mode's mean log pi_theta - log pi_base diagnostic.
        "kl_estimate": policy_reference_delta.detach(),
        "ratio": ratio.detach(),
        "prefix_ratio_max": prefix_ratio_max.detach(),
        "clipped": clipped_fraction.detach(),
    }
    if return_tensor_metrics:
        return loss, tensor_metrics
    return loss, {key: float(value.item()) for key, value in tensor_metrics.items()}


@contextmanager
def _policy_dropout_disabled(model):
    """Disable dropout while retaining training/checkpointing mode."""
    import torch
    changed = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            changed.append((module, "p", module.p))
            module.p = 0.0
        # Common attention implementations use functional dropout instead of
        # an nn.Dropout module. Generation is deterministic conditional on
        # sampled tokens; old/current likelihood forwards must match it.
        for name in ("attention_dropout", "attn_dropout", "dropout"):
            value = getattr(module, name, None)
            if isinstance(value, float) and value != 0.0:
                changed.append((module, name, value))
                setattr(module, name, 0.0)
    try:
        yield
    finally:
        for module, name, value in reversed(changed):
            setattr(module, name, value)


def _clipped_policy_options(cfg):
    """Return PPO epsilon, lower epsilon, upper epsilon, and reference-KL coef."""
    epsilon = float(getattr(cfg, "spo_rs_clip_epsilon", SPO_RS_CLIP_EPSILON_DEFAULT))
    epsilon_low = float(getattr(cfg, "spo_rs_clip_epsilon_low", SPO_RS_CLIP_EPSILON_LOW_DEFAULT))
    epsilon_high = float(getattr(cfg, "spo_rs_clip_epsilon_high", SPO_RS_CLIP_EPSILON_HIGH_DEFAULT))
    return epsilon, epsilon_low, epsilon_high, 0.0


def historical_policy_loss(cfg, current_logprobs, example, clip_options, *,
                           return_tensor_metrics=False, check_finite=True):
    """Use one objective across the replicas and both overflow paths."""
    if uses_ttt_policy_loss(cfg.advantage_mode):
        from training.hybrid import importance_sampling_loss

        return importance_sampling_loss(
            current_logprobs, example, cfg.kl_penalty_coef,
            return_tensor_metrics=return_tensor_metrics, check_finite=check_finite)

    epsilon, low, high, kl = clip_options
    return clipped_surrogate_loss(
        current_logprobs, example["old_policy_logprobs"], None,
        example["advantage"], clip_epsilon=epsilon, clip_epsilon_low=low,
        clip_epsilon_high=high, kl_coef=kl,
        return_tensor_metrics=return_tensor_metrics)


def _historical_grad_clip(cfg):
    # Keep the nonfinite gradient check when the TTT update clips no gradient.
    return math.inf if uses_ttt_policy_loss(cfg.advantage_mode) else cfg.grad_clip


def _prepare_entropic_overlap_group(
        tokenizer, group_id, responses, reward_futures, prompt_jobs,
        prompt_ids_by_job, num_groups, *, reference_logprob_mean):
    """Materialize one reward-complete entropic group on CPU.

    Entropic advantages depend on every reward in their group, but they do not
    depend on rewards from any other group.  As soon as one group is complete,
    this builds the exact examples used by the ordinary post-evaluation path so
    process-distributed forward/backward can begin while other groups remain in
    the CPU evaluator.  Parameters are not updated here.
    """
    import torch

    if len(responses) != len(reward_futures):
        raise RuntimeError(
            f"entropic overlap group {group_id} has {len(responses)} "
            f"responses but {len(reward_futures)} reward futures")
    if any(not future.done() for future in reward_futures):
        raise RuntimeError(f'entropic overlap group {group_id} was prepared before all of its rewards completed')

    rewards = np.asarray([float(future.result().reward) for future in reward_futures], dtype=np.float64)
    if rewards.size == 0:
        return []
    advantages, _beta = compute_group_advantages(rewards)
    constant = float(rewards.max() - rewards.min()) < 1e-12

    prepared = []
    for rollout_index, (record, advantage) in enumerate(zip(responses, advantages)):
        token_ids = list(record.get("token_ids") or ())
        if not token_ids:
            continue
        job_idx = int(record["job_idx"])
        if job_idx not in prompt_ids_by_job:
            prompt_ids_by_job[job_idx] = tokenizer(
                prompt_jobs[job_idx]["prompt_text"],
                return_tensors="pt").input_ids.cpu()

        behavior_values = record.get("behavior_logprobs")
        reference_values = record.get("reference_logprobs")
        behavior_logprobs = (
            torch.as_tensor(behavior_values, dtype=torch.float32).cpu()
            if behavior_values is not None
            and len(behavior_values) == len(token_ids) else None)
        reference_logprobs = (
            torch.as_tensor(reference_values, dtype=torch.float32).cpu()
            if reference_values is not None
            and len(reference_values) == len(token_ids) else None)
        example = {
            "prompt_ids": prompt_ids_by_job[job_idx],
            "response_ids": torch.tensor([token_ids], dtype=torch.long, device="cpu"),
            "advantage": float(advantage),
            "behavior_logprobs": behavior_logprobs,
            "reference_logprobs": reference_logprobs,
            "reference_logprob_mean": reference_logprob_mean,
            "reward_constant": constant,
            "group_id": int(group_id),
            "rollout_index": int(rollout_index),
            "prompt_job_id": job_idx,
            "sample_weight": 1.0 / (int(num_groups) * len(responses)),
            "_overlap_key": (int(group_id), int(rollout_index)),
        }
        record["_entropic_overlap_example"] = example
        prepared.append(example)
    return prepared


def _prepare_spo_rs_overlap_inputs(tokenizer, group_responses, reward_futures, prompt_jobs, *,
                                   reference_logprob_mean=None):
    """Build every reward-independent SPO-RS training input on CPU.

    SPO-RS uses rollout-policy logprobs produced by vLLM and a frozen
    pre-step tracker baseline.  Prompt/response tensors, behavior scores, the
    exact step-wide trajectory weight, and scheduling keys therefore exist
    before sandbox rewards finish.  The reward future stays only on the local
    staging object and is removed before the example enters a multiprocessing
    queue.
    """
    import torch

    prompt_ids_by_job = {}
    prepared = []
    for group_id in sorted(group_responses):
        responses = group_responses[group_id]
        futures = reward_futures[group_id]
        if len(responses) != len(futures):
            raise RuntimeError(
                f"SPO-RS overlap group {group_id} has {len(responses)} "
                f"responses but {len(futures)} reward futures")
        for rollout_index, (record, reward_future) in enumerate(zip(responses, futures)):
            token_ids = list(record.get("token_ids") or ())
            behavior_values = record.get("behavior_logprobs")
            if not token_ids or behavior_values is None:
                continue
            try:
                behavior_array = np.asarray(behavior_values, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if (behavior_array.ndim != 1
                    or behavior_array.size != len(token_ids)
                    or not np.isfinite(behavior_array).all()):
                continue

            job_idx = int(record["job_idx"])
            if job_idx not in prompt_ids_by_job:
                prompt_ids_by_job[job_idx] = tokenizer(
                    prompt_jobs[job_idx]["prompt_text"],
                    return_tensors="pt").input_ids.cpu()
            example = {
                "prompt_ids": prompt_ids_by_job[job_idx],
                "response_ids": torch.tensor([token_ids], dtype=torch.long, device="cpu"),
                "advantage": None,
                "behavior_logprobs": torch.as_tensor(behavior_array, dtype=torch.float32).cpu(),
                "reference_logprobs": None,
                "reward_constant": False,
                "group_id": int(group_id),
                "rollout_index": int(rollout_index),
                "prompt_job_id": job_idx,
                "sample_weight": 0.0,
                "_overlap_key": (int(group_id), int(rollout_index)),
                "_overlap_reward_future": reward_future,
            }
            if reference_logprob_mean is not None:
                example["reference_logprobs"] = torch.as_tensor(
                    record["reference_logprobs"], dtype=torch.float32)
                example["reference_logprob_mean"] = reference_logprob_mean
            record["_spo_rs_overlap_example"] = example
            prepared.append(example)

    if prepared:
        trajectory_weight = 1.0 / len(prepared)
        for example in prepared:
            example["sample_weight"] = trajectory_weight
    return prepared


def _complete_spo_rs_overlap_examples(examples, spo_rs_tracker=None, *, sibling_futures=None,
                                      ttt_weight=DEFAULT_TTT_WEIGHT,
                                      spo_weight=DEFAULT_SPO_WEIGHT):
    """Attach advantages to reward-complete trajectories."""
    if not examples:
        return []
    futures = [example.get("_overlap_reward_future") for example in examples]
    if any(future is None or not future.done() for future in futures):
        raise RuntimeError("SPO-RS overlap received an example before its reward completed")
    if spo_rs_tracker is None:
        raise RuntimeError("SPO-RS overlap requires its historical tracker")
    results = [future.result() for future in (futures if sibling_futures is None else sibling_futures)]
    rewards = [float(result.reward) for result in results]
    advantages = spo_rs_tracker.rollout_advantages(rewards)
    if advantages is None:
        raise RuntimeError("SPO-RS overlap lost its frozen pre-step tracker baseline")
    if sibling_futures is not None:
        advantages, _ = hybrid_advantages(
            rewards, advantages, ttt_weight=ttt_weight, spo_weight=spo_weight)
        advantages = [advantages[example["rollout_index"]] for example in examples]
    for example, advantage in zip(examples, advantages):
        example["advantage"] = float(advantage)
        # Futures cannot be sent through the process trainer's work queues.
        example.pop("_overlap_reward_future", None)
    return examples


def _apply_precomputed_exact_gradients(model, overflow_state, step_idx):
    """Restore the durable gradients for the full process-trainer update."""
    import torch

    saved_gradients = _validate_precomputed_exact_gradients(model, overflow_state, step_idx)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            saved = saved_gradients[name]
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(saved.to(device=parameter.device, dtype=parameter.grad.dtype))


def _finalize_exact_spo_rs_overflow(
        model, optimizer, cfg, step_idx, overflow_state, *,
        completed_before=(), label="durable exact overflow recovery"):
    """Clip and apply the one full-batch update, never a partial overflow update."""
    if overflow_state.get("remaining_keys"):
        raise RuntimeError("cannot update the policy with unfinished trajectories")
    original_keys = overflow_state.get("deferred_keys", ())
    # Materialize the saved full-batch gradients before clipping and updating once.
    finalize_started = time.time()
    optimizer.zero_grad(set_to_none=True)
    _apply_precomputed_exact_gradients(model, overflow_state, step_idx)
    grad_norm = _clip_sharded_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        _historical_grad_clip(cfg), error_if_nonfinite=True)
    optimizer.step()

    metric_keys = (
        "loss", "policy_loss", "kl_estimate",
        "ratio", "clipped_fraction",
    )
    normal_stats = overflow_state.get("normal_stats", {})
    overflow_stats = overflow_state.get("overflow_stats", {})
    normal_totals = normal_stats.get("totals", {})
    overflow_totals = overflow_stats.get("totals", {})
    totals = {key: (float(normal_totals.get(key, 0.0)) + float(overflow_totals.get(key, 0.0))) for key in metric_keys}
    max_ratio = max(float(normal_stats.get("max_ratio", 0.0)), float(overflow_stats.get("max_ratio", 0.0)))
    max_prefix_ratio = max(
        float(normal_stats.get("max_prefix_ratio", 0.0)),
        float(overflow_stats.get("max_prefix_ratio", 0.0)))
    history = [{
        "epoch": 1,
        **totals,
        "ratio_max": max_ratio,
        "prefix_ratio_max": max_prefix_ratio,
        "oom_quarantined_examples": 0,
        "grad_norm": float(grad_norm),
    }]
    print(
        f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
        f"loss={totals['loss']:.6f} "
        f"reference={'on' if uses_ttt_policy_loss(cfg.advantage_mode) and cfg.kl_penalty_coef else 'off'} "
        f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
        f"clipped={totals['clipped_fraction']:.1%} "
        "OOM-quarantined=0 "
        f"({label})",
        flush=True,
    )
    return {
        "policy_updates": history,
        "training_seconds": (
            float(normal_stats.get("training_seconds", 0.0))
            + float(overflow_stats.get("training_seconds", 0.0))
            + (time.time() - finalize_started)),
        "oom_quarantined_examples": 0,
        "exact_overflow_reused_examples": (int(normal_stats.get("trained_examples", 0)) + len(completed_before)),
        "exact_overflow_sharded_examples": len(original_keys),
        "exact_overflow_resumed_examples": len(completed_before),
    }


class ProcessDistributedTrainer:
    """Fast LoRA trainer with one persistent Python process per GPU."""

    def __init__(self, cfg, exp_dir, dependency_log_path=None):
        import multiprocessing as mp
        import queue
        import uuid

        self.cfg = cfg
        self._exp_dir = Path(exp_dir).resolve()
        self._dependency_log_path = dependency_log_path
        self._process_label = "train-fast"
        self._offloaded = False
        self._offloaded_ranks = set()
        self._closed = False
        self._workers_idle = True
        self._entropic_overlap = None
        self._spo_rs_overlap = None
        self._dist_initialized = False
        self._queue_module = queue
        self._physical_ids = _parse_gpu_ids(cfg.training_gpu_ids)
        self._world_size = int(cfg.num_training_gpus)
        self._memory_fraction = float(getattr(cfg, "training_memory_fraction", 0.80))
        self._memory_percentage = round(100.0 * self._memory_fraction, 1)
        if self._world_size != len(self._physical_ids):
            raise ValueError("num_training_gpus does not match training_gpu_ids")
        if self._world_size < 1:
            raise ValueError("process-distributed training requires at least one GPU")

        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._work_queue = self._context.Queue()
        self._command_queues = {}
        self._processes = {}

        from runtime.host_memory import (
            GIB, TokenWeightedActivationOffloadGate,
            automatic_activation_offload_budget_bytes,
        )
        hardware_profile = getattr(cfg, "hardware_profile", "RTX6000")
        self._activation_offload_budget_bytes = automatic_activation_offload_budget_bytes()
        self._activation_offload_semaphore = (
            TokenWeightedActivationOffloadGate(
                self._context,
                self._activation_offload_budget_bytes,
                max(1, int(cfg.max_seq_length)),
            ))
        self._activation_offload_slots = (self._activation_offload_semaphore.full_context_concurrency)
        self._activation_offload_semaphore.host_admission.hardware_profile = hardware_profile
        self._activation_offload_budget_gib = (self._activation_offload_budget_bytes / GIB)

        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        sync_dir = self._exp_dir / ".fast_trainer"
        sync_dir.mkdir(parents=True, exist_ok=True)
        rendezvous_path = sync_dir / (f"nccl-{os.getpid()}-{uuid.uuid4().hex}")
        self._init_method = rendezvous_path.as_uri()

    def _start_replica(self, rank):
        from training.fast_distributed import worker_main

        cfg_dict = dict(vars(self.cfg))
        cfg_dict["_terminal_log_path"] = str(self._exp_dir / "temirnal.log")
        cfg_dict["_setting_log_path"] = str(self._exp_dir / "setting.log")
        command_queue = self._context.Queue(maxsize=2)
        self._command_queues[rank] = command_queue
        process = self._context.Process(
            target=worker_main,
            args=(
                rank, self._world_size, cfg_dict,
                self._init_method, self._work_queue, command_queue,
                self._result_queue, self._dependency_log_path,
                self._activation_offload_semaphore,
            ),
            name=f"ttt-fast-trainer-rank-{rank}",
        )
        process.start()
        self._processes[rank] = process

    @contextmanager
    def loading_replicas(self):
        """Start child loads before rank zero loads when host RAM permits."""
        from runtime.gpu_runtime import _estimated_weight_gib
        from runtime.host_memory import GIB, HostMemoryUnavailable

        try:
            with ExitStack() as reservations:
                if self.host_memory_admission.hardware_profile == "B200_full_ram":
                    model_gib = _estimated_weight_gib(self.cfg.coder_model_name, "auto")
                    # Include the loader's per-replica staging allowance and
                    # rank zero's saved state, which can be restored in parallel.
                    required = math.ceil((model_gib + 8) * self._world_size * GIB)
                    state_path = self._exp_dir / "training_state.pt"
                    if state_path.is_file():
                        required += state_path.stat().st_size
                    try:
                        reserved = reservations.enter_context(
                            self.host_memory_admission.reservation(
                                required, stage="concurrent trainer loading",
                                priority=True, blocking=False))
                    except HostMemoryUnavailable:
                        print(f"[{self._process_label}] loading trainers serially; "
                              f"{required / GIB:.1f} GiB of host RAM required for concurrent startup",
                              flush=True)
                    else:
                        log_loading = setting_log_only if getattr(self.cfg, "_startup_step", 0) > 0 else print
                        log_loading(f"[{self._process_label}] loading {self._world_size} trainer replicas "
                                    f"concurrently with {reserved / GIB:.1f} GiB of host RAM reserved",
                                    flush=True)
                        for rank in range(1, self._world_size):
                            self._start_replica(rank)
                yield
        except BaseException:
            self._abort_workers()
            raise

    def finish_startup(self, backend, model, tokenizer, optimizer):
        """Wait for every replica and synchronize rank zero's restored adapter."""
        from datetime import timedelta
        import torch.distributed as dist
        import torch.multiprocessing  # Register tensor reducers for the queues.
        from training.fast_distributed import (
            broadcast_trainable_parameters, initial_training_token_budget,
            set_total_memory_ceiling, trainable_parameter_signature, validate_model_device,
        )

        if dist.is_initialized():
            raise RuntimeError(
                "the process trainer cannot start because torch.distributed is already "
                "initialized in the main process")
        self.backend = backend
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self._token_budgets = [initial_training_token_budget(model, self.cfg)] * self._world_size

        torch.set_num_threads(max(1, int(os.cpu_count() or self._world_size) // self._world_size))
        torch.cuda.set_device(0)
        set_total_memory_ceiling(0, self._memory_fraction)
        validate_model_device(model, 0)
        expected_signature = trainable_parameter_signature(model)
        ranks = range(1, self._world_size)
        loaded = self._collect_event("loaded", ranks) if self._processes else {}
        if not loaded:
            setting_log_only(
                f"[{self._process_label}] loading trainer replicas one at a time",
                flush=True)
        for rank in ranks:
            message = loaded.get(rank)
            if message is None:
                self._start_replica(rank)
                message = self._collect_event("loaded", (rank,))[rank]
            if message.get("parameter_signature") != expected_signature:
                raise RuntimeError(f'fast trainer rank {rank} has a different trainable parameter layout')
            setting_log_only(f"[{self._process_label}] trainer replica {rank + 1}/"
                             f"{self._world_size} loaded on logical GPU {rank}", flush=True)

        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "init_distributed"})
        dist.init_process_group(
            backend="nccl", init_method=self._init_method,
            rank=0, world_size=self._world_size,
            timeout=timedelta(minutes=30))
        self._dist_initialized = True
        broadcast_trainable_parameters(model, source_rank=0)
        self._collect_event("ready", ranks)

        fallback_description = (
            "token-weighted CPU activation offload uses a "
            f"{self._activation_offload_budget_gib:.0f}-GiB RAM budget "
            f"({self._activation_offload_slots} full-context or up to "
            f"{self._world_size} shorter ranks)")
        setting_log_only(
            f"[{self._process_label}] process-distributed trainer active on "
            f"physical GPUs {self._physical_ids}; adaptive memory ceiling="
            f"{self._memory_percentage:g}%, "
            "exact long-rollout rescue staged at 96%, 98%, then 99%; "
            f"{fallback_description}; live RAM admission is shared with "
            "evaluation and model transfers", flush=True)

    @property
    def world_size(self):
        return self._world_size

    @property
    def is_offloaded(self):
        return self._offloaded

    def estimated_host_offload_bytes(self, *, frozen_to_disk=False):
        """RAM required to move every live trainer replica off its GPU."""
        from runtime.host_memory import (GIB, estimated_module_bytes, estimated_optimizer_bytes)

        model_bytes = estimated_module_bytes(self.model, frozen_to_disk=frozen_to_disk)
        optimizer_bytes = estimated_optimizer_bytes(self.optimizer)
        # All worker models have the same storage layout as rank zero.  The
        # optimizer exists only on rank zero; workers retain gradients/model
        # state but do not construct duplicate Adam state.
        return int(1.05 * (model_bytes * self._world_size + optimizer_bytes) + self._world_size * GIB)

    @property
    def host_memory_admission(self):
        return self._activation_offload_semaphore.host_admission

    def reclaim_host_memory(self):
        """Synchronize every trainer process and free unused host caches."""
        import torch
        from runtime.host_memory import reclaim_unused_host_memory

        if not self._workers_idle:
            raise RuntimeError("cannot reclaim trainer host caches while an update is active")
        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "purge_host_memory"})
        with torch.cuda.device(0):
            torch.cuda.synchronize(0)
        local_cleanup = reclaim_unused_host_memory(torch, cuda_devices=(0,), force=True)
        child_messages = self._collect_event("host_memory_purged", range(1, self._world_size))
        return [local_cleanup] + [child_messages[rank]["cleanup"] for rank in range(1, self._world_size)]

    def _dead_workers(self, pending):
        return [rank for rank in pending if rank in self._processes and not self._processes[rank].is_alive()]

    def _collect_event(self, event, ranks):
        pending = set(int(rank) for rank in ranks)
        results = {}
        while pending:
            try:
                message = self._result_queue.get(timeout=5.0)
            except self._queue_module.Empty:
                dead = self._dead_workers(pending)
                if dead:
                    codes = {rank: self._processes[rank].exitcode for rank in dead}
                    if any(code == -signal.SIGKILL for code in codes.values()):
                        from runtime.run_supervisor import request_restart
                        request_restart(
                            "trainer worker was SIGKILLed during "
                            f"{event}; restarting from durable artifacts"
                        )
                    raise RuntimeError(f'fast trainer worker(s) exited while waiting for {event}: {codes}')
                continue
            message_event = message.get("event")
            rank = int(message.get("rank", -1))
            if message_event == "error":
                traceback_text = message.get("traceback", "no traceback")
                if message.get("host_memory_timeout"):
                    from runtime.host_memory import HostMemoryTimeout
                    admission = getattr(self, "host_memory_admission", None)
                    if admission is not None:
                        admission.stop_waiting()
                    raise HostMemoryTimeout(f"trainer rank {rank}: RAM wait exceeded 2 hours")
                if "ExactLongContextTrainingOOM" in traceback_text:
                    raise ExactLongContextTrainingOOM(
                        f"fast trainer rank {rank} exhausted exact "
                        "single-GPU long-context fallbacks:\n"
                        f"{traceback_text}")
                raise RuntimeError(f'fast trainer rank {rank} failed:\n{traceback_text}')
            if message_event != event or rank not in pending:
                raise RuntimeError(f'unexpected fast trainer message while waiting for {event}: {message}')
            pending.remove(rank)
            results[rank] = message
        return results

    @staticmethod
    def _cpu_examples(examples):
        import torch

        tensor_cache = {}
        copied_examples = []
        for example in examples:
            copied = {}
            for key, value in example.items():
                if torch.is_tensor(value):
                    cache_key = id(value)
                    if cache_key not in tensor_cache:
                        tensor_cache[cache_key] = value.detach().cpu()
                    copied[key] = tensor_cache[cache_key]
                else:
                    copied[key] = value
            copied_examples.append(copied)
        return copied_examples

    def _queue_examples(self, examples, *, seal=True, announce=True, weighted_offload=False):
        """Queue expensive examples first; idle ranks steal the next work."""

        maximum_length = max(1, int(self.cfg.max_seq_length))

        def estimated_cost(example):
            prompt = int(example["prompt_ids"].shape[1])
            response = int(example["response_ids"].shape[1])
            total = prompt + response
            return total * total + response * maximum_length

        copied = self._cpu_examples(examples)
        overflow_floor = getattr(self, "_warm_overflow_min_tokens", None)
        if overflow_floor is not None:
            for example in copied:
                example["_cooperative_overflow"] = bool(
                    int(example["prompt_ids"].shape[1] + example["response_ids"].shape[1]) >= overflow_floor)
        # GPT-OSS-120B's attention sinks and sliding attention require independent batch rows.
        shared_prefix_packing = getattr(self.cfg, "coder_model_profile", "") != "gpt-oss-120b"
        for example in copied:
            example["_shared_prefix_packing_allowed"] = shared_prefix_packing

        ordered = sorted(copied, key=estimated_cost, reverse=True)
        offload_waves = None
        if weighted_offload:
            # First-fit decreasing packs recovery trajectories into
            # RAM-safe concurrent waves. Unlike the former fixed rank
            # count, a wave can use every GPU when its shorter activation
            # graphs fit the same node-wide budget. Flattening the waves
            # retains asynchronous work stealing; the shared weighted
            # gate remains the authoritative safety mechanism.
            gate = self._activation_offload_semaphore
            waves = []
            for example in ordered:
                work = int(example["prompt_ids"].shape[1]) + int(example["response_ids"].shape[1])
                units = gate.request_units(work)
                if units is None:
                    waves.append({
                        "units": gate.capacity_units + 1,
                        "items": [example],
                    })
                    continue
                placed = False
                for wave in waves:
                    if (len(wave["items"]) < self._world_size
                            and int(wave["units"]) + int(units)
                            <= int(gate.capacity_units)):
                        wave["items"].append(example)
                        wave["units"] += int(units)
                        placed = True
                        break
                if not placed:
                    waves.append({
                        "units": int(units),
                        "items": [example],
                    })
            ordered = [example for wave in waves for example in wave["items"]]
            offload_waves = [len(wave["items"]) for wave in waves]
        for example in ordered:
            self._work_queue.put(example)
        if announce:
            if offload_waves is None:
                setting_log_only(f"[{self._process_label}] queued {len(ordered)} "
                                 "examples longest-first; each free GPU claims the "
                                 "next adaptive batch", flush=True)
            else:
                preview = ",".join(str(value) for value in offload_waves[:8])
                suffix = "..." if len(offload_waves) > 8 else ""
                setting_log_only(
                    f"[{self._process_label}] queued {len(ordered)} "
                    "oversized examples longest-first; conservative "
                    f"startup ranks/wave={preview}{suffix}, then measured "
                    "token-weighted admission expands concurrency; idle "
                    "ranks steal work as RAM capacity is freed",
                    flush=True,
                )
        if seal:
            for _ in range(self._world_size):
                self._work_queue.put(None)
        return ordered


    def train_spo_rs(self, examples, cfg, step_idx):
        import torch
        from training.fast_distributed import (
            broadcast_trainable_parameters, historical_training_memory_fraction,
            local_spo_rs_update, reduce_trainable_gradients,
        )

        if self._offloaded:
            print(f'[train-fast] restoring {self._world_size} process trainers for the update', flush=True)
            self.restore_after_generation()

        started = time.time()
        queued_examples = self._queue_examples(examples)
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued_examples)
        reference_label = (
            f"{sum(_valid_example_token_logprobs(ex, 'reference_logprobs') for ex in queued_examples)}/{len(examples)}"
            if uses_ttt_policy_loss(cfg.advantage_mode) and cfg.kl_penalty_coef else "off")
        setting_log_only(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
                         f"vLLM supplied old={supplied_old}/{len(examples)}, reference={reference_label}",
                         flush=True)
        update_memory_fraction = historical_training_memory_fraction(
            self.model, cfg, self._memory_fraction)
        high_memory_update = update_memory_fraction > self._memory_fraction
        rescue = "98%->99%" if high_memory_update else "96%->98%->99%"
        packed_expert_training = getattr(self.model, "_ttt_gpt_oss_shared_lora", False)
        # Leave multiple work claims when short rollouts fit in big batches,
        # allowing work stealing to balance the update.
        max_examples_per_batch = (
            max(1, math.ceil(len(queued_examples) / (2 * self._world_size)))
            if packed_expert_training else 64)
        setting_log_only(f"[train-fast] one process/GPU; memory ceiling="
                         f"{100.0 * update_memory_fraction:g}%; "
                         f"exact singleton rescue={rescue}; "
                         f"initial padded-token budgets/GPU={self._token_budgets}", flush=True)

        # Packed GPT-OSS retries complete microbatches on the live replicas.
        defer_exact_oom = not packed_expert_training and _uses_replayable_policy_update(cfg.advantage_mode)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "max_examples_per_batch": max_examples_per_batch,
                "defer_exact_oom": defer_exact_oom,
            })

        local_stats = local_spo_rs_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            max_examples_per_batch=max_examples_per_batch,
            activation_offload_semaphore=(self._activation_offload_semaphore),
            defer_exact_oom=defer_exact_oom)
        child_messages = self._collect_event("computed", range(1, self._world_size))
        rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
        accounted_examples = sum(
            int(stats["trained_examples"])
            + int(stats["quarantined_examples"])
            + int(stats.get("deferred_examples", 0))
            for stats in rank_stats)
        if accounted_examples != len(queued_examples):
            raise RuntimeError(
                "fast trainer shared queue lost or duplicated work: "
                f"accounted for {accounted_examples}/{len(queued_examples)} "
                "examples")

        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "finish_update"})
        reduce_trainable_gradients(self.model, destination_rank=0)

        update_error = None
        overflow_error = None
        grad_norm = None
        try:
            deferred_examples = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            if deferred_examples:
                gradient_path, marker, deferred_keys, _normal_stats = (
                    _save_exact_spo_rs_overflow(
                        self._exp_dir, step_idx, self.model, rank_stats,
                        normal_train_seconds=time.time() - started))
                longest = max(
                    (int(length) for stats in rank_stats for length in stats.get("deferred_token_lengths", ())),
                    default=0)
                print(
                    f"[train-overflow] step {step_idx}: retained exact "
                    f"gradients for "
                    f"{len(queued_examples) - deferred_examples}/"
                    f"{len(queued_examples)} trajectories; routing only "
                    f"{deferred_examples} oversized trajectory/trajectories "
                    "to cooperative GPU overflow "
                    f"(longest={longest} tokens, "
                    f"state={gradient_path.name}, marker={marker.name})",
                    flush=True,
                )
                overflow_error = ExactLongContextTrainingOOM(
                    f"step {step_idx} deferred {len(deferred_keys)} exact "
                    "trajectory/trajectories to the selective multi-GPU "
                    "overflow lane")
            else:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    max_norm=_historical_grad_clip(cfg), error_if_nonfinite=True)
                self.optimizer.step()
                grad_norm = float(grad_norm_tensor.item())
        except BaseException as error:
            update_error = error
        finally:
            self.model.zero_grad(set_to_none=True)
            broadcast_trainable_parameters(self.model, source_rank=0)
            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
        if update_error is not None:
            raise update_error
        if overflow_error is not None:
            from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, queued_examples, cfg, step_idx,
                    _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        metric_keys = (
            "loss", "policy_loss", "kl_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {key: sum(stats["totals"][key] for stats in rank_stats) for key in metric_keys}
        max_ratio = max(stats["max_ratio"] for stats in rank_stats)
        max_prefix_ratio = max(stats["max_prefix_ratio"] for stats in rank_stats)
        quarantined_examples = sum(stats["quarantined_examples"] for stats in rank_stats)
        trained_per_gpu = [stats["trained_examples"] for stats in rank_stats]
        batches_per_gpu = [stats["backward_batches"] for stats in rank_stats]
        peak_fractions = [stats["peak_memory_fraction"] for stats in rank_stats]
        allocator_fractions = [stats["allocator_memory_fraction"] for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        setting_log_only(
            f"[train-fast] trained examples/GPU={trained_per_gpu}; "
            f"backward batches/GPU={batches_per_gpu}; final padded-token "
            f"budgets/GPU={self._token_budgets}; peak memory/GPU="
            f"{peak_percentages}%; allocator caps/GPU={allocator_percentages}%", flush=True)
        print(f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"reference={reference_label} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        return {
            "policy_updates": history,
            "training_seconds": time.time() - started,
            "training_parallel_gpus": self._world_size,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def recover_spo_rs_overflow(self, examples, cfg, step_idx, overflow_state):
        """Finish saved SPO-RS overflow without replaying completed gradients.

        Normal gradients already stored in ``overflow_state`` are never
        replayed. Remaining trajectories are work-stolen by all replicas. A
        trajectory that exhausts card memory may use the shared host-RAM
        admission gate; successful gradients are reduced once and atomically
        merged into the durable checkpoint. Cooperative recovery can finish the
        residual trajectories on supported checkpoints.
        """
        from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
        if supports_warm_overflow(cfg):
            return complete_warm_overflow(self, examples, cfg, step_idx, overflow_state)

        import torch
        from training.fast_distributed import (
            broadcast_trainable_parameters, local_spo_rs_update,
            reduce_trainable_gradients,
        )

        if not _uses_replayable_policy_update(getattr(cfg, "advantage_mode", "")):
            raise ValueError("exact overflow recovery requires a replayable policy objective")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for exact overflow recovery")
        if self._offloaded:
            print(
                f"[{self._process_label}] restoring {self._world_size} "
                "process trainers for exact overflow recovery",
                flush=True,
            )
            self.restore_after_generation()

        _validate_precomputed_exact_gradients(self.model, overflow_state, step_idx)
        requested_keys = {tuple(int(value) for value in example["_overlap_key"]) for example in examples}
        remaining_keys = {tuple(int(value) for value in key) for key in overflow_state.get("remaining_keys", ())}
        if not requested_keys or requested_keys != remaining_keys:
            raise RuntimeError(
                "replicated exact-overflow inputs do not match the durable "
                f"remaining keys: inputs={len(requested_keys)}, "
                f"remaining={len(remaining_keys)}")

        started = time.time()
        recovery_cfg = SimpleNamespace(**dict(vars(cfg)))
        recovery_cfg._exact_replica_force_activation_offload = True
        queued_examples = self._queue_examples(examples, weighted_offload=True)
        print(
            f"[recovery] step {step_idx}: dispatching "
            f"{len(queued_examples)} saved oversized trajectories across "
            f"{self._world_size} GPU replicas. CPU activation admission "
            f"shares a {self._activation_offload_budget_gib:.0f}-GiB RAM budget "
            "across every rank whose current trajectory fits",
            flush=True,
        )
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(recovery_cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "defer_exact_oom": True,
            })

        try:
            local_stats = local_spo_rs_update(
                self.backend, self.model, self.tokenizer, (), recovery_cfg, 0,
                self._token_budgets[0],
                memory_fraction=self._memory_fraction,
                work_queue=self._work_queue,
                activation_offload_semaphore=(self._activation_offload_semaphore),
                defer_exact_oom=True,
            )
            child_messages = self._collect_event("computed", range(1, self._world_size))
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
            accounted_examples = sum(
                int(stats["trained_examples"])
                + int(stats["quarantined_examples"])
                + int(stats.get("deferred_examples", 0))
                for stats in rank_stats
            )
            if accounted_examples != len(queued_examples):
                raise RuntimeError(
                    "replicated exact-overflow queue lost or duplicated "
                    f"work: accounted for {accounted_examples}/"
                    f"{len(queued_examples)} trajectories")

            deferred_keys = {
                (int(key[0]), int(key[1]))
                for stats in rank_stats
                for key in stats.get("deferred_keys", ())
            }
            deferred_count = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            quarantined_count = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
            trained_count = sum(int(stats["trained_examples"]) for stats in rank_stats)
            if (deferred_count != len(deferred_keys)
                    or not deferred_keys.issubset(requested_keys)
                    or quarantined_count):
                raise RuntimeError(
                    "replicated exact-overflow recovery produced invalid "
                    "trajectory accounting: "
                    f"trained={trained_count}, deferred={deferred_count}, "
                    f"unique deferred={len(deferred_keys)}, "
                    f"quarantined={quarantined_count}")
            successful_keys = requested_keys - deferred_keys
            if trained_count != len(successful_keys):
                raise RuntimeError(
                    "replicated exact-overflow successful-key accounting "
                    f"changed: trained={trained_count}, "
                    f"keys={len(successful_keys)}")
            metric_keys = (
                "loss", "policy_loss", "kl_estimate",
                "ratio", "clipped_fraction",
            )
            wave_stats = {
                "totals": {key: sum(float(stats["totals"][key]) for stats in rank_stats) for key in metric_keys},
                "max_ratio": max(float(stats["max_ratio"]) for stats in rank_stats),
                "max_prefix_ratio": max(float(stats["max_prefix_ratio"]) for stats in rank_stats),
                "backward_batches": sum(int(stats["backward_batches"]) for stats in rank_stats),
            }

            # Workers reduce their local sums into rank 0, then wait at the
            # parameter broadcast while rank 0 commits or applies the update.
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "finish_update"})
            reduce_trainable_gradients(self.model, destination_rank=0)

            update_error = None
            overflow_error = None
            grad_norm = None
            try:
                if successful_keys:
                    _commit_exact_spo_rs_overflow_batch(
                        self.model, overflow_state, step_idx,
                        successful_keys, wave_stats,
                        training_seconds=time.time() - started,
                    )

                # The durable checkpoint is now authoritative. Clear the
                # just-reduced wave before materializing the complete saved
                # gradient, otherwise successful recovery work is doubled.
                self.model.zero_grad(set_to_none=True)
                if deferred_keys:
                    overflow_state["replica_offload_exhausted"] = True
                    _write_exact_spo_rs_overflow_state(overflow_state)
                    marker = _read_exact_sharded_recovery(self._exp_dir) or {}
                    _write_exact_sharded_recovery(
                        self._exp_dir, step_idx,
                        marker.get("reason", "bounded replicated overflow recovery exhausted"),
                        replica_offload_exhausted=True,
                        remaining_deferred_keys=[list(key) for key in sorted(deferred_keys)],
                    )
                    overflow_error = ExactLongContextTrainingOOM(
                        f"step {step_idx} retained "
                        f"{len(successful_keys)} additional exact gradient(s) "
                        f"but {len(deferred_keys)} trajectory/trajectories "
                        "still exceed bounded replica activation offload")
                else:
                    _apply_precomputed_exact_gradients(self.model, overflow_state, step_idx)
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                        max_norm=_historical_grad_clip(cfg),
                        error_if_nonfinite=True,
                    )
                    self.optimizer.step()
                    grad_norm = float(grad_norm_tensor.item())
            except BaseException as error:
                update_error = error
            finally:
                self.model.zero_grad(set_to_none=True)
                broadcast_trainable_parameters(self.model, source_rank=0)
                self._collect_event("updated", range(1, self._world_size))
                self._workers_idle = True
            if update_error is not None:
                raise update_error
            if overflow_error is not None:
                raise overflow_error
        except BaseException:
            if not self._workers_idle:
                self._abort_workers()
            raise

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        normal_stats = overflow_state.get("normal_stats", {})
        overflow_stats = overflow_state.get("overflow_stats", {})
        normal_totals = normal_stats.get("totals", {})
        overflow_totals = overflow_stats.get("totals", {})
        totals = {
            key: (float(normal_totals.get(key, 0.0)) + float(overflow_totals.get(key, 0.0)))
            for key in metric_keys
        }
        max_ratio = max(float(normal_stats.get("max_ratio", 0.0)), float(overflow_stats.get("max_ratio", 0.0)))
        max_prefix_ratio = max(
            float(normal_stats.get("max_prefix_ratio", 0.0)),
            float(overflow_stats.get("max_prefix_ratio", 0.0)),
        )
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        history = [{
            "epoch": 1,
            **totals,
            "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "oom_quarantined_examples": 0,
            "grad_norm": grad_norm,
        }]
        print(
            f"[{self._process_label}] exact overflow trained examples/GPU="
            f"{trained_per_gpu}; backward batches/GPU={batches_per_gpu}; "
            f"final padded-token budgets/GPU={self._token_budgets}",
            flush=True,
        )
        print(
            f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
            f"loss={totals['loss']:.6f} "
            f"reference={'on' if uses_ttt_policy_loss(cfg.advantage_mode) and cfg.kl_penalty_coef else 'off'} "
            f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
            f"clipped={totals['clipped_fraction']:.1%} "
            "OOM-quarantined=0 (RAM-bounded replicated exact overflow recovery)",
            flush=True,
        )
        return {
            "policy_updates": history,
            "training_seconds": (
                float(normal_stats.get("training_seconds", 0.0))
                + float(overflow_stats.get("training_seconds", 0.0))),
            "training_parallel_gpus": self._world_size,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "oom_quarantined_examples": 0,
            "exact_overflow_reused_examples": int(normal_stats.get("trained_examples", 0))
                + len(overflow_state.get("completed_keys", ()))
                - len(successful_keys),
            "exact_overflow_replica_examples": len(successful_keys),
            "exact_overflow_sharded_examples": 0,
        }

    def begin_entropic_overlap(self, cfg, step_idx):
        """Start a streaming entropic backward before CPU evaluation ends."""
        from concurrent.futures import ThreadPoolExecutor
        from training.fast_distributed import local_policy_update

        if str(getattr(cfg, "advantage_mode", "")).lower() != "entropic":
            raise ValueError("streaming policy overlap is entropic-only")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for an entropic overlap update")
        if self._entropic_overlap is not None:
            raise RuntimeError("an entropic overlap update is already active")
        if self._offloaded:
            print(f"[{self._process_label}] restoring {self._world_size} "
                  "process trainers while CPU evaluation continues",
                  flush=True)
            self.restore_after_generation()

        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_policy",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
            })

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="entropic-rank0-overlap")
        state = {
            "step_idx": int(step_idx),
            "started_at": time.time(),
            "computed_at": None,
            "executor": executor,
            "future": None,
            "queued_examples": 0,
            "scheduled_keys": set(),
            "supplied_old": 0,
            "supplied_reference": 0,
            "sealed": False,
        }
        future = executor.submit(
            local_policy_update,
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            activation_offload_semaphore=(self._activation_offload_semaphore))
        state["future"] = future

        def _record_completion(_future):
            state["computed_at"] = time.time()

        future.add_done_callback(_record_completion)
        self._entropic_overlap = state
        return state

    def queue_entropic_overlap(self, examples):
        """Feed one reward-complete group to the active GPU consumers."""
        state = self._entropic_overlap
        if state is None or state["sealed"]:
            raise RuntimeError("entropic overlap queue is not open")
        queued = self._queue_examples(examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued)
        state["supplied_reference"] += sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued)
        state["scheduled_keys"].update(example["_overlap_key"] for example in queued)
        return len(queued)

    def seal_entropic_overlap(self):
        """Tell every GPU consumer that no more completed groups remain."""
        state = self._entropic_overlap
        if state is None:
            return
        if not state["sealed"]:
            for _ in range(self._world_size):
                self._work_queue.put(None)
            state["sealed"] = True

    def finish_entropic_overlap(self, cfg, step_idx, expected_examples, *, apply_update=True):
        """Sum across replicas and optionally apply streamed TTT gradients."""
        import torch
        from training.fast_distributed import (broadcast_trainable_parameters, reduce_trainable_gradients)

        state = self._entropic_overlap
        if state is None:
            raise RuntimeError("no entropic overlap update is active")
        self.seal_entropic_overlap()
        expected_examples = int(expected_examples)
        queued_examples = int(state["queued_examples"])
        if apply_update and expected_examples != queued_examples:
            raise RuntimeError(
                "entropic overlap final example count changed: "
                f"queued={queued_examples}, final={expected_examples}")

        try:
            local_stats = state["future"].result()
            child_messages = self._collect_event("computed", range(1, self._world_size))
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
            accounted_examples = sum(
                int(stats["trained_examples"])
                + int(stats["quarantined_examples"])
                for stats in rank_stats)
            if accounted_examples != queued_examples:
                raise RuntimeError(
                    "streaming entropic queue lost or duplicated work: "
                    f"accounted for {accounted_examples}/{queued_examples} "
                    "examples")

            for command_queue in self._command_queues.values():
                command_queue.put({
                    "kind": "finish_update",
                    "apply_update": bool(apply_update),
                })

            grad_norm = None
            update_error = None
            if apply_update:
                reduce_trainable_gradients(self.model, destination_rank=0)
                try:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                        max_norm=_historical_grad_clip(cfg), error_if_nonfinite=True)
                    self.optimizer.step()
                    grad_norm = float(grad_norm_tensor.item())
                except BaseException as error:
                    update_error = error
                finally:
                    self.model.zero_grad(set_to_none=True)
                    broadcast_trainable_parameters(self.model, source_rank=0)
            else:
                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)

            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
            if update_error is not None:
                raise update_error
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise
        finally:
            state["executor"].shutdown(wait=True, cancel_futures=False)
            self._entropic_overlap = None

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        total_loss = sum(float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        setting_log_only(
            f"[{self._process_label}] trained examples/GPU={trained_per_gpu}; "
            f"backward batches/GPU={batches_per_gpu}; final padded-token "
            f"budgets/GPU={self._token_budgets}; peak memory/GPU="
            f"{peak_percentages}%; allocator caps/GPU={allocator_percentages}%", flush=True)
        completed_at = state["computed_at"] or time.time()
        elapsed = max(0.0, completed_at - state["started_at"])
        ratio_message = ""
        if ratio_count:
            ratio_message = (f'  IS ratio mean={ratio_sum / ratio_count:.9f} max={ratio_max:.3f}')
        print(f"[step {step_idx}] train time: {elapsed:.1f}s  "
              f"TTT loss sum: {total_loss:.9f}  "
              "avg logpi_old - logpi_base: "
              f"{total_logp_delta / expected_examples:.9f}{ratio_message}  "
              f"OOM-quarantined={quarantined_examples} "
              "(forward/backward overlapped CPU evaluation)", flush=True)
        return {
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_grad_norm": grad_norm,
            "training_overlapped_evaluation": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def begin_spo_rs_overlap(self, cfg, step_idx):
        """Start exact clipped SPO-RS backward while CPU evaluation runs."""
        from concurrent.futures import ThreadPoolExecutor
        from training.fast_distributed import local_spo_rs_update

        if not _uses_reward_history(getattr(cfg, "advantage_mode", "")):
            raise ValueError("streaming historical overlap requires a historical advantage mode")
        if self._closed or not self._workers_idle:
            raise RuntimeError("process trainer is not idle for an SPO-RS overlap update")
        if self._spo_rs_overlap is not None:
            raise RuntimeError("an SPO-RS overlap update is already active")
        if self._entropic_overlap is not None:
            raise RuntimeError("an entropic overlap update is already active")
        if self._offloaded:
            print(f"[{self._process_label}] restoring {self._world_size} "
                  "process trainers while CPU evaluation continues",
                  flush=True)
            self.restore_after_generation()

        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_spo_rs",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
                "defer_exact_oom": True,
            })

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spo-rs-rank0-overlap")
        state = {
            "overflow_examples": {},
            "step_idx": int(step_idx),
            "started_at": time.time(),
            "computed_at": None,
            "executor": executor,
            "future": None,
            "queued_examples": 0,
            "scheduled_keys": set(),
            "supplied_old": 0,
            "sealed": False,
        }
        future = executor.submit(
            local_spo_rs_update,
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            activation_offload_semaphore=(self._activation_offload_semaphore),
            defer_exact_oom=True)
        state["future"] = future

        def _record_completion(_future):
            state["computed_at"] = time.time()

        future.add_done_callback(_record_completion)
        self._spo_rs_overlap = state
        return state

    def queue_spo_rs_overlap(self, examples):
        """Feed reward-complete SPO-RS trajectories to all free GPUs."""
        state = self._spo_rs_overlap
        if state is None or state["sealed"]:
            raise RuntimeError("SPO-RS overlap queue is not open")
        if any(example.get("advantage") is None for example in examples):
            raise RuntimeError("SPO-RS overlap cannot queue an unresolved advantage")
        queued = self._queue_examples(examples, seal=False, announce=False)
        state["queued_examples"] += len(queued)
        state["supplied_old"] += sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued)
        state["scheduled_keys"].update(example["_overlap_key"] for example in queued)
        state["overflow_examples"].update((tuple(example["_overlap_key"]), example) for example in queued)
        return len(queued)

    def seal_spo_rs_overlap(self):
        """Tell every clipped-policy GPU consumer the stream is complete."""
        state = self._spo_rs_overlap
        if state is None:
            return
        if not state["sealed"]:
            for _ in range(self._world_size):
                self._work_queue.put(None)
            state["sealed"] = True

    def finish_spo_rs_overlap(self, cfg, step_idx, expected_examples, *, apply_update=True):
        """Reduce and apply the exact streamed SPO-RS gradients once."""
        import torch
        from training.fast_distributed import (broadcast_trainable_parameters, reduce_trainable_gradients)

        state = self._spo_rs_overlap
        if state is None:
            raise RuntimeError("no SPO-RS overlap update is active")
        self.seal_spo_rs_overlap()
        expected_examples = int(expected_examples)
        queued_examples = int(state["queued_examples"])
        if apply_update and expected_examples != queued_examples:
            raise RuntimeError(
                "SPO-RS overlap final example count changed: "
                f"queued={queued_examples}, final={expected_examples}")

        try:
            local_stats = state["future"].result()
            child_messages = self._collect_event("computed", range(1, self._world_size))
            # Rank 0 can finish before a longer worker. Record completion only
            # after every GPU has drained the shared queue.
            state["computed_at"] = time.time()
            rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
            accounted_examples = sum(
                int(stats["trained_examples"])
                + int(stats["quarantined_examples"])
                + int(stats.get("deferred_examples", 0))
                for stats in rank_stats)
            if accounted_examples != queued_examples:
                raise RuntimeError(
                    "streaming SPO-RS queue lost or duplicated work: "
                    f"accounted for {accounted_examples}/{queued_examples} "
                    "examples")

            deferred_examples = sum(int(stats.get("deferred_examples", 0)) for stats in rank_stats)
            for command_queue in self._command_queues.values():
                command_queue.put({
                    "kind": "finish_update",
                    "apply_update": bool(apply_update),
                })

            grad_norm = None
            update_error = None
            overflow_error = None
            if apply_update:
                reduce_trainable_gradients(self.model, destination_rank=0)
                try:
                    if deferred_examples:
                        gradient_path, marker, deferred_keys, _normal_stats = (
                            _save_exact_spo_rs_overflow(
                                self._exp_dir, step_idx, self.model,
                                rank_stats,
                                normal_train_seconds=max(0.0, state["computed_at"] - state["started_at"])))
                        longest = max(
                            (int(length) for stats in rank_stats for length in stats.get("deferred_token_lengths", ())),
                            default=0)
                        print(
                            f"[train-overflow] step {step_idx}: retained "
                            f"exact gradients for "
                            f"{queued_examples - deferred_examples}/"
                            f"{queued_examples} trajectories; routing only "
                            f"{deferred_examples} oversized trajectory/"
                            "trajectories to cooperative GPU overflow "
                            f"(longest={longest} tokens, "
                            f"state={gradient_path.name}, "
                            f"marker={marker.name})",
                            flush=True,
                        )
                        overflow_error = ExactLongContextTrainingOOM(
                            f"step {step_idx} deferred "
                            f"{len(deferred_keys)} exact trajectory/"
                            "trajectories to the selective multi-GPU "
                            "overflow lane")
                    else:
                        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                            max_norm=_historical_grad_clip(cfg),
                            error_if_nonfinite=True)
                        self.optimizer.step()
                        grad_norm = float(grad_norm_tensor.item())
                except BaseException as error:
                    update_error = error
                finally:
                    self.model.zero_grad(set_to_none=True)
                    broadcast_trainable_parameters(self.model, source_rank=0)
            else:
                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad(set_to_none=True)

            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
            if update_error is not None:
                raise update_error
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise
        finally:
            state["executor"].shutdown(wait=True, cancel_futures=False)
            self._spo_rs_overlap = None

        if overflow_error is not None:
            from training.exact_overflow import supports_warm_overflow, complete_warm_overflow
            if supports_warm_overflow(cfg):
                return complete_warm_overflow(
                    self, list(state["overflow_examples"].values()), cfg,
                    step_idx, _load_exact_spo_rs_overflow(self._exp_dir, step_idx))
            raise overflow_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        if not apply_update:
            return {}

        metric_keys = (
            "loss", "policy_loss", "kl_estimate",
            "ratio", "clipped_fraction",
        )
        totals = {key: sum(float(stats["totals"][key]) for stats in rank_stats) for key in metric_keys}
        max_ratio = max(float(stats["max_ratio"]) for stats in rank_stats)
        max_prefix_ratio = max(float(stats["max_prefix_ratio"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        completed_at = state["computed_at"] or time.time()
        elapsed = max(0.0, completed_at - state["started_at"])
        history = [{
            "epoch": 1, **totals, "ratio_max": max_ratio,
            "prefix_ratio_max": max_prefix_ratio,
            "oom_quarantined_examples": quarantined_examples,
            "grad_norm": grad_norm,
        }]
        setting_log_only(
            f"[{self._process_label}] trained examples/GPU={trained_per_gpu}; "
            f"backward batches/GPU={batches_per_gpu}; final padded-token "
            f"budgets/GPU={self._token_budgets}; peak memory/GPU="
            f"{peak_percentages}%; allocator caps/GPU={allocator_percentages}%", flush=True)
        print(f"[step {step_idx}] {cfg.advantage_mode} epoch 1/1: "
              f"loss={totals['loss']:.6f} "
              f"reference={'on' if cfg.advantage_mode == 'hybrid' and cfg.kl_penalty_coef else 'off'} "
              f"ratio={totals['ratio']:.6f} max={max_ratio:.6f} "
              f"clipped={totals['clipped_fraction']:.1%} "
              f"OOM-quarantined={quarantined_examples} "
              "(forward/backward overlapped CPU evaluation)", flush=True)
        return {
            "policy_updates": history,
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_overlapped_evaluation": True,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def train_policy(self, examples, cfg, step_idx):
        """Run the shared TTT loss for entropic and Hinge on every replica."""
        import torch
        from training.hybrid import reference_logprob_mean
        from training.fast_distributed import (broadcast_trainable_parameters,
                                      local_policy_update,
                                      reduce_trainable_gradients)

        process_label = self._process_label
        if self._offloaded:
            print(f'[{process_label}] restoring {self._world_size} process trainers for the update', flush=True)
            self.restore_after_generation()

        started = time.time()
        if not examples:
            return {
                "training_seconds": 0.0,
                "training_parallel_gpus": self._world_size,
                "oom_quarantined_examples": 0,
            }
        if cfg.generation_backend == "hf":
            # HF generation omits scores; freeze them before any backward or update.
            self.backend.set_training_mode()
            with _policy_dropout_disabled(self.model):
                for example in examples:
                    prompt = example["prompt_ids"].to(self.model.device)
                    response = example["response_ids"].to(self.model.device)
                    if not _valid_example_token_logprobs(example, "behavior_logprobs"):
                        example["behavior_logprobs"] = compute_token_logprobs(
                            self.model, prompt, response, False, cfg.logprob_chunk).detach().cpu()
                    if cfg.kl_penalty_coef and not _valid_example_token_logprobs(example, "reference_logprobs"):
                        with self.backend.disable_adapter():
                            example["reference_logprobs"] = compute_token_logprobs(
                                self.model, prompt, response, False, cfg.logprob_chunk).detach().cpu()
        reference_mean = None
        if cfg.kl_penalty_coef:
            reference_mean = reference_logprob_mean(
                (int(example["response_ids"].shape[1]),
                 example["behavior_logprobs"], example["reference_logprobs"])
                for example in examples)
            for example in examples:
                example["reference_logprob_mean"] = reference_mean
        queued_examples = self._queue_examples(examples)
        total_examples = len(queued_examples)
        supplied_old = sum(_valid_example_token_logprobs(example, "behavior_logprobs") for example in queued_examples)
        supplied_reference = sum(
            _valid_example_token_logprobs(example, "reference_logprobs")
            for example in queued_examples)
        setting_log_only(f"[step {step_idx}] {cfg.advantage_mode} logprobs: "
                         f"frozen old={supplied_old}/{total_examples}, "
                         f"reference={supplied_reference}/{total_examples}", flush=True)
        setting_log_only(f"[{process_label}] one process/GPU with adaptive batching; "
                         f"memory ceiling={self._memory_percentage:g}%; "
                         "exact singleton rescue=96%->98%->99%; "
                         f"initial padded-token budgets/GPU={self._token_budgets}", flush=True)

        self._workers_idle = False
        for rank in range(1, self._world_size):
            self._command_queues[rank].put({
                "kind": "train_policy",
                "step_cfg": dict(vars(cfg)),
                "token_budget": self._token_budgets[rank],
                "memory_fraction": self._memory_fraction,
            })

        local_stats = local_policy_update(
            self.backend, self.model, self.tokenizer, (), cfg, 0,
            self._token_budgets[0],
            memory_fraction=self._memory_fraction,
            work_queue=self._work_queue,
            activation_offload_semaphore=(self._activation_offload_semaphore))
        child_messages = self._collect_event("computed", range(1, self._world_size))
        rank_stats = [local_stats] + [child_messages[rank]["stats"] for rank in range(1, self._world_size)]
        accounted_examples = sum(
            int(stats["trained_examples"])
            + int(stats["quarantined_examples"])
            for stats in rank_stats)
        if accounted_examples != total_examples:
            raise RuntimeError(
                "fast trainer shared queue lost or duplicated work: "
                f"accounted for {accounted_examples}/{total_examples} "
                "examples")

        for command_queue in self._command_queues.values():
            command_queue.put({"kind": "finish_update"})
        reduce_trainable_gradients(self.model, destination_rank=0)

        update_error = None
        grad_norm = None
        try:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                max_norm=_historical_grad_clip(cfg), error_if_nonfinite=True)
            self.optimizer.step()
            grad_norm = float(grad_norm_tensor.item())
        except BaseException as error:
            update_error = error
        finally:
            self.model.zero_grad(set_to_none=True)
            broadcast_trainable_parameters(self.model, source_rank=0)
            self._collect_event("updated", range(1, self._world_size))
            self._workers_idle = True
        if update_error is not None:
            raise update_error

        for rank, stats in enumerate(rank_stats):
            self._token_budgets[rank] = max(1, int(stats["token_budget"]))
        total_loss = sum(float(stats["total_loss"]) for stats in rank_stats)
        total_logp_delta = sum(float(stats["total_logp_delta"]) for stats in rank_stats)
        ratio_sum = sum(float(stats["ratio_sum"]) for stats in rank_stats)
        ratio_max = max(float(stats["ratio_max"]) for stats in rank_stats)
        ratio_count = sum(int(stats["ratio_count"]) for stats in rank_stats)
        quarantined_examples = sum(int(stats["quarantined_examples"]) for stats in rank_stats)
        trained_per_gpu = [int(stats["trained_examples"]) for stats in rank_stats]
        batches_per_gpu = [int(stats["backward_batches"]) for stats in rank_stats]
        peak_fractions = [float(stats["peak_memory_fraction"]) for stats in rank_stats]
        allocator_fractions = [float(stats["allocator_memory_fraction"]) for stats in rank_stats]
        peak_percentages = [round(100.0 * fraction, 1) for fraction in peak_fractions]
        allocator_percentages = [round(100.0 * fraction, 1) for fraction in allocator_fractions]
        setting_log_only(
            f"[{process_label}] trained examples/GPU={trained_per_gpu}; "
            f"backward batches/GPU={batches_per_gpu}; final padded-token "
            f"budgets/GPU={self._token_budgets}; peak memory/GPU="
            f"{peak_percentages}%; allocator caps/GPU={allocator_percentages}%", flush=True)
        elapsed = time.time() - started
        ratio_message = ""
        if ratio_count:
            ratio_message = (f'  IS ratio mean={ratio_sum / ratio_count:.9f} max={ratio_max:.3f}')
        print(f"[step {step_idx}] train time: {elapsed:.1f}s  "
              f"TTT loss sum: {total_loss:.9f}  "
              "avg logpi_old - logpi_base: "
              f"{total_logp_delta / total_examples:.9f}{ratio_message}  "
              f"OOM-quarantined={quarantined_examples}", flush=True)
        return {
            "reference_kl_mean": reference_mean,
            "training_seconds": elapsed,
            "training_parallel_gpus": self._world_size,
            "training_grad_norm": grad_norm,
            "fast_trained_examples_per_gpu": trained_per_gpu,
            "fast_backward_batches_per_gpu": batches_per_gpu,
            "fast_padded_token_budgets": list(self._token_budgets),
            "fast_peak_memory_fraction": peak_fractions,
            "fast_allocator_memory_fraction": allocator_fractions,
            "oom_quarantined_examples": quarantined_examples,
        }

    def offload_for_generation(self, *, purge_host_cache=True, disk_directory=None,
                               host_admission=None, deadline=None):
        if self._offloaded:
            return
        import gc
        import torch

        # Long-context saved tensors use PyTorch's *host* pinned allocator.
        # CUDA empty_cache() cannot release that cache.  Clear it on every
        # replica before allocating eight complete CPU model copies; doing so
        # at this phase boundary does not change the update and avoids the
        # step-over-step host-RAM growth that can OOM-kill the whole VM.
        if purge_host_cache:
            self.reclaim_host_memory()
        self._offloaded = True
        self._offloaded_ranks = set(range(self._world_size))
        try:
            for command_queue in self._command_queues.values():
                command = {"kind": "offload", "disk_directory": disk_directory}
                if host_admission is not None:
                    command.update(incremental=True, deadline=deadline)
                command_queue.put(command)
            torch.cuda.synchronize(0)
            if host_admission is not None:
                from runtime.incremental_offload import offload_optimizer
                offload_optimizer(self.optimizer, host_admission, deadline=deadline)
                self.backend.offload_for_generation(host_admission=host_admission, deadline=deadline)
            else:
                _move_optimizer_state(self.optimizer, "cpu", hardware_profile=getattr(
                    getattr(self, "cfg", None), "hardware_profile", "RTX6000"))
                if disk_directory:
                    self.backend.offload_for_generation(disk_directory=disk_directory)
                else:
                    self.backend.offload_for_generation()
            gc.collect()
            with torch.cuda.device(0):
                if self.cfg.evaluation_shares_generation:
                    torch.cuda.synchronize(0)
                    torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            self._collect_event("offloaded", range(1, self._world_size))
        except BaseException:
            # A failed worker has already left the NCCL process group. Do not
            # attempt a graceful barrier/restore against a partial group.
            self._workers_idle = False
            self._abort_workers()
            raise

    def restore_after_generation(self):
        self.restore_on_gpus(self._physical_ids)

    def restore_on_gpus(self, physical_gpu_ids):
        """Move only released GPU groups back to training, in parallel.

        Called between coder sleep waves, so a complete set of CPU trainer
        copies never has to coexist with all new inference sleep backups.
        No collective or parameter update occurs until every rank is ready.
        """
        if not self._offloaded:
            return
        physical_gpu_ids = set(int(gpu) for gpu in physical_gpu_ids)
        ranks = {rank for rank in self._offloaded_ranks if self._physical_ids[rank] in physical_gpu_ids}
        if not ranks:
            return
        if not self._dist_initialized:
            raise RuntimeError('process-distributed trainer is unavailable after a worker failure')
        import gc
        import torch
        from training.fast_distributed import (set_total_memory_ceiling, validate_model_device)
        from runtime.host_memory import reclaim_unused_host_memory

        try:
            print(f"[{self._process_label}] restoring trainer GPUs "
                  f"{[self._physical_ids[rank] for rank in sorted(ranks)]} "
                  "as coder memory is released; CPU evaluation continues",
                  flush=True)
            for rank in sorted(ranks - {0}):
                self._command_queues[rank].put({"kind": "restore"})
            if 0 in ranks:
                gc.collect()
                with torch.cuda.device(0):
                    torch.cuda.empty_cache()
                    set_total_memory_ceiling(0, self._memory_fraction)
                    self.backend.restore_after_generation()
                    self.backend.set_training_mode()
                    torch.cuda.synchronize(0)
                validate_model_device(self.model, 0)
                _restore_optimizer_state_to_parameters(self.optimizer)
                reclaim_unused_host_memory(torch, cuda_devices=(0,), force=True)
            self._collect_event("restored", ranks - {0})
            self._offloaded_ranks.difference_update(ranks)
            self._offloaded = bool(self._offloaded_ranks)
        except BaseException:
            self._workers_idle = False
            self._abort_workers()
            raise

    def _abort_workers(self):
        was_initialized = self._dist_initialized
        self._terminate_workers_for_restart()
        if was_initialized:
            import torch.distributed as dist
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
        self._dist_initialized = False

    def _close_queues(self):
        # After consumers exit, pending rollout tensors cannot drain from the
        # feeder pipe. Joining that feeder at interpreter exit hangs forever.
        # Discard pending messages only when this trainer has stopped/aborted.
        queues = [getattr(self, "_work_queue", None),
                  getattr(self, "_result_queue", None)]
        queues.extend(getattr(self, "_command_queues", {}).values())
        for queue in queues:
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()

    def _terminate_workers_for_restart(self):
        """Stop child ranks without entering another distributed collective.

        The saved state is already durable. The owner exits or replaces itself
        to release rank zero's CUDA/NCCL state. Use a bounded deadline for the
        other ranks so a missing rank cannot hang a shutdown collective.
        """
        import time

        for process in self._processes.values():
            if process.is_alive():
                process.terminate()

        deadline = time.monotonic() + 10.0
        for process in self._processes.values():
            process.join(timeout=max(0.0, deadline - time.monotonic()))

        for process in self._processes.values():
            if not process.is_alive():
                continue
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
            else:
                process.terminate()

        deadline = time.monotonic() + 5.0
        for process in self._processes.values():
            if process.is_alive():
                process.join(timeout=max(0.0, deadline - time.monotonic()))
        self._dist_initialized = False
        self._close_queues()

    def shutdown(self, *, for_restart=False):
        if self._closed:
            return
        self._closed = True

        if for_restart:
            self._terminate_workers_for_restart()
            return

        import torch.distributed as dist

        # A downstream save/evaluation exception can bypass the normal overlap
        # finalizer. Unblock rank 0's queue consumer before aborting worker
        # processes; otherwise ThreadPoolExecutor's non-daemon thread could keep
        # the interpreter alive indefinitely during error shutdown.
        if self._entropic_overlap is not None:
            state = self._entropic_overlap
            self.seal_entropic_overlap()
            try:
                state["future"].result()
            except Exception:
                pass
            finally:
                state["executor"].shutdown(wait=True, cancel_futures=False)
                self._entropic_overlap = None

        if self._spo_rs_overlap is not None:
            state = self._spo_rs_overlap
            self.seal_spo_rs_overlap()
            try:
                state["future"].result()
            except Exception:
                pass
            finally:
                state["executor"].shutdown(wait=True, cancel_futures=False)
                self._spo_rs_overlap = None

        if (not self._workers_idle or not self._dist_initialized or self._dead_workers(range(1, self._world_size))):
            self._abort_workers()
            return
        try:
            for command_queue in self._command_queues.values():
                command_queue.put({"kind": "stop"})
            dist.barrier()
            self._collect_event("stopped", range(1, self._world_size))
        finally:
            if self._dist_initialized and dist.is_initialized():
                dist.destroy_process_group()
            self._dist_initialized = False
            for process in self._processes.values():
                process.join(timeout=10.0)
            for process in self._processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10.0)
            self._close_queues()
