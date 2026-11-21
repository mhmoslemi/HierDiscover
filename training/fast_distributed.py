"""Process-per-GPU implementation for opt-in fast LoRA training.

The main process is distributed rank 0. Persistent spawned workers own the
remaining GPUs, so transformer forward/checkpoint recomputation on different
cards does not contend for one Python interpreter. Only LoRA gradients and
parameters are communicated, using bounded FP32 NCCL buffers.
"""

from types import SimpleNamespace
import math
import traceback

from config_validation import uses_hybrid_policy_loss

STANDARD_TRAINING_MEMORY_FRACTION = 0.80
MAX_STANDARD_TRAINING_MEMORY_FRACTION = 0.90
LONG_ROLLOUT_MEMORY_FRACTION = 0.96
LONG_ROLLOUT_MEMORY_FRACTIONS = (0.96, 0.98, 0.99)
MAX_LONG_ROLLOUT_MEMORY_FRACTION = max(LONG_ROLLOUT_MEMORY_FRACTIONS)
_FP32_BUCKET_ELEMENTS = 32 * 1024**2


def _shared_prefix_examples(batch):
    batch = list(batch or ())
    if len(batch) < 2:
        return False
    if not all(
            bool(example.get("_shared_prefix_packing_allowed", True))
            for example in batch):
        return False
    prompt_job_id = batch[0].get("prompt_job_id")
    if prompt_job_id is None:
        return False
    prompt_shape = tuple(batch[0]["prompt_ids"].shape)
    return all(
        example.get("prompt_job_id") == prompt_job_id
        and tuple(example["prompt_ids"].shape) == prompt_shape
        for example in batch[1:]
    )


def _shared_prefix_token_count(batch):
    """Physical packed tokens for one prompt and several response branches."""
    batch = list(batch)
    if not batch:
        return 0
    prompt = int(batch[0]["prompt_ids"].shape[1])
    return prompt + sum(max(
        0, int(example["response_ids"].shape[1]) - 1)
        for example in batch)


def trainable_parameters(model):
    return [parameter for parameter in model.parameters()
            if parameter.requires_grad]


def trainable_parameter_signature(model):
    """Plain-data layout used to reject mismatched replicas before NCCL."""
    return tuple(
        (name, tuple(parameter.shape), str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def validate_model_device(model, logical_id):
    """A fast replica must be wholly resident on its one assigned GPU."""
    import torch

    expected = torch.device(f"cuda:{int(logical_id)}")
    wrong = {
        str(parameter.device) for parameter in model.parameters()
        if parameter.device != expected
    }
    if wrong:
        raise RuntimeError(
            f"fast trainer rank {logical_id} expected every parameter on "
            f"{expected}, but also found {sorted(wrong)}")


def _set_allocator_memory_ceiling(logical_id, fraction, *, maximum, label):
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= float(maximum):
        raise ValueError(
            f"{label} allocator ceiling must be in (0, {maximum:.2f}]")
    with torch.cuda.device(int(logical_id)):
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=int(logical_id))


def set_allocator_memory_ceiling(
        logical_id, fraction=STANDARD_TRAINING_MEMORY_FRACTION):
    """Apply the configured normal-training allocator limit."""
    return _set_allocator_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_STANDARD_TRAINING_MEMORY_FRACTION,
        label="fast trainer")


def _set_total_memory_ceiling(logical_id, fraction, *, maximum, label):
    """Cap this allocator plus allocations owned by other GPU processes."""
    import torch

    fraction = float(fraction)
    if not 0.0 < fraction <= float(maximum):
        raise ValueError(
            f"{label} total memory ceiling must be in (0, {maximum:.2f}]")
    with torch.cuda.device(int(logical_id)):
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        outside_allocator = max(
            0, int(total_bytes) - int(free_bytes) - reserved)
        process_limit = int(fraction * int(total_bytes)) - outside_allocator
        if process_limit <= allocated:
            occupied = (outside_allocator + allocated) / max(1, int(total_bytes))
            raise RuntimeError(
                f"{label} rank {logical_id} already needs "
                f"{100.0 * occupied:.1f}% GPU memory before a batch, so the "
                f"{100.0 * fraction:.0f}% total ceiling cannot be honored")
        if reserved > process_limit:
            torch.cuda.empty_cache()
        allocator_fraction = min(
            fraction, process_limit / max(1, int(total_bytes)))
        _set_allocator_memory_ceiling(
            logical_id, allocator_fraction,
            maximum=maximum, label=label)
        return allocator_fraction


def set_total_memory_ceiling(
        logical_id, fraction=STANDARD_TRAINING_MEMORY_FRACTION):
    """Apply the normal conservative limit used by adaptive microbatches."""
    return _set_total_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_STANDARD_TRAINING_MEMORY_FRACTION,
        label="fast trainer")


def set_long_rollout_memory_ceiling(
        logical_id, fraction=LONG_ROLLOUT_MEMORY_FRACTION):
    """Use additional card memory for exact training.

    Callers start at 96% and can retry singletons at 98% and 99% before
    resorting to saved-activation CPU offload. External allocations, including
    sleeping vLLM processes and NCCL, remain part of every ceiling.
    """
    return _set_total_memory_ceiling(
        logical_id, fraction,
        maximum=MAX_LONG_ROLLOUT_MEMORY_FRACTION,
        label="long-rollout rescue")


def parameter_buckets(parameters):
    """Keep small adapters in one collective and group large ones by memory."""
    bucket = []
    elements = 0
    for parameter in parameters:
        count = parameter.numel()
        if bucket and elements + count > _FP32_BUCKET_ELEMENTS:
            yield bucket
            bucket, elements = [], 0
        bucket.append(parameter)
        elements += count
    if bucket:
        yield bucket


def broadcast_trainable_parameters(model, source_rank=0):
    import torch
    import torch.distributed as dist

    parameters = trainable_parameters(model)
    if not parameters:
        raise RuntimeError("fast trainer found no trainable parameters")
    is_source = dist.get_rank() == int(source_rank)
    with torch.no_grad():
        for bucket in parameter_buckets(parameters):
            sizes = [parameter.numel() for parameter in bucket]
            flat = torch.empty(sum(sizes), device=bucket[0].device, dtype=torch.float32)
            parts = flat.split(sizes)
            if is_source:
                for parameter, part in zip(bucket, parts):
                    part.view_as(parameter).copy_(parameter)
            dist.broadcast(flat, src=int(source_rank))
            if not is_source:
                for parameter, part in zip(bucket, parts):
                    parameter.copy_(part.view_as(parameter))
            del part, parts, flat


def reduce_trainable_gradients(model, destination_rank=0, *, group=None):
    """Sum LoRA gradients in FP32 before copying back to parameter dtype."""
    import torch
    import torch.distributed as dist

    parameters = trainable_parameters(model)
    if not parameters:
        raise RuntimeError("fast trainer found no trainable parameters")
    is_destination = dist.get_rank() == int(destination_rank)
    with torch.no_grad():
        for bucket in parameter_buckets(parameters):
            sizes = [parameter.numel() for parameter in bucket]
            flat = torch.empty(sum(sizes), device=bucket[0].device, dtype=torch.float32)
            parts = flat.split(sizes)
            for parameter, part in zip(bucket, parts):
                if parameter.grad is None:
                    part.zero_()
                else:
                    part.view_as(parameter).copy_(parameter.grad)
            dist.reduce(flat, dst=int(destination_rank), op=dist.ReduceOp.SUM, group=group)
            for parameter, part in zip(bucket, parts):
                if is_destination:
                    if parameter.grad is None:
                        parameter.grad = torch.empty_like(parameter)
                    parameter.grad.copy_(part.view_as(parameter))
                else:
                    parameter.grad = None
            del part, parts, flat


def _padded_token_count(batch):
    if not batch:
        return 0
    if _shared_prefix_examples(batch):
        return _shared_prefix_token_count(batch)
    return max(
        int(example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])
        for example in batch
    ) * len(batch)


def _move_examples(examples, logical_id):
    import torch

    device = torch.device(f"cuda:{logical_id}")
    tensor_cache = {}
    moved = []
    for example in examples:
        local = {}
        for key, value in example.items():
            if torch.is_tensor(value):
                cache_key = id(value)
                if cache_key not in tensor_cache:
                    tensor_cache[cache_key] = value.to(
                        device, non_blocking=True)
                local[key] = tensor_cache[cache_key]
            else:
                local[key] = value
        moved.append(local)
    return moved


def historical_training_memory_fraction(model, cfg, configured_fraction):
    if (getattr(model, "_ttt_gpt_oss_shared_lora", False)
            and getattr(cfg, "hardware_profile", None) == "B200_full_ram"):
        return LONG_ROLLOUT_MEMORY_FRACTION
    return float(configured_fraction)


def initial_training_token_budget(model, cfg):
    context = int(cfg.max_seq_length)
    if getattr(model, "_ttt_gpt_oss_shared_lora", False):
        # Start with one long-context workload. Batch size grows from actual
        # backward peaks, including after a fresh TriMul trainer launch.
        return min(context, int(cfg.long_training_headroom_min_tokens))
    return context


class _PackedExpertBatchMemory:
    """Fit padded GPT-OSS work with space for fused decode and FP32 gradients."""

    def __init__(self, model, parameters, logprob_chunk):
        config = model.config
        hidden = int(config.hidden_size)
        intermediate = int(config.intermediate_size)
        experts = int(config.num_local_experts)
        routed = int(config.num_experts_per_tok)
        bf16_bytes = 2
        # Gate/up is the largest decoded BF16 stack. Leave room for a
        # second copy during grouped GEMM or transposed backward decode.
        self.decode_bytes = experts * hidden * (2 * intermediate) * bf16_bytes
        self.gradient_bytes = sum(p.numel() * p.element_size() for p in parameters)
        # One BF16 logits tile and two FP32 buffers for log-softmax.
        head_bytes = max(1, int(logprob_chunk or 256)) * int(config.vocab_size) * 10
        self.fixed_bytes = 2 * self.decode_bytes + self.gradient_bytes + head_bytes
        self.minimum_bytes_per_token = bf16_bytes * routed * (hidden + 2 * intermediate)
        layers = int(config.num_hidden_layers)
        # Eager backward retains router, expert and attention activations
        # across all layers, even when no single layer is especially large.
        self.eager_bytes_per_token = bf16_bytes * layers * (
            routed * (3 * hidden + 5 * intermediate) + 12 * hidden
        )
        # Before a measurement, budget the routed input/output and intermediate
        # gradients, plus hidden states at layer checkpoint boundaries.
        self.bytes_per_token = bf16_bytes * (
            routed * (8 * hidden + 12 * intermediate)
            + hidden * (layers + 8)
        )
        self.failed_work_limit = math.inf

    def available_work(self, room_bytes):
        return max(1, int((room_bytes - self.fixed_bytes) // self.bytes_per_token))

    def limit(self, budget, room_bytes):
        return min(budget, self.available_work(room_bytes), self.failed_work_limit)

    def needs_checkpointing(self, work, room_bytes):
        return work * self.eager_bytes_per_token > room_bytes - self.fixed_bytes

    def observe(self, work, used_bytes):
        # Incoming grads and one decode stack are already covered by the fixed
        # reserve. Learn their remaining cost from the whole forward/backward.
        variable = max(0, used_bytes - self.gradient_bytes - self.decode_bytes)
        self.bytes_per_token = max(
            self.minimum_bytes_per_token,
            self.bytes_per_token * 0.85,
            variable * 1.15 / work,
        )

    def backoff(self, failed_work):
        self.failed_work_limit = min(self.failed_work_limit, failed_work * 7 // 8)


def _rank_memory(cuda):
    allocated = int(cuda.memory_allocated())
    reserved = int(cuda.memory_reserved())
    free, total = cuda.mem_get_info()
    return allocated, reserved, int(free), int(total)


def local_spo_rs_update(backend, model, tokenizer, examples, cfg, logical_id,
                      token_budget, *, memory_fraction=0.80, work_queue=None,
                      max_examples_per_batch=64,
                      activation_offload_semaphore=None,
                      defer_exact_oom=False):
    """Accumulate one rank's gradients and leave them attached to the model."""
    import torch
    import training.trainers as training
    from runtime.host_memory import reclaim_unused_host_memory
    from models.model_backend import fused_long_attention_is_active
    from training.exact_overflow import supports_warm_overflow

    if not 0.0 < float(memory_fraction) <= (
            MAX_STANDARD_TRAINING_MEMORY_FRACTION):
        raise ValueError(
            "fast trainer memory_fraction must be in (0, 0.90]")

    backend.set_training_mode()
    initial_fraction = historical_training_memory_fraction(model, cfg, memory_fraction)
    headroom_preallocated = initial_fraction > float(memory_fraction)
    if headroom_preallocated:
        allocator_fraction = set_long_rollout_memory_ceiling(
            logical_id, initial_fraction)
    else:
        allocator_fraction = set_total_memory_ceiling(
            logical_id, initial_fraction)
    hybrid = uses_hybrid_policy_loss(cfg.advantage_mode)
    clip_options = None if hybrid else training._clipped_policy_options(cfg)
    parameters = trainable_parameters(model)
    packed_memory = (
        _PackedExpertBatchMemory(model, parameters, cfg.logprob_chunk)
        if getattr(model, "_ttt_gpt_oss_shared_lora", False) else None
    )
    gradient_accumulators = [None for _ in parameters]
    metric_keys = (
        "loss", "policy_loss", "kl_estimate",
        "ratio", "clipped_fraction",
    )
    metric_count = len(metric_keys)
    step_metrics = None
    quarantined_examples = 0
    deferred_keys = []
    deferred_token_lengths = []
    backward_batches = 0
    trained_examples = 0
    peak_memory_fraction = 0.0
    budget = max(1, int(token_budget or cfg.max_seq_length))
    packing_budget = budget
    pending_batches = []
    emergency_allocator_fraction = None
    emergency_allocator_stage = int(headroom_preallocated)

    def expand_for_long_rollout():
        nonlocal allocator_fraction, emergency_allocator_fraction
        nonlocal emergency_allocator_stage
        while emergency_allocator_stage < len(
                LONG_ROLLOUT_MEMORY_FRACTIONS):
            requested_fraction = LONG_ROLLOUT_MEMORY_FRACTIONS[
                emergency_allocator_stage]
            emergency_allocator_stage += 1
            previous_fraction = allocator_fraction
            try:
                expanded = set_long_rollout_memory_ceiling(
                    logical_id, requested_fraction)
            except RuntimeError as error:
                print(f"[train-oom] cuda:{logical_id}: "
                      f"{100.0 * requested_fraction:.0f}% reserved GPU "
                      f"headroom is unavailable ({error})", flush=True)
                continue
            allocator_fraction = expanded
            if expanded <= previous_fraction:
                continue
            emergency_allocator_fraction = expanded
            return expanded
        return None

    examples = list(examples or ())
    if work_queue is not None and examples:
        raise ValueError(
            "fast trainer accepts either local examples or a shared queue")

    def example_length(example):
        return int(
            example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])

    def record_deferred(example):
        key = example.get("_overlap_key")
        if (not isinstance(key, (tuple, list)) or len(key) != 2):
            raise RuntimeError(
                "deferred exact SPO-RS trajectory has no stable overlap key")
        deferred_keys.append((int(key[0]), int(key[1])))
        deferred_token_lengths.append(example_length(example))

    def needs_fused_long_singleton(example):
        return bool(
            fused_long_attention_is_active()
            and training._requires_fused_long_singleton(
                cfg, example_length(example))
        )

    def fused_pack_limit():
        if (fused_long_attention_is_active()
                and getattr(cfg, "fused_long_attention", False)):
            return training._FUSED_LONG_SINGLETON_MIN_TOKENS
        return None

    max_examples_per_batch = max(1, min(64, int(max_examples_per_batch)))
    if work_queue is None:
        pending = sorted(examples, key=example_length)

        def take_batch():
            if not pending:
                return []
            batch = [pending.pop()]
            maximum = example_length(batch[0])
            if needs_fused_long_singleton(batch[0]):
                return batch
            while pending and len(batch) < max_examples_per_batch:
                candidate = pending[-1]
                if needs_fused_long_singleton(candidate):
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > packing_budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
                    break
                batch.append(pending.pop())
                maximum = next_maximum
            return batch
    else:
        # None is one end marker per rank. A single deferred example lets a rank
        # stop growing the current padded batch without putting work back or
        # disturbing the global queue's exactly-once behavior.
        queue_finished = False
        deferred = None

        def take_batch():
            nonlocal queue_finished, deferred
            if queue_finished and deferred is None:
                return []
            if deferred is not None:
                first, deferred = deferred, None
            else:
                first = work_queue.get()
                if first is None:
                    queue_finished = True
                    return []
            batch = [first]
            maximum = example_length(first)
            if needs_fused_long_singleton(first):
                return batch
            while (not queue_finished
                   and len(batch) < max_examples_per_batch):
                candidate = work_queue.get()
                if candidate is None:
                    queue_finished = True
                    break
                if needs_fused_long_singleton(candidate):
                    deferred = candidate
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > packing_budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
                    deferred = candidate
                    break
                batch.append(candidate)
                maximum = next_maximum
            return batch

    model.zero_grad(set_to_none=True)
    while True:
        measurement = None
        if packed_memory is not None:
            with torch.cuda.device(logical_id):
                measurement = _rank_memory(torch.cuda)
            allocated, reserved, free, total = measurement
            room = min(int(allocator_fraction * total), free + reserved) - allocated
            packing_budget = packed_memory.limit(budget, room)
        else:
            packing_budget = budget

        claimed = pending_batches.pop() if pending_batches else take_batch()
        if packed_memory is not None and len(claimed) > 1:
            fit = max(1, packing_budget // max(map(example_length, claimed)))
            if fit < len(claimed):
                pending_batches.append(claimed[fit:])
                claimed = claimed[:fit]
        cpu_batch = claimed
        if not cpu_batch:
            break
        forced_overflow_min_tokens = int(getattr(cfg, '_exact_replica_overflow_min_tokens', 0) or 0)
        if defer_exact_oom:
            retained = []
            for example in cpu_batch:
                if (bool(example.get("_cooperative_overflow", False))
                        or (forced_overflow_min_tokens > 0
                            and example_length(example) >= forced_overflow_min_tokens)):
                    record_deferred(example)
                else:
                    retained.append(example)
            claimed = cpu_batch = retained
            if not cpu_batch:
                continue
        requested_padded_tokens = _padded_token_count(cpu_batch)
        fused_long_singleton = bool(
            len(cpu_batch) == 1
            and needs_fused_long_singleton(cpu_batch[0]))
        packed_checkpointing = packed_memory is not None and (
            requested_padded_tokens >= cfg.long_training_checkpoint_min_tokens
            or packed_memory.needs_checkpointing(requested_padded_tokens, room)
        )
        packed_needs_headroom = (
            packed_memory is not None and len(cpu_batch) == 1
            and requested_padded_tokens > packed_memory.available_work(room)
        )
        cpu_batch = [dict(example) for example in cpu_batch]
        missing_behavior = training._initialize_behavior_logprobs(cpu_batch)
        if hybrid and missing_behavior:
            raise ValueError("hybrid training requires the original sampler logprobs")
        for example in cpu_batch:
            example.pop("behavior_logprobs", None)

        with torch.cuda.device(logical_id):
            base_allocated, base_reserved, free_bytes, total_bytes = (
                measurement if measurement is not None else _rank_memory(torch.cuda))
            external_bytes = max(
                0, int(total_bytes) - int(free_bytes) - base_reserved)
            torch.cuda.reset_peak_memory_stats()
            local_batch = _move_examples(cpu_batch, logical_id)

            def attempt(active_batches):
                attempt_values = {key: [] for key in metric_keys}
                attempt_ratio_values = []
                attempt_prefix_ratio_values = []

                with training._policy_dropout_disabled(model):
                    for batch in active_batches:
                        for example in batch:
                            if example["old_policy_logprobs"] is not None:
                                continue
                            old_logprobs = training.compute_batched_token_logprobs(
                                model, [example], with_grad=False,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id)[0]
                            if not torch.isfinite(old_logprobs).all():
                                raise FloatingPointError(
                                    "nonfinite old-policy logprobs")
                            example["old_policy_logprobs"] = old_logprobs.detach()

                        current_logprobs = training.compute_batched_token_logprobs(
                            model, batch, with_grad=True,
                            chunk=cfg.logprob_chunk,
                            pad_token_id=tokenizer.pad_token_id)

                        weighted_losses = []
                        for example, current_lp in zip(
                                batch, current_logprobs):
                            loss, metrics = training.historical_policy_loss(
                                cfg, current_lp, example, clip_options,
                                return_tensor_metrics=True, check_finite=False)
                            weight = float(example["sample_weight"])
                            loss_weight = 1.0 if hybrid else weight
                            weighted_losses.append(loss_weight * loss)
                            attempt_values["loss"].append(
                                loss_weight * loss.detach())
                            attempt_values["policy_loss"].append(
                                loss_weight * metrics["policy_loss"])
                            for key in ("kl_estimate", "ratio"):
                                attempt_values[key].append(
                                    weight * metrics[key])
                            attempt_values["clipped_fraction"].append(
                                weight * metrics["clipped"])
                            attempt_ratio_values.append(metrics["ratio"])
                            attempt_prefix_ratio_values.append(
                                metrics["prefix_ratio_max"])
                        if weighted_losses:
                            if hybrid and not torch.isfinite(
                                    torch.stack([loss.detach() for loss in weighted_losses])).all():
                                raise FloatingPointError("nonfinite hybrid importance-sampling loss")
                            sum(weighted_losses[1:],
                                weighted_losses[0]).backward()

                zero = torch.zeros(
                    (), dtype=torch.float64,
                    device=torch.device(f"cuda:{logical_id}"))
                reduced = [
                    (torch.stack(attempt_values[key]).sum()
                     if attempt_values[key] else zero)
                    for key in metric_keys
                ]
                reduced.extend((
                    (torch.stack(attempt_ratio_values).max()
                     if attempt_ratio_values else zero),
                    (torch.stack(attempt_prefix_ratio_values).max()
                     if attempt_prefix_ratio_values else zero),
                ))
                return torch.stack([
                    value.to(dtype=torch.float64) for value in reduced
                ]).detach()

            try:
                result, effective_batches, unprocessed = training._run_oom_resilient_backward(
                    model, [local_batch], attempt,
                    device_label=f"cuda:{logical_id}",
                    expand_memory=expand_for_long_rollout,
                    start_checkpointed=fused_long_singleton or packed_checkpointing,
                    start_with_headroom=(
                        (fused_long_singleton and not headroom_preallocated)
                        or packed_needs_headroom
                    ),
                    headroom_preallocated=headroom_preallocated,
                    # Selectively resumed trajectories already exhausted the
                    # complete GPU-only ladder in an earlier process. Enter
                    # bounded offload immediately for this dispatch only;
                    # ordinary later steps retain their GPU-first behavior.
                    start_activation_offload=bool(
                        defer_exact_oom and getattr(
                            cfg,
                            "_exact_replica_force_activation_offload",
                            False)),
                    activation_offload_semaphore=(
                        activation_offload_semaphore),
                    # The warm cooperative lane supplies more GPU memory
                    # without retaining an enormous host activation graph.
                    # Other objectives/backends keep their existing policy.
                    allow_activation_offload=not (
                        defer_exact_oom and supports_warm_overflow(cfg)),
                    defer_exact_oom=bool(defer_exact_oom),
                    split_failed_batches=(packed_memory is None),
                )
            except RuntimeError as error:
                if (packed_memory is None or len(claimed) == 1
                        or not (training._is_cuda_oom(error)
                                or training._is_attention_kernel_unavailable(error))):
                    raise
                # Split only the failed work. Each successful child batch
                # commits independently to this rank's gradient accumulation.
                packed_memory.backoff(requested_padded_tokens)
                midpoint = (len(claimed) + 1) // 2
                pending_batches.extend((claimed[midpoint:], claimed[:midpoint]))
                print(f"[train-oom] cuda:{logical_id}: split failed "
                      f"{requested_padded_tokens:,}-token batch into "
                      f"{midpoint}+{len(claimed) - midpoint} examples", flush=True)
                del local_batch
                continue
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            used_at_peak = min(
                int(total_bytes), external_bytes + peak_reserved)
            observed_fraction = used_at_peak / max(1, int(total_bytes))
            peak_memory_fraction = max(
                peak_memory_fraction, observed_fraction)

            if result is not None:
                with torch.no_grad():
                    accumulated = []
                    incoming = []
                    for parameter_index, parameter in enumerate(parameters):
                        gradient = parameter.grad
                        if gradient is None:
                            continue
                        accumulator = gradient_accumulators[parameter_index]
                        if accumulator is None:
                            gradient_accumulators[
                                parameter_index] = gradient.detach()
                        else:
                            accumulated.append(accumulator)
                            incoming.append(gradient)
                        parameter.grad = None
                    if incoming:
                        torch._foreach_add_(accumulated, incoming)
                    del incoming, accumulated
                    if step_metrics is None:
                        step_metrics = result
                    else:
                        step_metrics[:metric_count].add_(result[:metric_count])
                        maxima = step_metrics[metric_count:]
                        torch.maximum(maxima, result[metric_count:], out=maxima)

            model.zero_grad(set_to_none=True)
            if defer_exact_oom:
                for deferred_example in unprocessed:
                    record_deferred(deferred_example)
            else:
                quarantined_examples += len(unprocessed)
            backward_batches += len(effective_batches)
            trained_examples += sum(
                len(batch) for batch in effective_batches)
            successful_padded_tokens = max(
                (_padded_token_count(batch) for batch in effective_batches),
                default=0)
            longest_successful = max(
                (int(example["prompt_ids"].shape[1]
                     + example["response_ids"].shape[1])
                 for batch in effective_batches for example in batch),
                default=1)
            backed_off = bool(len(effective_batches) > 1 or unprocessed)
            if packed_memory is not None:
                if effective_batches and not backed_off:
                    if packed_checkpointing:
                        packed_memory.observe(
                            successful_padded_tokens, peak_allocated - base_allocated)
                    # A batch can leave unused capacity simply because the
                    # next complete rollout would not fit. Allow growth even
                    # if the current batch did not fill its entire token cap.
                    budget = min(int(cfg.max_seq_length), math.ceil(budget * 1.25))
                budget = min(budget, packed_memory.failed_work_limit)
            elif not effective_batches:
                budget = max(1, budget // 2)
            elif backed_off:
                budget = max(
                    longest_successful,
                    min(budget, successful_padded_tokens))
            else:
                active_bytes = max(1, peak_allocated - base_allocated)
                target_process_bytes = max(
                    base_allocated + 1,
                    int(allocator_fraction * int(total_bytes)))
                available_for_batch = max(
                    1, target_process_bytes - base_allocated)
                desired_budget = int(
                    requested_padded_tokens
                    * available_for_batch / active_bytes * 0.95)
                lower = max(longest_successful, int(budget * 0.70))
                upper = max(lower, int(budget * 1.20))
                desired_budget = min(upper, max(lower, desired_budget))
                budget = max(
                    longest_successful,
                    int(round(0.5 * budget + 0.5 * desired_budget)))

        del local_batch, effective_batches, unprocessed

    with torch.cuda.device(logical_id), torch.no_grad():
        for parameter, accumulator in zip(
                parameters, gradient_accumulators):
            parameter.grad = accumulator
        if step_metrics is None:
            torch.cuda.synchronize(logical_id)
            packed = [0.0] * (metric_count + 2)
        else:
            packed = step_metrics.cpu().tolist()
    reclaim_unused_host_memory(
        torch, cuda_devices=(logical_id,), force=True)

    return {
        "totals": dict(zip(metric_keys, packed[:metric_count])),
        "max_ratio": packed[metric_count],
        "max_prefix_ratio": packed[metric_count + 1],
        "quarantined_examples": quarantined_examples,
        "deferred_examples": len(deferred_keys),
        "deferred_keys": deferred_keys,
        "deferred_token_lengths": deferred_token_lengths,
        "backward_batches": backward_batches,
        "trained_examples": trained_examples,
        "peak_memory_fraction": peak_memory_fraction,
        "allocator_memory_fraction": allocator_fraction,
        "long_rollout_memory_rescue": (
            emergency_allocator_fraction is not None),
        "token_budget": budget,
    }


def local_policy_update(backend, model, tokenizer, examples, cfg, logical_id,
                        token_budget, total_examples, *,
                        memory_fraction=0.80, work_queue=None,
                        activation_offload_semaphore=None):
    """Accumulate one rank's exact entropic gradients.

    Rank 0 supplies the global example count so summing LoRA gradients
    across ranks reproduces the global mean loss.
    """
    from contextlib import nullcontext
    import torch
    import training.trainers as training
    from runtime.host_memory import reclaim_unused_host_memory
    from models.model_backend import fused_long_attention_is_active

    if not 0.0 < float(memory_fraction) <= (
            MAX_STANDARD_TRAINING_MEMORY_FRACTION):
        raise ValueError(
            "fast trainer memory_fraction must be in (0, 0.90]")
    total_examples = int(total_examples)
    if total_examples < 1:
        raise ValueError("fast policy update requires at least one example")

    backend.set_training_mode()
    allocator_fraction = set_total_memory_ceiling(
        logical_id, float(memory_fraction))
    parameters = trainable_parameters(model)
    gradient_accumulators = [None for _ in parameters]
    total_loss = 0.0
    total_logp_delta = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    ratio_count = 0
    quarantined_examples = 0
    backward_batches = 0
    trained_examples = 0
    peak_memory_fraction = 0.0
    kl_errors = []
    budget = max(1, int(token_budget or cfg.max_seq_length))
    emergency_allocator_fraction = None
    emergency_allocator_stage = 0

    def expand_for_long_rollout():
        nonlocal allocator_fraction, emergency_allocator_fraction
        nonlocal emergency_allocator_stage
        while emergency_allocator_stage < len(
                LONG_ROLLOUT_MEMORY_FRACTIONS):
            requested_fraction = LONG_ROLLOUT_MEMORY_FRACTIONS[
                emergency_allocator_stage]
            emergency_allocator_stage += 1
            try:
                expanded = set_long_rollout_memory_ceiling(
                    logical_id, requested_fraction)
            except RuntimeError as error:
                print(f"[train-oom] cuda:{logical_id}: "
                      f"{100.0 * requested_fraction:.0f}% reserved GPU "
                      f"headroom is unavailable ({error})", flush=True)
                continue
            if (emergency_allocator_fraction is not None
                    and expanded <= emergency_allocator_fraction):
                continue
            emergency_allocator_fraction = expanded
            allocator_fraction = max(allocator_fraction, expanded)
            return expanded
        return None

    examples = list(examples or ())
    if work_queue is not None and examples:
        raise ValueError(
            "fast trainer accepts either local examples or a shared queue")

    def example_length(example):
        return int(
            example["prompt_ids"].shape[1]
            + example["response_ids"].shape[1])

    def needs_fused_long_singleton(example):
        return bool(
            fused_long_attention_is_active()
            and training._requires_fused_long_singleton(
                cfg, example_length(example))
        )

    def fused_pack_limit():
        if (fused_long_attention_is_active()
                and getattr(cfg, "fused_long_attention", False)):
            return training._FUSED_LONG_SINGLETON_MIN_TOKENS
        return None

    max_examples_per_batch = 64
    if work_queue is None:
        pending = sorted(examples, key=example_length)
        max_examples_per_batch = min(
            max_examples_per_batch, max(1, len(examples)))

        def take_batch():
            if not pending:
                return []
            batch = [pending.pop()]
            maximum = example_length(batch[0])
            if needs_fused_long_singleton(batch[0]):
                return batch
            while pending and len(batch) < max_examples_per_batch:
                candidate = pending[-1]
                if needs_fused_long_singleton(candidate):
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
                    break
                batch.append(pending.pop())
                maximum = next_maximum
            return batch
    else:
        queue_finished = False
        deferred = None

        def take_batch():
            nonlocal queue_finished, deferred
            if queue_finished and deferred is None:
                return []
            if deferred is not None:
                first, deferred = deferred, None
            else:
                first = work_queue.get()
                if first is None:
                    queue_finished = True
                    return []
            batch = [first]
            maximum = example_length(first)
            if needs_fused_long_singleton(first):
                return batch
            while (not queue_finished
                   and len(batch) < max_examples_per_batch):
                candidate = work_queue.get()
                if candidate is None:
                    queue_finished = True
                    break
                if needs_fused_long_singleton(candidate):
                    deferred = candidate
                    break
                candidate_length = example_length(candidate)
                next_maximum = max(maximum, candidate_length)
                next_padded_tokens = next_maximum * (len(batch) + 1)
                pack_limit = fused_pack_limit()
                if (next_padded_tokens > budget
                        or (pack_limit is not None
                            and next_padded_tokens >= pack_limit)):
                    deferred = candidate
                    break
                batch.append(candidate)
                maximum = next_maximum
            return batch

    model.zero_grad(set_to_none=True)
    while True:
        cpu_batch = take_batch()
        if not cpu_batch:
            break
        requested_padded_tokens = _padded_token_count(cpu_batch)
        fused_long_singleton = bool(
            len(cpu_batch) == 1
            and needs_fused_long_singleton(cpu_batch[0]))

        with torch.cuda.device(logical_id):
            base_allocated = int(torch.cuda.memory_allocated())
            base_reserved = int(torch.cuda.memory_reserved())
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            external_bytes = max(
                0, int(total_bytes) - int(free_bytes) - base_reserved)
            torch.cuda.reset_peak_memory_stats()
            local_batch = _move_examples(cpu_batch, logical_id)

            def attempt(active_batches):
                attempt_losses = []
                attempt_logp_deltas = []
                attempt_ratio_means = []
                attempt_ratio_maxima = []
                attempt_ratio_count = 0
                attempt_kl_error = None

                # Match the existing standard-policy trainer exactly. Unlike
                # PPO's frozen-policy likelihood pass, this objective retains
                # the model's ordinary training-mode dropout behavior.
                with nullcontext():
                    for batch in active_batches:
                        base_logprobs = [
                            example.get("reference_logprobs")
                            for example in batch
                        ]
                        supplied_reference = all(
                            training._valid_example_token_logprobs(
                                example, "reference_logprobs")
                            for example in batch)
                        if not supplied_reference:
                            try:
                                with (backend.disable_adapter(),
                                      torch.no_grad()):
                                    base_logprobs = (
                                        training.compute_batched_token_logprobs(
                                            model, batch, with_grad=False,
                                            chunk=cfg.logprob_chunk,
                                            pad_token_id=(
                                                tokenizer.pad_token_id)))
                            except Exception as error:
                                raise RuntimeError(
                                    "exact base-policy logprob fallback "
                                    "failed; refusing to replace the "
                                    "configured KL penalty with the current "
                                    "policy") from error
                        current_logprobs = (
                            training.compute_batched_token_logprobs(
                                model, batch, with_grad=True,
                                chunk=cfg.logprob_chunk,
                                pad_token_id=tokenizer.pad_token_id))

                        batch_losses = []
                        for example, current_lp, base_lp in zip(
                                batch, current_logprobs, base_logprobs):
                            base_lp = base_lp.to(current_lp.device)
                            advantage = example["advantage"]
                            logp_difference = (
                                current_lp - base_lp).detach()
                            average_difference = logp_difference.mean()
                            kl_advantage = cfg.kl_penalty_coef * (
                                average_difference
                                - (current_lp - base_lp))
                            effective_advantage = advantage + kl_advantage

                            has_behavior = (
                                training._valid_example_token_logprobs(
                                    example, "behavior_logprobs"))
                            behavior_lp = (
                                example["behavior_logprobs"].to(
                                    current_lp.device)
                                if has_behavior else None)
                            if has_behavior:
                                importance_ratio = (
                                    training.
                                    _detached_behavior_importance_ratio(
                                        current_lp, behavior_lp))
                                loss = -(
                                    importance_ratio
                                    * effective_advantage.detach()
                                    * current_lp).mean()
                            else:
                                importance_ratio = 1.0
                                loss = -(
                                    effective_advantage.detach()
                                    * current_lp).mean()
                            if has_behavior:
                                attempt_ratio_means.append(
                                    importance_ratio.mean().detach())
                                attempt_ratio_maxima.append(
                                    importance_ratio.max().detach())
                                attempt_ratio_count += 1
                            if not torch.isfinite(loss).all():
                                raise FloatingPointError(
                                    "nonfinite policy loss")
                            batch_losses.append(loss / total_examples)
                            attempt_losses.append(loss.detach())
                            attempt_logp_deltas.append(
                                logp_difference.mean().detach())
                        if batch_losses:
                            sum(batch_losses[1:],
                                batch_losses[0]).backward()

                zero = torch.zeros(
                    (), dtype=torch.float64,
                    device=torch.device(f"cuda:{logical_id}"))
                packed = torch.stack([
                    (torch.stack(attempt_losses).sum()
                     if attempt_losses else zero).to(torch.float64),
                    (torch.stack(attempt_logp_deltas).sum()
                     if attempt_logp_deltas else zero).to(torch.float64),
                    (torch.stack(attempt_ratio_means).sum()
                     if attempt_ratio_means else zero).to(torch.float64),
                    (torch.stack(attempt_ratio_maxima).max()
                     if attempt_ratio_maxima else zero).to(torch.float64),
                ]).detach().cpu().tolist()
                return (
                    packed[0], packed[1], packed[2], packed[3],
                    attempt_ratio_count,
                    attempt_kl_error,
                )

            result, effective_batches, quarantined = (
                training._run_oom_resilient_backward(
                    model, [local_batch], attempt,
                    device_label=f"cuda:{logical_id}",
                    expand_memory=expand_for_long_rollout,
                    start_checkpointed=fused_long_singleton,
                    start_with_headroom=fused_long_singleton,
                    activation_offload_semaphore=(
                        activation_offload_semaphore)))
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            used_at_peak = min(
                int(total_bytes), external_bytes + peak_reserved)
            observed_fraction = used_at_peak / max(1, int(total_bytes))
            peak_memory_fraction = max(
                peak_memory_fraction, observed_fraction)

            if result is not None:
                with torch.no_grad():
                    for parameter_index, parameter in enumerate(parameters):
                        gradient = parameter.grad
                        if gradient is None:
                            continue
                        accumulator = gradient_accumulators[parameter_index]
                        if accumulator is None:
                            gradient_accumulators[
                                parameter_index] = gradient.detach()
                        else:
                            accumulator.add_(gradient)
                        parameter.grad = None
                total_loss += result[0]
                total_logp_delta += result[1]
                ratio_sum += result[2]
                ratio_max = max(ratio_max, result[3])
                ratio_count += result[4]
                if result[5] is not None:
                    kl_errors.append(result[5])

            model.zero_grad(set_to_none=True)
            quarantined_examples += len(quarantined)
            backward_batches += len(effective_batches)
            trained_examples += sum(
                len(batch) for batch in effective_batches)
            successful_padded_tokens = max(
                (_padded_token_count(batch) for batch in effective_batches),
                default=0)
            longest_successful = max(
                (int(example["prompt_ids"].shape[1]
                     + example["response_ids"].shape[1])
                 for batch in effective_batches for example in batch),
                default=1)
            backed_off = bool(len(effective_batches) > 1 or quarantined)
            if not effective_batches:
                budget = max(1, budget // 2)
            elif backed_off:
                budget = max(
                    longest_successful,
                    min(budget, successful_padded_tokens))
            else:
                active_bytes = max(1, peak_allocated - base_allocated)
                target_process_bytes = max(
                    base_allocated + 1,
                    int(float(memory_fraction) * int(total_bytes))
                    - external_bytes)
                available_for_batch = max(
                    1, target_process_bytes - base_allocated)
                desired_budget = int(
                    requested_padded_tokens
                    * available_for_batch / active_bytes * 0.95)
                if observed_fraction >= float(memory_fraction):
                    desired_budget = min(
                        desired_budget,
                        int(budget * float(memory_fraction)
                            / max(observed_fraction, 1e-9) * 0.95))
                lower = max(longest_successful, int(budget * 0.70))
                upper = max(lower, int(budget * 1.20))
                desired_budget = min(upper, max(lower, desired_budget))
                budget = max(
                    longest_successful,
                    int(round(0.5 * budget + 0.5 * desired_budget)))

        del local_batch, effective_batches, quarantined

    with torch.cuda.device(logical_id), torch.no_grad():
        for parameter, accumulator in zip(
                parameters, gradient_accumulators):
            parameter.grad = accumulator
        torch.cuda.synchronize(logical_id)
    reclaim_unused_host_memory(
        torch, cuda_devices=(logical_id,), force=True)

    return {
        "total_loss": total_loss,
        "total_logp_delta": total_logp_delta,
        "ratio_sum": ratio_sum,
        "ratio_max": ratio_max,
        "ratio_count": ratio_count,
        "quarantined_examples": quarantined_examples,
        "backward_batches": backward_batches,
        "trained_examples": trained_examples,
        "peak_memory_fraction": peak_memory_fraction,
        "allocator_memory_fraction": allocator_fraction,
        "long_rollout_memory_rescue": (
            emergency_allocator_fraction is not None),
        "token_budget": budget,
        "kl_errors": kl_errors,
    }


def worker_main(rank, world_size, cfg_dict, init_method, work_queue,
                command_queue, result_queue, dependency_log_path=None,
                activation_offload_semaphore=None):
    """Persistent rank>0 worker command loop."""
    dist_initialized = False
    try:
        import gc
        import os
        from datetime import timedelta
        import warnings
        import diagnostics.console as console

        warnings.filterwarnings("ignore", category=FutureWarning)
        console._install_console_timestamps()
        terminal_log = console._install_terminal_log()
        terminal_log_path = cfg_dict.get("_terminal_log_path")
        if terminal_log_path:
            terminal_log.bind(terminal_log_path)
        setting_log_path = cfg_dict.get("_setting_log_path")
        if setting_log_path:
            # Use the compatibility binding exported by the console module.
            # Logging-version skew must not abort a distributed update after
            # rollout generation and evaluation have completed.
            console.bind_setting_log(setting_log_path)

        from diagnostics.loading_logs import quiet_replica_load

        rank_log = (f"{dependency_log_path}.trainer-rank{rank}"
                    if dependency_log_path else None)
        with quiet_replica_load(rank_log):
            import torch
            import torch.distributed as dist
            from models.model_backend import load_backend

            torch.set_num_threads(max(
                1, int(os.cpu_count() or world_size) // int(world_size)))
            torch.cuda.set_device(int(rank))
            memory_fraction = float(
                cfg_dict.get("training_memory_fraction", 0.80))
            set_total_memory_ceiling(int(rank), memory_fraction)
            cfg = SimpleNamespace(**dict(cfg_dict))
            cfg.training_replica_device = int(rank)
            cfg.num_training_gpus = 1
            backend = load_backend(cfg.backend, cfg)
            model, tokenizer = backend.load()
        validate_model_device(model, int(rank))
        result_queue.put({
            "event": "loaded", "rank": int(rank),
            "parameter_signature": trainable_parameter_signature(model),
        })

        command = command_queue.get()
        if command.get("kind") != "init_distributed":
            raise RuntimeError("fast worker expected init_distributed")
        dist.init_process_group(
            backend="nccl", init_method=init_method,
            rank=int(rank), world_size=int(world_size),
            timeout=timedelta(minutes=30))
        dist_initialized = True
        broadcast_trainable_parameters(model, source_rank=0)
        result_queue.put({"event": "ready", "rank": int(rank)})
        offloaded = False

        while True:
            command = command_queue.get()
            kind = command.get("kind")
            if kind == "purge_host_memory":
                from runtime.host_memory import reclaim_unused_host_memory

                with torch.cuda.device(int(rank)):
                    torch.cuda.synchronize(int(rank))
                cleanup = reclaim_unused_host_memory(
                    torch, cuda_devices=(int(rank),), force=True)
                result_queue.put({
                    "event": "host_memory_purged",
                    "rank": int(rank),
                    "cleanup": cleanup,
                })
            elif kind == "offload":
                if not offloaded:
                    if command.get("incremental"):
                        backend.offload_for_generation(
                            host_admission=activation_offload_semaphore.host_admission,
                            deadline=command.get("deadline"))
                    elif command.get("disk_directory"):
                        backend.offload_for_generation(disk_directory=command["disk_directory"])
                    else:
                        backend.offload_for_generation()
                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        if cfg.evaluation_shares_generation:
                            torch.cuda.synchronize(int(rank))
                            torch.cuda.ipc_collect()
                        torch.cuda.empty_cache()
                    offloaded = True
                result_queue.put({"event": "offloaded", "rank": int(rank)})
            elif kind == "restore":
                if offloaded:
                    from runtime.host_memory import reclaim_unused_host_memory

                    gc.collect()
                    with torch.cuda.device(int(rank)):
                        torch.cuda.empty_cache()
                        set_total_memory_ceiling(
                            int(rank), memory_fraction)
                        backend.restore_after_generation()
                        backend.set_training_mode()
                        torch.cuda.synchronize(int(rank))
                    validate_model_device(model, int(rank))
                    # ``Module.to(cuda)`` releases its former CPU storages,
                    # but glibc and the pinned allocator can retain those
                    # arenas. Return them before long-sequence training starts.
                    reclaim_unused_host_memory(
                        torch, cuda_devices=(int(rank),), force=True)
                    offloaded = False
                result_queue.put({"event": "restored", "rank": int(rank)})
            elif kind == "train_spo_rs":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received training while offloaded")
                step_cfg = SimpleNamespace(**dict(
                    command.get("step_cfg") or vars(cfg)))
                stats = local_spo_rs_update(
                    backend, model, tokenizer, (), step_cfg,
                    int(rank), command["token_budget"],
                    memory_fraction=command.get("memory_fraction", 0.80),
                    work_queue=work_queue,
                    max_examples_per_batch=command.get("max_examples_per_batch", 64),
                    activation_offload_semaphore=(
                        activation_offload_semaphore),
                    defer_exact_oom=bool(command.get(
                        "defer_exact_oom", False)))
                result_queue.put({
                    "event": "computed", "rank": int(rank),
                    "stats": stats,
                })
            elif kind == "train_policy":
                if offloaded:
                    raise RuntimeError(
                        "fast worker received training while offloaded")
                step_cfg = SimpleNamespace(**dict(
                    command.get("step_cfg") or vars(cfg)))
                stats = local_policy_update(
                    backend, model, tokenizer, (), step_cfg,
                    int(rank), command["token_budget"],
                    command["total_examples"],
                    memory_fraction=command.get("memory_fraction", 0.80),
                    work_queue=work_queue,
                    activation_offload_semaphore=(
                        activation_offload_semaphore))
                result_queue.put({
                    "event": "computed", "rank": int(rank),
                    "stats": stats,
                })
            elif kind == "overflow_backward":
                if not offloaded:
                    raise RuntimeError("cooperative overflow requires parked replicas")
                from training.exact_overflow import compute_overflow_work

                step_cfg = SimpleNamespace(**dict(command["step_cfg"]))
                result = compute_overflow_work(
                    backend, tokenizer, command["example"], step_cfg,
                    command["gpu_ids"])
                result_queue.put({
                    "event": "overflow_computed", "rank": int(rank),
                    "result": result,
                })
            elif kind == "sync_update":
                broadcast_trainable_parameters(model, source_rank=0)
                model.zero_grad(set_to_none=True)
                result_queue.put({"event": "updated", "rank": int(rank)})
            elif kind == "finish_update":
                gradient_scale = float(command.get("gradient_scale", 1.0))
                apply_update = bool(command.get("apply_update", True))
                if not math.isfinite(gradient_scale) or gradient_scale <= 0.0:
                    raise ValueError(
                        "finish_update gradient_scale must be finite and "
                        "positive")
                if apply_update:
                    if gradient_scale != 1.0:
                        with torch.no_grad():
                            for parameter in model.parameters():
                                if (parameter.requires_grad
                                        and parameter.grad is not None):
                                    parameter.grad.mul_(gradient_scale)
                    reduce_trainable_gradients(model, destination_rank=0)
                    broadcast_trainable_parameters(model, source_rank=0)
                model.zero_grad(set_to_none=True)
                result_queue.put({"event": "updated", "rank": int(rank)})
            elif kind == "stop":
                dist.barrier()
                result_queue.put({"event": "stopped", "rank": int(rank)})
                break
            else:
                raise RuntimeError(f"unknown fast-worker command: {kind!r}")
    except BaseException as error:
        try:
            from runtime.host_memory import HostMemoryTimeout
            result_queue.put({
                "event": "error", "rank": int(rank),
                "traceback": traceback.format_exc(),
                "host_memory_timeout": isinstance(error, HostMemoryTimeout),
            })
        except Exception:
            pass
    finally:
        if dist_initialized:
            try:
                import torch.distributed as dist
                dist.destroy_process_group()
            except Exception:
                pass


def _move_optimizer_state(optimizer, device, *, hardware_profile="RTX6000") -> None:
    """Move Adam state recursively without replacing Parameter identities."""
    import torch
    from runtime.host_memory import retry_host_allocation

    def move(value):
        if torch.is_tensor(value):
            return retry_host_allocation(lambda: value.to(device), hardware_profile=hardware_profile,
                                         stage="optimizer offload")
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)

def _restore_optimizer_state_to_parameters(optimizer) -> None:
    """Return each optimizer tensor to the device of its sharded parameter."""
    if getattr(optimizer, "moments_on_host", False):
        return
    import torch

    def move(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state, parameter.device)
