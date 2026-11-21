"""Durable evaluated batches and the GPU handoff for LLM credit assignment."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from experiment_io import _atomic_write_text


CRITIC_MODEL = "openai/gpt-oss-120b"
CRITIC_CONTEXT = 131_072


def rollout_id(step, group, rollout):
    return f"s{step + 1}g{group}r{rollout}"


def _raw_score(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def rollout_row(meta):
    return {"id": meta["node_id"], "parent_id": meta["parent_id"],
            "score": _raw_score(meta["raw_score"]) if meta["valid"] else None}


def step_context(problem, seeds, adapter_path):
    # Denoising's target is a Poisson constraint, while its raw score is MSE.
    sota = (None if problem.name == "denoising" else
            problem.cfg.get("target_print", problem.target))
    return {
        "higher_is_better": bool(problem.maximize), "sota": _raw_score(sota),
        "adapter": Path(adapter_path).name,
        "seed": int(problem.cfg["seed"]) if problem.cfg.get("deterministic") else None,
        "seeds": [{"id": state.id, "step": 0, "parent_id": None,
                   "score": _raw_score(state.raw_score)} for state in seeds],
    }


def batch_path(root, step):
    return Path(root) / f"step{step:02d}" / "critic.json"


def save_batch(path, batch):
    _atomic_write_text(path, json.dumps(batch, allow_nan=False, separators=(",", ":")))


def load_batch(root, step):
    path = batch_path(root, step)
    if not path.is_file():
        return None
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch["step"] != step:
        raise ValueError(f"critic batch step differs from {path}")
    return batch


def prepare_batch(root, step, context, rows):
    request = {**context, "step": step, "rollouts": rows}
    saved = load_batch(root, step)
    if saved is None:
        save_batch(batch_path(root, step), request)
    elif any(saved.get(key) != value for key, value in request.items()):
        raise ValueError(f"evaluated rollouts changed for the saved critic batch at step {step}")


def load_tree(path, batch):
    root = Path(path).parent.parent
    step = batch["step"]
    tree = list(batch["seeds"])
    for previous_step in range(step):
        previous = load_batch(root, previous_step)
        if previous is None or "advantages" not in previous:
            raise ValueError(f"missing accepted critic history for step {previous_step}")
        tree.extend({**row, "step": previous_step + 1} for row in previous["rollouts"])
    tree.extend({**row, "step": step + 1} for row in batch["rollouts"])
    return tree


def score_batch(root, step, cfg, host_admission=None):
    """Start one transient critic after its GPUs have been released."""
    from generation.gen_workers import (
        VLLM_WRAPPER_SHUTDOWN_TIMEOUT_S, _cap_vllm_utilization_for_gpus,
        _terminate_vllm_process_group, _vllm_engine_kwargs,
    )
    from runtime.gpu_runtime import (
        allocate_gpu_roles, derive_vllm_parallel_layout, known_attention_heads,
        parse_gpu_ids, query_gpu_memory,
    )
    from runtime.hardware_profiles import local_checkpoint

    batch = load_batch(root, step)
    if "advantages" in batch:
        return batch

    model = (cfg.coder_model_name if Path(cfg.coder_model_name).name.lower() == "gpt-oss-120b"
             else local_checkpoint(CRITIC_MODEL, cfg.hardware_profile))
    available = parse_gpu_ids(cfg.gpu_ids)
    roles = allocate_gpu_roles(available, "critic")
    memory = query_gpu_memory()
    layout = derive_vllm_parallel_layout(
        {"max_seq_length": CRITIC_CONTEXT, "vllm_gpu_memory_utilization": 0.96,
         "vllm_runtime_reserve_gib": 5.0, "hardware_profile": cfg.hardware_profile},
        roles, memory, known_attention_heads(model), model_name=model)
    tp, pp = layout.tensor_parallel_size, layout.pipeline_parallel_size
    gpu_ids = available[:tp * pp]
    utilization = _cap_vllm_utilization_for_gpus(
        0.96, gpu_ids, reserve_gib=5.0, reserve_from_free=True)
    options = _vllm_engine_kwargs(
        model_name=model, max_seq_length=CRITIC_CONTEXT, load_in_4bit=False, lora_rank=1,
        gpu_memory_utilization=utilization, enforce_eager=True, max_num_seqs=1,
        tensor_parallel_size=tp, pipeline_parallel_size=pp, max_num_batched_tokens=4096,
        engine_options={"enable_lora": False, "moe_backend": "marlin", "generation_config": "vllm"},
        hardware_profile=cfg.hardware_profile)
    script = Path(__file__).resolve().parents[1] / "llm-critic.py"
    path = batch_path(root, step).resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_ids)),
               TTT_CRITIC_PARENT_PID=str(os.getpid()))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    reservation = (nullcontext() if host_admission is None else host_admission.reservation(
        int(4 * len(gpu_ids) * 2**30 * 1.05), stage="critic worker initialization", priority=True))
    print(f"[step {step}] launching one GPT-OSS critic on GPUs {gpu_ids} (TP={tp}, PP={pp})", flush=True)
    with reservation:
        worker = subprocess.Popen(
            [sys.executable, str(script), "--score-step", str(path), json.dumps(options)],
            env=env, start_new_session=True)
        try:
            while worker.poll() is None:
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if "advantages" not in load_batch(root, step):
                        continue
                    try:
                        worker.wait(timeout=VLLM_WRAPPER_SHUTDOWN_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        print("[llm-critic] reaping the worker after its shutdown deadline", flush=True)
                    break
        finally:
            _terminate_vllm_process_group(worker.pid)
            if worker.poll() is None:
                worker.wait(timeout=5)
    batch = load_batch(root, step)
    if "advantages" not in batch:
        raise RuntimeError(f"critic worker exited with status {worker.returncode} without accepting step {step}")
    return batch
