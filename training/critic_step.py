"""Durable evaluated batches and the GPU handoff for LLM credit assignment."""

import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

from diagnostics.console import setting_log_only
from experiment_io import _atomic_write_text


CRITIC_MODEL = "openai/gpt-oss-120b"
CRITIC_CONTEXT = 131_072
CRITIC_WORKERS = 2
CRITIC_RESPONSES = 8


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
    return Path(root) / "critic" / f"step{step:02d}" / "critic.json"


def save_batch(path, batch):
    _atomic_write_text(path, json.dumps(batch, allow_nan=False, separators=(",", ":")))


def load_batch(root, step):
    path = batch_path(root, step)
    if not path.is_file():
        path = Path(root) / f"step{step:02d}" / "critic.json"
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


def load_tree(root, batch):
    step = batch["step"]
    tree = list(batch["seeds"])
    for previous_step in range(step):
        previous = load_batch(root, previous_step)
        if previous is None or "advantages" not in previous:
            raise ValueError(f"missing accepted critic history for step {previous_step}")
        tree.extend(
            {**row, "step": previous_step + 1, "assigned_advantage": previous["advantages"][row["id"]]}
            for row in previous["rollouts"]
        )
    tree.extend({**row, "step": step + 1} for row in batch["rollouts"])
    return tree


def save_rollout_report(directory, batch, tree):
    scores = {row["id"]: row["score"] for row in tree}
    parents = {}
    for row in batch["rollouts"]:
        parent_id = row["parent_id"]
        parent = parents.setdefault(parent_id, {"raw_score": scores.get(parent_id), "children": {}})
        parent["children"][row["id"]] = {
            "raw_score": row["score"],
            "advantage": batch["advantages"][row["id"]],
        }
    selected = None
    if "selected_batch" in batch and "selected_rollout" in batch:
        selected = f"batch{batch['selected_batch']:03d}/rollout{batch['selected_rollout']:03d}.txt"
    path = Path(directory) / "rollouts.json"
    report = {"step": batch["step"], "selected_reply": selected, "parents": parents}
    _atomic_write_text(path, json.dumps(report, indent=2, allow_nan=False) + "\n")
    return path


def score_batch(root, step, cfg, host_admission=None):
    """Score an evaluated batch on two ordinary rollout workers."""
    from generation.gen_workers import (
        _terminate_vllm_process_group, gpt_oss_coder_engine_options, rollout_engine_options,
    )
    from runtime.gpu_runtime import _estimated_weight_gib, parse_gpu_ids
    from runtime.hardware_profiles import local_checkpoint
    from runtime.host_memory import GIB, HostMemoryAdmission

    path = batch_path(root, step).resolve()
    batch = load_batch(root, step)
    if batch is None:
        raise FileNotFoundError(path)
    if "advantages" in batch:
        return batch
    if not path.is_file():
        save_batch(path, batch)

    model = (cfg.coder_model_name if Path(cfg.coder_model_name).name.lower() == "gpt-oss-120b"
             else local_checkpoint(CRITIC_MODEL, cfg.hardware_profile))
    gpu_ids = parse_gpu_ids(cfg.gpu_ids)[:CRITIC_WORKERS]
    if len(gpu_ids) != CRITIC_WORKERS:
        raise ValueError(f"llm-critic requires {CRITIC_WORKERS} rollout GPUs, one per worker")
    log_path = path.parent.parent / "critic.log"
    seed = batch.get("seed")
    if seed is not None:
        seed = (seed * 1_000_003 + step * 1009) % (2**31 - 1)
    per_worker = CRITIC_RESPONSES // CRITIC_WORKERS
    b200 = cfg.hardware_profile == "B200_full_ram"
    options = rollout_engine_options(cfg)
    options.update(
        model_name=model, backend="vllm", max_seq_length=CRITIC_CONTEXT,
        seed=seed, num_workers=CRITIC_WORKERS, gpu_ids=gpu_ids, gen_micro_batch=per_worker,
        vllm_tensor_parallel_size=1, vllm_pipeline_parallel_size=1, vllm_staged_loading=False,
        vllm_base_policy_only=True, vllm_log_path=str(log_path),
        vllm_engine_options=gpt_oss_coder_engine_options(per_worker, use_cuda_graphs=b200),
    )
    if model != cfg.coder_model_name:
        options["vllm_quantization"] = None
    script = Path(__file__).resolve().parents[1] / "llm-critic.py"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_ids)),
               TTT_CRITIC_PARENT_PID=str(os.getpid()))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if host_admission is None:
        host_admission = HostMemoryAdmission(multiprocessing.get_context("spawn"),
                                             hardware_profile=cfg.hardware_profile)
    staging_bytes = int(4 * CRITIC_WORKERS * GIB * 1.05)
    if b200:
        staging_bytes += int(CRITIC_WORKERS * _estimated_weight_gib(model, "auto") * GIB)
    reservation = host_admission.reservation(
        staging_bytes, stage="parallel critic loading", priority=True, wait_for_external=False)
    setting_log_only(f"[step {step}] critic: {CRITIC_WORKERS} workers on GPUs {gpu_ids}, "
                     f"{CRITIC_RESPONSES} replies per batch. Log: {log_path}", flush=True)
    with reservation, log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n=== critic step {step}, GPUs {gpu_ids}, {model} ===\n")
        worker = subprocess.Popen(
            [sys.executable, str(script), "--score-step", str(path), json.dumps(options)],
            env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = worker.wait()
        finally:
            _terminate_vllm_process_group(worker.pid)
            if worker.poll() is None:
                worker.wait(timeout=5)
    batch = load_batch(root, step)
    if returncode or batch is None or "advantages" not in batch:
        raise RuntimeError(f"critic step {step} failed (exit {returncode}); see {log_path}")
    print(f"[step {step}] critic accepted batch {batch['selected_batch']}, "
          f"rollout {batch['selected_rollout']:03d}, after {batch['attempts']} replies", flush=True)
    return batch
