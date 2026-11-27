import json
import multiprocessing
from pathlib import Path
import sys

import pytest

from generation import gen_workers
from problems.base import ParentContext
from problems.gpu_mode import GpuMode
from runtime import configuration, gpu_runtime, host_memory
from runtime.run_supervisor import _resume_command


ROOT = Path(__file__).resolve().parents[1]


def _launch_config(monkeypatch, profile, *, problem="ac1", actual=None, extra=()):
    total = {None: 178.0, "H100": 80.0, "H200": 140.0}[profile]
    name = actual or f"NVIDIA {profile or 'B200'}"
    gpus = (2, 4)
    config = ROOT / "configs" / f"{problem}.yaml"
    argv = ["orchestrator.py", "--config", str(config), "--advantage-mode", "hinge",
            "--fused-long-attention"]
    if profile is not None:
        argv += ["--hardware-profile", profile]
    argv += extra
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("AVAILABLE_GPUS", "2,4")
    monkeypatch.delenv("TTT_SUPERVISOR_CONFIG_SNAPSHOT", raising=False)
    monkeypatch.setattr(gpu_runtime, "query_gpu_memory", lambda: {
        gpu: gpu_runtime.GPUMemory(gpu, name, total, total) for gpu in gpus
    })
    # Linux affinity is orthogonal to the config and engine wiring.
    monkeypatch.setattr(configuration, "_isolated_evaluation_cpu_ids", lambda: [0, 1])
    monkeypatch.setattr(configuration, "_isolated_ac1_cpu_pairs", lambda cfg: [(0, 1)])
    return configuration.load_config()


@pytest.mark.parametrize(("profile", "expected_engines", "batch_tokens", "attention"), [
    ("H100", [[2, 4]], 4096, {"backend": "FLASH_ATTN", "flash_attn_version": 3}),
    ("H200", [[2], [4]], 0, {"backend": "FLASH_ATTN", "flash_attn_version": 3}),
    (None, [[2], [4]], 0, {"backend": "FLASH_ATTN"}),
])
def test_two_gpu_launch_configures_vllm_and_replica_training(
        monkeypatch, profile, expected_engines, batch_tokens, attention):
    cfg, _ = _launch_config(monkeypatch, profile)
    assert cfg.hardware_profile == (profile or "B200_full_ram")
    assert cfg.advantage_mode == "hinge"
    assert cfg.generation_backend == "vllm"
    assert cfg.backend == "unsloth" and not cfg.load_in_4bit
    assert cfg.training_gpu_ids == "2,4"
    assert cfg.num_training_gpus == 2
    assert cfg.max_seq_length == cfg.max_new_tokens == 131_072
    assert cfg.isolate_eval and cfg.evaluation_gpu_id is None

    # Stop at the actual worker boundary. No CUDA or model weights needed.
    engine_groups = []
    monkeypatch.setattr(gen_workers.GenerationPool, "_start_workers",
                        lambda pool, ranks, initial: engine_groups.extend(
                            pool._gpu_groups[rank] for rank in ranks))
    monkeypatch.setattr(gen_workers, "_cap_vllm_utilization_for_gpus",
                        lambda utilization, *args, **kwargs: utilization)
    monkeypatch.setattr(gen_workers, "_assert_vllm_memory_available", lambda *args: None)
    pool = gen_workers.GenerationPool(
        cfg.coder_model_name, cfg.num_gpus, gpu_ids=[2, 4], max_seq_length=cfg.max_seq_length,
        backend=cfg.generation_backend, gen_micro_batch=cfg.gen_micro_batch,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        vllm_pipeline_parallel_size=cfg.vllm_pipeline_parallel_size,
        **gen_workers.rollout_engine_options(cfg),
    )
    try:
        assert engine_groups == expected_engines
        options = pool._worker_options
        assert options["use_v2_model_runner"]
        kwargs = gen_workers._vllm_engine_kwargs(
            cfg.coder_model_name, pool.max_seq_length, cfg.load_in_4bit, cfg.lora_rank,
            options["gpu_memory_utilization"], max_num_seqs=options["gen_micro_batch"],
            tensor_parallel_size=options["tensor_parallel_size"],
            max_num_batched_tokens=options["max_num_batched_tokens"],
            performance_mode=options["performance_mode"], hardware_profile=options["hardware_profile"],
        )
        assert kwargs["attention_config"] == attention
        assert kwargs["max_model_len"] == 131_072
        assert kwargs["performance_mode"] == "throughput"
        assert kwargs["enable_lora"] and kwargs["enable_prefix_caching"]
        assert kwargs.get("max_num_batched_tokens", 0) == batch_tokens
        assert kwargs.get("fully_sharded_loras", False) == (profile == "H100")
    finally:
        pool.shutdown()


def test_trimul_h200_uses_matching_evaluator_and_prompt(monkeypatch):
    cfg, merged = _launch_config(
        monkeypatch, "H200", problem="gpu_mode_trimul", extra=("--kernel-python", sys.executable))
    problem = GpuMode(merged)
    assert cfg.gpu_type == "H200" and cfg.reward_workers == 2
    assert problem.kernel_gpu_ids == [2, 4]
    assert problem.kernel_python == sys.executable
    assert "run on an H200." in problem.build_prompt(ParentContext())[0]["content"]


def test_hopper_replans_on_resume(monkeypatch, tmp_path):
    _, saved = _launch_config(monkeypatch, "H200")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    cfg, _ = _launch_config(monkeypatch, "H100", extra=("--resume", str(tmp_path)))
    assert cfg.hardware_profile == "H100"
    assert cfg.vllm_tensor_parallel_size == 2
    assert cfg.vllm_max_num_batched_tokens == 4096
    assert cfg.max_seq_length == saved["max_seq_length"]
    assert _resume_command(sys.argv, tmp_path) == sys.argv


def test_hopper_profile_rejects_wrong_gpu(monkeypatch):
    with pytest.raises(ValueError, match="hardware_profile H100 requires H100 GPUs"):
        _launch_config(monkeypatch, "H100", actual="NVIDIA H200")


def test_hopper_can_stage_loads_when_parallel_load_exceeds_job_ram(monkeypatch):
    gib = host_memory.GIB
    monkeypatch.setattr(host_memory, "host_memory_info", lambda: {
        "total": 100 * gib, "available": 40 * gib,
    })
    admission = host_memory.HostMemoryAdmission(
        multiprocessing.get_context("spawn"), hardware_profile="H200")
    with admission.reservation(150 * gib, minimum_bytes=8 * gib,
                               wait_for_external=False) as granted:
        assert 8 * gib <= granted < 150 * gib
    with pytest.raises(host_memory.HostMemoryUnavailable, match="job's .* RAM limit"):
        with admission.reservation(150 * gib):
            pytest.fail("a required transfer that exceeds the job limit was admitted")
