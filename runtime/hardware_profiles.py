"""Launch-scoped hardware capacity, checkpoint locations and residency."""

from itertools import zip_longest
import os
from pathlib import Path


B200_MODEL_DIR = Path("/root/nas/Mohammad/models")
HOPPER_GPU_MEMORY_GIB = {"H100": 80.0, "H200": 140.0}


def uses_throughput_profile(profile):
    return profile == "B200_full_ram" or profile in HOPPER_GPU_MEMORY_GIB


def local_checkpoint(model_name, profile):
    if profile == "B200_full_ram" and str(model_name).lower() == "openai/gpt-oss-120b":
        path = B200_MODEL_DIR / "gpt-oss-120b"
        if path.is_dir():
            print(f"[hardware] local checkpoint: {path}")
            return str(path)
    return model_name


def hardware_profile(value):
    name = str(value).strip().upper()
    if name == "B200_FULL_RAM":
        return "B200_full_ram"
    if name != "RTX6000" and name not in HOPPER_GPU_MEMORY_GIB:
        raise ValueError("hardware_profile must be RTX6000, B200_full_ram, H100, or H200")
    return name


def generation_cpu_ids(profile, gpu_count):
    """Keep evaluation off inference cores, including their SMT siblings."""
    if not uses_throughput_profile(profile) or not hasattr(os, "sched_getaffinity"):
        return []
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
        except (OSError, ValueError):
            # Without physical topology, do not accidentally reserve half a core.
            return []
        cores.setdefault(key, []).append(cpu)
    reserve = min(2 + 2 * int(gpu_count), len(cores) // 2)
    if len(cores) < 4 or int(gpu_count) < 1:
        return []
    sockets = {}
    for (socket, _core), threads in cores.items():
        sockets.setdefault(socket, []).append(threads)
    # Spread inference across sockets rather than trapping all engines on one.
    ordered = [threads for row in zip_longest(*sockets.values())
               for threads in row if threads is not None]
    return sorted(cpu for threads in ordered[:reserve] for cpu in threads)


def restore_hardware_settings(merged, current_yaml, saved, *, profile_override=None):
    """Do not reuse an old machine's resolved auto capacities on resume."""
    selected = hardware_profile(
        current_yaml.get("hardware_profile", "RTX6000")
        if profile_override is None else profile_override)
    previous = hardware_profile(saved.get("hardware_profile", "RTX6000"))
    merged["hardware_profile"] = selected
    if selected == "B200_full_ram" and previous != selected:
        # Use the current launch's coder residency setting when changing hardware.
        merged["reload_coder_between_phases"] = current_yaml.get("reload_coder_between_phases", False)
    if (saved and previous != selected) or selected in HOPPER_GPU_MEMORY_GIB:
        for key, default in {
            "gen_micro_batch": "auto", "vllm_gpu_memory_utilization": "auto",
            "vllm_max_num_batched_tokens": "auto", "logprob_chunk": "auto",
        }.items():
            merged[key] = current_yaml.get(key, default)
    if selected == "B200_full_ram":
        # Resolved checkpoint values must not freeze vLLM's automatic B200
        # scheduler when the current launch still requests auto.
        merged["gen_micro_batch"] = current_yaml.get("gen_micro_batch", "auto")
        merged["vllm_max_num_batched_tokens"] = current_yaml.get(
            "vllm_max_num_batched_tokens", "auto")
