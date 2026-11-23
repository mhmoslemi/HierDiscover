"""Launch-scoped hardware capacity, checkpoint locations and residency."""

from itertools import zip_longest
import os
from pathlib import Path


B200_MODEL_DIR = Path("/root/nas/Mohammad/models")


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
    if name != "RTX6000":
        raise ValueError("hardware_profile must be RTX6000 or B200_full_ram")
    return name


def cpu_threads_by_core(cpu_ids):
    cores = {}
    for cpu in cpu_ids:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (int((topology / "physical_package_id").read_text()),
                   (topology / "thread_siblings_list").read_text().strip())
        except (OSError, ValueError):
            return {}
        cores.setdefault(key, []).append(cpu)
    return cores


def generation_cpu_ids(profile, gpu_count):
    """Keep evaluation off inference cores, including their SMT siblings."""
    if profile != "B200_full_ram" or not hasattr(os, "sched_getaffinity"):
        return []
    cores = cpu_threads_by_core(sorted(os.sched_getaffinity(0)))
    reserve = min(2 + 2 * int(gpu_count), len(cores) // 2)
    if len(cores) < 4 or int(gpu_count) < 1:
        return []
    sockets = {}
    for (socket, _siblings), threads in cores.items():
        sockets.setdefault(socket, []).append(threads)
    # Spread inference across sockets rather than trapping all engines on one.
    ordered = [threads for row in zip_longest(*sockets.values())
               for threads in row if threads is not None]
    return sorted(cpu for threads in ordered[:reserve] for cpu in threads)


def restore_hardware_settings(merged, current_yaml, saved):
    """Do not reuse an old machine's resolved auto capacities on resume."""
    selected = hardware_profile(current_yaml.get("hardware_profile", "RTX6000"))
    previous = hardware_profile(saved.get("hardware_profile", "RTX6000"))
    merged["hardware_profile"] = selected
    if selected == "B200_full_ram" and previous != selected:
        # Use the current launch's coder residency setting when changing hardware.
        merged["reload_coder_between_phases"] = current_yaml.get("reload_coder_between_phases", False)
    if saved and previous != selected:
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
