"""Cross-process exclusive leases for physical GPU evaluation devices."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import time
from xml.etree import ElementTree


# The idle B200s report 5 MiB through nvidia-smi even with no GPU processes.
_IDLE_VRAM_MIB = 5


class GPUNotIdle(RuntimeError):
    """Benchmark admission failed before timing any program."""


def _gpu_busy_reason(gpu_id: int) -> str:
    try:
        output = subprocess.run(
            ["nvidia-smi", "-q", "-x", "-i", str(gpu_id)],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        gpu, = ElementTree.fromstring(output).findall("gpu")
        gpu_util = int(gpu.findtext("utilization/gpu_util", "").split()[0])
        memory_util = int(gpu.findtext("utilization/memory_util", "").split()[0])
        used_mib = int(gpu.findtext("fb_memory_usage/used", "").split()[0])
        processes = gpu.find("processes")
        if processes is None or (processes.text or "").strip():
            raise ValueError("process list is unavailable")
        pids = sorted({int(process.findtext("pid", "")) for process in processes})
    except (OSError, subprocess.SubprocessError, ElementTree.ParseError, ValueError, IndexError) as exc:
        raise GPUNotIdle(f"Could not verify physical GPU {gpu_id} before benchmarking: {exc}") from exc

    if not (gpu_util or memory_util or pids or used_mib > _IDLE_VRAM_MIB):
        return ""
    return (f"GPU utilization {gpu_util}%, memory activity {memory_util}%, "
            f"used VRAM {used_mib} MiB, GPU PIDs {pids}")


def wait_for_idle_gpu(gpu_id: int, *, on_busy=None) -> None:
    """Require zero activity, no GPU processes, and only baseline VRAM usage."""
    deadline = time.monotonic() + 30.0
    reason = _gpu_busy_reason(gpu_id)
    if reason:
        print(f"[gpu-eval] waiting for physical GPU {gpu_id}: {reason}", flush=True)
        if on_busy is not None:
            on_busy()
    while reason and time.monotonic() < deadline:
        time.sleep(0.5)
        reason = _gpu_busy_reason(gpu_id)
    if reason:
        raise GPUNotIdle(f"Physical GPU {gpu_id} stayed busy before benchmarking: {reason}")


@contextmanager
def gpu_lease(gpu_id: int, timeout_s: float = 0.0,
              purpose: str = "evaluation"):
    """Hold an advisory lease for one physical GPU.

    A separate lock file is used for every physical id, so independent runs and
    reward-worker processes serialize benchmark evaluations on the same card.
    timeout_s <= 0 waits indefinitely.
    """
    gpu_id = int(gpu_id)
    lease_root = Path(os.environ.get(
        "TTT_GPU_LEASE_DIR",
        str(Path(tempfile.gettempdir()) / "ttt-gpu-leases"),
    ))
    lease_root.mkdir(parents=True, exist_ok=True)
    path = lease_root / f"gpu-{gpu_id}.lock"
    handle = path.open("a+")
    deadline = time.monotonic() + float(timeout_s) if timeout_s > 0 else None
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for physical GPU {gpu_id} "
                        f"{purpose} lease")
                time.sleep(0.1)

        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} purpose={purpose}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
