"""
Run model-generated code in a subprocess with a hard timeout.

Write the code to a temp file, then spawn a Python subprocess that:
  1. imports the file
  2. calls the named entrypoint function
  3. pickles the return value to a results file

The parent process reads the pickle when the child exits cleanly, or
kills the child (and its process group) on timeout.

"""

import codecs
import io
import os
import pickle
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time


# Generated programs are untrusted with respect to output volume.  A single
# ``print`` inside an optimization loop can otherwise write gigabytes into the
# temporary capture file; Linux then fills host RAM with page cache and the
# parent used to read the complete file back into a Python string.  The reward
# metadata persists only a short diagnostic prefix, so retain a bounded prefix
# at the source and discard everything after it.
_CAPTURE_LIMIT_BYTES = 256 * 1024


# Creating a Popen briefly allocates several parent-side descriptors. Serialize
# only that millisecond-scale setup; the child processes themselves still run
# fully concurrently on their assigned CPUs.
_SANDBOX_LAUNCH_LOCK = threading.Lock()


# Placeholders __PROGRAM_PATH__ / __FUNCTION_NAME__ / __RESULTS_PATH__
# are substituted before launch.
RUNNER_TEMPLATE = r'''
import os
import sys
import pickle
import traceback
import importlib.util

STDOUT_PATH = "__STDOUT_PATH__"
STDERR_PATH = "__STDERR_PATH__"
CAPTURE_LIMIT_BYTES = int(os.environ.pop(
    "TTT_SANDBOX_CAPTURE_LIMIT_BYTES", "262144"))


class _BoundedCapture:
    """Text/binary-compatible prefix capture with a hard byte ceiling."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, path, fd):
        self._stream = open(path, "wb", buffering=0)
        self._fd = int(fd)
        self._remaining = max(0, CAPTURE_LIMIT_BYTES)
        self._truncated = False
        # Support the common ``sys.stdout.buffer.write(...)`` form without
        # exposing the real file descriptor to native libraries. Native fd 1/2
        # remains /dev/null, so C extensions cannot bypass the byte ceiling.
        self.buffer = self

    def write(self, value):
        original_length = len(value) if hasattr(value, "__len__") else 0
        if self._remaining <= 0:
            return original_length
        if isinstance(value, bytes):
            truncated = len(value) > self._remaining
            data = value[:self._remaining]
        else:
            rendered = str(value)
            # Bound encoding work too: after capture fills, pathological print
            # loops must not keep allocating a huge temporary bytes object.
            prefix = rendered[:self._remaining]
            data = prefix.encode(self.encoding, errors=self.errors)
            truncated = (
                len(prefix) < len(rendered) or len(data) > self._remaining)
        if not truncated and len(data) <= self._remaining:
            self._stream.write(data)
            self._remaining -= len(data)
            return original_length
        marker = b"\n[... sandbox output truncated ...]\n"
        keep = max(0, self._remaining - len(marker))
        if keep:
            self._stream.write(data[:keep])
        marker_room = self._remaining - keep
        if marker_room:
            self._stream.write(marker[:marker_room])
        self._remaining = 0
        self._truncated = True
        return original_length

    def flush(self):
        self._stream.flush()

    def close(self):
        self._stream.close()

    def fileno(self):
        # fd 1/2 is deliberately /dev/null in the Popen configuration below.
        return self._fd

    def isatty(self):
        return False


if not os.environ.pop("TTT_SANDBOX_STDOUT_FEEDBACK", ""):
    sys.stdout = _BoundedCapture(STDOUT_PATH, 1)
sys.stderr = _BoundedCapture(STDERR_PATH, 2)

# Apply the limit before importing NumPy/SciPy or the generated program.  It is
# inherited by descendants.  RLIMIT_AS is a per-process last line of defence;
# normal Erdős candidates are tiny, while a runaway dense allocation fails
# that rollout instead of forcing the whole host into swap thrashing.
MEMORY_LIMIT_BYTES = int(os.environ.pop(
    "TTT_SANDBOX_MEMORY_LIMIT_BYTES", "0") or 0)
if MEMORY_LIMIT_BYTES > 0 and sys.platform.startswith("linux"):
    try:
        import resource
        _current_soft, _current_hard = resource.getrlimit(
            resource.RLIMIT_AS)
        _effective_limit = MEMORY_LIMIT_BYTES
        if _current_hard != resource.RLIM_INFINITY:
            _effective_limit = min(_effective_limit, int(_current_hard))
        resource.setrlimit(
            resource.RLIMIT_AS,
            (_effective_limit, _effective_limit),
        )
    except Exception as limit_error:
        raise RuntimeError(
            "failed to apply sandbox memory limit: " + str(limit_error))

# Pin before importing the generated program. The affinity is inherited by
# every subprocess it creates, so one candidate and its whole process tree stay
# on the CPU assigned by the evaluation scheduler.
EVAL_CPU_ID = os.environ.pop("TTT_EVAL_CPU_ID", "")
if EVAL_CPU_ID:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("isolated evaluation requires Linux CPU affinity")
    os.sched_setaffinity(0, {int(EVAL_CPU_ID)})

# Force spawn for any multiprocessing the child code might do
try:
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
except Exception:
    pass

PROGRAM_PATH = "__PROGRAM_PATH__"
FUNCTION_NAME = "__FUNCTION_NAME__"
RESULTS_PATH = "__RESULTS_PATH__"

sys.path.insert(0, os.path.dirname(PROGRAM_PATH))

try:
    spec = importlib.util.spec_from_file_location("program", PROGRAM_PATH)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    fn = getattr(program, FUNCTION_NAME)
    result = fn()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"ok": True, "value": result}, f)
except Exception as e:
    tb = traceback.format_exc()
    try:
        with open(RESULTS_PATH, "wb") as f:
            pickle.dump({"ok": False, "error": str(e), "traceback": tb}, f)
    except Exception:
        pass
    sys.stderr.write(tb)
finally:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
'''


def _kill_tree(proc, pgid, hard=False):
    """Best-effort kill of the entire process tree."""
    sig = signal.SIGKILL if hard else signal.SIGTERM
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            # The session no longer exists, so it has no surviving children.
            return
        except Exception:
            pass
    # start_new_session normally makes killpg sufficient. Use pkill only on
    # platforms/races where no usable process group was available; spawning a
    # pkill process for every successful evaluation wastes hundreds of PIDs.
    if proc.poll() is None and shutil.which("pkill"):
        try:
            subprocess.run(
                ["pkill", "-KILL" if hard else "-TERM", "-P", str(proc.pid)],
                check=False,
            )
        except Exception:
            pass


class _StdoutFeedback:
    """Drain the full stdout stream while keeping its feedback and log excerpts."""

    def __init__(self, pipe):
        self._pipe = pipe
        self._decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")("ignore"), translate=True)
        self._selector = selectors.DefaultSelector()
        self._selector.register(pipe, selectors.EVENT_READ)
        self._head = self._tail = self._whitespace = ""

    def _append(self, text):
        if not self._tail:
            text = text.lstrip()
        self._head += text[:995 - len(self._head)]
        body = text.rstrip()
        if body:
            self._tail = (self._tail + self._whitespace + body)[-2001:]
            self._whitespace = text[len(body):][-2001:]
        else:
            self._whitespace = (self._whitespace + text)[-2001:]

    def _read(self, timeout):
        if not self._selector.get_map() or not self._selector.select(timeout):
            return False
        data = os.read(self._pipe.fileno(), 64 * 1024)
        if not data:
            self._selector.unregister(self._pipe)
        self._append(self._decoder.decode(data))
        return bool(data)

    def wait(self, proc, timeout, *, deadline=None):
        if deadline is None and timeout is not None:
            deadline = time.monotonic() + timeout
        while self._selector.get_map() and proc.poll() is None:
            remaining = 0.05 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            self._read(min(0.05, remaining))
        proc.wait(timeout=None if deadline is None else max(0, deadline - time.monotonic()))

    def finish(self):
        # A descendant that outlives cleanup must not keep the reader open.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and self._read(0):
            pass
        self._append(self._decoder.decode(b"", final=True))
        if len(self._tail) > 2000:
            return self._head + "\n...\n" + self._tail[-1000:]
        return self._tail

    def close(self):
        self._selector.close()
        self._pipe.close()


def run_code(code: str, entrypoint: str, timeout_s: float | None, max_cpus: int | None = 1,
             *, cpu_id=None, memory_limit_bytes=None, on_sandbox_deadline=None,
             stdout_feedback=False, hard_timeout=False):
    """
    Execute `code` (Python source) in a subprocess. Calls `entrypoint()`
    and returns whatever it returns.

    A None timeout allows a trusted check to finish without a deadline. The
    optional callback reports a configured deadline on launch and None on exit.
    A None CPU limit preserves the inherited thread environment when unpinned.
    A hard timeout sends SIGKILL at the deadline.

    Returns a dict:
      {"ok": True,  "value": <return value>, "stdout": "..."}
      {"ok": False, "error": "...", "stdout": "..."}
    """
    pinned_cpu = None
    if cpu_id is not None:
        if not (hasattr(os, "sched_getaffinity")
                and hasattr(os, "sched_setaffinity")):
            raise RuntimeError(
                "isolated evaluation requires Linux CPU affinity support")
        pinned_cpu = int(cpu_id)
        allowed_cpus = set(os.sched_getaffinity(0))
        if pinned_cpu not in allowed_cpus:
            raise ValueError(
                f"CPU {pinned_cpu} is outside this process's allowed CPU set")

    paths = []
    proc = None
    pgid = None
    feedback = None
    try:
        # Write code to a temp file.
        with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w") as f:
            program_path = f.name
            paths.append(program_path)
            f.write(code)

        results_path = program_path + ".pkl"
        stdout_path = program_path + ".stdout"
        stderr_path = program_path + ".stderr"
        paths.extend((results_path, stdout_path, stderr_path))

        # Write the runner script.
        runner_src = (
            RUNNER_TEMPLATE
            .replace("__PROGRAM_PATH__", program_path)
            .replace("__FUNCTION_NAME__", entrypoint)
            .replace("__RESULTS_PATH__", results_path)
            .replace("__STDOUT_PATH__", stdout_path)
            .replace("__STDERR_PATH__", stderr_path)
        )
        with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w") as f:
            runner_path = f.name
            paths.append(runner_path)
            f.write(runner_src)

        # Limit BLAS threads in the child so generated code cannot fork one
        # thread per host CPU.
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
        env.pop("TTT_EVAL_CPU_ID", None)
        if pinned_cpu is not None or max_cpus is not None:
            t = "1" if pinned_cpu is not None else str(max(1, int(max_cpus)))
            for key in [
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "BLIS_NUM_THREADS"]:
                env[key] = t
        if pinned_cpu is not None:
            env["TTT_EVAL_CPU_ID"] = str(pinned_cpu)
        env["TTT_SANDBOX_CAPTURE_LIMIT_BYTES"] = str(
            _CAPTURE_LIMIT_BYTES)
        if stdout_feedback:
            env["TTT_SANDBOX_STDOUT_FEEDBACK"] = "1"
        else:
            env.pop("TTT_SANDBOX_STDOUT_FEEDBACK", None)
        if memory_limit_bytes is not None:
            resolved_memory_limit = int(memory_limit_bytes)
            if resolved_memory_limit <= 0:
                raise ValueError("memory_limit_bytes must be positive")
            env["TTT_SANDBOX_MEMORY_LIMIT_BYTES"] = str(
                resolved_memory_limit)
        else:
            env.pop("TTT_SANDBOX_MEMORY_LIMIT_BYTES", None)

        # AC1 drains stdout directly, including writes to fd 1 from native code.
        # The default sandbox uses the bounded capture files.
        deadline = None
        with _SANDBOX_LAUNCH_LOCK:
            proc = subprocess.Popen(
                [sys.executable, runner_path],
                stdout=subprocess.PIPE if stdout_feedback else subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            if hard_timeout and timeout_s is not None:
                deadline = time.monotonic() + timeout_s

        if stdout_feedback or hard_timeout:
            pgid = proc.pid  # start_new_session creates this group even for fast exits.
        else:
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = None

        if stdout_feedback:
            feedback = _StdoutFeedback(proc.stdout)
        if on_sandbox_deadline is not None and timeout_s is not None:
            if deadline is None:
                deadline = time.monotonic() + timeout_s
            on_sandbox_deadline(deadline)
        timed_out = False
        try:
            wait_timeout = timeout_s if deadline is None else max(0, deadline - time.monotonic())
            if feedback is None:
                proc.wait(timeout=wait_timeout)
            else:
                feedback.wait(proc, wait_timeout, deadline=deadline if hard_timeout else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc, pgid, hard=hard_timeout)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _kill_tree(proc, pgid, hard=True)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if on_sandbox_deadline is not None:
                on_sandbox_deadline(None)

        # Belt-and-suspenders: make sure no descendant outlives evaluation.
        _kill_tree(proc, pgid, hard=True)

        def _read_capture(path):
            try:
                with open(path, "rb") as capture:
                    # Defensive bound as well as the child-side bound: never
                    # materialize an unexpectedly large file in the trainer.
                    return capture.read(_CAPTURE_LIMIT_BYTES + 1)[
                        :_CAPTURE_LIMIT_BYTES]
            except OSError:
                return b""

        stdout_text = (_read_capture(stdout_path).decode(errors="ignore") if feedback is None
                       else feedback.finish())
        stderr_bytes = _read_capture(stderr_path)
        stderr_text = (
            stderr_bytes.decode(errors="ignore") if stderr_bytes else "")

        if timed_out:
            return {
                "ok": False,
                "error": f"Timeout after {timeout_s}s",
                "stdout": stdout_text,
            }
        if proc.returncode != 0:
            # The runner normally pickles its exception even though it exits
            # successfully; this branch handles import/interpreter crashes.
            if os.path.exists(results_path):
                try:
                    with open(results_path, "rb") as f:
                        payload = pickle.load(f)
                    payload["stdout"] = stdout_text
                    return payload
                except Exception as error:
                    return {
                        "ok": False,
                        "error": f"Failed to read results: {error}",
                        "stdout": stdout_text,
                    }
            return {
                "ok": False,
                "error": f"Process exited with code {proc.returncode}",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if not os.path.exists(results_path):
            return {
                "ok": False,
                "error": "No results file written",
                "stdout": stdout_text,
            }
        try:
            with open(results_path, "rb") as f:
                payload = pickle.load(f)
            payload["stdout"] = stdout_text
            return payload
        except Exception as error:
            return {
                "ok": False,
                "error": f"Failed to read results: {error}",
                "stdout": stdout_text,
            }
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc, pgid, hard=True)
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
        if feedback is not None:
            feedback.close()
        for path in paths:
            try:
                os.unlink(path)
            except (FileNotFoundError, OSError):
                pass


if __name__ == "__main__":
    # Quick self-test
    code = """
import numpy as np
def my_entry():
    return np.array([[0.5, 0.5]]), np.array([0.4]), 0.4
"""
    print(run_code(code, "my_entry", timeout_s=10))

    bad_code = """
def my_entry():
    while True:
        pass
"""
    print(run_code(bad_code, "my_entry", timeout_s=2))

    long_stdout = """
def my_entry():
    print('x' * 270000)
    print('final feedback')
"""
    result = run_code(long_stdout, "my_entry", timeout_s=10, stdout_feedback=True)
    assert result["ok"] and result["stdout"].endswith("final feedback")
