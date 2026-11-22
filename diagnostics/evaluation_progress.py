"""Periodic CPU evaluation status, including when no sandbox finishes."""

import math
import sys
import threading
import time
from contextlib import nullcontext

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def write_progress_line(message):
    output = sys.stderr
    bar_output = getattr(output, "progress_stream", output)
    with (tqdm.external_write_mode(file=bar_output) if tqdm is not None else nullcontext()):
        print(message, file=output, flush=True)


class CpuEvaluationProgress:
    def __init__(self, step, total, *, interval=30):
        self._step = step
        self._total = total
        self._interval = interval
        self._condition = threading.Condition()
        self._submitted = 0
        self._queued = 0
        self._deadlines = {}
        self._done = 0
        self._closed = False
        self._reporter = None

    def submit(self, executor, evaluate, *args, **kwargs):
        job = object()
        launched = False

        def report_deadline(deadline):
            nonlocal launched
            with self._condition:
                if deadline is None:
                    self._deadlines.pop(job, None)
                else:
                    launched = True
                    self._queued -= 1
                    self._deadlines[job] = deadline
                    if self._queued == 0:
                        self._condition.notify()

        def finished(_future):
            with self._condition:
                if not launched:
                    self._queued -= 1
                self._deadlines.pop(job, None)
                self._done += 1
                if self._queued == 0:
                    self._condition.notify()

        with self._condition:
            future = executor.submit(evaluate, *args, on_sandbox_deadline=report_deadline, **kwargs)
            self._queued += 1
            self._record_submission()
        future.add_done_callback(finished)
        return future

    def complete_without_worker(self):
        with self._condition:
            self._done += 1
            self._record_submission()

    def _record_submission(self):
        self._submitted += 1
        if self._reporter is None:
            self._reporter = threading.Thread(target=self._report, name="cpu-eval-progress", daemon=True)
            self._reporter.start()
        elif self._submitted == self._total:
            self._condition.notify()

    def _report(self):
        submitted = queued = 0
        while True:
            with self._condition:
                if submitted:
                    self._condition.wait_for(
                        lambda: self._closed or self._done == self._total
                        or submitted < self._total == self._submitted
                        or (queued > 0 and self._queued == 0 and self._submitted == self._total),
                        timeout=self._interval)
                if self._closed and self._done < self._total:
                    return
                submitted, queued, done = self._submitted, self._queued, self._done
                running = len(self._deadlines)
                deadline = max(self._deadlines.values(), default=None)
            not_submitted = self._total - submitted
            activity = f"{running} running, {queued} queued"
            finishing = submitted - queued - running - done
            if finishing:
                activity += f", {finishing} finishing"
            waiting = queued > 0 or not_submitted > 0
            if deadline is None:
                cutoff = "sandbox cutoff pending" if waiting else "sandboxes finished"
            elif not math.isfinite(deadline):
                cutoff = "sandbox cutoff unavailable"
            else:
                seconds_left = max(0, math.ceil(deadline - time.monotonic()))
                minutes, seconds = divmod(seconds_left, 60)
                scope = "active sandbox cutoff" if waiting else "sandbox cutoff"
                cutoff = f"{scope} in ≤{minutes}m{seconds:02d}s"
            write_progress_line(
                f"[step {self._step}] CPU eval {done}/{self._total} done, {self._total - done} left | "
                f"submitted {submitted}/{self._total} ({not_submitted} not submitted) | "
                f"{activity} | {cutoff}")
            if done == self._total:
                return

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._reporter is not None:
            self._reporter.join()
