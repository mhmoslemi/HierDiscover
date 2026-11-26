"""
Per-experiment file I/O.

Creates a directory under runs/ named from the benchmark and objective, then
writes one .txt and one .meta.json per rollout. All rollouts are
saved, including ones that failed extraction or validation.

Filenames:
    step03_group2_rollout17.txt        ← raw model response
    step03_group2_rollout17.tokens.zlib ← exact generated token IDs
    step03_group2_rollout17.behavior_logprobs.zlib ← rollout-policy scores
    step03_group2_rollout17.meta.json  ← reward, valid, msg, beta, advantage, etc.

Directory name (TRIOUT is an optional shell-only label):
    runs/trimul-entropic-0602-2201-triout1/

A config.json is also dumped at the root of the run dir.
"""

import json
import math
import os
import re
import struct
import time
import zlib
from dataclasses import asdict
from pathlib import Path

from config_validation import reject_retired_coder_cap


_ROLLOUT_TOKEN_MAGIC = b"TTTtok1\0"
_ROLLOUT_LOGPROB_MAGIC = b"TTTlp1\0\0"
_ROLLOUT_IO_RETRY_SECONDS = 5.0
_ROLLOUT_IO_RETRY_TIMEOUT_SECONDS = 2 * 60 * 60

ROLLOUT_BUDGET_KEYS = (
    "groups_per_step", "group_size",
)
ROLLOUT_CONFIG_KEYS = ROLLOUT_BUDGET_KEYS
_ADVANTAGE_PLOT_FIELDS = (
    "group", "rollout", "valid", "reward", "raw_score", "advantage_mode",
    "advantage", "ttt_advantage", "spo_rs_advantage", "spo_rs_v_t",
)


def _retry_rollout_filesystem(operation, path):
    """Wait out transient NAS write failures instead of killing a step."""
    path = Path(path)
    deadline = time.monotonic() + _ROLLOUT_IO_RETRY_TIMEOUT_SECONDS
    next_notice = 0.0
    while True:
        try:
            return operation()
        except OSError as error:
            now = time.monotonic()
            if now >= deadline:
                raise OSError(
                    error.errno,
                    f"rollout artifact remained unwritable for 2 hours: "
                    f"{path} ({error})",
                    str(path),
                ) from error
            if now >= next_notice:
                remaining = max(0, int((deadline - now) / 60))
                print(
                    f"[rollout-io] write unavailable for {path.name}: "
                    f"{error}; retrying every 5s, {remaining} min remaining",
                    flush=True,
                )
                next_notice = now + 60.0
            time.sleep(min(_ROLLOUT_IO_RETRY_SECONDS, deadline - now))


def _atomic_write_bytes(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(payload)
        tmp.replace(path)

    _retry_rollout_filesystem(write, path)


def _atomic_write_text(path, value, *, errors=None):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        if errors is None:
            tmp.write_text(str(value))
        else:
            tmp.write_text(str(value), errors=errors)
        tmp.replace(path)

    _retry_rollout_filesystem(write, path)


def save_rollout_token_ids(path, token_ids):
    """Atomically persist exact generated IDs in a compact portable format."""
    values = [int(token_id) for token_id in token_ids]
    if any(token_id < 0 or token_id > 0xFFFFFFFF for token_id in values):
        raise ValueError("rollout token IDs must fit unsigned 32-bit storage")
    payload = _ROLLOUT_TOKEN_MAGIC + struct.pack("<Q", len(values))
    if values:
        payload += struct.pack(f"<{len(values)}I", *values)
    compressed = zlib.compress(payload, level=1)
    _atomic_write_bytes(path, compressed)


def load_rollout_token_ids(path):
    """Load and validate a sidecar produced by save_rollout_token_ids."""
    payload = zlib.decompress(Path(path).read_bytes())
    header_size = len(_ROLLOUT_TOKEN_MAGIC) + 8
    if (len(payload) < header_size
            or payload[:len(_ROLLOUT_TOKEN_MAGIC)] != _ROLLOUT_TOKEN_MAGIC):
        raise ValueError(f"invalid rollout token sidecar: {path}")
    count = struct.unpack_from("<Q", payload, len(_ROLLOUT_TOKEN_MAGIC))[0]
    expected_size = header_size + 4 * count
    if len(payload) != expected_size:
        raise ValueError(f"truncated rollout token sidecar: {path}")
    if not count:
        return []
    return list(struct.unpack_from(f"<{count}I", payload, header_size))


def save_rollout_logprobs(path, logprobs):
    """Atomically persist chosen-token rollout logprobs as exact float32."""
    values = [float(value) for value in logprobs]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rollout logprobs must be finite")
    payload = _ROLLOUT_LOGPROB_MAGIC + struct.pack("<Q", len(values))
    if values:
        payload += struct.pack(f"<{len(values)}f", *values)
    compressed = zlib.compress(payload, level=1)
    _atomic_write_bytes(path, compressed)


def load_rollout_logprobs(path):
    """Load and validate a sidecar produced by save_rollout_logprobs."""
    payload = zlib.decompress(Path(path).read_bytes())
    header_size = len(_ROLLOUT_LOGPROB_MAGIC) + 8
    if (len(payload) < header_size
            or payload[:len(_ROLLOUT_LOGPROB_MAGIC)]
            != _ROLLOUT_LOGPROB_MAGIC):
        raise ValueError(f"invalid rollout logprob sidecar: {path}")
    count = struct.unpack_from(
        "<Q", payload, len(_ROLLOUT_LOGPROB_MAGIC))[0]
    expected_size = header_size + 4 * count
    if len(payload) != expected_size:
        raise ValueError(f"truncated rollout logprob sidecar: {path}")
    if not count:
        return []
    values = list(struct.unpack_from(f"<{count}f", payload, header_size))
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"nonfinite rollout logprob sidecar: {path}")
    return values


class StepPlotter:
    """Refresh step plots off the training thread, in order.

    Use the same Python environment, headless CPU-only subprocesses, and a
    bounded timeout per script. A single worker avoids competing plot writes;
    --max-step prevents a queued plot from including a later unfinished step.
    Diagnostics go to plots.log, and failures never invalidate training.
    """

    def __init__(self, run_dir, problem):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        self.run_dir = Path(run_dir).resolve()
        self.problem = str(problem).strip().lower()
        self.stopping = Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="step-plots")

    def submit(self, step):
        scripts = [("analysis.plot_erdos_best_curve", []),
                   ("analysis.plot_parent_tree", [])]
        if self.problem in {"erdos", "erdos_min_overlap", "erdos_minimum_overlap"}:
            scripts.insert(0, ("analysis.plot_erdos", ["--saved-only"]))
        elif self.problem in {"ac1", "ac2"}:
            scripts.insert(0, ("analysis.plot_ac", []))
        self.executor.submit(self._refresh, int(step), scripts)

    def submit_advantages(self, step):
        self.executor.submit(self._refresh, int(step), [
            ("analysis.plot_advantages", ["--snapshots", str(step)]),
        ])

    def close(self, *, wait=True):
        # On interruption, cancel queued plots; an already-running plot is
        # still bounded by the per-script timeout.
        if not wait:
            self.stopping.set()
        self.executor.shutdown(wait=wait, cancel_futures=not wait)

    def _refresh(self, step, scripts):
        import os
        import subprocess
        import sys

        environment = dict(os.environ)
        environment.update({
            "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        })
        script_dir = Path(__file__).resolve().parent
        try:
            with (self.run_dir / "plots.log").open("a", encoding="utf-8") as log:
                for script, options in scripts:
                    if self.stopping.is_set():
                        break
                    log.write(f"\n=== step {step}: {script} ===\n")
                    log.flush()
                    try:
                        result = subprocess.run(
                            [sys.executable, "-m", script,
                             str(self.run_dir), "--max-step", str(step), *options],
                            cwd=script_dir, env=environment, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            timeout=120, check=False,
                        )
                        if result.returncode:
                            raise RuntimeError(f"exit status {result.returncode}")
                    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
                        log.write(f"plot refresh failed: {error}\n")
                        log.flush()
                        print(f"[plots] step {step}: {script} not updated "
                              f"({error}); see {self.run_dir / 'plots.log'}",
                              flush=True)
        except OSError as error:
            print(f"[plots] step {step}: cannot write plots.log: {error}", flush=True)


def _slugify(s: str) -> str:
    """Make a string safe for use in a directory name."""
    return re.sub(r"[^A-Za-z0-9._\-]", "_", str(s)).strip("_")


def make_experiment_dir(cfg, root: str = "runs", resume_dir=None,
                        config_dict=None) -> Path:
    """
    Build a directory whose name encodes the key identifiers of this run.
    Full hyperparameters are always in config.json inside the directory.

    When ``resume_dir`` is provided, reuse it without rewriting the original
    configuration. ``config_dict`` lets callers persist problem-specific YAML
    keys in addition to the Config dataclass fields.
    """
    if resume_dir is not None:
        path = Path(resume_dir).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"resume directory not found: {path}")
        return path

    benchmark = getattr(cfg, "problem_type", None) or getattr(cfg, "problem", "run")
    objective = getattr(cfg, "advantage_mode", None) or "entropic"
    name_parts = [
        _slugify(benchmark),
        _slugify(objective),
        time.strftime("%m%d-%H%M"),
    ]
    triout = os.environ.get("TRIOUT")
    if triout is not None:
        if not re.fullmatch(r"[1-9][0-9]*", triout):
            raise ValueError("TRIOUT must be a positive integer used only in the run directory name")
        name_parts.append(f"triout{triout}")
    name = "-".join(name_parts)
    path = Path(root) / name
    try:
        path.mkdir(parents=True)
    except FileExistsError:
        raise FileExistsError(
            f"Run directory already exists: {path}. "
            "Choose a different TRIOUT for a new run or use --resume for the existing run."
        ) from None

    if config_dict is not None:
        cfg_dict = dict(config_dict)
    else:
        try:
            cfg_dict = asdict(cfg)
        except TypeError:
            cfg_dict = {k: getattr(cfg, k) for k in dir(cfg)
                        if not k.startswith("_")
                        and not callable(getattr(cfg, k))}
    (path / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    return path


def restart_changed_rollout_step(cfg, exp_dir, step):
    """Archive an unfinished step whose prompt or rollout budget changed."""
    run = Path(exp_dir)
    directory = run / f"step{step:02d}"
    parents_path = directory / f"step{step:02d}.parents.json"
    if ((directory / f"step{step:02d}.summary.json").is_file()
            or not parents_path.is_file()):
        return None
    event = json.loads(parents_path.read_text())
    saved = event.get("rollout_config")
    source = parents_path
    if saved is None:
        source = run / "config.json"
        saved = json.loads(source.read_text())
    reject_retired_coder_cap(saved, source=source)
    values = {key: int(saved[key]) for key in ROLLOUT_BUDGET_KEYS if key in saved}
    budget_changed = any(getattr(cfg, key, None) != value for key, value in values.items())
    prompt_changed = (saved.get("rollout_prompt_revision")
                      != getattr(cfg, "rollout_prompt_revision", None))
    if not (budget_changed or prompt_changed):
        return None

    backup = archive_unfinished_rollout_step(exp_dir, step)
    reasons = []
    if budget_changed:
        reasons.append("rollout counts changed")
    if prompt_changed:
        reasons.append("task prompt revision changed")
    print(f"[resume] {'; '.join(reasons)}: restarting step {step} with "
          f"{cfg.groups_per_step} parents x {cfg.group_size} rollouts; "
          f"previous step artifacts moved to {backup}", flush=True)
    return backup


def archive_unfinished_rollout_step(exp_dir, step):
    import tempfile

    run = Path(exp_dir)
    directory = run / f"step{step:02d}"
    if not directory.is_dir() or (directory / f"step{step:02d}.summary.json").is_file():
        return None
    # Partial gradients belong to the discarded trajectories.
    sidecars = [run / f".exact_spo_rs_overflow_step{step:04d}.pt"]
    marker = run / ".exact_sharded_recovery.json"
    if marker.is_file():
        recovery = json.loads(marker.read_text())
        if int(recovery["step"]) == step:
            sidecars.insert(0, marker)
            filename = recovery.get("overflow_gradient_file")
            if filename:
                if Path(filename).name != filename:
                    raise ValueError("invalid overflow gradient filename")
                sidecars.append(run / filename)
    backup_root = run / "_restarted_steps"
    backup_root.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=f"step{step:02d}_", dir=backup_root))
    # Move the directory last so an interruption during these moves remains
    # recognizable on the next resume. No checkpoint/adapter is changed.
    for path in dict.fromkeys(sidecars):
        if path.is_file():
            path.rename(backup / path.name)
    critic_dir = run / "critic" / directory.name
    if critic_dir.is_dir():
        (backup / "critic").mkdir()
        critic_dir.rename(backup / "critic" / directory.name)
    if directory.is_dir():
        directory.rename(backup / directory.name)

    return backup


def save_rollout(
    exp_dir: Path,
    step: int,
    group: int,
    rollout: int,
    response_text: str,
    meta: dict,
    prompt_text: str = None,
    artifacts_already_saved: bool = False,
):
    """
    Save one rollout as a .txt + .meta.json pair, plus the optional prompt.

    meta should include at least: reward, valid, parsed, ran, msg.
    Anything JSON-serializable is fine.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    _retry_rollout_filesystem(
        lambda: step_dir.mkdir(parents=True, exist_ok=True), step_dir)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"
    if not artifacts_already_saved:
        save_rollout_artifacts(
            exp_dir, step, group, rollout, response_text,
            prompt_text=prompt_text)

    # Make sure we can dump everything (numpy floats, bools, etc.)
    def _coerce(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if hasattr(v, "item"):  
            try:
                return v.item()
            except Exception:
                return str(v)
        if hasattr(v, "tolist"):  
            try:
                return v.tolist()
            except Exception:
                return str(v)
        return str(v)

    safe_meta = {k: _coerce(v) for k, v in meta.items()}
    meta_path = step_dir / f"{base}.meta.json"
    _atomic_write_text(meta_path, json.dumps(safe_meta, indent=2))


def save_rollout_artifacts(
    exp_dir: Path,
    step: int,
    group: int,
    rollout: int,
    response_text: str,
    prompt_text: str = None,
    pending_meta: dict = None,
    response_token_ids=None,
    behavior_logprobs=None,
):
    """Persist generated text immediately, before reward evaluation finishes."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    _retry_rollout_filesystem(
        lambda: step_dir.mkdir(parents=True, exist_ok=True), step_dir)
    base = f"step{step:02d}_group{group:02d}_rollout{rollout:03d}"

    def _atomic_text(path, value):
        _atomic_write_text(path, value or "", errors="replace")

    _atomic_text(step_dir / f"{base}.txt", response_text)
    if response_token_ids is not None:
        save_rollout_token_ids(
            step_dir / f"{base}.tokens.zlib", response_token_ids)
    if behavior_logprobs is not None:
        values = list(behavior_logprobs)
        if (response_token_ids is not None
                and len(values) != len(response_token_ids)):
            raise ValueError(
                "rollout behavior-logprob count must match token-ID count")
        save_rollout_logprobs(
            step_dir / f"{base}.behavior_logprobs.zlib", values)
    if prompt_text is not None:
        _atomic_text(step_dir / f"{base}.prompt.txt", prompt_text)
    if pending_meta is not None:
        path = step_dir / f"{base}.meta.json"
        _atomic_write_text(
            path, json.dumps(pending_meta, indent=2, default=str))


def save_parent_selections(exp_dir: Path, step: int, sampler_type: str,
                           parents: list, picks_info: list, *, rollout_config=None):
    """Persist the exact parents selected for a step before generation/training.

    The sampler checkpoint only contains nodes that survived archive pruning.
    This event file is deliberately independent of that checkpoint so a later
    tree plot can also reconstruct selections whose children were invalid,
    duplicates, or eventually pruned.
    """
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for group, parent in enumerate(parents):
        info = picks_info[group] if group < len(picks_info) else {}
        selected.append({
            "group": group,
            "parent_id": str(parent.id),
            "parent_timestep": int(parent.timestep),
            "parent_reward": (float(parent.value)
                              if parent.value is not None else None),
            "parent_raw_score": (float(parent.raw_score)
                                 if parent.raw_score is not None else None),
            "parent_is_seed": bool(parent.is_seed),
            "ancestor_ids": [str(p.get("id")) for p in (parent.parents or [])
                             if p.get("id") is not None],
            "visit_count": int(info.get("n", 0)),
            "q_value": (float(info["Q"]) if info.get("Q") is not None else None),
            "prior": (float(info["P"]) if info.get("P") is not None else None),
            "exploration_bonus": (float(info["bonus"])
                                  if info.get("bonus") is not None else None),
            "selection_score": (float(info["score"])
                                if info.get("score") is not None else None),
        })
        if parent.is_seed and getattr(rollout_config, "ac1_ttt_rollouts", False):
            selected[-1]["seed_construction"] = parent.construction
    payload = {
        "version": 1,
        "step": int(step),
        "sampler_type": str(sampler_type),
        "parents": selected,
    }
    if rollout_config is not None:
        payload["rollout_config"] = {
            key: int(getattr(rollout_config, key)) for key in ROLLOUT_CONFIG_KEYS
            if hasattr(rollout_config, key)
        }
        payload["rollout_config"]["rollout_prompt_revision"] = getattr(
            rollout_config, "rollout_prompt_revision", None)
    path = step_dir / f"step{step:02d}.parents.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def advantage_plot_rollout(meta):
    """Keep the plot inputs without copying saved constructions or logs."""
    row = {key: meta[key] for key in _ADVANTAGE_PLOT_FIELDS if key in meta}
    if "hybrid_invariant_mixture_rms" in (meta.get("spo_rs") or {}):
        row["legacy_invariant_rms"] = True
    return row


def save_step_summary(exp_dir: Path, step: int, summary: dict):
    """Write a per-step summary (group stats, best so far, timings)."""
    step_dir = Path(exp_dir) / f"step{step:02d}"
    _atomic_write_text(step_dir / f"step{step:02d}.summary.json",
                       json.dumps(summary, indent=2, default=str))


def save_best_state_snapshot(exp_dir: Path, step: int, state, *,
                             metric_name: str, maximize: bool,
                             construction=None):
    """Persist the sampler's authoritative best-ever state after evaluation.

    Rollout metadata is not authoritative here: the best state may be an
    unbeaten seed or a valid child later removed by archive deduplication or
    pruning.  Keeping one small, step-scoped snapshot also lets asynchronous
    plotting reproduce exactly the best state known at that evaluated step.
    """
    if state is None:
        return None
    payload = {
        "version": 1,
        "completed_step": int(step),
        "metric_name": str(metric_name),
        "maximize": bool(maximize),
        "state_id": str(getattr(state, "id", "")),
        "found_step": int(getattr(state, "timestep", step)),
        "is_seed": bool(getattr(state, "is_seed", False)),
        "raw_score": (float(state.raw_score)
                      if getattr(state, "raw_score", None) is not None
                      else None),
        "reward": (float(state.value)
                   if getattr(state, "value", None) is not None else None),
        "code": str(getattr(state, "code", "") or ""),
        "construction": construction,
        "construction_saved": construction is not None,
    }
    step_dir = Path(exp_dir) / f"step{step:02d}"
    step_dir.mkdir(exist_ok=True)
    path = step_dir / f"step{step:02d}.best.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    return path


def load_best_state_snapshot(exp_dir: Path, max_step=None):
    """Load the latest authoritative best-state snapshot through max_step."""
    candidates = []
    for path in Path(exp_dir).glob("step*/step*.best.json"):
        try:
            payload = json.loads(path.read_text())
            completed_step = int(payload["completed_step"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if max_step is not None and completed_step > int(max_step):
            continue
        candidates.append((completed_step, path, payload))
    if not candidates:
        return None
    _step, path, payload = max(candidates, key=lambda item: item[0])
    payload["_snapshot_path"] = path
    return payload


def append_step_result(exp_dir: Path, step: int, summary: dict):
    """Append one idempotent per-step result line to the run-level log."""
    step = int(step)
    path = Path(exp_dir) / "result.txt"
    marker = f"step={step:04d}\t"
    if path.is_file():
        try:
            if any(line.startswith(marker)
                   for line in path.read_text(errors="replace").splitlines()):
                return path
        except OSError:
            pass

    def _number(value):
        if value is None:
            return "unavailable"
        try:
            return f"{float(value):.12f}"
        except (TypeError, ValueError):
            return "unavailable"

    metric = str(summary.get("result_metric_name") or "raw metric")
    direction = ("maximize" if bool(summary.get("result_maximize", True))
                 else "minimize")
    step_raw = _number(summary.get("step_best_raw_score"))
    step_reward = _number(summary.get("step_best_reward"))
    run_raw = _number(summary.get("best_seen_raw_score"))
    run_reward = _number(summary.get("best_seen_reward"))
    found_step = summary.get("best_seen_step")
    found_step = ("unavailable" if found_step is None
                  else str(int(found_step)))
    line = (
        f"{marker}metric={metric}\tdirection={direction}\t"
        f"step_best_raw={step_raw}\tstep_best_reward={step_reward}\t"
        f"best_seen_raw={run_raw}\tbest_seen_reward={run_reward}\t"
        f"best_seen_found_step={found_step}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    return path


def save_final_summary(exp_dir: Path, best_value, best_code, best_step,
                       best_construction=None, best_raw_score=None):
    """
    Write the end-of-run summary.

    best_construction is the actual solution object (the Erdos h array or an
    AC coefficient sequence). Written both into the summary and to its own
    file, because a plot or verifier should never have to re-run a stochastic,
    wall-clock-bounded program to recover the answer.
    """
    out = {
        "best_value": float(best_value) if best_value is not None else None,
        "best_raw_score": (float(best_raw_score)
                           if best_raw_score is not None else None),
        "best_step": int(best_step) if best_step is not None else None,
        "best_code": best_code or "",
        "best_construction": best_construction,
    }
    (exp_dir / "final.summary.json").write_text(json.dumps(out, indent=2))
    if best_code:
        (exp_dir / "best_code.py").write_text(best_code)
    if best_construction:
        (exp_dir / "best_construction.json").write_text(
            json.dumps(best_construction))

if __name__ == "__main__":
    # Self-test
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        coder_model_name="openai/gpt-oss-120b",
        problem="erdos", problem_type=None,
        num_steps=50, groups_per_step=8, group_size=64,
        learning_rate=4e-5, temperature=1.0, kl_penalty_coef=0.1,
    )
    p = make_experiment_dir(cfg, root="/tmp/runs_test")
    print(f"Created: {p}")
    save_rollout(p, step=0, group=0, rollout=0,
                 response_text="```python\nprint('hello')\n```",
                 meta={"reward": 0.0, "valid": False, "msg": "demo",
                       "advantage": 1.234})
    print("Saved demo rollout. Contents:")
    for f in sorted(p.iterdir()):
        print(" ", f.name)
