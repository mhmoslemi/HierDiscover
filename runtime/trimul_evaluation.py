"""Durable handoff to a TriMul evaluator with no trainer running."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time


class EvaluationHandoff(Exception):
    def __init__(self, path):
        self.path = path
        super().__init__(str(path))


def batch_path(root, step):
    return Path(root) / f"step{step:02d}" / "trimul_evaluation.json"


def _save_batch(path, batch):
    from experiment_io import _atomic_write_text

    _atomic_write_text(path, json.dumps(batch, allow_nan=False))


def completed_batch(root, step):
    path = batch_path(root, step)
    if not path.is_file():
        return None
    batch = json.loads(path.read_text(encoding="utf-8"))
    if not batch["completed"]:
        raise EvaluationHandoff(path)
    return batch


def request_batch(root, step, problem, timeout_s, candidates, **scoring):
    path = batch_path(root, step)
    _save_batch(path, {
        "step": step, "config": problem.cfg, "timeout_s": timeout_s,
        "candidates": candidates, "scoring": scoring, "completed": False,
    })
    raise EvaluationHandoff(path)


def rollout_path(root, step, row):
    return (Path(root) / f"step{step:02d}" /
            f"step{step:02d}_group{row['group']:02d}_rollout{row['rollout']:03d}.txt")


def check_parents(root, step, parents):
    path = batch_path(root, step).with_name(f"step{step:02d}.parents.json")
    saved = json.loads(path.read_text(encoding="utf-8"))["parents"]
    if [row["parent_id"] for row in saved] != [str(parent.id) for parent in parents]:
        raise RuntimeError(f"Cannot restore the selected TriMul parents from {path}")


def read_rollout(root, step, row):
    path = rollout_path(root, step, row)
    prompt = path.with_suffix(".prompt.txt").read_bytes().decode("utf-8")
    return path, path.read_bytes().decode("utf-8"), prompt


def load_result(path, text, final_answer):
    from evaluation.reward import extract_python_code
    from problems.base import RewardResult

    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    code = extract_python_code(text, final_answer_text=final_answer) if meta["valid"] else ""
    if meta["valid"] and not code:
        raise ValueError(f"Saved TriMul result has no kernel source: {path}")
    return RewardResult(
        reward=meta["reward"], raw_score=meta["raw_score"], valid=meta["valid"],
        parsed=meta["parsed"], ran=meta["ran"], msg=meta["msg"],
        stdout=meta["sandbox_stdout"], code=code, failure_kind=meta["failure_kind"],
        construction=[] if meta["valid"] else None, poisson=meta.get("poisson"),
    )


def evaluate_batch(path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("TriMul evaluation coordinator requires an empty CUDA_VISIBLE_DEVICES")

    from experiment_io import save_rollout
    from generation.gen_workers import make_progress_bar
    from problems.base import ParentContext, RewardResult
    from problems.gpu_mode import GpuMode

    path = Path(path)
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch["completed"]:
        return
    root, step = path.parent.parent, int(batch["step"])
    critic_context = batch["scoring"].get("critic")
    problem = GpuMode(batch["config"])
    if problem.problem_type != "trimul" or not problem.kernel_gpu_ids:
        raise ValueError("isolated evaluation requires TriMul and at least one evaluation GPU")
    problem.begin_evaluation_batch()

    def evaluate(row):
        response_path = rollout_path(root, step, row)
        metadata = json.loads(response_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if metadata.get("trimul_isolated_evaluation"):
            return metadata["raw_score"] if metadata["valid"] else None
        response = response_path.read_bytes().decode("utf-8")
        if row["format_issue"] is not None:
            result = RewardResult(reward=problem.fail_score, msg=row["format_issue"], failure_kind="code")
        else:
            result = problem.compute_reward(response, ParentContext(), batch["timeout_s"],
                                            final_answer_text=row["final_answer"])
        if result.failure_kind == "infrastructure":
            raise RuntimeError(f"TriMul evaluation infrastructure failed: {result.msg}")
        outcome = asdict(result)
        if critic_context is not None:
            outcome["archive_eligible"] = bool(result.valid and result.code)
        outcome.pop("code")
        outcome.pop("construction")
        outcome["sandbox_stdout"] = outcome.pop("stdout")
        metadata.update(outcome, status="evaluated", trimul_isolated_evaluation=True)
        save_rollout(root, step, row["group"], row["rollout"], response, metadata,
                     artifacts_already_saved=True)
        return result.raw_score if result.valid else None

    begin = time.monotonic()
    best = batch["scoring"].get("previous_best_raw")
    bar = make_progress_bar(len(batch["candidates"]), desc="evaluating TriMul")
    try:
        with ThreadPoolExecutor(max_workers=len(problem.kernel_gpu_ids)) as pool:
            futures = [pool.submit(evaluate, row) for row in batch["candidates"]]
            try:
                for future in as_completed(futures):
                    raw = future.result()
                    if raw is not None and (best is None or raw < best):
                        getattr(bar, "write", print)(f"[step {step}] best kernel runtime: {raw:.9f} µs")
                        best = raw
                    bar.update(1)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        bar.close()
    batch["evaluation_seconds"] = time.monotonic() - begin
    if critic_context is not None:
        from types import SimpleNamespace
        from training import critic_step

        rows = []
        for row in sorted(batch["candidates"], key=lambda item: (item["group"], item["rollout"])):
            meta = rollout_path(root, step, row).with_suffix(".meta.json")
            rows.append(critic_step.rollout_row(json.loads(meta.read_text(encoding="utf-8"))))
        critic_step.prepare_batch(root, step, critic_context, rows)
        critic_step.score_batch(root, step, SimpleNamespace(**batch["config"]))
    batch["completed"] = True
    _save_batch(path, batch)


if __name__ == "__main__":
    from diagnostics.console import _install_console_timestamps, _install_terminal_log

    _install_console_timestamps()
    log = _install_terminal_log()
    run_dir = Path(os.environ["TTT_SUPERVISOR_RUN_FILE"]).read_text(encoding="utf-8").strip()
    log.bind(Path(run_dir) / "temirnal.log")
    evaluate_batch(sys.argv[1])
