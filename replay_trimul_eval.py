"""Replay 20 saved TriMul kernels with a warm setup process on each GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
from functools import cache, wraps
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace


REPO = Path(__file__).resolve().parent
STEP_DIR = Path("/root/nas/Mohammad/HierDiscover/runs/gpu_mode_gpt-oss-120b_1120-1258-prompt/step18")
COUNT = 20


def write_json(path, data):
    from experiment_io import _atomic_write_text

    _atomic_write_text(path, json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_environment():
    # Share the launcher’s evaluator caches and GPU lease directory.
    runtime = REPO / ".runtime"
    shm = Path("/dev/shm")
    fast = Path(os.environ.get("TTT_FAST_RUNTIME_DIR") or (
        shm / "ttt-small-runtime" if shm.is_dir() and os.access(shm, os.W_OK) else runtime / "fast"))
    os.environ["TTT_FAST_RUNTIME_DIR"] = str(fast)
    os.environ["TMPDIR"] = str(fast / "tmp")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for key, directory in {
        "TMPDIR": fast / "tmp",
        "TORCH_EXTENSIONS_DIR": fast / "torch_extensions",
        "CUDA_CACHE_PATH": fast / "cuda",
        "XDG_CACHE_HOME": runtime / "cache",
        "HF_HOME": runtime / "cache" / "huggingface",
    }.items():
        Path(os.environ.setdefault(key, str(directory))).mkdir(parents=True, exist_ok=True)


def prepare_candidates(step_dir, output, gpu_ids, config, saved_batch):
    from evaluation.reward import extract_python_code
    from experiment_io import load_rollout_token_ids
    from generation.gen_workers import native_final_answer_scope
    from runtime.trimul_evaluation import rollout_path

    final_answers = {
        rollout_path(step_dir.parent, saved_batch["step"], row).stem: row["final_answer"]
        for row in saved_batch.get("candidates", [])
    }
    candidates = []
    missing = []
    tokenizer = None
    for meta_path in sorted(step_dir.glob(f"{step_dir.name}_group*_rollout*.meta.json")):
        meta = read_json(meta_path)
        if meta.get("valid") is not True:
            continue
        name = meta_path.name.removesuffix(".meta.json")
        response_path = step_dir / f"{name}.txt"
        prompt_path = step_dir / f"{name}.prompt.txt"
        tokens_path = step_dir / f"{name}.tokens.zlib"
        if not response_path.is_file() or (name not in final_answers
                and not (prompt_path.is_file() and tokens_path.is_file())):
            missing.append(name)
            continue
        response = response_path.read_bytes().decode("utf-8")
        source = "saved final answer"
        if name in final_answers:
            final_answer = final_answers[name]
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    config.get("coder_model_name") or config["model_name"],
                    trust_remote_code=True, local_files_only=True)
            final_answer = native_final_answer_scope(
                tokenizer, load_rollout_token_ids(tokens_path),
                prompt_text=prompt_path.read_bytes().decode("utf-8"))
            source = "saved token IDs"
        code = extract_python_code(response, final_answer_text=final_answer)
        if not code:
            raise ValueError(f"Cannot recover the submitted Python for valid rollout {name}")
        directory = output / "candidates" / name
        directory.mkdir(parents=True)
        source_bytes = code.encode("utf-8")
        (directory / "submission.py").write_bytes(source_bytes)
        shutil.copyfile(meta_path, directory / "original.meta.json")
        candidates.append({
            "id": name, "gpu": gpu_ids[len(candidates) % len(gpu_ids)],
            "original_raw_score_us": meta["raw_score"],
            "source_sha256": sha256(source_bytes).hexdigest(), "source_recovered_from": source,
        })
        if len(candidates) == COUNT:
            return candidates
    raise ValueError(f"Found {len(candidates)} of {COUNT} replayable valid rollouts in {step_dir}. "
                     f"Valid rollouts missing source artifacts: {', '.join(missing) or 'none'}")


def logged(function):
    @wraps(function)
    def call(*args, **kwargs):
        label = function.__name__
        if function.__name__ == "run_program":
            label = " ".join(map(str, args[0]))
        begin = time.monotonic()
        print(f"[{time.strftime('%H:%M:%S')}] starting {label}", flush=True)
        try:
            result = function(*args, **kwargs)
            if label == "verify_benchmark_environment":
                print(f"Verified evaluator environment {result}", flush=True)
            return result
        finally:
            print(f"[{time.strftime('%H:%M:%S')}] ended {label} after {time.monotonic() - begin:.3f}s",
                  flush=True)
    return call


def replay_worker(connection, owner_pid, output, gpu_id, options):
    # The trusted wrapper stays alive. The runner still launches isolated
    # submission, correctness, and leaderboard processes for each candidate.
    from problems import gpu_mode

    gpu_mode._isolate_eval_process_tree(owner_pid)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, options["lib_dir"])
    with (output / f"gpu{gpu_id}.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log), connection:
            from libkernelbot import run_eval, task

            gpu_mode.ensure_python3_on_path = cache(logged(gpu_mode.ensure_python3_on_path))
            gpu_mode.verify_benchmark_environment = cache(logged(gpu_mode.verify_benchmark_environment))
            task.make_task_definition = cache(logged(task.make_task_definition))
            run_eval.make_system_info = cache(logged(run_eval.make_system_info))
            run_eval.run_program = logged(run_eval.run_program)
            run_config = run_eval.run_config

            def save_runner_result(config):
                write_json(directory / "benchmark_config.json", {
                    key: value for key, value in config.items() if key != "sources"
                })
                result = run_config(config)
                write_json(directory / "runner.json", asdict(result))
                return result

            run_eval.run_config = save_runner_result
            print(f"GPU {gpu_id} setup worker PID {os.getpid()}", flush=True)
            while (candidate := connection.recv()) is not None:
                directory = output / "candidates" / candidate["id"]
                source = (directory / "submission.py").read_bytes()
                if sha256(source).hexdigest() != candidate["source_sha256"]:
                    raise ValueError(f"Submitted bytes changed for {candidate['id']}")
                print(f"\n=== {candidate['id']} ===", flush=True)
                reply = {}
                sink = SimpleNamespace(send=reply.update, close=lambda: None)
                begin = time.monotonic()
                gpu_mode._eval_child(sink, source.decode("utf-8"), gpu_id=gpu_id, **options)
                reply.update(gpu=gpu_id, worker_pid=os.getpid(),
                             elapsed_seconds=time.monotonic() - begin)
                write_json(directory / "replay.json", reply)
                print(f"{reply['msg']} ({reply['elapsed_seconds']:.3f}s)", flush=True)
                connection.send(reply)


def replay_gpu(output, gpu_id, candidates, options, timeout, stop):
    from problems.gpu_mode import _eval_process_group_exists, _terminate_eval_process_tree
    from runtime.gpu_lease import wait_for_idle_gpu

    context = mp.get_context("spawn")
    next_index = 0
    while next_index < len(candidates) and not stop.is_set():
        if next_index:
            wait_for_idle_gpu(gpu_id)
        parent_end, worker_end = context.Pipe()
        worker = context.Process(
            target=replay_worker,
            args=(worker_end, os.getpid(), output, gpu_id, options), daemon=True)
        with parent_end:
            with worker_end:
                worker.start()
            try:
                for candidate in candidates[next_index:]:
                    start = time.monotonic()
                    deadline = start + timeout
                    result = None
                    try:
                        parent_end.send(candidate)
                        while not stop.is_set():
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            if parent_end.poll(min(1.0, remaining)):
                                result = parent_end.recv()
                                break
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        pass
                    if stop.is_set():
                        return
                    next_index += 1
                    if result is None:
                        message = ("replay timed out" if time.monotonic() >= deadline
                                   else "replay worker exited without a result")
                        directory = output / "candidates" / candidate["id"]
                        if not (directory / "replay.json").is_file():
                            write_json(directory / "replay.json", {
                                "ok": False, "score_us": None, "msg": message, "gpu": gpu_id,
                                "worker_pid": worker.pid, "elapsed_seconds": time.monotonic() - start,
                            })
                        print(f"GPU {gpu_id} {candidate['id']} {message}", flush=True)
                        break
                    print(f"GPU {gpu_id} {candidate['id']} {result['msg']}", flush=True)
                else:
                    parent_end.send(None)
                    worker.join(5)
            finally:
                if worker.is_alive() or _eval_process_group_exists(worker.pid):
                    _terminate_eval_process_tree(worker)
                else:
                    worker.join()
                if worker.is_alive():
                    raise RuntimeError(f"GPU {gpu_id} setup process group {worker.pid} did not exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-dir", type=Path, default=STEP_DIR)
    parser.add_argument("--gpu-ids", default="6,7")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    step_dir = args.step_dir.resolve()
    default_output = REPO / "outputs" / f"trimul-{step_dir.name}-replay-{time.strftime('%Y%m%d-%H%M%S')}"
    output = (args.output or default_output).resolve()
    if output.is_relative_to(step_dir.parent):
        parser.error("--output must be separate from the source run")
    os.chdir(REPO)
    configure_environment()

    from problems.gpu_mode import GpuMode
    from runtime.gpu_lease import gpu_lease, wait_for_idle_gpu
    from runtime.gpu_runtime import parse_gpu_ids

    gpu_ids = parse_gpu_ids(args.gpu_ids, field="--gpu-ids")
    batch_path = step_dir / "trimul_evaluation.json"
    saved_batch = read_json(batch_path) if batch_path.is_file() else {}
    config = saved_batch.get("config") or read_json(step_dir.parent / "config.json")
    problem = GpuMode(config)
    if problem.problem_type != "trimul":
        raise ValueError(f"{step_dir} is not a TriMul step")
    options = {
        "lib_dir": str(Path(problem.lib_dir).resolve()),
        "task_yaml": str(Path(problem.task_yaml).resolve()),
        "problem_type": "trimul", "log_chars": problem.log_chars,
        "expected_gpu": problem.gpu_type, "expected_triton": problem.triton_version,
        "kernel_python": problem.kernel_python,
    }
    timeout = problem.kernel_timeout_s or float(saved_batch.get("timeout_s", config.get("sandbox_timeout_s", 0)))
    if timeout <= 0:
        raise ValueError("Replay needs a positive evaluator timeout")
    output.mkdir(parents=True)
    print(f"Saving replay to {output}", flush=True)
    candidates = prepare_candidates(step_dir, output, gpu_ids, config, saved_batch)
    write_json(output / "manifest.json", {
        "step_dir": str(step_dir), "gpu_ids": gpu_ids, "candidates": candidates,
        "setup": "one trusted wrapper per GPU, original test and leaderboard subprocesses",
        "evaluator_options": options, "timeout_s": timeout, "wrapper_python": sys.executable,
        "config": config,
    })
    shutil.copyfile(options["task_yaml"], output / "task.yml")

    stop = threading.Event()
    begin = time.monotonic()
    with ExitStack() as stack:
        for gpu_id in sorted(gpu_ids):
            stack.enter_context(gpu_lease(gpu_id, 1.0, purpose="TriMul replay"))
            wait_for_idle_gpu(gpu_id)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(replay_gpu, output, gpu_id,
                                   [row for row in candidates if row["gpu"] == gpu_id],
                                   options, timeout, stop) for gpu_id in gpu_ids]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                raise
    elapsed = time.monotonic() - begin
    results = [read_json(output / "candidates" / row["id"] / "replay.json") for row in candidates]
    scored = sum(result["ok"] for result in results)
    write_json(output / "summary.json", {
        "attempted": len(results), "scored": scored, "elapsed_seconds": elapsed,
    })
    print(f"{scored}/{len(results)} returned scores in {elapsed:.1f}s. Artifacts saved in {output}", flush=True)
    return 0 if scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
