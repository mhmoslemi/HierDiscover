"""Erdos' Minimum Overlap Problem and its exact evaluator."""

from __future__ import annotations
import inspect
from typing import Any, List
import numpy as np
from problems.base import Problem, ParentContext, RewardResult, SeedState
from prompts.task import erdos
from prompts.utils import append_user_instruction, render_state_context


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        # Invalid candidates can have an all-zero or numerically tiny sum. The
        # verifier rejects the resulting non-finite normalization below; keep
        # NumPy from also emitting a noisy RuntimeWarning for that candidate.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            h_values = h_values * (target_sum / current_sum)
        if not np.all(np.isfinite(h_values)):
            raise ValueError("Normalization produced NaN or inf values")
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-9, rtol=0.0):
        raise ValueError(
            f"C5 mismatch: reported {c5_achieved:.12f}, "
            f"computed {computed_c5:.12f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    return float(verify_c5_solution(h_values, c5_bound, n_points))


def verify_erdos_solution(result) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


_VERIFIER_SRC = (
    "import numpy as np\n\n"
    + inspect.getsource(verify_c5_solution) + "\n\n"
    + inspect.getsource(evaluate_erdos_solution) + "\n\n"
)


class ErdosMinOverlap(Problem):
    two_stage_rollouts = True
    name = "erdos"
    entrypoint = "run"
    metric_name = "C\u2085 bound"
    maximize = False
    # The h array is the solution and cannot be recovered by replay: the
    # optimizer is stochastic and budget-bounded, and a mid-run rollout's parent
    # array is not otherwise stored either. Save both.
    saves_construction = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.3808
        self.budget_s = float(cfg.get("budget_s", 60.0))

    def build_strategy_messages(
            self, messages: List[dict],
            previous_strategies: List[str] | None = None) -> List[dict]:
        staged = append_user_instruction(messages, erdos.STRATEGY_NOTES.strip())
        return super().build_strategy_messages(
            staged, previous_strategies=previous_strategies)

    def build_code_messages(self, messages: List[dict],
                            strategy: str) -> List[dict]:
        staged = super().build_code_messages(messages, strategy)
        return append_user_instruction(staged, erdos.STAGED_OUTPUT.strip())

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        state_ctx = render_state_context(
            self.metric_name, self.target, parent, maximize=self.maximize,
        )
        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = erdos.PARENT_CONSTRUCTION.format(
                size=len(parent.construction),
            )
        code_section = (
            erdos.IMPROVE_PROGRAM if parent.code and parent.code.strip()
            else erdos.WRITE_PROGRAM
        )
        direct_output_section = "" if self.two_stage_rollouts else erdos.DIRECT_OUTPUT

        user = erdos.TASK_PROMPT.format(
            budget_s=self.budget_s,
            eval_cpus=self.eval_cpus,
            state_ctx=state_ctx,
            construction_section=construction_section,
            code_section=code_section,
            direct_output_section=direct_output_section,
        )
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = _VERIFIER_SRC
        if parent.construction is not None:
            prelude += f"initial_h_values = np.array({list(parent.construction)!r})\n\n"
        return prelude + "# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, (tuple, list)) and len(output) == 3):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        if not verify_erdos_solution(output):
            res.msg = "Invalid solution."
            return res
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        res.valid = True
        res.raw_score = float(c5_bound)
        res.reward = float(1.0 / c5_bound)
        res.construction = list(np.asarray(h_values).ravel())
        res.msg = f"C5 bound: {c5_bound}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        seeds: List[SeedState] = []
        min_points = self.cfg.get("seed_min_points", 40)
        max_points = self.cfg.get("seed_max_points", 99)
        for i in range(self.num_seed_states):
            rng = np.random.default_rng(self.seed + i)
            n_points = int(rng.integers(min_points, max_points + 1))
            construction = np.ones(n_points) * 0.5
            perturbation = rng.uniform(-0.4, 0.4, n_points)
            perturbation = perturbation - np.mean(perturbation)
            construction = construction + perturbation
            dx = 2.0 / n_points
            correlation = np.correlate(construction, 1 - construction, mode="full") * dx
            c5_bound = float(np.max(correlation))
            seeds.append(SeedState(
                code="",
                value=float(1.0 / c5_bound),
                raw_score=c5_bound,
                construction=list(construction),
            ))
        return seeds
