"""
Base classes for problems.
The reward convention across ALL problems is "higher is better", so the PUCT
sampler and the entropic advantage do not need to know whether the underlying
metric is minimized or maximized. 
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

from evaluation.reward import extract_python_code
from evaluation.sandbox import run_code


ERDOS_REWARD_ANCHORS = {
    "ac1": (2.0, 1.5026),
}

# (runtime in microseconds, reward), from slowest to fastest.
TRIMUL_REWARD_POINTS = ((6500.0, 2.0), (2800.0, 2.4), (550.0, 5.0))


def erdos_scaled_reward(raw_score: float, seed_score: float, goal_score: float) -> float:
    """Map a valid minimization score to the Erdős seed and goal reward scale."""
    fraction_from_goal = (raw_score - goal_score) / (seed_score - goal_score)
    erdos_score = 0.38086 + (0.5 - 0.38086) * fraction_from_goal
    return 1.0 / erdos_score


# ----------------------------------------------------------------------
# Data carried between the engine and the problems
# ----------------------------------------------------------------------
@dataclass
class SeedState:
    """One initial archive entry."""
    code: str = ""
    value: float = 0.0                 # reward (higher=better), used by the sampler
    raw_score: Optional[float] = None  # true metric, shown in the prompt
    construction: Optional[list] = None  # injected global (height_sequence_1 / initial_h_values)
    poisson: Optional[float] = None


@dataclass
class ParentContext:
    """Everything a prompt/preprocess needs from the parent state."""
    code: str = ""
    value: float = 0.0
    raw_score: Optional[float] = None
    construction: Optional[list] = None
    observation: str = ""
    previous_raw_score: Optional[float] = None
    poisson: Optional[float] = None


@dataclass
class RewardResult:
    reward: float = 0.0
    raw_score: Optional[float] = None
    valid: bool = False
    parsed: bool = False
    ran: bool = False
    msg: str = ""
    stdout: str = ""
    code: str = ""
    construction: Optional[list] = None  
    # Failure stage preserved in rollout diagnostics.
    # code = malformed source/runtime/interface bug; constraint = a runnable
    # program rejected by the scientific verifier; timeout/infrastructure are
    # distinct from code failures.
    failure_kind: str = ""
    poisson: Optional[float] = None


def is_code_failure(res) -> bool:
    """Classify code failures separately from verifier, timeout and infra failures.

    Explicit failure kinds are authoritative. The fallback supports older
    RewardResult-like objects without losing their original classification.
    """
    kind = str(getattr(res, "failure_kind", "") or "").strip().lower()
    if kind:
        return kind == "code"
    if not bool(getattr(res, "parsed", False)):
        return True
    if bool(getattr(res, "ran", False)):
        return False
    msg = str(getattr(res, "msg", "") or "").lower()
    return "timeout" not in msg


# ----------------------------------------------------------------------
# Problem ABC
# ----------------------------------------------------------------------
class Problem(ABC):
    name: str = "base"
    entrypoint: str = "run"         
    metric_name: str = "score"
    maximize: bool = True

    # Whether RewardResult.construction is the actual solution object and
    # therefore worth writing into every rollout's meta. Enable it where the
    # construction cannot be recovered by re-running the program: Erdos and the
    # AC problems return arrays produced by stochastic, wall-clock-bounded
    # optimizers, so a replay need not reproduce them. Circle packing and
    # gpu_mode carry no construction at all, and their programs are the artifact.
    saves_construction: bool = False

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.target = self.cfg.get("target")
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.num_seed_states = int(self.cfg.get("num_seed_states", 8))
        self.seed = int(self.cfg.get("seed", 42))
        self.eval_cpus = int(self.cfg.get("eval_cpus", 1))
        if self.eval_cpus < 1:
            raise ValueError("eval_cpus must be >= 1")
        # H100 schedules sandboxes without pinning, but native thread pools
        # still have to fit inside each sandbox's address-space limit.
        self.sandbox_max_cpus = (
            1 if self.cfg.get("os_scheduled_cpu_evaluation") else self.eval_cpus)

    # ---- prompt / sandbox program / scoring (subclasses implement) ----
    @abstractmethod
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        """Build the chat messages for one parent state."""
        ...

    @abstractmethod
    def preprocess(self, code: str, parent: ParentContext) -> str:
        """Return the full program to execute (prelude + verifier + construction + code)."""
        ...

    @abstractmethod
    def score(self, output: Any, stdout: str) -> RewardResult:
        """Validate the sandbox return value and turn it into a RewardResult."""
        ...

    @abstractmethod
    def seed_states(self) -> List[SeedState]:
        ...

    def initial_spo_rs_rewards(self, seeds: List[SeedState]) -> List[float]:
        """Rewards available before the first policy rollout."""
        return [seed.value for seed in seeds]

    # ---- default reward path (subprocess sandbox) --------------------
    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float, *, cpu_id: Optional[int] = None,
                       memory_limit_bytes: Optional[int] = None,
                       final_answer_text: Optional[str] = None,
                       on_sandbox_deadline=None,
                       ) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        ac1_ttt_rollouts = (getattr(self, "problem_type", None) == "ac1"
                            and self.cfg.get("ac1_ttt_rollouts", False))
        if ac1_ttt_rollouts:
            from generation.output_validation import ttt_ac1_program, ttt_ac1_python_block

            stored_code = (ttt_ac1_python_block(response_text) if final_answer_text is None
                           else final_answer_text)
            code = ttt_ac1_program(stored_code)
        else:
            code = stored_code = extract_python_code(response_text, final_answer_text=final_answer_text)
        if code is None:
            res.msg = "no_code_block"
            res.failure_kind = "code"
            return res
        res.parsed = True
        res.code = stored_code

        full_code = self.preprocess(code, parent)
        out = run_code(
            full_code,
            entrypoint=self.entrypoint,
            timeout_s=timeout_s,
            max_cpus=(1 if cpu_id is not None else self.sandbox_max_cpus),
            cpu_id=cpu_id,
            memory_limit_bytes=memory_limit_bytes,
            on_sandbox_deadline=on_sandbox_deadline,
            hard_timeout=bool(self.cfg.get("os_scheduled_cpu_evaluation")),
            **({"stdout_feedback": True} if ac1_ttt_rollouts else {}),
        )
        diagnostics = [out.get("stdout", ""), out.get("traceback", ""),
                       out.get("stderr", "")]
        res.stdout = "\n".join(str(x).strip() for x in diagnostics if x).strip()
        if not out.get("ok"):
            error = str(out.get("error", "unknown"))
            res.msg = f"run_failed: {error}"
            res.failure_kind = ("timeout" if "timeout" in error.lower()
                                else "code")
            return res
        res.ran = True

        scored = self.score(out.get("value"), res.stdout)
        # carry engine-level fields the scorer does not set
        scored.parsed = True
        scored.ran = True
        scored.code = stored_code
        if not scored.stdout:
            scored.stdout = res.stdout
        if not scored.valid and not scored.msg:
            scored.msg = "invalid"
        if not scored.valid and not scored.failure_kind:
            scored.failure_kind = "constraint"
        return scored
