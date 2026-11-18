"""
Autocorrelation Inequalities (AC1 and AC2).

  - problem_type "ac1": entrypoint propose_candidate, minimize upper bound,
                        reward = 1/result
  - problem_type "ac2": entrypoint construct_function, maximize lower bound,
                        reward = result

"""

from __future__ import annotations

import inspect
import re
from typing import Any, List

import numpy as np

from examples.ac_inequalities.seeds import (
    example_ae_program_random_init,
    thetaevolve_initial_program_prev_init,
)
from problems.base import Problem, ParentContext, RewardResult, SeedState
from prompts.task import ac1, ac2, ac_common
from prompts.utils import append_user_instruction, render_state_context


def _ac_bytelevel_decoder_table():
    """Inverse GPT-2/ByteLevel alphabet used only for AC plan recovery."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return {
        chr(character): value
        for value, character in zip(byte_values, unicode_values)
    }


_AC_BYTELEVEL_DECODE = _ac_bytelevel_decoder_table()
_AC_BYTELEVEL_MARKERS = ("\u0120", "\u010a", "\u0109", "\u010d")


def _repair_ac_bytelevel_surface(text: str) -> str:
    """Repair a serialized byte-token surface for AC strategy text only."""
    original = str(text or "")
    if not any(marker in original for marker in _AC_BYTELEVEL_MARKERS):
        return original
    payload = bytearray()
    for character in original:
        value = _AC_BYTELEVEL_DECODE.get(character)
        if value is None:
            payload.extend(character.encode("utf-8"))
        else:
            payload.append(value)
    return payload.decode("utf-8", errors="replace")

def evaluate_sequence(sequence: list) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    if not isinstance(sequence, list):
        return np.inf
    if not sequence:
        return np.inf
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]
    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return np.inf
    return float(2 * n * max_b / (sum_a**2))


evaluate_sequence_ac1 = evaluate_sequence


def evaluate_sequence(sequence: list) -> float:
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")
    if not sequence:
        raise ValueError("Empty sequence")
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


evaluate_sequence_ac2 = evaluate_sequence


class ACInequalities(Problem):
    two_stage_rollouts = True
    saves_construction = True
    name = "ac_inequalities"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.problem_type = str(cfg.get("problem_type", "ac1")).lower()
        if self.problem_type not in ("ac1", "ac2"):
            raise ValueError(f"problem_type must be 'ac1' or 'ac2', got {self.problem_type}")
        self.budget_s = int(cfg.get("budget_s", 1000))

        if self.problem_type == "ac1":
            self.entrypoint = "propose_candidate"
            self.metric_name = "upper bound"
            self.maximize = False
            if self.target is None:
                self.target = 1.5030
            self._evaluate = evaluate_sequence_ac1
            self._eval_src = inspect.getsource(evaluate_sequence_ac1)
        else:
            self.entrypoint = "construct_function"
            self.metric_name = "lower bound"
            self.maximize = True
            if self.target is None:
                self.target = 0.97
            self._evaluate = evaluate_sequence_ac2
            self._eval_src = inspect.getsource(evaluate_sequence_ac2)

    # ------------------------------------------------------------------
    def build_strategy_messages(
            self, messages: List[dict],
            previous_strategies: List[str] | None = None) -> List[dict]:
        task = ac1 if self.problem_type == "ac1" else ac2
        notes = (ac_common.STRATEGY_NOTES + task.STRATEGY_NOTES).strip()
        staged = append_user_instruction(messages, notes)
        return super().build_strategy_messages(
            staged, previous_strategies=previous_strategies)

    def strategy_output_issue(self, strategy: str) -> str | None:
        """Reject corrupted AC plans before they reach another model.

        DeepSeek/Qwen occasionally returns a syntactically valid strategy block
        whose prose is still a byte-token surface or a collapsed backend-error
        fragment.  Those strings are not useful exploration: the coder cannot
        reliably interpret them.  This gate is deliberately AC-only and checks
        text integrity rather than mathematical quality.
        """
        text = str(strategy or "")
        byte_markers = sum(text.count(marker)
                           for marker in ("\u0120", "\u010a", "\u0109", "\u010d"))
        if byte_markers or "\ufffd" in text:
            return "AC strategy contained tokenizer-decoding artifacts"
        if re.search(r"\S{240,}", text):
            return "AC strategy contained collapsed or unreadable text"
        if re.search(
                r"\bErrorMessage\s*:|\bError during execution\b|"
                r"\bChatCoder\b|/opt/[^\s]+/project\b",
                text, flags=re.IGNORECASE):
            return "AC strategy contained backend error text"
        return None

    def normalize_strategy_response(self, response: str) -> str:
        """Repair byte-token surfaces only for AC strategy generation."""
        return _repair_ac_bytelevel_surface(response)


    def build_prompt(self, parent: ParentContext) -> List[dict]:
        task = ac1 if self.problem_type == "ac1" else ac2
        state_ctx = render_state_context(
            self.metric_name, self.target, parent, maximize=self.maximize,
        )
        if parent.construction is not None and len(parent.construction) > 0:
            state_ctx += ac_common.CONSTRUCTION_LENGTH.format(size=len(parent.construction))
        user = task.TASK_PROMPT.format(
            evaluator=task.EVALUATOR,
            literature=task.LITERATURE,
            budget_s=self.budget_s,
            eval_cpus=self.eval_cpus,
            state_ctx=state_ctx,
        )
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = "import numpy as np\n\n" + self._eval_src + "\n\n"
        if parent.construction is not None:
            prelude += f"height_sequence_1 = np.array({list(parent.construction)!r})\n\n"
        return prelude + "# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not isinstance(output, list) or not output:
            res.msg = "bad_return_sequence"
            res.failure_kind = "code"
            return res
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not np.isfinite(x) for x in output):
            res.msg = "bad_return_values"
            res.failure_kind = "code"
            return res
        # AC entrypoints return only the construction, not a model-reported
        # score. Unlike Erdos there is therefore no claimed-vs-computed scalar
        # and no numerical comparison tolerance to tune: independently run the
        # authoritative evaluator once and trust only that result.
        try:
            result = float(self._evaluate(output))
        except Exception as exc:
            res.msg = f"invalid_solution: {exc}"
            res.failure_kind = "constraint"
            return res
        if not np.isfinite(result) or result <= 0.0:
            res.msg = f"invalid_score: {result}"
            res.failure_kind = "constraint"
            return res
        res.valid = True
        res.raw_score = result
        res.construction = list(output)
        res.msg = f"Success; raw_score={result}"
        if self.problem_type == "ac1":
            res.reward = float(1.0 / result)
        else:
            res.reward = result
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        seeds: List[SeedState] = []
        for i in range(self.num_seed_states):
            rng = np.random.default_rng(self.seed + i)
            construction = [float(rng.random())] * int(rng.integers(1000, 8000))
            if self.problem_type == "ac1":
                raw = float(evaluate_sequence_ac1(construction))
                value = float(1.0 / raw)
                code = example_ae_program_random_init(self.budget_s)
            else:
                raw = float(evaluate_sequence_ac2(construction))
                value = raw
                code = thetaevolve_initial_program_prev_init
            seeds.append(SeedState(code=code, value=value, raw_score=raw,
                                   construction=construction))
        return seeds
