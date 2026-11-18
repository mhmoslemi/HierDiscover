"""
Circle packing.

  - entrypoint:  run_packing
  - validator:   validates and scores the returned packing
  - reward:      sum of radii if valid else 0   (maximize)
"""

from __future__ import annotations
from typing import Any, List
import numpy as np

from evaluation.reward import validate_packing
from problems.base import Problem, ParentContext, RewardResult, SeedState
from prompts.task import circle_packing
from prompts.utils import render_state_context


class CirclePacking(Problem):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_circles = int(cfg.get("num_circles", 26))
        if self.target is None:
            self.target = 2.636 if self.num_circles == 26 else 2.940

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        target = circle_packing.TARGET_26 if self.num_circles == 26 else circle_packing.TARGET_32
        state_ctx = render_state_context(self.metric_name, target, parent)
        return [{"role": "user", "content": circle_packing.TASK_PROMPT.format(
            n=self.num_circles, validator=circle_packing.VALIDATOR, state_ctx=state_ctx,
        )}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = (
            "import numpy as np\n"
            "import math\n"
            "try:\n"
            "    from scipy.optimize import minimize\n"
            "except ImportError:\n"
            "    minimize = None\n\n"
            + circle_packing.VALIDATOR + "\n"
        )
        return prelude + "\n# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, tuple) and len(output) == 3):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        centers, radii, _ = output
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array: {e}"
            res.failure_kind = "code"
            return res

        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != self.num_circles:
            res.msg = f"bad_centers_shape: {centers.shape}"
            res.failure_kind = "code"
            return res
        if radii.shape != (self.num_circles,):
            res.msg = f"bad_radii_shape: {radii.shape}"
            res.failure_kind = "code"
            return res

        valid, msg = validate_packing(centers, radii)

        # A packing the validator accepts but whose radii are all (near) zero is
        # not a solution: it is a program that detected its own failure and
        # returned something harmless. Accepting it at reward 0 makes defensive
        # coding free and inflates the valid rate. Reject it instead.
        if valid:
            s = float(np.sum(radii))
            if s <= self.degenerate_threshold:
                res.valid = False
                res.msg = (f"degenerate_packing: sum of radii {s:.3e} <= "
                           f"{self.degenerate_threshold:.3e}")
                return res
            res.valid = True
            res.msg = msg
            res.reward = s
            res.raw_score = s
            return res

        res.valid = False
        res.msg = msg
        return res

    @property
    def degenerate_threshold(self) -> float:
        """
        Below this the packing is treated as invalid. Default is deliberately
        tiny: it should catch all-zero and numerically-collapsed returns without
        rejecting a genuinely poor but real packing. Override with
        `degenerate_threshold` in the problem config.
        """
        return float(self.cfg.get("degenerate_threshold", 1e-6))

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        return [SeedState(code="", value=0.0, raw_score=0.0)
                for _ in range(self.num_seed_states)]
