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
        return code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        try:
            centers, radii, _ = output
        except (TypeError, ValueError):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        try:
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii)
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

        try:
            valid, msg = validate_packing(centers, radii)
        except (TypeError, ValueError) as e:
            res.msg = f"bad_array: {e}"
            res.failure_kind = "code"
            return res

        res.valid = valid
        res.msg = msg
        if valid:
            res.raw_score = res.reward = float(np.sum(radii))
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        return [SeedState(code="", value=0.0, raw_score=0.0)
                for _ in range(self.num_seed_states)]
