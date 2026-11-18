"""Format the parent fields used by the shared task messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompts import base

if TYPE_CHECKING:
    from problems.base import ParentContext


def render_state_context(metric_name: str, target: float, parent: ParentContext,
                         maximize: bool = True) -> str:
    context = base.STATE_INTRO.format(metric_name=metric_name)
    if parent.code and parent.code.strip():
        code = parent.code
        if not code.startswith("```python\n"):
            code = base.PYTHON_BLOCK.format(code=code)
        context += base.LAST_CODE.format(code=code)
    else:
        context += base.NO_CODE

    score = parent.raw_score
    if score is None:
        context += base.TARGET.format(metric_name=metric_name, target=target)
    else:
        if (parent.previous_raw_score is not None
                and parent.construction is not None and len(parent.construction)):
            context += base.BEFORE_AFTER.format(
                metric_name=metric_name, direction="higher" if maximize else "lower",
                before=parent.previous_raw_score, after=score,
            )
        else:
            context += base.CURRENT_VALUE.format(metric_name=metric_name, after=score)
        gap = target - score if maximize else score - target
        context += base.TARGET_GAP.format(target=target, gap=gap)

    stdout = parent.observation.strip() if parent.observation else ""
    if stdout:
        if len(stdout) > 500:
            stdout = base.TRUNCATED_OUTPUT.format(stdout=stdout[-500:])
        context += base.PREVIOUS_OUTPUT.format(stdout=stdout)
    return context
