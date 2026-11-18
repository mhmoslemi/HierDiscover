"""Assemble task, parent, and stage instructions into chat messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompts.coder_base import (
    CODE_STAGE_PROMPT, NO_PARENT_PROGRAM, PARENT_PROGRAM, TARGET_CONTEXT,
)
from prompts.strategy_base import LOCAL_STRATEGY_PROMPT, PREVIOUS_STRATEGIES

if TYPE_CHECKING:
    from problems.base import ParentContext


def render_state_context(metric_name: str, target, parent: ParentContext,
                         maximize: bool = True) -> str:
    direction = "higher is better" if maximize else "lower is better"
    target_context = TARGET_CONTEXT.format(
        metric_name=metric_name, target=target, direction=direction,
    )
    if parent.code and parent.code.strip():
        score = parent.raw_score if parent.raw_score is not None else parent.value
        return target_context + PARENT_PROGRAM.format(
            metric_name=metric_name, score=score, code=parent.code,
        )
    return target_context + NO_PARENT_PROGRAM


def append_user_instruction(messages: list[dict], instruction: str) -> list[dict]:
    staged = [dict(message) for message in messages]
    if staged and staged[-1].get("role") == "user":
        staged[-1]["content"] = (
            str(staged[-1].get("content", "")).rstrip() + "\n\n" + instruction + "\n"
        )
    else:
        staged.append({"role": "user", "content": instruction})
    return staged


def strategy_messages(messages: list[dict],
                      previous_strategies: list[str] | None = None) -> list[dict]:
    previous_strategies = list(previous_strategies or [])
    history = ""
    if previous_strategies:
        plans = "\n\n".join(
            f"<previous_strategy_{index}>\n{strategy.strip()}\n"
            f"</previous_strategy_{index}>"
            for index, strategy in enumerate(previous_strategies, start=1)
        )
        history = PREVIOUS_STRATEGIES.format(plans=plans)
    instruction = LOCAL_STRATEGY_PROMPT.replace("{previous_strategies}", history, 1)
    return append_user_instruction(messages, instruction)


def coder_messages(messages: list[dict], strategy: str) -> list[dict]:
    instruction = CODE_STAGE_PROMPT.format(strategy=strategy.strip())
    return append_user_instruction(messages, instruction)
