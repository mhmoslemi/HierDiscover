"""Shared text for the coder and parent context."""


CODE_STAGE_PROMPT = """## Strategy from the reasoning-model planning stage

<strategy>
{strategy}
</strategy>

## Code-stage output

Use the task information and strategy above to produce the complete solution.
Read the strategy critically and extract useful mathematical insights,
algorithmic ideas, and implementation details. Build on the parts you judge
helpful. Treat its proposals as suggestions, not requirements: you are free to
adapt, simplify, combine, replace, or discard any part, including choosing a
different overall approach. Use your own judgment to produce the best valid
solution, while following the original task, evaluator, required interface,
and execution constraints.
Do not output analysis, reasoning, a strategy, notes, or example
usage. Return only exactly one fenced Python code block, beginning with
```python and ending with ```."""


FALLBACK_STRATEGY = (
    "No usable strategy was returned. Derive the implementation directly "
    "from the task specification and current parent construction."
)


TARGET_CONTEXT = "Target {metric_name}: {target} ({direction}).\n"

PARENT_PROGRAM = """Your previous program achieved {metric_name} = {score:.6f}.
Here is the previous program:
```python
{code}
```
"""

NO_PARENT_PROGRAM = "No previous program. Write one from scratch.\n"
