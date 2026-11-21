"""Shared parent text from test-time-training/discover."""


UPSTREAM_COMMIT = "6c40e82dab9d5de7416ac873ad5cd3106084aaed"


STATE_INTRO = "You are iteratively optimizing {metric_name}."


LAST_CODE = "\nHere is the last code we ran:\n{code}"


PYTHON_BLOCK = "```python\n{code}\n```"


NO_CODE = "\nNo previous code available."


BEFORE_AFTER = (
    "\nHere is the {metric_name} before and after running the code above "
    "({direction} is better): {before:.6f} -> {after:.6f}"
)


CURRENT_VALUE = "\nCurrent {metric_name} (lower is better): {after:.6f}"


TARGET_GAP = (
    "\nTarget: {target}. Current gap: {gap:.6f}. "
    "Further improvements will also be generously rewarded."
)


TARGET = "\nTarget {metric_name}: {target}"


PREVIOUS_OUTPUT = "\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"


TRUNCATED_OUTPUT = "\n\n\t\t ...(TRUNCATED)...\n{stdout}"
