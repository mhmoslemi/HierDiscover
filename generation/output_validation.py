"""Final Python program format checks."""

import re


def final_answer_scope(response, *, require_final_marker=False):
    text = str(response or "")
    lowered = text.lower()
    end = -1
    for marker in ("<|channel|>final<|message|>", "assistantfinal",
                   "</think>", "</analysis>", "</reasoning>"):
        position = lowered.rfind(marker)
        if position >= 0:
            end = max(end, position + len(marker))
    if end >= 0:
        text = text[end:]
    elif require_final_marker:
        return ""
    if re.search(r"<(?:think|analysis|reasoning)\b", text, re.IGNORECASE):
        return ""
    return text


def coder_output_issue(response, *, require_final_marker=False, final_answer_text=None):
    """Check formatting only, never program quality, syntax, or reward."""
    text = (final_answer_scope(response, require_final_marker=require_final_marker)
            if final_answer_text is None else final_answer_text)
    openings = list(re.finditer(r"```python\b[^\S\n]*\n?", text, re.IGNORECASE))
    if not openings:
        return "missing final Python block"
    start = openings[-1].end()
    end = text.find("```", start)
    if end < 0:
        return "unclosed final Python block"
    if not text[start:end].strip():
        return "empty final Python block"
    return None
