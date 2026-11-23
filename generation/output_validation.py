"""Final Python program format checks."""

import re


def ttt_ac1_python_block(text):
    """Use the pinned TTT parser's Python-only block boundaries."""
    matches = list(re.finditer(r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", text, re.DOTALL))
    if matches:
        body = matches[-1].group(1).rstrip()
        if body.strip():
            return f"```python\n{body}\n```"
    return ""


def ttt_ac1_response_code(tokenizer, token_ids):
    (return_id,) = tokenizer.encode("<|return|>", add_special_tokens=False)
    return_count = token_ids.count(return_id)
    if return_count > 1:
        raise ValueError(f"TTT AC1 response has {return_count} return tokens")
    if return_count:
        token_ids = token_ids[:token_ids.index(return_id)]
    return ttt_ac1_python_block(tokenizer.decode(token_ids))


def ttt_ac1_program(block):
    match = re.search(r"```python\s+([\s\S]*?)\s*```", block)
    return match[1].strip() if match else None


def final_answer_scope(response):
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
    if re.search(r"<(?:think|analysis|reasoning)\b", text, re.IGNORECASE):
        return ""
    return text


def coder_output_issue(response, *, final_answer_text=None):
    """Check formatting only, never program quality, syntax, or reward."""
    text = final_answer_scope(response) if final_answer_text is None else final_answer_text
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
