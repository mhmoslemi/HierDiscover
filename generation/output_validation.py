"""Coder and strategist output-contract checks."""

import json
import re


_INLINE_STRATEGY_QUOTE = (
    r"(?P<ticks>`+)(?!`)[^\n]*?(?P=ticks)(?!`)|"
    r'"(?:\\.|[^"\\\n])*"|(?<!\w)\'(?:\\.|[^\'\\\n])*\'|'
    r"“[^”\n]*”|‘[^’\n]*’"
)
_STRATEGY_BOUNDARIES = re.compile(
    r"</?(?:strategy|think|analysis|reasoning)\b[^>]*>|"
    r"<\|channel\|>final<\|message\|>|assistantfinal",
    flags=re.IGNORECASE,
)
_STRATEGY_LEXEME = re.compile(
    r"(?P<quote>" + _INLINE_STRATEGY_QUOTE + r")|(?P<fence>`{3,}|~{3,})|"
    r"(?P<boundary>" + _STRATEGY_BOUNDARIES.pattern + r")",
    flags=re.IGNORECASE,
)
_REASONING_END = re.compile(
    r"^[ \t]*(?:</(?:think|analysis|reasoning)>|"
    r"<\|channel\|>final<\|message\|>|assistantfinal)",
    flags=re.IGNORECASE | re.MULTILINE,
)
_FINAL_STRATEGY_END = re.compile(r"</strategy\s*>\s*\Z", re.IGNORECASE)


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


def strategy_boundary_view(text):
    """Hide quoted protocol examples without changing original text offsets."""
    chunks = []
    cursor = 0
    in_plan = reasoning = False
    while match := _STRATEGY_LEXEME.search(text, cursor):
        start, end = match.span()
        hidden = match.group("quote") is not None
        if match.group("fence"):
            fence = match.group("fence")
            closing = re.search(r"^[ \t]*" + re.escape(fence[0]) +
                                rf"{{{len(fence)},}}[ \t]*\r?$", text[end:], re.MULTILINE)
            close = end + closing.start() if closing else -1
            end = end + closing.end() if closing else len(text)
            # Markdown cannot consume a transport boundary or the outer plan's final tag.
            for boundary in _REASONING_END.finditer(text, match.end(), end):
                example = text[boundary.end():close] if close >= 0 else ""
                if close >= 0 and (in_plan and "<strategy" not in example.lower()
                                   or re.search(r"<strategy\b[^>]*>.*?</strategy\s*>", example, re.I | re.S)):
                    continue
                end = boundary.start()
                break
            if in_plan:
                boundary = _FINAL_STRATEGY_END.search(text, match.end())
                if boundary and boundary.start() < end:
                    end = boundary.start()
            hidden = True
        elif match.group("boundary"):
            tag = match.group().lower()
            if tag.startswith("<strategy"):
                in_plan = not reasoning
            elif tag.startswith("</strategy"):
                in_plan = False
            else:
                reasoning = tag.startswith(("<think", "<analysis", "<reasoning"))
                if not reasoning:
                    in_plan = False
        chunks.append(text[cursor:start])
        part = text[start:end]
        if hidden:
            part = _STRATEGY_BOUNDARIES.sub(lambda tag: " " * len(tag.group()), part)
        chunks.append(part)
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def strategy_answer_scope(response):
    """Scope planner output without treating quoted reasoning tags as boundaries."""
    text = str(response or "")
    scope = final_answer_scope(strategy_boundary_view(text))
    return text[len(text) - len(scope):] if scope else ""


def normalize_strategy_plan(response):
    """Adapt explicit JSON/plan fences; leave the existing XML path intact.

    Also accepts these final-answer formats from compatible open-weight
    models. Arbitrary prose, Python, incomplete JSON, or reasoning alone is
    never promoted to a strategy. The final plan still goes through validation.
    """
    original = str(response or "")
    text = strategy_answer_scope(original).strip()
    fence = re.fullmatch(r"```(json|plan|strategy)\s*\n(.*?)\n```",
                         text, flags=re.DOTALL | re.IGNORECASE)
    plan = None
    if fence:
        text = fence.group(2).strip()
        if fence.group(1).lower() in {"plan", "strategy"}:
            plan = text
    if plan is None:
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return "" if fence or text.startswith(("{", "[")) else original
        if not isinstance(data, dict) or set(data) != {"plan"}:
            return ""
        plan = data["plan"]
    if not isinstance(plan, str) or not plan.strip():
        return ""
    # Reject embedded boundaries so an injected or nested block cannot
    # silently replace the complete plan during the existing extraction step.
    if re.search(r"</?(?:strategy|think|analysis|reasoning)\b|assistantfinal|"
                 r"<\|channel\|>", plan, re.IGNORECASE):
        return ""
    return f"<strategy>\n{plan.strip()}\n</strategy>"


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
