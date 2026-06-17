r"""Parse a boxed numeric answer out of an LLM completion — the shared answer contract for both
verifiable-reasoning environments (forecasting and interventional). Each policy is prompted to end
with \boxed{number}; this extracts it from raw text or a chat-format completion, so the reward
functions and LLM policies don't each reinvent the parse.
"""

import re

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_answer(text: str) -> float | None:
    r"""Extract the predicted number: prefer the last \boxed{...}, else the last number."""
    boxed = _BOXED.findall(text)
    nums = _NUM.findall(boxed[-1]) if boxed else _NUM.findall(text)
    return float(nums[-1]) if nums else None


def completion_text(completion: object) -> str:
    """The text of a TRL completion — a raw string or a chat-format [{role, content}, ...] list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return str(completion[-1].get("content", ""))
    return str(completion)
