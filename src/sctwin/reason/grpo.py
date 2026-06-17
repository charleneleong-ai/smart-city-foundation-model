"""GRPO (RLVR) glue: turn the verifiable-reasoning environment into a TRL reward function.

TRL's `GRPOTrainer` is the RLVR algorithm; Unsloth gives fast LoRA + generation on one GPU
(see apps/train_reasoner.py). The policy LLM is prompted to reason and emit a boxed number;
`make_reward_fn` parses it (via `reason.boxed`) and scores it with the environment — the
verifiable reward GRPO maximises. This module is pure (the env + the shared parser), so it's
testable without a GPU.
"""

from sctwin.reason.boxed import completion_text, parse_answer
from sctwin.reason.environment import ReasoningEnvironment

_PROMPT = (
    "You are an urban energy analyst. From the context, predict the electricity load "
    "(arbitrary units) for the given H3 cell and hour. Reason step by step, then give the "
    "final answer on its own line as \\boxed{{number}}.\n\nContext: {context}\n"
)


def prompt_for(context: str) -> str:
    return _PROMPT.format(context=context)


def make_reward_fn(env: ReasoningEnvironment, *, wrong: float = 0.0):
    """A TRL-compatible reward function: reward_fn(completions, actual, lo, hi, **kwargs) ->
    list[float]. `actual`/`lo`/`hi` are dataset columns TRL passes aligned with completions;
    unparseable answers get `wrong`."""

    def reward_fn(
        completions: list, actual: list, lo: list, hi: list, **kwargs: object
    ) -> list[float]:
        out = []
        for comp, a, lo_i, hi_i in zip(completions, actual, lo, hi, strict=True):
            ans = parse_answer(completion_text(comp))
            out.append(wrong if ans is None else env.score(ans, float(a), float(lo_i), float(hi_i)))
        return out

    return reward_fn
