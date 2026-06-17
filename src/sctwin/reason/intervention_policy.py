"""LLM policy + RLVR glue for the interventional environment — the reasoner that *predicts an
intervention's effect*, replacing the `InterventionQuestion -> float` baselines with a real model.

Mirrors `reason/grpo.py` (forecasting env): a prompt describing the intervention, a boxed-number
parse (reused from grpo), an injectable text generator (`Callable[[str], str]`) so the policy is
testable and box-ready without a GPU, and a TRL-compatible reward function. `training_records`
turns the environment into RLVR rows (prompt + hidden verifiable targets) that
`apps/train_reasoner.py` feeds to GRPO.
"""

from collections.abc import Callable

from sctwin.reason.boxed import completion_text, parse_answer
from sctwin.reason.intervention import IPolicy, InterventionEnvironment, InterventionQuestion
from sctwin.reason.reward import interventional_reward

Generate = Callable[[str], str]

_PROMPT = (
    "You are an urban energy planner. Estimate the effect of an intervention on a district's "
    "electricity {metric} load, as a signed change (negative = a reduction). "
    "Intervention: {kind} at intensity {factor:.2f} on H3 cell {cell}. "
    "Reason step by step, then give the final answer — the change in {metric} load — on its own "
    "line as \\boxed{{number}}.\n"
)


def intervention_prompt(q: InterventionQuestion) -> str:
    iv = q.intervention
    return _PROMPT.format(kind=iv.kind, factor=iv.factor, cell=iv.cell, metric=iv.metric)


def oracle_effect(q: InterventionQuestion) -> float:
    return q.true_delta  # ceiling: the counterfactual oracle's own answer


def zero_effect(q: InterventionQuestion) -> float:
    return 0.0  # naive floor: "the intervention changes nothing"


def llm_policy(generate: Generate, *, wrong: float = 0.0) -> IPolicy:
    """An `IPolicy` backed by a text generator: prompt the LLM with the intervention, parse the
    boxed Δ. Unparseable output scores `wrong` (0.0 = predicts no effect)."""

    def policy(q: InterventionQuestion) -> float:
        answer = parse_answer(generate(intervention_prompt(q)))
        return wrong if answer is None else answer

    return policy


def training_records(env: InterventionEnvironment) -> list[dict[str, float | str]]:
    """Environment -> RLVR rows: the prompt the policy sees plus the *hidden* verifiable targets
    (`true_delta`, `scale`) that TRL passes to `make_reward_fn` aligned with completions."""
    return [
        {"prompt": intervention_prompt(q), "true_delta": q.true_delta, "scale": q.scale}
        for q in env.questions()
    ]


def make_reward_fn(*, wrong: float = 0.0) -> Callable[..., list[float]]:
    """A TRL reward function: `reward_fn(completions, true_delta, scale, **kwargs) -> list[float]`.
    Parses each completion's boxed Δ and scores it against the oracle Δ via `interventional_reward`
    (half sign, half magnitude); unparseable completions get `wrong`."""

    def reward_fn(
        completions: list, true_delta: list, scale: list, **kwargs: object
    ) -> list[float]:
        out = []
        for comp, td, sc in zip(completions, true_delta, scale, strict=True):
            answer = parse_answer(completion_text(comp))
            out.append(
                wrong
                if answer is None
                else interventional_reward(answer, float(td), scale=float(sc))
            )
        return out

    return reward_fn
