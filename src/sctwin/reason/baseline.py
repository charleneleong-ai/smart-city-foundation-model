"""Reference baselines for the verifiable reasoning environment — the numbers a trained
(RLVR) policy must beat. Establish these *before* training: the foundation model's gain is
only meaningful against the forecast it competes with, an oracle ceiling, and a naive floor.

- oracle: answers the held-out actual (ceiling; ~1.0).
- interval_centre: the midpoint of the calibrated interval — i.e. the GBM forecast the
  conformal band is centred on. The strong reference the reasoner has to approach or beat.
- constant(v): always predicts v (use the mean actual for a naive floor).
"""

import functools

from sctwin.reason.environment import Policy, Question, ReasoningEnvironment


def oracle(q: Question) -> float:
    return q.actual


def interval_centre(q: Question) -> float:
    return (q.lo + q.hi) / 2


def _const(value: float, q: Question) -> float:
    return value


def constant(value: float) -> Policy:
    return functools.partial(_const, value)  # picklable, unlike a lambda


def reference_policies(env: ReasoningEnvironment) -> dict[str, Policy]:
    """The standard reference set: oracle ceiling, the forecast (interval centre), and a
    constant-mean floor computed from the environment's own held-out actuals."""
    qs = env.questions()
    mean = sum(q.actual for q in qs) / len(qs) if qs else 0.0
    return {
        "oracle (perfect)": oracle,
        "forecast (interval centre)": interval_centre,
        "constant (mean)": constant(mean),
    }


def evaluate(env: ReasoningEnvironment, policies: dict[str, Policy]) -> dict[str, float]:
    """Mean reward of each named policy over the environment — the baseline table."""
    return {name: env.rollout(p)["mean_reward"] for name, p in policies.items()}
