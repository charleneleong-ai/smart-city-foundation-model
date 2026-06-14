from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import polars as pl

from sctwin.reason.reward import accuracy_reward, coverage_reward


@dataclass(frozen=True)
class Question:
    """A held-out query the reasoner must answer: predict the value at (cell, time). The
    actual + calibrated interval are the *verifiable* ground truth, hidden from the policy."""

    cell: str
    time: datetime
    actual: float
    lo: float
    hi: float


Policy = Callable[[Question], float]


class ReasoningEnvironment:
    """The city as a verifiable-reward (RLVR) environment, built from an SP5 verification
    results frame (cell, time, y_true, lo, hi). A reasoning policy — later an RLVR-trained
    LLM, here any `Question -> float` callable — is scored against held-out actuals and their
    calibrated intervals. `rollout` returns the policy's mean reward (its RLVR return)."""

    def __init__(self, results: pl.DataFrame, *, scale: float | None = None) -> None:
        self._questions = [
            Question(r["cell"], r["time"], r["y_true"], r["lo"], r["hi"])
            for r in results.iter_rows(named=True)
        ]
        std = results["y_true"].std()  # float at runtime; std stubs widen to float | timedelta | None
        self._scale = scale if scale is not None else (cast(float, std) if std else 1.0)

    def questions(self) -> list[Question]:
        return self._questions

    def score(self, answer: float, actual: float, lo: float, hi: float) -> float:
        """Verifiable per-step reward: agreement with the actual (0.7) + inside-interval (0.3)."""
        return 0.7 * accuracy_reward(answer, actual, scale=self._scale) + 0.3 * coverage_reward(answer, lo, hi)

    def reward(self, q: Question, answer: float) -> float:
        return self.score(answer, q.actual, q.lo, q.hi)

    def rollout(self, policy: Policy) -> dict[str, float]:
        """Score a policy over every question — the mean reward is what RLVR maximises."""
        rewards = [self.reward(q, policy(q)) for q in self._questions]
        return {"mean_reward": sum(rewards) / len(rewards), "n": float(len(rewards))}
