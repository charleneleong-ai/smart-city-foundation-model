from datetime import datetime, timezone

import numpy as np
import polars as pl

from sctwin.reason.environment import ReasoningEnvironment


def _results(n: int = 30) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    yt = np.random.default_rng(0).uniform(0, 100, n)
    return pl.DataFrame({"cell": ["a"] * n, "time": [t0] * n, "y_true": yt, "lo": yt - 5, "hi": yt + 5})


def test_better_policy_earns_higher_reward():
    env = ReasoningEnvironment(_results())
    perfect = env.rollout(lambda q: q.actual)  # answers the held-out actual exactly
    constant = env.rollout(lambda q: 0.0)  # ignores the question
    assert perfect["mean_reward"] > constant["mean_reward"]  # the RLVR signal discriminates
    assert perfect["mean_reward"] > 0.9
    assert perfect["n"] == 30.0


def test_reward_is_bounded_and_rewards_being_inside_the_interval():
    env = ReasoningEnvironment(_results())
    q = env.questions()[0]
    assert 0.0 <= env.reward(q, q.actual) <= 1.0
    assert env.reward(q, q.actual) > env.reward(q, q.hi + 100.0)  # exact+covered beats far+outside
