from datetime import datetime, timezone

import polars as pl

from sctwin.reason.environment import ReasoningEnvironment
from sctwin.reason.grpo import make_reward_fn, parse_answer, prompt_for


def test_parse_answer_prefers_boxed_then_falls_back_to_last_number():
    assert parse_answer("...reasoning... \\boxed{42.5}") == 42.5
    assert parse_answer("the load is about 17 units") == 17.0
    assert parse_answer("first 3 then \\boxed{-8.2}") == -8.2
    assert parse_answer("no number at all") is None


def _env() -> ReasoningEnvironment:
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame({"cell": ["a"], "time": [t], "y_true": [50.0], "lo": [45.0], "hi": [55.0]})
    return ReasoningEnvironment(frame, scale=10.0)


def test_reward_fn_scores_right_answer_highest_and_punishes_unparseable():
    reward_fn = make_reward_fn(_env())
    completions = ["...\\boxed{50}", "...\\boxed{250}", "I am not sure"]
    r = reward_fn(completions, actual=[50.0] * 3, lo=[45.0] * 3, hi=[55.0] * 3)
    assert r[0] == 1.0  # exact + inside the interval
    assert r[0] > r[1] > r[2]  # far guess beats an unparseable answer
    assert r[2] == 0.0  # unparseable -> no reward


def test_prompt_carries_context_and_boxed_instruction():
    p = prompt_for("H3 cell a, hour 12, HDD 3.2")
    assert "boxed" in p and "H3 cell a" in p
