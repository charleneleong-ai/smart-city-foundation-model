from datetime import datetime

import polars as pl
import pytest

from sctwin.reason.intervention import Intervention, InterventionEnvironment, InterventionQuestion
from sctwin.reason.intervention_policy import (
    intervention_prompt,
    llm_policy,
    make_reward_fn,
    oracle_effect,
    training_records,
    zero_effect,
)

_C = "abc"
_IV = Intervention("retrofit", _C, factor=0.5, metric="mean")
_Q = InterventionQuestion(_IV, true_delta=-7.0, scale=10.0)


def _boxed(generate_text: str):
    return lambda _prompt: generate_text  # an injectable stub "LLM"


@pytest.fixture
def retrofit_env() -> InterventionEnvironment:
    # a 1-cell retrofit env whose hidden true Δ(mean) = -7 (load = 100 + 2*HDD, factor 0.5)
    times = [datetime(2023, 1, 1, h) for h in range(4)]
    demand = pl.DataFrame(
        {"cell": _C, "time": times, "layer": "load", "value": [116.0, 108.0, 132.0, 100.0]}
    )
    weather = pl.DataFrame(
        {"cell": _C, "time": times, "layer": "weather.t2m", "value": [10.0, 14.0, 2.0, 18.0]}
    )
    return InterventionEnvironment(demand, weather, [Intervention("retrofit", _C, 0.5, "mean")])


def test_prompt_describes_the_intervention_and_asks_for_boxed_delta():
    p = intervention_prompt(_Q)
    assert "retrofit" in p and _C in p and "mean" in p and "0.50" in p
    assert "\\boxed{" in p  # the parseable answer format the reward depends on


class TestLLMPolicy:
    """An LLM (any text generator) acts as the env's policy via boxed-Δ parsing."""

    def test_parses_boxed_delta(self):
        assert llm_policy(_boxed("...so \\boxed{-7.0}"))(_Q) == -7.0

    def test_unparseable_returns_wrong_floor(self):
        assert llm_policy(_boxed("I cannot tell"), wrong=0.0)(_Q) == 0.0

    def test_drives_the_environment_to_full_reward(self, retrofit_env):
        # an LLM that answers the hidden true Δ scores ~1.0 through the env
        out = retrofit_env.rollout(llm_policy(_boxed("answer: \\boxed{-7}")))
        assert out["mean_reward"] == pytest.approx(1.0)


def test_baselines_bracket_the_reward():
    assert oracle_effect(_Q) == _Q.true_delta and zero_effect(_Q) == 0.0


class TestRewardFn:
    """The TRL reward fn scores parsed completions against the hidden true Δ + scale columns."""

    def test_scores_correct_wrong_and_unparseable(self):
        reward_fn = make_reward_fn(wrong=0.0)
        completions = [
            "\\boxed{-7}",
            "\\boxed{7}",
            "no number here",
        ]  # right, wrong-sign, unparseable
        rewards = reward_fn(completions, true_delta=[-7.0, -7.0, -7.0], scale=[10.0, 10.0, 10.0])
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] < 0.5  # right magnitude, wrong sign -> below the 0.5 sign floor
        assert rewards[2] == 0.0

    def test_handles_chat_format_completions(self):
        reward_fn = make_reward_fn()
        rewards = reward_fn(
            [[{"role": "assistant", "content": "\\boxed{-7}"}]], true_delta=[-7.0], scale=[10.0]
        )
        assert rewards[0] == pytest.approx(1.0)  # the chat-format unwrap path is exercised


def test_training_records_carry_prompt_and_hidden_targets(retrofit_env):
    [row] = training_records(retrofit_env)
    assert set(row) == {"prompt", "true_delta", "scale"}
    assert "retrofit" in row["prompt"] and row["true_delta"] == pytest.approx(-7.0)
