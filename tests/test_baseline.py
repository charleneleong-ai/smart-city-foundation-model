from datetime import datetime, timezone

import polars as pl

from sctwin.reason.baseline import constant, evaluate, interval_centre, oracle, reference_policies
from sctwin.reason.environment import ReasoningEnvironment


def _env() -> ReasoningEnvironment:
    t = datetime(2020, 1, 1, tzinfo=timezone.utc)
    # interval centred 1 off the actual (a decent-but-imperfect forecast); mean is far for the tails
    frame = pl.DataFrame(
        {"cell": ["a", "b", "c"], "time": [t] * 3, "y_true": [10.0, 20.0, 30.0],
         "lo": [8.0, 18.0, 28.0], "hi": [14.0, 24.0, 34.0]}
    )
    return ReasoningEnvironment(frame, scale=5.0)


def test_baseline_table_ranks_oracle_over_forecast_over_constant():
    env = _env()
    scores = evaluate(env, reference_policies(env))
    assert scores["oracle (perfect)"] > scores["forecast (interval centre)"] > scores["constant (mean)"]
    assert scores["oracle (perfect)"] > 0.99  # exact + inside interval


def test_interval_centre_is_covered_but_constant_mean_escapes_the_band():
    env = _env()
    q = env.questions()[0]  # actual 10, band [8,14] -> centre 11 covered; mean 20 outside
    assert env.reward(q, interval_centre(q)) > env.reward(q, constant(20.0)(q))
    assert env.reward(q, oracle(q)) == 1.0  # exact actual + inside interval -> ceiling
