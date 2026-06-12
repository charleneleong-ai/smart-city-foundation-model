from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from sctwin.forecast.backtest import backtest, metrics, temporal_split
from sctwin.forecast.baselines import DegreeDayRegressor


def _frame(n: int) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, size=n)
    return pl.DataFrame(
        {
            "cell": ["a"] * n,
            "time": [t0 + timedelta(hours=i) for i in range(n)],
            "y": 3.0 * hdd + rng.normal(0, 0.1, size=n),
            "hdd": hdd,
            "cdd": np.zeros(n),
        }
    )


def test_temporal_split_has_no_leakage():
    train, test = temporal_split(_frame(100), test_frac=0.25)
    assert train.height == 75 and test.height == 25
    assert train["time"].max() < test["time"].min()


def test_metrics_known_values():
    m = metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["mae"] == 2.0 / 3.0
    assert m["rmse"] == (4.0 / 3.0) ** 0.5


def test_backtest_returns_metrics_and_predictions():
    res = backtest(DegreeDayRegressor(), _frame(200), ["hdd", "cdd"], test_frac=0.25)
    assert res.metrics["mae"] < 0.5
    assert len(res.predictions) == 50
    # 200 hourly rows, 25% test -> split at the 150th hour
    assert res.split_time == datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=150)
