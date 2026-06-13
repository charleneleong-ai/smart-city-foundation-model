from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from sctwin.forecast.baselines import DegreeDayRegressor
from sctwin.verify.results import as_layer, verification_frame


def _frame(n: int) -> pl.DataFrame:
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(0)
    hdd = rng.uniform(0, 20, n)
    return pl.DataFrame(
        {
            "cell": ["a"] * n,
            "time": [t0 + timedelta(hours=i) for i in range(n)],
            "y": 3.0 * hdd + rng.normal(0, 0.3, n),
            "hdd": hdd,
            "cdd": np.zeros(n),
        }
    )


def test_results_frame_has_expected_columns_and_covered_logic():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"], alpha=0.1)
    for col in ["cell", "time", "y_true", "y_pred", "abs_error", "lo", "hi", "covered"]:
        assert col in res.columns
    # independent check: a point is covered iff its abs_error is within the interval
    # half-width — cross-checks covered against abs_error + (hi-lo), not the lo/hi formula
    chk = res.with_columns(
        (pl.col("abs_error") <= (pl.col("hi") - pl.col("lo")) / 2 + 1e-9).alias("exp")
    )
    assert (chk["covered"] == chk["exp"]).all()


def test_empirical_coverage_near_target():
    res = verification_frame(DegreeDayRegressor(), _frame(600), ["hdd", "cdd"], alpha=0.1)
    assert 0.8 <= res["covered"].mean() <= 1.0


def test_as_layer_projects_to_canonical_schema():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"])
    layer = as_layer(res, "abs_error")
    assert layer.columns == ["cell", "time", "layer", "value"]
    assert layer["layer"].unique().to_list() == ["abs_error"]


def test_as_layer_rejects_unknown_field():
    res = verification_frame(DegreeDayRegressor(), _frame(400), ["hdd", "cdd"])
    with pytest.raises(ValueError, match="unknown field"):
        as_layer(res, "nope")
