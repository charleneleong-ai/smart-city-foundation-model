from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.forecast.features import (
    BASE_FEATURES,
    CALENDAR_BASE,
    FEATURE_COLS,
    build_calendar_supervised,
    feature_cols,
    resample,
)


def _hourly(*, layer: str = "load", value: float = 1.0, hours: int = 48, cell: str = "a") -> pl.DataFrame:
    t0 = datetime(2013, 1, 1, tzinfo=timezone.utc)
    return pl.DataFrame(
        {"cell": [cell] * hours, "time": [t0 + timedelta(hours=h) for h in range(hours)],
         "layer": [layer] * hours, "value": [value] * hours}
    )


def test_feature_cols_track_lags_and_weather_flag():
    assert feature_cols((1, 24)) == FEATURE_COLS  # hourly default unchanged
    assert feature_cols((1, 7)) == [*BASE_FEATURES, "y_lag_1", "y_lag_7"]  # daily lags
    assert feature_cols((1, 7), weather=False) == [*CALENDAR_BASE, "y_lag_1", "y_lag_7"]


def test_resample_sums_demand_means_levels_and_no_ops_hourly():
    daily = resample(_hourly(value=1.0, hours=48), "day", agg="sum")
    assert daily.height == 2 and (daily["value"] == 24.0).all()  # 24 hourly -> 1 daily sum
    wx = _hourly(layer="t2m", hours=24).with_columns(pl.col("time").dt.hour().cast(pl.Float64).alias("value"))
    assert abs(resample(wx, "day", agg="mean")["value"][0] - 11.5) < 1e-9  # mean of 0..23
    assert resample(_hourly(), "hour").equals(_hourly())  # hourly is a no-op


def test_build_supervised_uses_frequency_aware_lags():
    daily = resample(_hourly(value=2.0, hours=24 * 14), "day", agg="sum")  # 14 daily points
    sup = build_calendar_supervised(daily, lags=(1, 7))
    assert "y_lag_7" in sup.columns and "y_lag_24" not in sup.columns  # weekly lag, not the hourly one
    assert sup.height == 14 - 7  # 7 days dropped for the weekly-lag warmup
