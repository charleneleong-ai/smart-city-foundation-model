from datetime import datetime, timedelta, timezone

import polars as pl

from sctwin.forecast.features import (
    BASE_FEATURES,
    CALENDAR_BASE,
    FEATURE_COLS,
    build_calendar_supervised,
    feature_cols,
    regularize,
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


def test_regularize_fills_within_series_gaps_with_forward_fill():
    t0 = datetime(2013, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame(  # cell "a" is missing hour 2
        {"cell": ["a", "a", "a"], "time": [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=3)],
         "layer": ["load"] * 3, "value": [10.0, 11.0, 13.0]}
    )
    out = regularize(frame, "hour").sort("time")
    assert out.height == 4  # the hour-2 gap is filled onto a complete grid
    assert out["value"].to_list() == [10.0, 11.0, 11.0, 13.0]  # hour 2 forward-filled from hour 1


def test_regularize_aligns_ragged_cells_to_one_shared_grid():
    t0 = datetime(2013, 1, 1, tzinfo=timezone.utc)

    def h(n: int) -> datetime:
        return t0 + timedelta(hours=n)

    frame = pl.DataFrame(  # cell a spans hours 0..3, cell b spans 2..5 -> common window is 2..3
        {"cell": ["a", "a", "b", "b"], "time": [h(0), h(3), h(2), h(5)],
         "layer": ["load"] * 4, "value": [1.0, 4.0, 20.0, 50.0]}
    )
    out = regularize(frame, "hour")
    hours = {(r["cell"], r["time"].hour) for r in out.iter_rows(named=True)}
    assert hours == {("a", 2), ("a", 3), ("b", 2), ("b", 3)}  # both cells, same 2-hour grid
    assert out.filter((pl.col("cell") == "a") & (pl.col("time") == h(2)))["value"].item() == 1.0  # ffill from h0


def test_build_supervised_uses_frequency_aware_lags():
    daily = resample(_hourly(value=2.0, hours=24 * 14), "day", agg="sum")  # 14 daily points
    sup = build_calendar_supervised(daily, lags=(1, 7))
    assert "y_lag_7" in sup.columns and "y_lag_24" not in sup.columns  # weekly lag, not the hourly one
    assert sup.height == 14 - 7  # 7 days dropped for the weekly-lag warmup
