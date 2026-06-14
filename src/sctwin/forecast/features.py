from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

BASE_TEMP_C = 18.0
LAGS = (1, 24)  # hourly default: previous hour + same hour yesterday
BASE_FEATURES = ["t2m", "hdd", "cdd", "hour", "dow", "month"]  # weather + calendar
CALENDAR_BASE = ["hour", "dow", "month"]

# (previous, seasonal) lags per forecast frequency — last step + same-step-one-cycle-ago
LAGS_BY_FREQ = {"hour": (1, 24), "day": (1, 7), "week": (1, 52), "month": (1, 12)}
_EVERY = {"hour": "1h", "day": "1d", "week": "1w", "month": "1mo"}


def feature_cols(lags: tuple[int, ...] = LAGS, *, weather: bool = True) -> list[str]:
    """The model's feature columns for the given lags (and with/without weather)."""
    return [*(BASE_FEATURES if weather else CALENDAR_BASE), *(f"y_lag_{lag}" for lag in lags)]


FEATURE_COLS = feature_cols(LAGS)  # hourly, with weather (back-compat)
CALENDAR_COLS = feature_cols(LAGS, weather=False)  # hourly, no weather (back-compat)


def resample(frame: pl.DataFrame, freq: str, *, agg: str = "mean") -> pl.DataFrame:
    """Aggregate a canonical (cell, time, layer, value) frame to a coarser frequency — `sum` for
    flows (demand), `mean` for levels (temperature). 'hour' is a no-op (already hourly)."""
    if freq == "hour":
        return frame
    reducer = pl.col("value").sum() if agg == "sum" else pl.col("value").mean()
    return (
        frame.sort(["cell", "layer", "time"])
        .group_by_dynamic("time", every=_EVERY[freq], group_by=["cell", "layer"])
        .agg(reducer.alias("value"))
        .select("cell", "time", "layer", "value")
    )


def regularize(frame: pl.DataFrame, freq: str) -> pl.DataFrame:
    """Put *all* (cell) series on one shared, gap-free grid at `freq` — Chronos `predict_df`
    needs every id on the same regular index. Clip to the window common to all cells, build the
    full grid, and forward-fill the value (persistence imputation for real-meter dropouts /
    ragged date ranges). Returns empty if the cells share no overlapping window."""
    if frame.is_empty():
        return frame
    bounds = frame.group_by("cell").agg(pl.col("time").min().alias("lo"), pl.col("time").max().alias("hi"))
    common_lo, common_hi = cast(datetime, bounds["lo"].max()), cast(datetime, bounds["hi"].min())
    if common_lo > common_hi:  # no window common to all cells
        return frame.clear()
    # forward-fill over the union range (so the fill sees readings before the common window), then clip
    grid = pl.datetime_range(
        cast(datetime, bounds["lo"].min()), cast(datetime, bounds["hi"].max()),
        interval=_EVERY[freq], time_zone="UTC", eager=True,
    ).dt.cast_time_unit("us")
    full = frame.select("cell").unique().join(pl.DataFrame({"time": grid}), how="cross")
    return (
        full.join(frame, on=["cell", "time"], how="left")
        .sort("cell", "time")
        .with_columns(pl.col("value").forward_fill().over("cell"))
        .filter((pl.col("time") >= common_lo) & (pl.col("time") <= common_hi))
        .drop_nulls("value")
        .with_columns(pl.lit("load").alias("layer"))
        .select("cell", "time", "layer", "value")
    )


def _lagged(df: pl.DataFrame, lags: tuple[int, ...]) -> pl.DataFrame:
    df = df.with_columns(*[pl.col("y").shift(lag).over("cell").alias(f"y_lag_{lag}") for lag in lags])
    return df.drop_nulls(subset=[f"y_lag_{lag}" for lag in lags])


def build_supervised(target: pl.DataFrame, weather: pl.DataFrame, *, lags: tuple[int, ...] = LAGS) -> pl.DataFrame:
    tgt = target.select("cell", "time", pl.col("value").alias("y"))
    wx = weather.select("cell", "time", pl.col("value").alias("t2m"))
    df = tgt.join(wx, on=["cell", "time"], how="inner").sort("cell", "time")
    df = df.with_columns(
        pl.max_horizontal(BASE_TEMP_C - pl.col("t2m"), 0.0).alias("hdd"),
        pl.max_horizontal(pl.col("t2m") - BASE_TEMP_C, 0.0).alias("cdd"),
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.weekday().alias("dow"),
        pl.col("time").dt.month().alias("month"),
    )
    return _lagged(df, lags)


def build_calendar_supervised(target: pl.DataFrame, *, lags: tuple[int, ...] = LAGS) -> pl.DataFrame:
    """Calendar + lag features only (no weather) — for demand series not geo-aligned to weather."""
    df = target.select("cell", "time", pl.col("value").alias("y")).sort("cell", "time")
    df = df.with_columns(
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.weekday().alias("dow"),
        pl.col("time").dt.month().alias("month"),
    )
    return _lagged(df, lags)


def to_xy(frame: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return frame.select(cols).to_numpy(), frame["y"].to_numpy()
