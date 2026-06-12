import numpy as np
import polars as pl

BASE_TEMP_C = 18.0
LAGS = [1, 24]
FEATURE_COLS = ["t2m", "hdd", "cdd", "hour", "dow", "month", "y_lag_1", "y_lag_24"]


def build_supervised(target: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
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
    df = df.with_columns(
        *[pl.col("y").shift(lag).over("cell").alias(f"y_lag_{lag}") for lag in LAGS]
    )
    return df.drop_nulls(subset=[f"y_lag_{lag}" for lag in LAGS])


def to_xy(frame: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return frame.select(feature_cols).to_numpy(), frame["y"].to_numpy()
