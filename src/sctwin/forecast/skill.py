"""Weather-forecast skill: score a forecast against realized observations (the cached ground
truth), versus the naive baselines any real forecast must beat. This is what turns a forecast
*source* (Open-Meteo NWP today, an AI weather FM later) into a *measured baseline*.

All frames are canonical (cell, time, layer, value). `truth` is the realized reanalysis/
observation; `forecast` is what was predicted for the same (cell, time).
"""

from datetime import timedelta

import numpy as np
import polars as pl


def skill(forecast: pl.DataFrame, truth: pl.DataFrame) -> dict[str, float]:
    """MAE / RMSE / bias of forecast vs realized truth, joined on (cell, time)."""
    j = forecast.join(truth.select("cell", "time", "value"), on=["cell", "time"], suffix="_true", validate="1:1")
    err = (j["value"] - j["value_true"]).to_numpy()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(err.mean()),
        "n": float(err.size),
    }


def persistence(truth: pl.DataFrame, *, lag_hours: int = 24) -> pl.DataFrame:
    """Naive forecast: the observation from `lag_hours` ago carried forward (per cell). Uses only
    past observations — the bar a real forecast must clear."""
    return truth.with_columns((pl.col("time") + timedelta(hours=lag_hours)).alias("time"))


def climatology(truth: pl.DataFrame) -> pl.DataFrame:
    """Forecast = the per-(cell, hour-of-day) mean — the diurnal climatology baseline."""
    hourly = truth.with_columns(pl.col("time").dt.hour().alias("_h"))
    means = hourly.group_by("cell", "_h").agg(pl.col("value").mean().alias("value"))
    return hourly.drop("value").join(means, on=["cell", "_h"]).drop("_h")


def benchmark(forecast: pl.DataFrame, truth: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Skill of the NWP forecast vs persistence vs climatology, all against realized truth."""
    return {
        "nwp forecast": skill(forecast, truth),
        "persistence (24 h)": skill(persistence(truth), truth),
        "climatology (hourly mean)": skill(climatology(truth), truth),
    }
