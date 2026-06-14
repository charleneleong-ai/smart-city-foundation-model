"""Derived demand channels — the physical-AI-facing predictions an embodied fleet acts on
(charging operators, depot planners, grid controllers plan against demand, not weather).

Also the loader for *real* demand: Monash electricity_hourly (321 real meters, hourly), so the
GBM-vs-Chronos comparison runs on genuine load instead of a synthetic sinusoid.
"""

import math
from datetime import datetime

import numpy as np
import polars as pl

from sctwin.geo import Cell, center_of

# Monash electricity_hourly (real hourly load, 321 meters, 2012-2015) via the Chronos datasets repo
ELECTRICITY_URL = (
    "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
    "monash_electricity_hourly/train-00000-of-00001.parquet"
)


def ev_charging_load(weather: pl.DataFrame, res: int, *, seed: int = 1) -> pl.DataFrame:
    """EV-charging demand (kW) per (cell, time) derived from 2 m temperature: an evening-peaked
    charging profile (people plug in after the commute), amplified in the cold (more driving +
    earlier returns + battery/heater draw), scaled by a per-cell fleet-size proxy. Canonical
    (cell, time, layer, value); non-negative."""
    cells = weather["cell"].unique().to_list()
    lon = {c: center_of(Cell(c, res))[1] for c in cells}
    lo, hi = min(lon.values()), max(lon.values())
    fleet = {c: 0.5 + 7.0 * ((lon[c] - lo) / ((hi - lo) or 1.0)) for c in cells}  # local EV fleet proxy
    rng = np.random.default_rng(seed)
    rows = weather.select("cell", "time", "value").to_dict(as_series=False)  # value = 2 m temp (°C)
    kw = [
        max(
            0.0,
            fleet[c] * 6.0
            * math.exp(-((t.hour - 19) ** 2) / 18.0)  # Gaussian evening peak (~19:00, ~3 h wide)
            * (1.0 + 0.05 * max(18.0 - temp, 0.0))  # cold amplification via heating-degrees
            + rng.normal(0, 0.4),
        )
        for c, t, temp in zip(rows["cell"], rows["time"], rows["value"], strict=True)
    ]
    return pl.DataFrame({"cell": rows["cell"], "time": rows["time"], "layer": "ev_charging", "value": kw})


def electricity_to_long(raw: pl.DataFrame, *, start: datetime, end: datetime, n_meters: int) -> pl.DataFrame:
    """Reshape the Monash electricity_hourly parquet (id, timestamp[], target[]) into canonical
    (cell, time, layer, value) for the first `n_meters` meters, windowed to [start, end]. Each
    real meter is its own 'cell' — real, heterogeneous load, no synthetic geography."""
    long = (
        raw.head(n_meters)
        .explode(["timestamp", "target"])
        .rename({"id": "cell", "timestamp": "time", "target": "value"})
        .filter((pl.col("time") >= start) & (pl.col("time") <= end))
    )
    return long.with_columns(pl.lit("load").alias("layer")).select("cell", "time", "layer", "value")
