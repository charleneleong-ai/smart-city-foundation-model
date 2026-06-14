"""Derived demand channels — the physical-AI-facing predictions an embodied fleet acts on
(charging operators, depot planners, grid controllers plan against demand, not weather).

Also the loaders for *real* demand: Monash electricity_hourly (321 meters) and Low Carbon
London smart-meter households (overlaid with real London weather), so the GBM-vs-Chronos
comparison runs on genuine load instead of a synthetic sinusoid.
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
# Low Carbon London smart-meter households (real UK load, half-hourly, 2012-2014) via Chronos datasets
LONDON_SMART_METERS_URL = (
    "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/"
    "monash_london_smart_meters/train-00000-of-00003.parquet"
)
# AEMO NEM regional demand (real AU load, MW) — public monthly CSV per region (NSW1, VIC1, ...)
AEMO_URL = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"


def _fleet(cells: list[str], res: int, population: dict[str, float] | None) -> dict[str, float]:
    """Per-cell EV-fleet scale. With GHSL population (geo_features.population_by_cell), fleet ∝
    the people actually in the cell; otherwise a synthetic west→east longitude gradient."""
    if population:
        pops = {c: population.get(c, 0.0) for c in cells}
        mean = (sum(pops.values()) / len(pops)) or 1.0
        return {c: 0.5 + 7.0 * (pops[c] / mean) for c in cells}  # fleet ~ population
    lon = {c: center_of(Cell(c, res))[1] for c in cells}
    lo, hi = min(lon.values()), max(lon.values())
    return {c: 0.5 + 7.0 * ((lon[c] - lo) / ((hi - lo) or 1.0)) for c in cells}


def ev_charging_load(
    weather: pl.DataFrame, res: int, *, population: dict[str, float] | None = None, seed: int = 1
) -> pl.DataFrame:
    """EV-charging demand (kW) per (cell, time) derived from 2 m temperature: an evening-peaked
    charging profile (people plug in after the commute), amplified in the cold (more driving +
    earlier returns + battery/heater draw), scaled by a per-cell fleet size. With `population`
    (real GHSL counts) the scale is the people in the cell; else a synthetic gradient. Canonical
    (cell, time, layer, value); non-negative."""
    cells = weather["cell"].unique().to_list()
    fleet = _fleet(cells, res, population)
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


def aemo_to_long(raw: pl.DataFrame, *, cell: str, start: datetime, end: datetime) -> pl.DataFrame:
    """AEMO price-and-demand CSV (REGION, SETTLEMENTDATE, TOTALDEMAND, …) → hourly canonical
    (cell, time, layer, value=MW). SETTLEMENTDATE is AEST (UTC+10); converted to UTC and averaged
    to the hour so it pairs with the (UTC) weather frame. Windowed to [start, end] on one cell —
    AEMO demand is a single regional aggregate."""
    return (
        raw.select(
            pl.col("SETTLEMENTDATE").str.to_datetime("%Y/%m/%d %H:%M:%S")
            .dt.replace_time_zone("Etc/GMT-10").dt.convert_time_zone("UTC")  # AEST (UTC+10) -> UTC
            .dt.truncate("1h").dt.cast_time_unit("us").alias("time"),
            pl.col("TOTALDEMAND").alias("value"),
        )
        .group_by("time")
        .agg(pl.col("value").mean())
        .filter((pl.col("time") >= start) & (pl.col("time") <= end))
        .with_columns(pl.lit(cell).alias("cell"), pl.lit("load").alias("layer"))
        .select("cell", "time", "layer", "value")
        .sort("time")
    )


def london_smart_meters_to_long(
    raw: pl.DataFrame, cells: list[Cell], *, start: datetime, end: datetime
) -> pl.DataFrame:
    """Real London household load (id, timestamp[], target[]) → hourly canonical (cell, time,
    layer, value). Each meter is assigned a distinct London `cell` so it pairs with that cell's
    weather; the half-hourly readings are averaged to the hour (UTC). Windowed to [start, end]."""
    ids = raw["id"].to_list()[: len(cells)]
    mapping = pl.DataFrame({"_id": ids, "cell": [c.h3 for c in cells[: len(ids)]]})
    long = (
        raw.head(len(cells))
        .explode(["timestamp", "target"])
        .select(
            pl.col("id").alias("_id"),
            pl.col("timestamp").dt.truncate("1h").dt.replace_time_zone("UTC")
            .dt.cast_time_unit("us").alias("time"),  # match the weather adapter's us precision for joins
            pl.col("target").alias("value"),
        )
        .filter(pl.col("value").is_finite())  # LCL meters have missing readings (NaN) — drop them
        .group_by("_id", "time")
        .agg(pl.col("value").mean())
        .join(mapping, on="_id")
        .filter((pl.col("time") >= start) & (pl.col("time") <= end))
    )
    return long.with_columns(pl.lit("load").alias("layer")).select("cell", "time", "layer", "value")
