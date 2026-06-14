"""Demand adapters — one per source, all conforming to `LayerAdapter`
(`fetch(cells, start, end) -> canonical (cell, time, layer, value)`), so any region's demand
plugs into the same forecast → verify → baseline pipeline as the weather layers do.

Weather is already global (Open-Meteo / ERA5 cover the planet); demand is the only
region-specific piece, so this is the seam that makes the twin portable to any city: add one
adapter per source. Today: real research datasets (Low Carbon London, Monash electricity) from
the Chronos datasets parquet. The same interface fits grid-operator APIs as drop-in adapters —
EIA (US balancing authorities), ENTSO-E (EU bidding zones), NESO (GB), AEMO (AU).
"""

from collections.abc import Callable
from datetime import datetime

import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.demand import (
    ELECTRICITY_URL,
    LONDON_SMART_METERS_URL,
    electricity_to_long,
    london_smart_meters_to_long,
)
from sctwin.geo import Cell


class LondonSmartMeterAdapter:
    """Real London household load (Low Carbon London) mapped onto the requested cells — pairs
    with that cell's weather for a weather-coupled forecast."""

    name = "demand.load"

    def __init__(self, url: str = LONDON_SMART_METERS_URL) -> None:
        self._url = url

    def _read(self, n: int) -> pl.DataFrame:
        return pl.scan_parquet(self._url).head(n).collect()  # slice pushdown — no full download

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        return london_smart_meters_to_long(self._read(len(cells)), cells, start=start, end=end)


class ElectricityMeterAdapter:
    """Real heterogeneous load (Monash electricity_hourly), one real meter per requested cell."""

    name = "demand.load"

    def __init__(self, url: str = ELECTRICITY_URL) -> None:
        self._url = url

    def _read(self) -> pl.DataFrame:
        return pl.read_parquet(self._url)

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        # the source timestamps are tz-naive; filter naive, then normalise to UTC like the weather frames
        long = electricity_to_long(
            self._read(), start=start.replace(tzinfo=None), end=end.replace(tzinfo=None), n_meters=len(cells)
        )
        meters = long["cell"].unique().sort().to_list()
        remap = pl.DataFrame({"cell": meters, "_c": [c.h3 for c in cells[: len(meters)]]})  # meter id -> cell
        return long.join(remap, on="cell").select(
            pl.col("_c").alias("cell"),
            pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
            "layer", "value",
        )


_ADAPTERS: dict[str, Callable[[], LayerAdapter]] = {
    "london": LondonSmartMeterAdapter,
    "electricity": ElectricityMeterAdapter,
}


def demand_source(name: str) -> LayerAdapter:
    """A demand adapter by name — the per-region selector (mirrors the weather --source switch)."""
    if name not in _ADAPTERS:
        raise ValueError(f"demand source must be one of {', '.join(_ADAPTERS)}")
    return _ADAPTERS[name]()
