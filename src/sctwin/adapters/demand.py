"""Demand adapters — one per source, all conforming to `LayerAdapter`
(`fetch(cells, start, end) -> canonical (cell, time, layer, value)`), so any region's demand
plugs into the same forecast → verify → baseline pipeline as the weather layers do.

Weather is already global (Open-Meteo / ERA5 cover the planet); demand is the only
region-specific piece, so this is the seam that makes the twin portable to any city: add one
adapter per source. Today: real research datasets (Low Carbon London, Monash electricity) from
the Chronos datasets parquet. The same interface fits grid-operator APIs as drop-in adapters —
EIA (US balancing authorities), ENTSO-E (EU bidding zones), NESO (GB), AEMO (AU).
"""

import io
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.demand import (
    AEMO_URL,
    ELECTRICITY_MAPS_URL,
    ELECTRICITY_URL,
    LONDON_SMART_METERS_URL,
    aemo_to_long,
    el_maps_to_long,
    electricity_to_long,
    london_smart_meters_to_long,
)
from sctwin.geo import Cell, center_of

_FAR_PAST = datetime(1970, 1, 1, tzinfo=timezone.utc)  # accumulate every fetched row into the cache
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=timezone.utc)


def _months(start: datetime, end: datetime) -> list[str]:
    """The YYYYMM tags spanning [start, end] inclusive (AEMO ships one CSV per month)."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


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


class AEMODemandAdapter:
    """Real Australian NEM regional demand (AEMO) — a single aggregate series per region (MW),
    pinned to one cell (the region's city) so it pairs with that city's weather."""

    name = "demand.load"

    def __init__(self, region: str = "NSW1") -> None:
        self._region = region

    def _read(self, ym: str) -> pl.DataFrame:
        url = AEMO_URL.format(ym=ym, region=self._region)
        resp = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()  # AEMO's CDN 403s a bare request — needs the UA header
        return pl.read_csv(io.BytesIO(resp.content))

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        raw = pl.concat([self._read(ym) for ym in _months(start, end)])
        return aemo_to_long(raw, cell=cells[0].h3, start=start, end=end)  # one regional series -> one cell


class ElectricityMapsAdapter:
    """Real total load (MW) from Electricity Maps — ~200 zones globally, by zone code or the
    cell's lat/lon (geolocation). The free endpoint returns only the last 24 h, so fetch()
    *accumulates*: each pull is merged (deduped by timestamp) into a per-zone parquet cache, so
    repeated polling builds a growing real series. Needs ELECTRICITYMAPS_TOKEN in the env."""

    name = "demand.load"

    def __init__(
        self, zone: str = "GB", *, token: str | None = None, cache_dir: str = ".cache/electricitymaps"
    ) -> None:
        self._zone, self._cache = zone, Path(cache_dir)
        self._token = token or os.environ.get("ELECTRICITYMAPS_TOKEN", "")

    def _read(self, cell: Cell) -> list[dict]:
        lat, lon = center_of(cell)
        params = {"zone": self._zone} if self._zone else {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"}
        resp = httpx.get(ELECTRICITY_MAPS_URL, params=params, headers={"auth-token": self._token}, timeout=60.0)
        resp.raise_for_status()
        return resp.json().get("history", [])

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        fresh = el_maps_to_long(self._read(cells[0]), cell=cells[0].h3, start=_FAR_PAST, end=_FAR_FUTURE)
        path = self._cache / f"{self._zone or cells[0].h3}.parquet"
        prior = pl.read_parquet(path) if path.exists() else fresh.clear()
        merged = pl.concat([prior, fresh]).unique(subset=["time"], keep="last").sort("time")  # accumulate
        self._cache.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(path)
        return merged.filter((pl.col("time") >= start) & (pl.col("time") <= end))


_ADAPTERS: dict[str, Callable[[], LayerAdapter]] = {
    "london": LondonSmartMeterAdapter,
    "electricity": ElectricityMeterAdapter,
    "aemo": AEMODemandAdapter,
    "electricitymaps": ElectricityMapsAdapter,
}


def demand_source(name: str) -> LayerAdapter:
    """A demand adapter by name — the per-region selector (mirrors the weather --source switch)."""
    if name not in _ADAPTERS:
        raise ValueError(f"demand source must be one of {', '.join(_ADAPTERS)}")
    return _ADAPTERS[name]()
