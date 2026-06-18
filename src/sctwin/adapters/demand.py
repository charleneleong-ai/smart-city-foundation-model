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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.demand import (
    AEMO_URL,
    EIA_URL,
    ELECTRICITY_MAPS_URL,
    ELECTRICITY_URL,
    ENTSOE_URL,
    LONDON_SMART_METERS_URL,
    aemo_to_long,
    eia_load_to_long,
    el_maps_to_long,
    electricity_to_long,
    entsoe_load_to_long,
    lcl_group_profile,
    london_smart_meters_to_long,
    need_measure_split,
)
from sctwin.geo import Cell, center_of

# ENTSO-E bidding-zone EIC codes (a starter set; the platform covers all of Europe)
_ENTSOE_EIC = {
    "GB": "10YGB----------A",
    "FR": "10YFR-RTE------C",
    "DE-LU": "10Y1001A1001A82H",
    "ES": "10YES-REE------0",
    "NL": "10YNL----------L",
    "BE": "10YBE----------2",
    "IT-NORD": "10Y1001A1001A73I",
    "PL": "10YPL-AREA-----S",
    "SE-SE3": "10Y1001A1001A46L",
}


def _entsoe_windows(start: datetime, end: datetime) -> list[tuple[str, str]]:
    """(periodStart, periodEnd) yyyyMMddHHmm windows of ≤1 year — ENTSO-E caps each request."""
    out, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=365), end)
        out.append((cur.strftime("%Y%m%d%H%M"), nxt.strftime("%Y%m%d%H%M")))
        cur = nxt
    return out


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
            self._read(),
            start=start.replace(tzinfo=None),
            end=end.replace(tzinfo=None),
            n_meters=len(cells),
        )
        meters = long["cell"].unique().sort().to_list()
        remap = pl.DataFrame(
            {"cell": meters, "_c": [c.h3 for c in cells[: len(meters)]]}
        )  # meter id -> cell
        return long.join(remap, on="cell").select(
            pl.col("_c").alias("cell"),
            pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"),
            "layer",
            "value",
        )


class AEMODemandAdapter:
    """Real Australian NEM regional demand (AEMO) — a single aggregate series per region (MW),
    pinned to one cell (the region's city) so it pairs with that city's weather."""

    name = "demand.load"

    def __init__(self, region: str = "NSW1") -> None:
        self._region = region

    def _read(self, ym: str) -> pl.DataFrame:
        url = AEMO_URL.format(ym=ym, region=self._region)
        resp = httpx.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60.0, follow_redirects=True
        )
        resp.raise_for_status()  # AEMO's CDN 403s a bare request — needs the UA header
        return pl.read_csv(io.BytesIO(resp.content))

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        raw = pl.concat([self._read(ym) for ym in _months(start, end)])
        return aemo_to_long(
            raw, cell=cells[0].h3, start=start, end=end
        )  # one regional series -> one cell


class ElectricityMapsAdapter:
    """Real total load (MW) from Electricity Maps — ~200 zones globally, by zone code or the
    cell's lat/lon (geolocation). The free endpoint returns only the last 24 h, so fetch()
    *accumulates*: each pull is merged (deduped by timestamp) into a per-zone parquet cache, so
    repeated polling builds a growing real series. Needs ELECTRICITYMAPS_TOKEN in the env."""

    name = "demand.load"

    def __init__(
        self,
        zone: str = "GB",
        *,
        token: str | None = None,
        cache_dir: str = ".cache/electricitymaps",
    ) -> None:
        self._zone, self._cache = zone, Path(cache_dir)
        self._token = token or os.environ.get("ELECTRICITYMAPS_TOKEN", "")

    def _read(self, cell: Cell) -> list[dict]:
        lat, lon = center_of(cell)
        params = {"zone": self._zone} if self._zone else {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"}
        resp = httpx.get(
            ELECTRICITY_MAPS_URL, params=params, headers={"auth-token": self._token}, timeout=60.0
        )
        resp.raise_for_status()
        return resp.json().get("history", [])

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        fresh = el_maps_to_long(
            self._read(cells[0]), cell=cells[0].h3, start=_FAR_PAST, end=_FAR_FUTURE
        )
        path = self._cache / f"{self._zone or cells[0].h3}.parquet"
        prior = pl.read_parquet(path) if path.exists() else fresh.clear()
        merged = (
            pl.concat([prior, fresh]).unique(subset=["time"], keep="last").sort("time")
        )  # accumulate
        self._cache.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(path)
        return merged.filter((pl.col("time") >= start) & (pl.col("time") <= end))


class ENTSOEAdapter:
    """Real EU actual total load (MW) from the ENTSO-E Transparency Platform — multi-year history
    by bidding zone (free security token), looped in ≤1-year requests. Pinned to one cell."""

    name = "demand.load"

    def __init__(self, zone: str = "GB", *, token: str | None = None) -> None:
        self._zone = zone
        self._token = token or os.environ.get("ENTSOE_TOKEN", "")

    def _read(self, period_start: str, period_end: str) -> str:
        params = {
            "securityToken": self._token,
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": _ENTSOE_EIC[self._zone],
            "periodStart": period_start,
            "periodEnd": period_end,
        }
        resp = httpx.get(ENTSOE_URL, params=params, timeout=120.0)
        resp.raise_for_status()
        return resp.text

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        windows = _entsoe_windows(start, end)
        if not windows:
            return entsoe_load_to_long("<empty/>", cell=cells[0].h3)
        out = pl.concat(
            [entsoe_load_to_long(self._read(ps, pe), cell=cells[0].h3) for ps, pe in windows]
        )
        return (
            out.filter((pl.col("time") >= start) & (pl.col("time") <= end))
            .unique("time", keep="last")
            .sort("time")
        )


class EIADemandAdapter:
    """Real US hourly demand (MW) from the EIA Hourly Electric Grid Monitor — multi-year history
    by balancing authority (free API key), paged in ≤5000-row requests. Pinned to one cell.
    `respondent` is a BA code (CISO, PJM, MISO, ERCO, ISNE, NYIS, …); type=D is demand."""

    name = "demand.load"
    _PAGE = 5000  # EIA v2 caps each request at 5000 rows

    def __init__(self, respondent: str = "CISO", *, token: str | None = None) -> None:
        self._respondent = respondent
        self._token = token or os.environ.get("EIA_API_KEY", "")

    def _read(self, start: datetime, end: datetime) -> list[dict]:
        rows: list[dict] = []
        while True:
            params = {
                "api_key": self._token,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": self._respondent,
                "facets[type][]": "D",
                "start": start.strftime("%Y-%m-%dT%H"),
                "end": end.strftime("%Y-%m-%dT%H"),
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": str(len(rows)),
                "length": str(self._PAGE),
            }
            resp = httpx.get(EIA_URL, params=params, timeout=120.0)
            resp.raise_for_status()
            page = resp.json().get("response", {}).get("data", [])
            rows.extend(page)
            if len(page) < self._PAGE:
                return rows  # last (partial) page reached

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        out = eia_load_to_long(
            self._read(start, end), cell=cells[0].h3
        )  # one BA series -> one cell
        return out.filter((pl.col("time") >= start) & (pl.col("time") <= end))


# --- Natural-experiment oracles (real measured Δ for the interventional env) -------------------
# These return a before/after *pair* of profiles (not a canonical `fetch` series), so they're not
# LayerAdapters / not in `_ADAPTERS`; pair them with `reason.intervention.measured_question`.


class LCLTariffAdapter:
    """Real Low Carbon London dynamic-ToU trial (2013): mean half-hourly load profiles for the
    treated (ToU) and control (Std) groups — the natural-experiment ground truth for the *tariff*
    lever. The Datastore CSV (~760 MB, CC-BY) is downloaded once; pass its local path as `source`.
    The kWh column name / `stdorToU` labels are the published LCL schema — confirm on download.
    https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households/"""

    def __init__(self, source: str) -> None:
        self._source = source  # local path to the downloaded Datastore CSV

    def _read(self) -> pl.DataFrame:
        return pl.read_csv(self._source).rename({"KWH/hh (per half hour) ": "value"})

    def profiles(self, cell: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(control Std, treated ToU) mean profiles → feed to measured_question('tariff', ...)."""
        raw = self._read()
        return lcl_group_profile(raw, "Std", cell=cell), lcl_group_profile(raw, "ToU", cell=cell)


class NEEDRetrofitAdapter:
    """Real UK NEED (DESNZ) before/after: pre- vs post-measure annual consumption for properties
    that installed an efficiency measure — the natural-experiment ground truth for the *retrofit*
    lever. Open anonymised sample (50k/4M rows); pass its local path as `source`. NEED column names
    vary by release — set measure/pre/post cols to match the data dictionary.
    https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework"""

    def __init__(
        self,
        source: str,  # local path to the downloaded NEED sample CSV
        *,
        measure_col: str = "LOFT_FLAG",
        pre_col: str = "Econ2010",
        post_col: str = "Econ2013",
    ) -> None:
        self._source, self._measure, self._pre, self._post = source, measure_col, pre_col, post_col

    def _read(self) -> pl.DataFrame:
        return pl.read_csv(self._source)

    def split(self, cell: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        """(pre, post) annual-consumption frames → feed to measured_question('retrofit', ...)."""
        return need_measure_split(
            self._read(),
            measure_col=self._measure,
            pre_col=self._pre,
            post_col=self._post,
            cell=cell,
        )


_ADAPTERS: dict[str, Callable[[], LayerAdapter]] = {
    "london": LondonSmartMeterAdapter,
    "electricity": ElectricityMeterAdapter,
    "aemo": AEMODemandAdapter,
    "electricitymaps": ElectricityMapsAdapter,
    "entsoe": ENTSOEAdapter,
    "eia": EIADemandAdapter,
}


def demand_source(name: str) -> LayerAdapter:
    """A demand adapter by name — the per-region selector (mirrors the weather --source switch)."""
    if name not in _ADAPTERS:
        raise ValueError(f"demand source must be one of {', '.join(_ADAPTERS)}")
    return _ADAPTERS[name]()
