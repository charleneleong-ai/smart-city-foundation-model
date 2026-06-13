"""Gridded ERA5 adapter via the ECMWF Data Stores client (CDS).

Unlike the per-point Open-Meteo adapter, ERA5 is gridded: one CDS request returns a whole
lat/lon NetCDF (no per-location rate limit), so thousands of H3 cells come from a single
download. ERA5 is 0.25° (~28 km) native, so it suits regional/continental density (~H3
res 5–6); fine city detail still wants Open-Meteo.

Live fetch needs a free CDS account + API key in `~/.ecmwfdatastoresrc` (the datastores
client's config, *not* the legacy `~/.cdsapirc`) and the dataset's licence accepted once
(`Client.accept_licence`). It is queue-based (async) — hence the on-disk NetCDF reuse here
plus the registry-level CachingAdapter. Heavy deps
(`ecmwf-datastores-client`, `xarray`, `netcdf4`) are in the `gridded` extra and imported
lazily, so importing this module is cheap and `_sample` is testable on a synthetic grid.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from sctwin.geo import Cell, center_of
from sctwin.schema import empty_frame, validate_frame

DATASET = "reanalysis-era5-single-levels"
_CDS_VAR = "2m_temperature"  # ERA5 request name; NetCDF exposes it as a single data var (Kelvin)


def _sample(ds, cells: list[Cell]) -> pl.DataFrame:
    """Sample each cell's centroid to the nearest grid point of an ERA5 NetCDF Dataset,
    converting Kelvin -> Celsius. Pure: takes an open xarray Dataset, returns canonical."""
    import pandas as pd  # heavy dep, only present when xarray is

    var = next(iter(ds.data_vars))
    tname = "valid_time" if "valid_time" in ds.variables else "time"
    times = [pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc) for t in ds[tname].values]

    frames = []
    for c in cells:
        clat, clon = center_of(c)
        kelvin = ds[var].sel(latitude=clat, longitude=clon, method="nearest").values
        frames.append(
            pl.DataFrame(
                {
                    "cell": c.h3,
                    "time": times,
                    "layer": "t2m",
                    "value": [float(v) - 273.15 for v in kelvin],
                }
            )
        )
    return validate_frame(pl.concat(frames))


class ERA5Adapter:
    name = "weather.t2m"

    def __init__(self, dataset: str = DATASET, workdir: str | Path = ".era5") -> None:
        self._dataset = dataset
        self._workdir = Path(workdir)

    def _area(self, cells: list[Cell], pad: float = 0.3) -> list[float]:
        lats = [center_of(c)[0] for c in cells]
        lons = [center_of(c)[1] for c in cells]
        return [max(lats) + pad, min(lons) - pad, min(lats) - pad, max(lons) + pad]  # N, W, S, E

    def _download(self, cells: list[Cell], start: datetime, end: datetime) -> Path:
        from ecmwf.datastores import Client  # needs ~/.cdsapirc (CDS API key)

        self._workdir.mkdir(parents=True, exist_ok=True)
        area = self._area(cells)
        key = hashlib.md5(repr(area).encode()).hexdigest()[:8]  # area in the key — distinct grids
        out = self._workdir / f"era5_{start.date()}_{end.date()}_{key}.nc"
        if out.exists():
            return out
        request = {
            "product_type": "reanalysis",
            "variable": _CDS_VAR,
            "date": f"{start.date()}/{end.date()}",
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "data_format": "netcdf",
        }
        Client().retrieve(self._dataset, request, str(out))  # submits, waits in queue, downloads
        return out

    def fetch(self, cells: list[Cell], start: datetime, end: datetime) -> pl.DataFrame:
        if not cells:
            return empty_frame()
        import xarray as xr

        with xr.open_dataset(self._download(cells, start, end)) as ds:
            return _sample(ds, cells)
