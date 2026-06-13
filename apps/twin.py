"""Build map payloads for the 3D twin viewer: weather inputs and SP5 verification.

Each builder returns a `map` dict consumed by render_3d.to_self_contained_html — name,
view (lat/lon/zoom/pitch), elevation, and a list of selectable layers (each a time-series
of canonical (cell, time, layer, value) records).
"""

import math
from datetime import datetime, timedelta, timezone

import h3
import numpy as np
import polars as pl

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.geo import Cell, center_of
from sctwin.registry import Registry
from sctwin.verify.results import as_layer, verification_frame

from presets import bbox_and_zoom

MAX_CELLS = 400  # one Open-Meteo call per cell

# energy-domain field -> (display name, unit, range mode): zero=[0,max]; sym=[-M,M]; auto
_ENERGY_LAYERS = [
    ("y_true", "demand", "load", "auto"),
    ("y_pred", "forecast", "load", "auto"),
    ("abs_error", "|error|", "load", "zero"),
    ("error", "delta (forecast−actual)", "load", "sym"),
    ("covered", "covered", "in/out", "auto"),
]


def _resolve(preset: dict, radius: float | None, res: int | None) -> tuple[list, float, int]:
    south, west, north, east, zoom, r = bbox_and_zoom(preset, radius, res)
    cells = cells_in_bbox(south, west, north, east, r)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS} (one API call each); adjust --radius/--res")
    return cells, zoom, r


def _weather(cells: list, start: datetime, end: datetime) -> pl.DataFrame:
    reg = Registry()
    reg.register(OpenMeteoWeatherAdapter())
    print(f"fetching {len(cells)} cells {start.date()}..{end.date()} ...")
    return reg.get("weather.t2m", cells, start, end)


def _frames(frame: pl.DataFrame, times: list, vmin: float, vmax: float, fmt: str) -> list[dict]:
    return [
        {"label": t.strftime(fmt), "records": h3_layer_records(frame, at=t, vmin=vmin, vmax=vmax)}
        for t in times
    ]


def _view(preset: dict, zoom: float, res: int) -> dict:
    return {
        "lat": preset["lat"], "lon": preset["lon"], "zoom": zoom, "pitch": preset.get("pitch", 50.0),
        "elevation_scale": 4.0 * h3.average_hexagon_edge_length(res, unit="m"),
    }


def weather_map(name: str, preset: dict, date: str, *, radius=None, res=None) -> dict:
    cells, zoom, r = _resolve(preset, radius, res)
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    wx = _weather(cells, day, day)
    hours = wx.select(pl.col("time").unique().sort()).to_series().to_list()
    hdd = wx.with_columns(pl.max_horizontal(18.0 - pl.col("value"), 0.0).alias("value"))

    def layer(nm: str, unit: str, f: pl.DataFrame) -> dict:
        vmin, vmax = float(f["value"].min()), float(f["value"].max())
        return {"name": nm, "unit": unit, "frames": _frames(f, hours, vmin, vmax, "%H:%M")}

    return {
        "name": name, "subtitle": f"Open-Meteo · {date} · H3 res {r} · 24 h", **_view(preset, zoom, r),
        "layers": [layer("2m temperature", "°C", wx), layer("heating degrees", "°C", hdd)],
    }


def _synth_load(wx: pl.DataFrame, res: int) -> pl.DataFrame:
    cell_ids = wx["cell"].unique().to_list()
    lons = {c: center_of(Cell(c, res))[1] for c in cell_ids}
    lo, hi = min(lons.values()), max(lons.values())
    noise = {c: 0.5 + 7.0 * ((lons[c] - lo) / ((hi - lo) or 1.0)) for c in cell_ids}
    rng = np.random.default_rng(0)
    rows = wx.select("cell", "time").to_dict(as_series=False)
    load = [
        100.0 + 18.0 * math.sin(2 * math.pi * t.hour / 24) + rng.normal(0, noise[c])
        for c, t in zip(rows["cell"], rows["time"], strict=True)
    ]
    return pl.DataFrame({"cell": rows["cell"], "time": rows["time"], "layer": "load", "value": load})


def _range(values: pl.Series, mode: str) -> tuple[float, float]:
    if mode == "zero":
        return 0.0, float(values.max())
    if mode == "sym":
        m = float(values.abs().max()) or 1.0
        return -m, m  # diverging around 0: blue = under-predict, red = over-predict
    return float(values.min()), float(values.max())


def twin_map(name: str, preset: dict, start: str, days: int, *, radius=None, res=None) -> dict:
    """One map over a city's H3 grid with both Weather and Energy layers (grouped in a single
    dropdown), so every layer is visible at once. Layers align by canonical cell order."""
    cells, zoom, r = _resolve(preset, radius, res)
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    wx = _weather(cells, s, s + timedelta(days=days - 1))

    day1 = wx.filter(pl.col("time") < s + timedelta(days=1))  # weather layers: first day (24 h)
    hours = day1.select(pl.col("time").unique().sort()).to_series().to_list()
    hdd = day1.with_columns(pl.max_horizontal(18.0 - pl.col("value"), 0.0).alias("value"))

    def wlayer(nm: str, f: pl.DataFrame) -> dict:
        vmin, vmax = float(f["value"].min()), float(f["value"].max())
        return {"name": nm, "unit": "°C", "group": "Weather", "frames": _frames(f, hours, vmin, vmax, "%H:%M")}

    results = verification_frame(GBMForecaster(), build_supervised(_synth_load(wx, r), wx), FEATURE_COLS, alpha=0.1)
    mae = float(results["abs_error"].mean())
    cov = results["covered"].mean()
    res_num = results.with_columns(pl.col("covered").cast(pl.Float64))
    tt = res_num.select(pl.col("time").unique().sort()).to_series().to_list()
    elayers = []
    for field, nm, unit, mode in _ENERGY_LAYERS:
        fl = as_layer(res_num, field)
        vmin, vmax = _range(fl["value"], mode)
        elayers.append({"name": nm, "unit": unit, "group": "Energy",
                        "frames": _frames(fl, tt, vmin, vmax, "%m-%d %H:%M")})

    return {
        "name": name, "subtitle": f"weather + synthetic energy · MAE {mae:.1f} · coverage {cov:.0%}",
        **_view(preset, zoom, r),
        "layers": [wlayer("2m temperature", day1), wlayer("heating degrees", hdd), *elayers],
    }


