"""Build map payloads for the 3D twin viewer: weather inputs and SP5 verification.

Each builder returns a `map` dict consumed by render_3d.to_self_contained_html — name,
view (lat/lon/zoom/pitch), elevation, and a list of selectable layers (each a time-series
of canonical (cell, time, layer, value) records).
"""

import math
import os
from datetime import datetime, timedelta, timezone

import h3
import numpy as np
import polars as pl

from sctwin.adapters.base import LayerAdapter
from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox, global_cells
from sctwin.app.render import _ramp, h3_layer_records
from sctwin.demand import ev_charging_load
from sctwin.forecast.baselines import GBMForecaster
from sctwin.forecast.features import FEATURE_COLS, build_supervised
from sctwin.geo import Cell, center_of
from sctwin.registry import Registry
from sctwin.verify.results import as_layer, verification_frame

from presets import bbox_and_zoom

MAX_CELLS = 1000  # batched ~100 coords/request; Open-Meteo's free tier rate-limits by location


def _source() -> LayerAdapter:
    """WEATHER_SOURCE: era5 -> gridded reanalysis; open-meteo-forecast -> real NWP (future
    covariate); else Open-Meteo archive (reanalysis, past)."""
    kind = os.environ.get("WEATHER_SOURCE", "open-meteo")
    if kind == "era5":
        from sctwin.adapters.era5 import ERA5Adapter

        return ERA5Adapter()
    if kind == "open-meteo-forecast":
        return OpenMeteoForecastAdapter()
    return OpenMeteoWeatherAdapter()


def _max_cells() -> int:
    return 8000 if os.environ.get("WEATHER_SOURCE") == "era5" else MAX_CELLS  # ERA5 = one grid request

# output field -> (display name, unit, range mode): zero=[0,max]; sym=[-M,M]; auto
_ENERGY_LAYERS = [
    ("y_true", "demand", "load", "auto"),
    ("y_pred", "forecast", "load", "auto"),
    ("abs_error", "|error|", "load", "zero"),
    ("error", "delta (forecast−actual)", "load", "sym"),
    ("covered", "covered", "in/out", "auto"),
]
_WEATHER_FC_LAYERS = [
    ("y_true", "actual", "°C", "auto"),
    ("y_pred", "forecast", "°C", "auto"),
    ("abs_error", "|error|", "°C", "zero"),
    ("covered", "covered", "in/out", "auto"),
]
_EV_LAYERS = [  # the physical-AI consumable: where/when charging demand lands, + forecast & coverage
    ("y_true", "demand", "kW", "zero"),
    ("y_pred", "forecast", "kW", "zero"),
    ("abs_error", "|error|", "kW", "zero"),
    ("covered", "covered", "in/out", "auto"),
]
# forecast a future value from calendar + its own lags only (no concurrent / circular features)
_FORECAST_FEATURES = ["hour", "dow", "month", "y_lag_1", "y_lag_24"]


def _resolve(preset: dict, radius: float | None, res: int | None) -> tuple[list, float, int]:
    south, west, north, east, zoom, r = bbox_and_zoom(preset, radius, res)
    cells = global_cells(r) if preset.get("global") else cells_in_bbox(south, west, north, east, r)
    cap = _max_cells()
    if not 0 < len(cells) <= cap:
        raise SystemExit(f"{len(cells)} cells — keep 1..{cap}; adjust --radius/--res")
    return cells, zoom, r


def _weather(cells: list, start: datetime, end: datetime) -> pl.DataFrame:
    kind = os.environ.get("WEATHER_SOURCE", "open-meteo")
    reg = Registry()
    reg.register(CachingAdapter(_source(), f".cache/{kind}"))  # per-source cache (don't mix grids)
    print(f"fetching {len(cells)} cells {start.date()}..{end.date()} via {kind} (cached) ...")
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


def _verify_layers(results: pl.DataFrame, specs: list, group: str) -> list[dict]:
    res_num = results.with_columns(pl.col("covered").cast(pl.Float64))
    times = res_num.select(pl.col("time").unique().sort()).to_series().to_list()
    out = []
    for field, nm, unit, mode in specs:
        fl = as_layer(res_num, field)
        vmin, vmax = _range(fl["value"], mode)
        out.append({"name": nm, "unit": unit, "group": group, "mode": mode,
                    "frames": _frames(fl, times, vmin, vmax, "%m-%d %H:%M")})
    return out


def unify_ranges(maps: list[dict]) -> None:
    """Recolor each layer on a range shared across all maps so colour/height are absolute
    across months, not self-normalised per month. Mutates in place; assumes shared layer
    structure (all built by `twin_map`). No-op for single-map builds."""
    if len(maps) < 2:
        return
    for j, ref in enumerate(maps[0]["layers"]):
        values = [r["value"] for m in maps for fr in m["layers"][j]["frames"] for r in fr["records"]]
        vmin, vmax = _range(pl.Series(values), ref["mode"])
        span = (vmax - vmin) or 1.0
        for m in maps:
            layer = m["layers"][j]
            layer["vmin"], layer["vmax"] = vmin, vmax  # pin the legend to the shared range too
            for fr in layer["frames"]:
                for r in fr["records"]:
                    t = (r["value"] - vmin) / span
                    r["color"], r["height"] = list(_ramp(t)), t


def twin_map(name: str, preset: dict, start: str, days: int, *, radius=None, res=None) -> dict:
    """One map over a city's H3 grid with Inputs + per-domain forecast Outputs, grouped in a
    single dropdown so every layer is visible at once. Weather is both an input (observed) and
    an output (forecast), each verified — like the energy demand forecast. Layers align by cell."""
    cells, zoom, r = _resolve(preset, radius, res)
    s = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    wx = _weather(cells, s, s + timedelta(days=days - 1))

    # Inputs: observed weather (first day, 24 h)
    day1 = wx.filter(pl.col("time") < s + timedelta(days=1))
    hours = day1.select(pl.col("time").unique().sort()).to_series().to_list()
    hdd = day1.with_columns(pl.max_horizontal(18.0 - pl.col("value"), 0.0).alias("value"))

    def wlayer(nm: str, f: pl.DataFrame) -> dict:
        vmin, vmax = float(f["value"].min()), float(f["value"].max())
        return {"name": nm, "unit": "°C", "group": "Inputs", "mode": "auto",
                "frames": _frames(f, hours, vmin, vmax, "%H:%M")}

    # Output 1: weather forecast (predict t2m from calendar + its own lags)
    wres = verification_frame(GBMForecaster(), build_supervised(wx, wx), _FORECAST_FEATURES, alpha=0.1)
    weather_out = _verify_layers(wres, _WEATHER_FC_LAYERS, "Weather forecast")

    # Output 2: energy demand forecast (synthetic load from weather features)
    eres = verification_frame(GBMForecaster(), build_supervised(_synth_load(wx, r), wx), FEATURE_COLS, alpha=0.1)
    energy_out = _verify_layers(eres, _ENERGY_LAYERS, "Energy forecast")

    # Output 3: EV-charging demand surface — the physical-AI consumable (fleet routing, depot siting)
    evres = verification_frame(GBMForecaster(), build_supervised(ev_charging_load(wx, r), wx), FEATURE_COLS, alpha=0.1)
    ev_out = _verify_layers(evres, _EV_LAYERS, "EV charging")

    w_mae, e_mae = float(wres["abs_error"].mean()), float(eres["abs_error"].mean())
    return {
        "name": name,
        "subtitle": f"{start} · forecast MAE — weather {w_mae:.1f}°C · energy {e_mae:.1f}",
        **_view(preset, zoom, r),
        "layers": [
            wlayer("temperature", day1), wlayer("heating degrees", hdd),
            *weather_out, *energy_out, *ev_out,
        ],
    }


