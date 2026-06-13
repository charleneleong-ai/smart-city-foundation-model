"""Serve the live twin: a deck.gl map that loads tiles around a movable centre on demand.

Run: uv run --extra app python apps/serve.py            (Open-Meteo, per-point)
     WEATHER_SOURCE=era5 uv run --extra app --extra gridded python apps/serve.py   (gridded ERA5)
- click the map to move the centre; the radius slider / res / date re-query /tiles
- downloads are cached in .cache/ so panning reuses prior fetches
- ERA5 (gridded) needs a free CDS API key in ~/.cdsapirc and the `gridded` extra
"""

import os

import uvicorn

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.era5 import ERA5Adapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.service import build_app
from sctwin.registry import Registry

source = ERA5Adapter() if os.environ.get("WEATHER_SOURCE") == "era5" else OpenMeteoWeatherAdapter()
reg = Registry()
reg.register(CachingAdapter(source, ".cache"))
app = build_app(reg)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
