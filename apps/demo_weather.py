"""Render a day of Open-Meteo temperature over a city's H3 grid as a 3D WebGL map
with a time slider.

Run: uv run python apps/demo_weather.py
(hits the real Open-Meteo archive API; writes weather_3d.html — open it in a browser)
"""

from datetime import datetime, timezone

import polars as pl

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

from render_3d import to_self_contained_html

ABOUT = (
    "Each hexagon is an H3 res-8 cell (~0.7 km²). Colour and bar height both encode "
    "2 m air temperature (blue = cold → red = warm), normalised over the whole day so "
    "frames are comparable. Source: Open-Meteo reanalysis. Drag the slider to step "
    "through the 24 hours; use Toggle 2D/3D to flatten the bars."
)

reg = Registry()
reg.register(OpenMeteoWeatherAdapter())

cells = cells_in_bbox(south=51.46, west=-0.20, north=51.55, east=-0.05, res=8)
day = datetime(2020, 1, 15, tzinfo=timezone.utc)
frame = reg.get("weather.t2m", cells, day, day)  # 24 hourly rows per cell

# normalise color/height on the whole-day range so the slider is comparable across hours
gmin = float(frame["value"].min())
gmax = float(frame["value"].max())
hours = frame.select(pl.col("time").unique().sort()).to_series().to_list()
frames = [
    {"label": h.strftime("%H:%M"), "records": h3_layer_records(frame, at=h, vmin=gmin, vmax=gmax)}
    for h in hours
]

html = to_self_contained_html(
    frames,
    lat=51.505,
    lon=-0.12,
    title="London — 2 m air temperature",
    subtitle="Open-Meteo · 2020-01-15 · H3 res 8 · 24 h",
    about=ABOUT,
    unit="°C",
)
with open("weather_3d.html", "w") as f:
    f.write(html)
print(f"wrote weather_3d.html ({len(cells)} hexes × {len(frames)} hours) — open it in a browser")
