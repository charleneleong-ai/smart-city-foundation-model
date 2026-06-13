"""Render a day of Open-Meteo temperature over a city's H3 grid as a 3D WebGL map.

Run: uv run python apps/demo_weather.py
(hits the real Open-Meteo archive API; writes weather_3d.html — open it in a browser)
"""

from datetime import datetime, timezone
from pathlib import Path

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

from render_3d import to_self_contained_html

reg = Registry()
reg.register(OpenMeteoWeatherAdapter())

cells = cells_in_bbox(south=51.46, west=-0.20, north=51.55, east=-0.05, res=8)
day = datetime(2020, 1, 15, tzinfo=timezone.utc)
frame = reg.get("weather.t2m", cells, day, day)
records = h3_layer_records(frame, at=datetime(2020, 1, 15, 12, tzinfo=timezone.utc))

html = to_self_contained_html(
    records,
    lat=51.505,
    lon=-0.12,
    title="London — 2m temperature",
    subtitle="Open-Meteo · 2020-01-15 12:00 UTC · H3 res 8",
    unit="°C",
)
out = Path("weather_3d.html")
out.write_text(html)
print(f"wrote {out} ({len(records)} hexes) — open it in a browser")
