"""Render one day of Open-Meteo t2m over a city's H3 grid.

Run: uv run --extra app --extra forecast python apps/demo_weather.py
(hits the real Open-Meteo archive API; writes weather_map.html)
"""

from datetime import datetime, timezone

import pydeck as pdk

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.registry import Registry

reg = Registry()
reg.register(OpenMeteoWeatherAdapter())

cells = cells_in_bbox(south=51.46, west=-0.20, north=51.55, east=-0.05, res=7)
day = datetime(2020, 1, 15, tzinfo=timezone.utc)
frame = reg.get("weather.t2m", cells, day, day)
records = h3_layer_records(frame, at=datetime(2020, 1, 15, 12, tzinfo=timezone.utc))

layer = pdk.Layer(
    "H3HexagonLayer",
    records,
    get_hexagon="cell",
    get_fill_color="color",
    pickable=True,
    extruded=False,
)
view = pdk.ViewState(latitude=51.5, longitude=-0.12, zoom=10)
pdk.Deck(layers=[layer], initial_view_state=view, map_style="light").to_html("weather_map.html")
print(f"wrote weather_map.html ({len(records)} hexes)")
