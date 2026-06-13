"""Serve the live twin: a deck.gl map that loads tiles around a movable centre on demand.

Run: uv run --extra app python apps/serve.py   (then open http://127.0.0.1:8000)
- click the map to move the centre; the radius slider / res / date re-query /tiles
- downloads are cached in .cache/ so panning reuses prior fetches
"""

import uvicorn

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.service import build_app
from sctwin.registry import Registry

reg = Registry()
reg.register(CachingAdapter(OpenMeteoWeatherAdapter(), ".cache"))
app = build_app(reg)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
