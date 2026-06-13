"""Serve the twin API with the weather adapter wired in.

Run: uv run --extra app python apps/serve.py  (then open http://127.0.0.1:8000/docs)
"""

import uvicorn

from sctwin.adapters.open_meteo import OpenMeteoWeatherAdapter
from sctwin.app.service import build_app
from sctwin.registry import Registry

reg = Registry()
reg.register(OpenMeteoWeatherAdapter())
app = build_app(reg)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
