"""Serve the live twin: a deck.gl map that loads tiles around a movable centre on demand.

Run: uv run --extra app python apps/serve.py                       (Open-Meteo, per-point)
     uv run --extra app --extra gridded python apps/serve.py --source era5   (gridded ERA5)
- click the map to move the centre; the radius slider / res / date re-query /tiles
- downloads are cached in .cache/ so panning reuses prior fetches
- ERA5 (gridded) needs the `gridded` extra + a CDS key in ~/.ecmwfdatastoresrc, with the
  ERA5 licence accepted once (Client.accept_licence) — verified live: real data, ~12 C UK
"""

import os
from typing import Annotated

import typer
import uvicorn

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.era5 import ERA5Adapter
from sctwin.adapters.open_meteo import OpenMeteoForecastAdapter, OpenMeteoWeatherAdapter
from sctwin.app.service import build_app
from sctwin.registry import Registry


def _adapter(source: str):
    if source == "era5":
        return ERA5Adapter()
    if source == "open-meteo-forecast":
        return OpenMeteoForecastAdapter()
    return OpenMeteoWeatherAdapter()


def _app(source: str):
    reg = Registry()
    reg.register(CachingAdapter(_adapter(source), f".cache/{source}"))
    return build_app(reg)


app = _app(os.environ.get("WEATHER_SOURCE", "open-meteo"))  # ASGI app for `uvicorn apps.serve:app`


def main(
    host: Annotated[str, typer.Option(help="bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port")] = 8000,
    source: Annotated[str, typer.Option(
        help="open-meteo (archive), open-meteo-forecast (real NWP), or era5")] =
        os.environ.get("WEATHER_SOURCE", "open-meteo"),
) -> None:
    """Serve the live twin — tiles loaded around a movable centre on demand."""
    if source not in ("open-meteo", "open-meteo-forecast", "era5"):
        raise typer.BadParameter("--source must be open-meteo, open-meteo-forecast, or era5")
    uvicorn.run(_app(source), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
