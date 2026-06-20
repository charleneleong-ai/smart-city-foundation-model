"""Backtest the macro fire CA against an observed burn perimeter (the verification-spine join).

Load a fire perimeter GeoJSON, rasterise it to H3 (observed burn), pull the day's weather over
its bounding box, run the wind-driven CA from the real ignition point, and score predicted vs
observed (IoU / precision / recall / F1).

Get the Palisades perimeter (NIFC WFIGS, ~23,448 ac):
  curl -G "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query" \\
    --data-urlencode "where=poly_IncidentName='Palisades' AND poly_GISAcres>20000" \\
    --data-urlencode "returnGeometry=true" --data-urlencode "outSR=4326" --data-urlencode "f=geojson" > palisades.geojson

Run: uv run python apps/eval_fire.py --perimeter palisades.geojson --date 2025-01-07

Honest limit: the CA's burned *extent* is a tuned free parameter (--steps / --spread), so IoU and
recall mostly reflect directional overlap, not predictive size skill — the stub has no calibrated
rate-of-spread. See src/sctwin/fire.py.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from demo_fire import spread_from_weather

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import WEATHER_VARS, OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.geo import cell_of
from sctwin.verify.burn import cells_from_geojson, score

MAX_CELLS = 1500


def bbox(gj: dict) -> tuple[float, float, float, float]:
    """(south, west, north, east) over every coordinate in a GeoJSON (coords are [lng, lat])."""
    lats: list[float] = []
    lons: list[float] = []

    def walk(x: object) -> None:
        if isinstance(x, list) and x and isinstance(x[0], (int, float)):
            lons.append(x[0])
            lats.append(x[1])
        elif isinstance(x, list):
            for item in x:
                walk(item)

    for feat in gj.get("features", [gj]):
        walk(feat.get("geometry", feat)["coordinates"])
    return min(lats), min(lons), max(lats), max(lons)


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")],
    date: Annotated[str, typer.Option(help="YYYY-MM-DD of the fire weather")] = "2025-01-07",
    res: Annotated[int, typer.Option(help="H3 resolution for the join")] = 8,
    steps: Annotated[int, typer.Option(help="CA spread steps")] = 40,
    spread: Annotated[float, typer.Option(help="0..1 front threshold")] = 0.5,
    seed_lat: Annotated[float, typer.Option(help="ignition latitude")] = 34.0725,
    seed_lon: Annotated[float, typer.Option(help="ignition longitude")] = -118.5425,
    margin: Annotated[float, typer.Option(help="km padding around the perimeter bbox")] = 2.0,
) -> None:
    """Score the macro fire CA against an observed burn perimeter."""
    gj = json.loads(perimeter.read_text())
    observed = cells_from_geojson(gj, res)
    south, west, north, east = bbox(gj)
    d = margin / 111.0
    cells = cells_in_bbox(south - d, west - d, north + d, east + d, res)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS}; raise --res or shrink --margin")

    cached = CachingAdapter(OpenMeteoWeatherAdapter(variables=WEATHER_VARS), ".cache/open-meteo-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    print(f"observed burn {len(observed)} cells · fetching {len(cells)} weather cells {date} ...")
    wx = cached.fetch(cells, day, day)

    seed = cell_of(seed_lat, seed_lon, res).h3
    arrival, meta = spread_from_weather(wx, seed, steps=steps, spread_fraction=spread)
    s = score(set(arrival), observed)
    print(
        f"seed {seed_lat},{seed_lon} · wind from {meta['wind_from']:.0f}° @ {meta['wind_speed']:.0f} km/h\n"
        f"predicted {s['predicted']} · observed {s['observed']} · overlap {s['intersection']}\n"
        f"IoU {s['iou']:.2f} · precision {s['precision']:.2f} · recall {s['recall']:.2f} · F1 {s['f1']:.2f}"
    )


if __name__ == "__main__":
    typer.run(main)
