"""Bridge the smart-city fire model -> the Fire-Shield monitoring app.

Runs the macro Palisades fire CA (same spine as viz_fire_3d), picks a firefighter cell the
front reaches mid-animation, and emits the app's `Environment` per CA step — so Fire-Shield's
live risk scores are driven by *this model's* advancing front instead of the app's manual
sliders. Smoke/heat/CO/HCN/visibility rise as the front's neighbours ignite; wind comes
straight from the scenario. CO/HCN are smoke-scaled proxies (the CA has no combustion-gas
channel — see docs/specs/2026-06-20-fire-shield-bridge-design.md).

Run: uv run python apps/export_fireshield_feed.py --perimeter palisades.geojson \
       --out ../fire-shield-google-ai/src/lib/model-feed.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import h3
import typer

from demo_fire import _at, spread_from_weather
from eval_fire import bbox

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.elevation import fetch_elevation
from sctwin.adapters.open_meteo import WEATHER_VARS, OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.geo import cell_of
from sctwin.verify.burn import cells_from_geojson

MAX_CELLS = 1500
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass(deg: float) -> str:
    return _COMPASS[round(deg / 45.0) % 8]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _target_cell(arrival: dict[str, int], maxstep: int) -> str:
    """A cell the front reaches mid-animation, so the feed shows the calm-then-spike arc a
    firefighter would actually experience (not the seed, which burns from step 0)."""
    mid = maxstep / 2.0
    reached = {c: s for c, s in arrival.items() if 0 < s <= maxstep}
    return min(reached, key=lambda c: abs(reached[c] - mid)) if reached else next(iter(arrival))


def _environment(burned_frac: float, on_fire: bool, *, base_temp: float, wind_kph: float,
                 wind_from: str, spread_dir: str) -> dict:
    """Map local fire proximity -> the Fire-Shield app's `EnvData`. `burned_frac` is the share of the
    cell's 2-ring neighbourhood alight; `on_fire` is whether the cell itself has ignited. Temperature
    ramps from near-ambient to searing as the front arrives; toxicGasLevel (CO) is a smoke-scaled
    proxy (the CA has no combustion-gas channel — see the bridge design note)."""
    smoke = _clamp(8 + 82 * burned_frac + (12 if on_fire else 0), 0, 100)
    return {
        "smokeDensity": round(smoke),
        "temperature": round(max(base_temp, 35) + 55 * burned_frac + (180 if on_fire else 0)),
        "windSpeed": round(wind_kph),
        "windDirection": wind_from,            # compass enum, where the wind comes FROM
        "toxicGasLevel": round(5 + 75 * burned_frac + (20 if on_fire else 0)),  # CO ppm proxy
        "visibility": round(_clamp(30 - 28 * burned_frac, 1, 30)),
        "fireSpreadDirection": spread_dir,     # compass enum, where the front travels
    }


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")],
    out: Annotated[Path, typer.Option(help="output JSON feed")] = Path("../fire-shield-google-ai/src/lib/model-feed.json"),
    date: Annotated[str, typer.Option(help="YYYY-MM-DD of the fire weather")] = "2025-01-07",
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    steps: Annotated[int, typer.Option(help="CA spread steps")] = 40,
    spread: Annotated[float, typer.Option(help="0..1 front threshold")] = 0.5,
    seed_lat: Annotated[float, typer.Option(help="ignition latitude")] = 34.0725,
    seed_lon: Annotated[float, typer.Option(help="ignition longitude")] = -118.5425,
    margin: Annotated[float, typer.Option(help="km padding around the perimeter bbox")] = 1.2,
) -> None:
    """Export a per-CA-step Environment feed for the Fire-Shield app from the macro fire model."""
    gj = json.loads(perimeter.read_text())
    _ = cells_from_geojson(gj, res)
    south, west, north, east = bbox(gj)
    d = margin / 111.0
    cells = cells_in_bbox(south - d, west - d, north + d, east + d, res)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS}; raise --res or shrink --margin")

    land = {h for h, e in fetch_elevation(cells).items() if e > 0.0}
    cells = [c for c in cells if c.h3 in land]
    cached = CachingAdapter(OpenMeteoWeatherAdapter(variables=WEATHER_VARS), ".cache/open-meteo-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    print(f"fetching {len(cells)} land cells {date} ...")
    wx = cached.fetch(cells, day, day)
    seed = cell_of(seed_lat, seed_lon, res).h3
    arrival, meta = spread_from_weather(wx, seed, steps=steps, spread_fraction=spread)
    maxstep = max(arrival.values()) if arrival else 0

    target = _target_cell(arrival, maxstep)
    ring = h3.grid_disk(target, 2)
    base_temp = (lambda t: sum(t) / max(len(t), 1))(_at(wx, meta["at"], "t2m"))
    wind_from = _compass(meta["wind_from"])
    spread_dir = _compass((meta["wind_from"] + 180.0) % 360.0)  # smoke/fire travels downwind

    frames = []
    for s in range(maxstep + 1):
        burned = sum(1 for c in ring if arrival.get(c, maxstep + 1) <= s)
        env = _environment(
            burned / len(ring), arrival.get(target, maxstep + 1) <= s,
            base_temp=base_temp, wind_kph=meta["wind_speed"], wind_from=wind_from, spread_dir=spread_dir,
        )
        frames.append({"step": s, "env": env})

    t_lat, t_lng = h3.cell_to_latlng(target)
    feed = {
        "source": "smart-city-foundation-model · macro Palisades fire CA",
        "generated_for": "fire-shield-google-ai",
        "cell": target, "lat": round(t_lat, 5), "lng": round(t_lng, 5),
        "windFrom": wind_from, "windKph": round(meta["wind_speed"]),
        "steps": maxstep, "frames": frames,
    }
    out.write_text(json.dumps(feed, indent=2))
    arrival_at = arrival.get(target)
    print(f"wrote {out} — {len(frames)} frames · target cell front-arrival step "
          f"{arrival_at}/{maxstep} · wind from {wind_from} @ {round(meta['wind_speed'])} km/h")


if __name__ == "__main__":
    typer.run(main)
