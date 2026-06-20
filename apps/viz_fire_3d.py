"""Animated 3D backtest: the macro fire CA's spread vs the real burn, in the deck.gl viewer.

Unlike demo_fire (one static snapshot of the predicted front), this bakes BOTH the model and
the observed burn into a single animated layer over the fire's H3 cells — one frame per CA
step — so the viewer's Play control walks the front forward over the real Palisades burn on a
satellite basemap. Per cell, per step:
  - hot (yellow->red)  model HIT      (predicted & in the real burn, coloured by arrival step)
  - magenta            model OVER-REACH (predicted, outside the real burn)
  - blue               real burn the model hasn't reached
  - faint              unburned context

Run: uv run python apps/viz_fire_3d.py --perimeter palisades.geojson --out la_fire_3d.html
(fetch the perimeter as in apps/eval_fire.py; honest limits in src/sctwin/fire.py)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import h3
import typer

from demo_fire import crew_overlay, spread_from_weather
from eval_fire import bbox
from render_3d import to_self_contained_html

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.elevation import fetch_elevation
from sctwin.adapters.open_meteo import WEATHER_VARS, OpenMeteoWeatherAdapter
from sctwin.app.cells import cells_in_bbox
from sctwin.deploy import sample_roster
from sctwin.geo import cell_of, center_of
from sctwin.verify.burn import cells_from_geojson, score

MAX_CELLS = 1500
_ABOUT = (
    "Animated macro fire-spread backtest. Press Play to walk the cellular-automaton front "
    "forward step by step over the real burn. Hot (yellow->red) = model hit, magenta = "
    "over-reach, blue = real burn the model missed. A deliberately-simple stub — directional "
    "overlap only, not predictive size skill; no fuel-physics or terrain coupling."
)


def _record(cell: str, step: int | None, t: int, observed: bool, maxstep: int) -> dict:
    """Colour/height for one cell at CA step `t` (step = its ignition step, or None)."""
    burned = step is not None and step <= t
    if burned and observed:  # model hit — early=yellow, late=red
        u = step / maxstep
        color, height, value = [255, int(205 * (1 - u)), 0, 215], 0.55 + 0.45 * (1 - u), float(step)
    elif burned:  # model over-reach (false positive)
        color, height, value = [205, 70, 205, 200], 0.5, float(step)
    elif observed:  # real burn the front hasn't reached
        color, height, value = [44, 127, 184, 150], 0.28, -1.0
    else:  # unburned context
        color, height, value = [90, 92, 104, 26], 0.0, 0.0
    return {"cell": cell, "value": value, "color": color, "height": height}


def build_backtest_map(name: str, cells, observed, arrival, meta, sc, *, res: int) -> dict:
    """One animated layer (frame per CA step) over the full cell universe, recentred on it."""
    ids = sorted(c.h3 for c in cells)  # fixed universe, sorted so the renderer's cell index aligns
    maxstep = max(arrival.values()) if arrival else 1
    frames = [
        {
            "label": f"step {t}/{maxstep}",
            "records": [_record(c, arrival.get(c), t, c in observed, maxstep) for c in ids],
        }
        for t in range(maxstep + 1)
    ]
    centers = [center_of(c) for c in cells]
    lat = sum(la for la, _ in centers) / len(centers)
    lon = sum(lo for _, lo in centers) / len(centers)
    return {
        "name": name,
        "subtitle": f"wind {meta['wind_from']:.0f}° @ {meta['wind_speed']:.0f} km/h · "
        f"IoU {sc['iou']:.2f} · recall {sc['recall']:.2f} · {maxstep} steps — press Play",
        "lat": lat, "lon": lon, "zoom": 11.5, "pitch": 55.0,
        "elevation_scale": 6.0 * h3.average_hexagon_edge_length(res, unit="m"),
        "legend": [  # categorical swatches, shown instead of the gradient bar. Colours mirror _record;
            # hit is a representative of its yellow->red gradient, unburned is brightened from its faint alpha
            {"color": [255, 120, 0], "label": "model front — hit"},
            {"color": [205, 70, 205], "label": "model over-reach"},
            {"color": [44, 127, 184], "label": "real burn — missed"},
            {"color": [120, 122, 134], "label": "unburned"},
        ],
        "layers": [{"name": "fire vs real burn", "unit": "CA step", "vmin": 0.0, "vmax": float(maxstep), "frames": frames}],
    }


def overlay_crew(m: dict, wx, seed: str, arrival: dict[str, int], meta: dict) -> dict:
    """Merge a personalised firefighter deployment onto the backtest map: crew markers + roster
    panel, scored against this fire's own peak-hour wind/heat, advancing with the model front.
    `maxstep` mirrors build_backtest_map's frame count so the crew frames stay index-aligned."""
    maxstep = max(arrival.values()) if arrival else 1
    return {**m, **crew_overlay(wx, seed, arrival, meta["at"], meta["wind_from"], meta["wind_speed"],
                                sample_roster(), maxstep)}


def main(
    perimeter: Annotated[Path, typer.Option(help="observed burn perimeter GeoJSON")],
    out: Annotated[Path, typer.Option(help="output HTML")] = Path("la_fire_3d.html"),
    date: Annotated[str, typer.Option(help="YYYY-MM-DD of the fire weather")] = "2025-01-07",
    res: Annotated[int, typer.Option(help="H3 resolution")] = 8,
    steps: Annotated[int, typer.Option(help="CA spread steps")] = 40,
    spread: Annotated[float, typer.Option(help="0..1 front threshold")] = 0.5,
    seed_lat: Annotated[float, typer.Option(help="ignition latitude")] = 34.0725,
    seed_lon: Annotated[float, typer.Option(help="ignition longitude")] = -118.5425,
    margin: Annotated[float, typer.Option(help="km padding around the perimeter bbox")] = 1.2,
    basemap: Annotated[str, typer.Option(help="satellite or dark")] = "satellite",
    deploy_crew: Annotated[bool, typer.Option("--deploy/--no-deploy", help="overlay a personalised firefighter deployment that advances with the model front")] = True,
    mask_water: Annotated[bool, typer.Option("--mask-water/--no-mask-water", help="drop sea-level (DEM ≤ 0) cells so the model can't predict fire over the ocean")] = True,
) -> None:
    """Render the animated fire-spread-vs-real-burn backtest as a self-contained 3D HTML."""
    gj = json.loads(perimeter.read_text())
    observed = cells_from_geojson(gj, res)
    south, west, north, east = bbox(gj)
    d = margin / 111.0
    cells = cells_in_bbox(south - d, west - d, north + d, east + d, res)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS}; raise --res or shrink --margin")

    if mask_water:  # DEM: drop ocean cells from the model + render universe so no fire over the Pacific
        land = {h for h, e in fetch_elevation(cells).items() if e > 0.0}
        cells = [c for c in cells if c.h3 in land]

    cached = CachingAdapter(OpenMeteoWeatherAdapter(variables=WEATHER_VARS), ".cache/open-meteo-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    print(f"observed {len(observed)} cells · fetching {len(cells)} weather cells {date} ...")
    wx = cached.fetch(cells, day, day)
    seed = cell_of(seed_lat, seed_lon, res).h3
    arrival, meta = spread_from_weather(wx, seed, steps=steps, spread_fraction=spread)
    sc = score(set(arrival), observed)
    m = build_backtest_map("Palisades fire — CA vs real burn", cells, observed, arrival, meta, sc, res=res)
    if deploy_crew:
        m = overlay_crew(m, wx, seed, arrival, meta)
    suffix = " + firefighter deployment" if deploy_crew else ""
    out.write_text(to_self_contained_html([m], title=f"Palisades fire — macro CA vs real burn{suffix}", about=_ABOUT, basemap=basemap))
    n_steps = max(arrival.values()) if arrival else 0
    crew = f" · {len(m['plan'])} crew deployed" if deploy_crew else ""
    print(f"wrote {out} — IoU {sc['iou']:.2f} · recall {sc['recall']:.2f} · {n_steps} steps animated{crew}")


if __name__ == "__main__":
    typer.run(main)
