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
from sctwin.deploy import Constraints, FireScenario, deploy, sample_roster
from sctwin.geo import cell_of

MAX_CELLS = 1500
_WILDFIRE_PM25 = 180.0  # representative heavy wildfire-smoke PM2.5 for the deployment scenario
_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass(deg: float) -> str:
    return _COMPASS[round(deg / 45.0) % 8]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


_DAY_START_MIN, _DAY_END_MIN = 6 * 60, 20 * 60  # map the CA steps across an 06:00–20:00 operational day


def _clock(step: int, maxstep: int) -> str:
    """Map a CA step to a wall-clock time over the operational day, so the feed reads as a timeline."""
    mins = round(_DAY_START_MIN + (step / max(maxstep, 1)) * (_DAY_END_MIN - _DAY_START_MIN))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _target_cell(arrival: dict[str, int], maxstep: int) -> str:
    """A cell the front reaches mid-animation, so the feed shows the calm-then-spike arc a
    firefighter would actually experience (not the seed, which burns from step 0). Deterministic:
    among cells closest to the mid step, the H3 id breaks ties so re-runs pick the same cell."""
    mid = maxstep / 2.0
    reached = {c: s for c, s in arrival.items() if 0 < s <= maxstep}
    if not reached:
        return min(arrival)
    return min(sorted(reached), key=lambda c: abs(reached[c] - mid))


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


def run_model(
    perimeter: Path, *, date: str = "2025-01-07", res: int = 8, steps: int = 40,
    spread: float = 0.5, seed_lat: float = 34.0725, seed_lon: float = -118.5425, margin: float = 1.2,
) -> tuple:
    """Run the macro fire CA over the perimeter's land cells. Returns (arrival, meta, wx, seed) — the
    expensive part, computed once; per-cell feeds are then derived cheaply by feed_at_cell()."""
    gj = json.loads(perimeter.read_text())
    south, west, north, east = bbox(gj)
    d = margin / 111.0
    cells = cells_in_bbox(south - d, west - d, north + d, east + d, res)
    if not 0 < len(cells) <= MAX_CELLS:
        raise ValueError(f"{len(cells)} cells — keep 1..{MAX_CELLS}; raise res or shrink margin")
    land = {h for h, e in fetch_elevation(cells).items() if e > 0.0}
    cells = [c for c in cells if c.h3 in land]
    cached = CachingAdapter(OpenMeteoWeatherAdapter(variables=WEATHER_VARS), ".cache/open-meteo-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    wx = cached.fetch(cells, day, day)
    seed = cell_of(seed_lat, seed_lon, res).h3
    arrival, meta = spread_from_weather(wx, seed, steps=steps, spread_fraction=spread)
    return arrival, meta, wx, seed


def feed_at_cell(arrival: dict[str, int], meta: dict, wx, target: str) -> dict:
    """The Fire-Shield `EnvData` feed (one frame per CA step) for a single firefighter cell."""
    maxstep = max(arrival.values()) if arrival else 0
    ring = h3.grid_disk(target, 2)
    temps = _at(wx, meta["at"], "t2m")
    base_temp = sum(temps) / max(len(temps), 1)
    wind_from = _compass(meta["wind_from"])
    spread_dir = _compass((meta["wind_from"] + 180.0) % 360.0)  # smoke/fire travels downwind
    frames = []
    for s in range(maxstep + 1):
        burned = sum(1 for c in ring if arrival.get(c, maxstep + 1) <= s)
        env = _environment(burned / len(ring), arrival.get(target, maxstep + 1) <= s,
                           base_temp=base_temp, wind_kph=meta["wind_speed"], wind_from=wind_from, spread_dir=spread_dir)
        frames.append({"step": s, "clock": _clock(s, maxstep), "env": env})
    t_lat, t_lng = h3.cell_to_latlng(target)
    return {
        "source": "smart-city-foundation-model · macro Palisades fire CA",
        "generated_for": "fire-shield-google-ai",
        "cell": target, "lat": round(t_lat, 5), "lng": round(t_lng, 5),
        "windFrom": wind_from, "windKph": round(meta["wind_speed"]),
        "steps": maxstep, "dayStart": _clock(0, maxstep), "dayEnd": _clock(maxstep, maxstep),
        "frames": frames,
    }


def build_feed(perimeter: Path, *, target_cell: str | None = None, **model_kw) -> dict:
    """Run the model and return the EnvData feed at `target_cell` (default: a cell the front reaches
    mid-animation, giving the calm-then-spike arc)."""
    arrival, meta, wx, _ = run_model(perimeter, **model_kw)
    target = target_cell or _target_cell(arrival, max(arrival.values()) if arrival else 0)
    return feed_at_cell(arrival, meta, wx, target)


def build_deployment(perimeter: Path, **model_kw) -> tuple:
    """Run the model + the deployment engine. Returns (arrival, meta, wx, members): each member is a
    deployed firefighter placed at a cell along the front (on-task ahead, staging at the rear), so the
    app can monitor whichever member is selected on the operator map."""
    arrival, meta, wx, seed = run_model(perimeter, **model_kw)
    maxstep = max(arrival.values()) if arrival else 0
    temps = _at(wx, meta["at"], "t2m")
    scenario = FireScenario(
        cell=seed, fire_type="grass", size=float(maxstep or 1), pm25=_WILDFIRE_PM25,
        temp_c=sum(temps) / max(len(temps), 1), wind_speed=meta["wind_speed"],
        wind_dir=meta["wind_from"], duration_min=240.0,
    )
    # size the sector deployment to the fire (peak burned area), capped so the map stays legible
    peak_acres = len(arrival) * h3.average_hexagon_area(8, unit="km^2") * 247.105
    crew = max(8, min(round(peak_acres / 900), 24))
    plan = deploy(scenario, sample_roster(crew), Constraints(required_capacity=crew * 0.6))
    reached = sorted((c for c, s in arrival.items() if s > 0), key=lambda c: arrival[c])
    order = ([a for a in plan.assignments if a.role != "staging"] +
             [a for a in plan.assignments if a.role == "staging"])  # on-task ahead, staging at the rear
    members = []
    for i, a in enumerate(order):
        cell = reached[i * len(reached) // max(len(order), 1)] if reached else seed
        lat, lng = h3.cell_to_latlng(cell)
        r = plan.per_ff_risk[a.firefighter_id]
        members.append({
            "id": a.firefighter_id, "role": a.role, "ppe": a.ppe,
            "deployRisk": round(r.value, 3), "riskLow": round(r.low, 3), "riskHigh": round(r.high, 3),
            "drivers": {k: round(v, 3) for k, v in r.drivers.items()},  # acute / incident / career
            "cell": cell, "lat": round(lat, 5), "lng": round(lng, 5), "arrival": arrival.get(cell),
        })
    return arrival, meta, wx, members


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
    feed = build_feed(perimeter, date=date, res=res, steps=steps, spread=spread,
                      seed_lat=seed_lat, seed_lon=seed_lon, margin=margin)
    out.write_text(json.dumps(feed, indent=2))
    fr = feed["frames"]
    arrival_at = next((f["step"] for f in fr if f["env"]["smokeDensity"] > fr[0]["env"]["smokeDensity"] + 10), feed["steps"])
    print(f"wrote {out} — {len(fr)} frames · target cell {feed['cell']} front-arrival ~step "
          f"{arrival_at}/{feed['steps']} · wind from {feed['windFrom']} @ {feed['windKph']} km/h")


if __name__ == "__main__":
    typer.run(main)
