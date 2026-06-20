"""Macro fire-spread demo: pull an LA weather window, derive per-cell fuel dryness, run the
H3 cellular-automaton spread from an ignition seed, and render the arrival-time surface on
the 3D twin viewer.

Run: uv run python apps/demo_fire.py --city la --date 2025-01-07 --radius 8 --res 8
(hits the real Open-Meteo archive API; writes la_fire_3d.html — open it in a browser)

A deliberately-simple MACRO stub, NOT operational fire prediction — see src/sctwin/fire.py.
"""

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import h3
import polars as pl
import typer

from presets import PRESETS, bbox_and_zoom
from render_3d import to_self_contained_html
from twin import _view

from sctwin.adapters.cache import CachingAdapter
from sctwin.adapters.open_meteo import (
    WEATHER_VARS,
    OpenMeteoForecastAdapter,
    OpenMeteoWeatherAdapter,
)
from sctwin.adapters.elevation import fetch_elevation
from sctwin.app.cells import cells_in_bbox
from sctwin.app.render import h3_layer_records
from sctwin.deploy import Constraints, FireScenario, deploy, sample_roster
from sctwin.deploy.roster import Roster
from sctwin.deploy.viz import crew_records
from sctwin.fire import dryness_field, simulate
from sctwin.geo import cell_of

_WILDFIRE_PM25 = 180.0  # representative heavy wildfire-smoke PM2.5 (the CA has no smoke channel)

MAX_CELLS = 1000  # Open-Meteo batches ~100 coords/request and rate-limits by location count
_ABOUT = (
    "Macro fire-spread stub. 'Fuel dryness' is a per-cell ignitability proxy from temperature, "
    "humidity and recent rain; 'fire arrival' is the cellular-automaton burn front from the "
    "ignition seed, pushed downwind and uphill (DEM slope term), normalised to the window's mean "
    "dryness so the relative front shape shows. Colour/height encode the value. NOT an operational "
    "fire model — no fuel-physics, antecedent-drought memory, or ember spotting; relative macro front only."
)


def _at(wx: pl.DataFrame, at: datetime, layer: str) -> list[float]:
    return wx.filter((pl.col("time") == at) & (pl.col("layer") == layer))["value"].to_list()


def _circular_mean_deg(degs: list[float]) -> float:
    """Mean of compass bearings via unit vectors — degrees can't be linearly averaged."""
    s = sum(math.sin(math.radians(d)) for d in degs)
    c = sum(math.cos(math.radians(d)) for d in degs)
    return math.degrees(math.atan2(s, c)) % 360.0


def _peak_hour(wx: pl.DataFrame, times: list[datetime]) -> datetime:
    """The hour with the worst fire weather: highest mean fuel-dryness x mean wind speed."""

    def danger(t: datetime) -> float:
        dry = dryness_field(wx, t).values()
        spd = _at(wx, t, "wind_speed")
        return (sum(dry) / max(len(dry), 1)) * (sum(spd) / max(len(spd), 1))

    return max(times, key=danger)


def spread_from_weather(
    wx: pl.DataFrame,
    seed_cell: str,
    *,
    steps: int = 20,
    spread_fraction: float = 0.5,
    elevation: dict[str, float] | None = None,
    slope_coeff: float = 0.0,
) -> tuple[dict[str, int], dict]:
    """Peak fire-weather hour -> dryness field + mean wind -> normalised wind-driven CA spread
    from `seed_cell`. Returns (arrival_step_by_cell, meta) — the core the demo renders and the
    backtest scores. Spread is normalised to the window's mean dryness so the WIND-DRIVEN front
    shows regardless of absolute (e.g. winter) magnitude; it is the *relative* front, not absolute
    ignition — the instantaneous dryness proxy has no antecedent-drought (FWI) memory."""
    times = wx.select(pl.col("time").unique().sort()).to_series().to_list()
    at = _peak_hour(wx, times)
    dryness = dryness_field(wx, at)
    wind_from = _circular_mean_deg(_at(wx, at, "wind_dir"))
    speeds = _at(wx, at, "wind_speed")
    wind_speed = sum(speeds) / max(len(speeds), 1)
    mean_dry = sum(dryness.values()) / max(len(dryness), 1)
    denom = min(wind_speed / 40.0, 1.0) * mean_dry
    base_rate = 1.0 / denom if denom > 0 else 0.0  # 0 wind or soaked fuel -> no spread
    arrival = simulate(
        {seed_cell}, dryness, wind_from, steps,
        wind_speed=wind_speed, base_rate=base_rate, threshold=spread_fraction,
        elevation=elevation, slope_coeff=slope_coeff,
    )
    return arrival, {"at": at, "wind_from": wind_from, "wind_speed": wind_speed, "dryness": dryness}


def _spread_frames(cells: list[str], arrival: dict[str, int], n_steps: int, at: datetime) -> list[dict]:
    """One frame per CA step so the time slider animates the burn front: at step `s` each cell is
    the bright active front (arrival == s), a dimmer cooling scar (arrival < s), or unburned (0.0).
    Every frame carries all cells in the same order so the viewer's index-aligned colouring holds."""
    frames = []
    for s in range(n_steps + 1):
        values = []
        for c in cells:
            a = arrival.get(c)
            if a is None or a > s:
                values.append(0.0)  # not yet reached
            elif a == s:
                values.append(1.0)  # active front
            else:
                values.append(0.35)  # burn scar, cooling behind the front
        df = pl.DataFrame({"cell": cells, "time": [at] * len(cells), "layer": "v", "value": values})
        frames.append({"label": f"step {s}/{n_steps}", "records": h3_layer_records(df, at=at, vmin=0.0, vmax=1.0)})
    return frames


def _front_centroid(arrival: dict[str, int], step: int, default: tuple[float, float]) -> tuple[float, float]:
    """Mean lat/lon of the cells igniting at `step` (the active front); `default` if this step has
    no new ignitions (so callers can carry the front forward rather than snapping back)."""
    front = [h3.cell_to_latlng(c) for c, a in arrival.items() if a == step]
    if not front:
        return default
    return sum(la for la, _ in front) / len(front), sum(lo for _, lo in front) / len(front)


def _crew_frames(base: list[dict], seed: tuple[float, float], arrival: dict[str, int], n_steps: int) -> list[list[dict]]:
    """Per-CA-step crew records: on-task crew advance toward the active fire front each step (keeping
    their per-crew offset from the seed); staging crew hold at the rear. Only positions change —
    risk/role/colour are the fixed deployment decision."""
    frames, front = [], seed
    for s in range(n_steps + 1):
        front = _front_centroid(arrival, s, front)  # carry the front forward on empty steps
        frames.append([
            r if r["role"] == "staging"
            else {**r, "lat": round(front[0] + (r["lat"] - seed[0]), 6), "lon": round(front[1] + (r["lon"] - seed[1]), 6)}
            for r in base
        ])
    return frames


def build_fire_map(
    name: str,
    wx: pl.DataFrame,
    preset: dict,
    zoom: float,
    res: int,
    *,
    seed_cell: str,
    steps: int = 20,
    spread_fraction: float = 0.5,
    roster: Roster | None = None,
    constraints: Constraints | None = None,
    elevation: dict[str, float] | None = None,
    slope_coeff: float = 0.0,
) -> dict:
    """Run the spread and wrap it as a twin `map` (fuel-dryness + fire-arrival + animated spread).
    With `elevation` + `slope_coeff` the CA is terrain-aware (fire races uphill). If a `roster` is
    given, also overlay a personalised firefighter deployment (crew markers + roster panel) at the
    ignition point, scored against this fire's own wind/heat conditions."""
    arrival, meta = spread_from_weather(wx, seed_cell, steps=steps, spread_fraction=spread_fraction,
                                        elevation=elevation, slope_coeff=slope_coeff)
    at, dryness, wind_from, wind_speed = meta["at"], meta["dryness"], meta["wind_from"], meta["wind_speed"]
    label = at.strftime("%m-%d %H:%MZ")
    n_steps = max(arrival.values()) if arrival else 0

    def single(layer_name: str, unit: str, field: dict[str, float], vmax: float) -> dict:
        df = pl.DataFrame(
            {"cell": list(field), "time": [at] * len(field), "layer": "v", "value": list(field.values())}
        )
        recs = h3_layer_records(df, at=at, vmin=0.0, vmax=vmax) if field else []
        return {"name": layer_name, "unit": unit, "frames": [{"label": label, "records": recs}]}

    m = {
        "name": name,
        "subtitle": f"peak {label} · wind from {wind_from:.0f}° @ {wind_speed:.0f} km/h · "
        f"{len(arrival)}/{len(dryness)} cells burned in {n_steps} steps",
        **_view(preset, zoom, res),
        "layers": [
            # animated burn front FIRST so the time slider / Play work on load
            {"name": "fire spread", "unit": "CA step",
             "frames": _spread_frames(list(dryness), arrival, n_steps, at)},
            single("fire arrival", "CA step", {c: float(s) for c, s in arrival.items()}, float(n_steps or 1)),
            single("fuel dryness", "0..1", dryness, 1.0),
        ],
    }
    if roster is not None:
        temps = _at(wx, at, "t2m")
        scenario = FireScenario(
            cell=seed_cell, fire_type="grass", size=float(n_steps or 1), pm25=_WILDFIRE_PM25,
            temp_c=sum(temps) / max(len(temps), 1), wind_speed=wind_speed, wind_dir=wind_from,
            duration_min=240.0,
        )
        plan = deploy(scenario, roster, constraints or Constraints(required_capacity=4.0))
        base = crew_records(plan, roster, scenario)
        m["plan"] = base  # static crew markers + roster panel over the fire
        m["plan_frames"] = _crew_frames(base, h3.cell_to_latlng(seed_cell), arrival, n_steps)  # advance w/ front
    return m


def main(
    city: Annotated[str, typer.Option(help="preset region (la = Palisades fire)")] = "la",
    date: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2025-01-07",
    radius: Annotated[float | None, typer.Option(help="km around the preset centre")] = 8.0,
    res: Annotated[int | None, typer.Option(help="H3 resolution override (0..15)")] = None,
    steps: Annotated[int, typer.Option(help="CA spread steps to roll out")] = 20,
    spread: Annotated[float, typer.Option(help="0..1 front threshold (lower = wider fan)")] = 0.5,
    source: Annotated[str, typer.Option(help="open-meteo (archive) or open-meteo-forecast")] = "open-meteo",
    seed_lat: Annotated[float | None, typer.Option(help="ignition latitude (default: preset centre)")] = None,
    seed_lon: Annotated[float | None, typer.Option(help="ignition longitude")] = None,
    deploy_crew: Annotated[bool, typer.Option("--deploy/--no-deploy", help="overlay a personalised firefighter deployment at the ignition point")] = True,
    slope: Annotated[float, typer.Option(help="DEM uphill-spread coefficient (0 = flat / wind-only; ~8 = terrain visibly shapes the front)")] = 8.0,
) -> None:
    """Pull LA weather, derive fuel dryness, run the macro CA spread, render to 3D HTML."""
    if city not in PRESETS:
        raise typer.BadParameter(f"--city must be one of {', '.join(sorted(PRESETS))}")
    if source not in ("open-meteo", "open-meteo-forecast"):
        raise typer.BadParameter("--source must be open-meteo or open-meteo-forecast")

    preset = PRESETS[city]
    south, west, north, east, zoom, r = bbox_and_zoom(preset, radius, res)
    cells = cells_in_bbox(south, west, north, east, r)
    if not 0 < len(cells) <= MAX_CELLS:
        raise SystemExit(f"{len(cells)} cells — keep 1..{MAX_CELLS}; adjust --radius/--res")

    adapter = (
        OpenMeteoForecastAdapter(variables=WEATHER_VARS)
        if source == "open-meteo-forecast"
        else OpenMeteoWeatherAdapter(variables=WEATHER_VARS)
    )
    cached = CachingAdapter(adapter, f".cache/{source}-fire")
    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    print(f"fetching {len(cells)} cells {date} via {source} (fire-weather vars, cached) ...")
    wx = cached.fetch(cells, day, day)

    elevation = fetch_elevation(cells) if slope > 0 else None  # DEM for the terrain-aware slope term
    seed = cell_of(seed_lat or preset["lat"], seed_lon or preset["lon"], r).h3
    m = build_fire_map(
        f"{city.upper()} macro fire spread", wx, preset, zoom, r,
        seed_cell=seed, steps=steps, spread_fraction=spread,
        roster=sample_roster() if deploy_crew else None,
        elevation=elevation, slope_coeff=slope,
    )
    suffix = " + firefighter deployment" if deploy_crew else ""
    html = to_self_contained_html([m], title=f"{city.upper()} — macro fire spread{suffix}",
                                  about=_ABOUT, basemap="satellite")  # Esri satellite/terrain overlay
    out = Path(f"{city}_fire_3d.html")
    out.write_text(html)
    crew = f" · {len(m['plan'])} crew deployed" if "plan" in m else ""
    print(f"wrote {out} — {m['subtitle']}{crew}")


if __name__ == "__main__":
    typer.run(main)
