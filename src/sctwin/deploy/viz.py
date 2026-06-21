import math
from datetime import datetime, timezone

import h3
import polars as pl

from sctwin.app.render import _ramp, h3_layer_records
from sctwin.deploy.exposure import toxicant_dose
from sctwin.deploy.hazard import FireScenario
from sctwin.deploy.optimise import Plan
from sctwin.deploy.roster import Roster

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)  # single static frame (incident snapshot)
_CANON = {"cell": pl.String, "time": pl.Datetime("us", "UTC"), "layer": pl.String, "value": pl.Float64}


def downwind_alignment(bearing_to_cell_deg: float, wind_dir_deg: float) -> float:
    """1.0 if the cell lies straight downwind of the incident, 0.0 upwind/crosswind. Meteorological
    `wind_dir` is where wind comes FROM, so smoke travels toward `wind_dir + 180`."""
    smoke_dir = (wind_dir_deg + 180.0) % 360.0
    return max(math.cos(math.radians(bearing_to_cell_deg - smoke_dir)), 0.0)


def _bearing(src: str, dst: str) -> float:
    slat, slon = (math.radians(x) for x in h3.cell_to_latlng(src))
    dlat, dlon = (math.radians(x) for x in h3.cell_to_latlng(dst))
    y = math.sin(dlon - slon) * math.cos(dlat)
    x = math.cos(slat) * math.sin(dlat) - math.sin(slat) * math.cos(dlat) * math.cos(dlon - slon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def hazard_surface(scenario: FireScenario, rings: int = 2) -> pl.DataFrame:
    """smoke / heat / dose per H3 cell over the incident's k-ring — smoke skewed downwind, decaying."""
    incident = scenario.cell
    rows: list[dict] = []
    for c in h3.grid_disk(incident, rings):
        dist = h3.grid_distance(incident, c)
        decay = 1.0 / (1.0 + dist)
        align = 1.0 if c == incident else downwind_alignment(_bearing(incident, c), scenario.wind_dir)
        smoke = scenario.pm25 * (0.3 + 0.7 * align) * decay
        heat = scenario.temp_c * decay
        local = FireScenario(c, scenario.fire_type, scenario.size, smoke, heat,
                             scenario.wind_speed, scenario.wind_dir, scenario.duration_min)
        dose = toxicant_dose(local, scenario.duration_min, "standard")
        rows += [{"cell": c, "time": _T0, "layer": lyr, "value": v}
                 for lyr, v in (("smoke", smoke), ("heat", heat), ("dose", dose))]
    return pl.DataFrame(rows, schema=_CANON)


def crew_records(plan: Plan, roster: Roster, scenario: FireScenario) -> list[dict]:
    """One marker per firefighter: position (BA on the incident, staging pulled back north),
    risk + driver breakdown + a green→red colour relative to the plan's worst individual."""
    by_id = {f.id: f for f in roster}
    ilat, ilon = h3.cell_to_latlng(scenario.cell)
    worst = plan.max_individual_risk or 1.0
    recs = []
    for i, a in enumerate(plan.assignments):
        ff, s = by_id[a.firefighter_id], plan.per_ff_risk[a.firefighter_id]
        if a.role == "staging":
            lat, lon = ilat + 0.004, ilon  # display-only offset: held in reserve
        else:
            lat, lon = ilat + 0.0006 * math.cos(i), ilon + 0.0006 * math.sin(i)  # jitter on incident
        recs.append({
            "ff_id": ff.id, "lon": round(lon, 6), "lat": round(lat, 6),
            "role": a.role, "ppe": a.ppe, "rotation": a.time_on_scene_min,
            "age": ff.age, "cardiovascular": ff.cardiovascular, "respiratory": ff.respiratory,
            "career_dose": ff.career_dose,
            "risk": round(s.value, 3), "low": round(s.low, 3), "high": round(s.high, 3),
            "drivers": {k: round(v, 3) for k, v in s.drivers.items()},
            "color": list(_ramp(s.value / worst)),
        })
    return recs


def model_legend(scenario: FireScenario) -> list[dict]:
    """Self-documenting legend: crew colour ramp + every input and which risk driver it feeds.
    Swatch rows carry `color`; caption rows (no color) explain the model and inline the live scenario."""
    s = scenario
    return [
        {"label": "crew marker → combined risk:"},
        {"color": list(_ramp(0.15)), "label": "low"},
        {"color": list(_ramp(0.55)), "label": "moderate"},
        {"color": list(_ramp(1.0)), "label": "high (vs plan worst)"},
        {"label": "risk = acute + incident + career"},
        {"label": "acute ← heat × age/CV/resp/fitness/heat-tol/#conditions"},
        {"label": "incident ← PM2.5 dose × time (resp sensitises)"},
        {"label": "career ← (career dose + this incident)^1.2"},
        {"label": f"scenario: {s.fire_type} fire · size {s.size:g}"},
        {"label": f"PM2.5 {s.pm25:g} · {s.temp_c:g}°C · wind {s.wind_speed:g}km/h@{s.wind_dir:g}° · {s.duration_min:g}min"},
    ]


def deploy_map(scenario: FireScenario, plan: Plan, roster: Roster, *, preset: dict, rings: int = 2) -> dict:
    """Map payload for `to_self_contained_html`: a Fire domain (smoke/heat/dose hexes) + `plan` markers."""
    surf = hazard_surface(scenario, rings)

    def layer(nm: str, lyr: str) -> dict:
        f = surf.filter(pl.col("layer") == lyr)
        vmin, vmax = float(f["value"].min()), float(f["value"].max())
        return {"name": nm, "unit": "", "group": "Fire", "vmin": vmin, "vmax": vmax,
                "frames": [{"label": "now", "records": h3_layer_records(f, _T0, vmin=vmin, vmax=vmax)}]}

    res = h3.get_resolution(scenario.cell)
    edge = 4.0 * h3.average_hexagon_edge_length(res, unit="m")
    return {
        "name": preset.get("name", "Fire"),
        "subtitle": f"{scenario.fire_type} · feasible={plan.feasible} · total risk {plan.total_risk:.2f}",
        "lat": preset["lat"], "lon": preset["lon"], "zoom": preset.get("zoom", 12.5),
        "pitch": preset.get("pitch", 50.0), "elevation_scale": edge,
        "layers": [layer("smoke / PM2.5", "smoke"), layer("heat", "heat"), layer("exposure dose", "dose")],
        "plan": crew_records(plan, roster, scenario),
        "legend": model_legend(scenario),
        "gradient": True,  # keep the hex value-gradient bar visible alongside the model legend
    }
