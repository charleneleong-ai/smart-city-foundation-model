"""Macro fire-spread world-model layer (L4) — a deliberately simple, transparent stub.

NOT an operational fire model: no FARSITE/ELMFIRE fuel physics, ember spotting, or sub-30 m
resolution. It gives (1) a per-H3-cell fuel-dryness proxy from the weather adapter's layers
and (2) a wind- and slope-driven cellular-automaton spread step, so the twin can roll out a
macro burned-area / arrival-time surface to backtest against an observed burn perimeter.
Slope is optional (pass a per-cell elevation field); without it the step is wind-only.
"""

import math
from datetime import datetime

import h3
import polars as pl

from sctwin.geo import haversine_km


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def fuel_dryness(t2m_c: float, rh_pct: float, precip_mm: float = 0.0) -> float:
    """Per-cell fuel-dryness / ignitability in [0, 1] (simplified proxy, not the Canadian FWI):
    hotter and drier raises it, recent rain suppresses it. Monotone in each input."""
    heat = _clamp((t2m_c - 5.0) / 30.0)  # 5 C -> 0, 35 C -> 1
    dry = _clamp(1.0 - rh_pct / 100.0)  # RH 100% -> 0, 0% -> 1
    wet = _clamp(1.0 - precip_mm / 5.0)  # >= 5 mm recent rain -> fully suppressed
    return heat * dry * wet


def _bearing(h_from: str, h_to: str) -> float:
    """Initial compass bearing in degrees [0, 360) from one H3 cell centroid to another."""
    lat1, lon1 = (math.radians(v) for v in h3.cell_to_latlng(h_from))
    lat2, lon2 = (math.radians(v) for v in h3.cell_to_latlng(h_to))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def wind_factor(bearing_deg: float, wind_from_deg: float) -> float:
    """Directional spread multiplier in [0, 1]: 1 downwind, 0 upwind, 0.5 crosswind.
    Meteorological `wind_from_deg` is the direction the wind blows FROM; fire runs opposite."""
    wind_to = (wind_from_deg + 180.0) % 360.0
    return 0.5 * (1.0 + math.cos(math.radians(bearing_deg - wind_to)))


def slope_factor(
    h_from: str, h_to: str, elevation: dict[str, float] | None, slope_coeff: float
) -> float:
    """Terrain multiplier in [0.1, 3.0]: fire spreads faster uphill. 1.0 (neutral) when no
    elevation field, no coefficient, or either cell's elevation is unknown. `slope_coeff` scales
    the gradient (rise/run, m per m); positive gradient = uphill = >1."""
    if not elevation or slope_coeff <= 0:
        return 1.0
    e0, e1 = elevation.get(h_from), elevation.get(h_to)
    if e0 is None or e1 is None:
        return 1.0
    dist_m = haversine_km(*h3.cell_to_latlng(h_from), *h3.cell_to_latlng(h_to)) * 1000.0
    gradient = (e1 - e0) / dist_m if dist_m else 0.0
    return _clamp(1.0 + slope_coeff * gradient, 0.1, 3.0)


def spread_step(
    burning: set[str],
    dryness: dict[str, float],
    wind_from_deg: float,
    *,
    wind_speed: float = 20.0,
    base_rate: float = 1.0,
    threshold: float = 0.5,
    wind_ref: float = 40.0,
    elevation: dict[str, float] | None = None,
    slope_coeff: float = 0.0,
) -> set[str]:
    """One macro CA step. Each burning cell tries to ignite its six H3 neighbours; a neighbour
    ignites when `base_rate * speed_factor * wind_factor * slope_factor * neighbour_dryness >
    threshold`. Returns the NEWLY ignited cells (already-burning excluded). Pure & deterministic.
    Slope is neutral unless an `elevation` field and a positive `slope_coeff` are supplied."""
    speed_factor = _clamp(wind_speed / wind_ref)
    ignited: set[str] = set()
    for cell in burning:
        for nb in h3.grid_disk(cell, 1):
            if nb == cell or nb in burning or nb in ignited:
                continue
            directional = wind_factor(_bearing(cell, nb), wind_from_deg)
            terrain = slope_factor(cell, nb, elevation, slope_coeff)
            weight = base_rate * speed_factor * directional * terrain * dryness.get(nb, 0.0)
            if weight > threshold:
                ignited.add(nb)
    return ignited


def simulate(
    seeds: set[str], dryness: dict[str, float], wind_from_deg: float, steps: int, **kw
) -> dict[str, int]:
    """Roll the CA forward up to `steps` times from `seeds`; return each burned cell's ignition
    step (0 for seeds) — a macro arrival-time surface. Stops early once nothing new ignites."""
    arrival = {s: 0 for s in seeds}
    burning = set(seeds)
    for step in range(1, steps + 1):
        fresh = spread_step(burning, dryness, wind_from_deg, **kw)
        if not fresh:
            break
        for c in fresh:
            arrival.setdefault(c, step)
        burning |= fresh
    return arrival


def dryness_field(frame: pl.DataFrame, at: datetime) -> dict[str, float]:
    """Bridge from the weather adapter to the CA: collapse a canonical (cell, time, layer,
    value) frame at one timestamp into per-cell fuel-dryness, reading the t2m / rh / precip
    layers. Requires t2m + rh per cell; precip defaults to 0 (dry) when absent."""
    by_cell: dict[str, dict[str, float]] = {}
    for r in frame.filter(pl.col("time") == at).iter_rows(named=True):
        by_cell.setdefault(r["cell"], {})[r["layer"]] = r["value"]
    return {
        cell: fuel_dryness(v["t2m"], v["rh"], v.get("precip") or 0.0)
        for cell, v in by_cell.items()
    }
