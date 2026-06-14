"""Derived demand channels — the physical-AI-facing predictions an embodied fleet acts on
(charging operators, depot planners, grid controllers plan against demand, not weather)."""

import math

import numpy as np
import polars as pl

from sctwin.geo import Cell, center_of


def ev_charging_load(weather: pl.DataFrame, res: int, *, seed: int = 1) -> pl.DataFrame:
    """EV-charging demand (kW) per (cell, time) derived from 2 m temperature: an evening-peaked
    charging profile (people plug in after the commute), amplified in the cold (more driving +
    earlier returns + battery/heater draw), scaled by a per-cell fleet-size proxy. Canonical
    (cell, time, layer, value); non-negative."""
    cells = weather["cell"].unique().to_list()
    lon = {c: center_of(Cell(c, res))[1] for c in cells}
    lo, hi = min(lon.values()), max(lon.values())
    fleet = {c: 0.5 + 7.0 * ((lon[c] - lo) / ((hi - lo) or 1.0)) for c in cells}  # local EV fleet proxy
    rng = np.random.default_rng(seed)
    rows = weather.select("cell", "time", "value").to_dict(as_series=False)  # value = 2 m temp (°C)
    kw = [
        max(
            0.0,
            fleet[c] * 6.0
            * math.exp(-((t.hour - 19) ** 2) / 18.0)  # Gaussian evening peak (~19:00, ~3 h wide)
            * (1.0 + 0.05 * max(18.0 - temp, 0.0))  # cold amplification via heating-degrees
            + rng.normal(0, 0.4),
        )
        for c, t, temp in zip(rows["cell"], rows["time"], rows["value"], strict=True)
    ]
    return pl.DataFrame({"cell": rows["cell"], "time": rows["time"], "layer": "ev_charging", "value": kw})
